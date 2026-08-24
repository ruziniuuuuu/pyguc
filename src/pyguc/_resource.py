"""Race-safe local resource reads and fixed untrusted-input budgets."""

from __future__ import annotations

import base64
import binascii
import os
import stat
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any
from urllib.parse import unquote, urlsplit

_MAX_SINGLE_RESOURCE_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_RESOURCE_BYTES = 1024 * 1024 * 1024
_MAX_JSON_NODES = 1_000_000
_MAX_ACCESSOR_VALUES = 100_000_000
_MAX_IMAGE_PIXELS = 100_000_000
_MAX_OUTPUT_PRIMS = 2_000_000


class Budget:
    """Private fixed budget ledger; callers cannot disable safety limits."""

    __slots__ = ("_accessor_values", "_image_pixels", "_output_prims", "_resource_bytes")

    def __init__(self) -> None:
        self._resource_bytes = 0
        self._accessor_values = 0
        self._image_pixels = 0
        self._output_prims = 0

    def charge_resource(self, byte_count: int) -> None:
        self._resource_bytes += byte_count
        if self._resource_bytes > _MAX_TOTAL_RESOURCE_BYTES:
            raise ValueError("asset exceeds the 1 GiB cumulative resource limit")

    def charge_accessor_values(self, value_count: int) -> None:
        self._accessor_values += value_count
        if self._accessor_values > _MAX_ACCESSOR_VALUES:
            raise ValueError("asset exceeds the 100 million accessor-value limit")

    def charge_image_pixels(self, pixel_count: int) -> None:
        self._image_pixels += pixel_count
        if self._image_pixels > _MAX_IMAGE_PIXELS:
            raise ValueError("asset exceeds the 100 million image-pixel limit")

    def charge_output_prims(self, prim_count: int) -> None:
        self._output_prims += prim_count
        if self._output_prims > _MAX_OUTPUT_PRIMS:
            raise ValueError("asset exceeds the 2 million composed-prim limit")

    def check_json_nodes(self, value: object) -> None:
        count = 0
        pending = [value]
        while pending:
            current = pending.pop()
            count += 1
            if count > _MAX_JSON_NODES:
                raise ValueError("glTF JSON exceeds the 1 million node limit")
            if isinstance(current, dict):
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)


class ResourceReader:
    """Read one source and its local resources through anchored file handles."""

    __slots__ = ("_base_fd", "_budget", "_cache", "_closed", "_source_fd", "base", "source")

    def __init__(self, source: Path, budget: Budget) -> None:
        self.source = source.expanduser().resolve(strict=True)
        self.base = self.source.parent
        self._budget = budget
        self._cache: dict[str, tuple[bytes, str | None]] = {}
        self._base_fd: int | None = None
        self._closed = False
        if os.name == "nt":
            self._source_fd = os.open(self.source, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        else:
            directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            directory_flags |= getattr(os, "O_DIRECTORY", 0)
            self._base_fd = os.open(self.base, directory_flags)
            try:
                self._source_fd = _open_relative(self._base_fd, (self.source.name,))
            except Exception:
                os.close(self._base_fd)
                self._base_fd = None
                raise

    def __enter__(self) -> ResourceReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._source_fd)
        finally:
            if self._base_fd is not None:
                os.close(self._base_fd)
                self._base_fd = None

    def read_source(self) -> bytes:
        self._ensure_open()
        data = _read_bounded(self._source_fd)
        self._budget.charge_resource(len(data))
        return data

    def read_uri(self, uri: str, allowed_data_types: set[str]) -> tuple[bytes, str | None]:
        self._ensure_open()
        cached = self._cache.get(uri)
        if cached is not None:
            return cached
        if uri.startswith("data:"):
            result = self._read_data_uri(uri, allowed_data_types)
        else:
            result = self._read_local_uri(uri)
        self._budget.charge_resource(len(result[0]))
        self._cache[uri] = result
        return result

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("resource reader is closed")

    def _read_data_uri(self, uri: str, allowed_data_types: set[str]) -> tuple[bytes, str | None]:
        header, separator, payload = uri.partition(",")
        parameters = header[5:].split(";")
        if not separator or "base64" not in parameters[1:]:
            raise ValueError("only base64 data URIs are supported")
        mime_type = parameters[0] or "application/octet-stream"
        if mime_type not in allowed_data_types:
            raise ValueError(f"data URI MIME type {mime_type!r} is unsupported")
        try:
            data = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError(f"invalid base64 data URI: {error}") from error
        if len(data) > _MAX_SINGLE_RESOURCE_BYTES:
            raise ValueError("data URI exceeds the 512 MiB resource limit")
        return data, mime_type

    def _read_local_uri(self, uri: str) -> tuple[bytes, str | None]:
        split = urlsplit(uri)
        if split.scheme or split.netloc or split.query or split.fragment:
            raise ValueError("network, scheme, query, and fragment URIs are forbidden")
        decoded = unquote(split.path)
        if not decoded or "\\" in decoded or "\x00" in decoded:
            raise ValueError("resource URI is empty or unsafe")
        pure_path = PurePosixPath(decoded)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError("absolute and parent-directory resource URIs are forbidden")
        if os.name == "nt":
            fd = os.open(
                self.base.joinpath(*pure_path.parts),
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
        else:
            if self._base_fd is None:
                raise RuntimeError("resource reader is closed")
            fd = _open_relative(self._base_fd, pure_path.parts)
        try:
            if os.name == "nt":
                resolved = _windows_final_path(fd)
                if not resolved.is_relative_to(self.base):
                    raise ValueError("resource resolves outside the source directory")
            data = _read_bounded(fd)
        finally:
            os.close(fd)
        return data, None


def _open_relative(base_fd: int, parts: tuple[str, ...]) -> int:
    if not parts:
        raise ValueError("resource URI is empty")
    current_fd = os.dup(base_fd)
    try:
        for part in parts[:-1]:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.open(parts[-1], flags, dir_fd=current_fd)
    finally:
        os.close(current_fd)


def _read_bounded(fd: int) -> bytes:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("resource is not a regular file")
    if metadata.st_size > _MAX_SINGLE_RESOURCE_BYTES:
        raise ValueError("resource exceeds the 512 MiB limit")
    chunks: list[bytes] = []
    remaining = _MAX_SINGLE_RESOURCE_BYTES + 1
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) > _MAX_SINGLE_RESOURCE_BYTES:
        raise ValueError("resource grew beyond the 512 MiB limit while being read")
    return data


def _windows_final_path(fd: int) -> Path:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    windows_msvcrt: Any = msvcrt
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GetFinalPathNameByHandleW
    function.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    function.restype = wintypes.DWORD
    handle = windows_msvcrt.get_osfhandle(fd)
    size = function(handle, None, 0, 0)
    if not size:
        raise windows_ctypes.WinError(windows_ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(size + 1)
    if not function(handle, buffer, len(buffer), 0):
        raise windows_ctypes.WinError(windows_ctypes.get_last_error())
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value).resolve(strict=True)
