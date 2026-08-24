from __future__ import annotations

import ctypes
import errno
from pathlib import Path
from typing import Any

import pytest

import pyguc._atomic as atomic


class _NativeFunction:
    def __init__(self, result: int) -> None:
        self.result = result
        self.argtypes: Any = None
        self.restype: Any = None

    def __call__(self, *arguments: object) -> int:
        return self.result


class _Library:
    def __init__(self, name: str, result: int) -> None:
        setattr(self, name, _NativeFunction(result))


def test_linux_adapter_reports_missing_renameat2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(atomic.ctypes, "CDLL", lambda *args, **kwargs: object())

    with pytest.raises(OSError) as caught:
        atomic._rename_linux(tmp_path / "source", tmp_path / "destination")

    assert caught.value.errno == errno.ENOTSUP


def test_macos_adapter_success_and_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _Library("renamex_np", 0)
    monkeypatch.setattr(atomic.ctypes, "CDLL", lambda *args, **kwargs: library)
    atomic._rename_macos(tmp_path / "source", tmp_path / "destination")

    library.renamex_np.result = 1
    monkeypatch.setattr(atomic.ctypes, "get_errno", lambda: errno.EEXIST)
    with pytest.raises(FileExistsError):
        atomic._rename_macos(tmp_path / "source", tmp_path / "destination")


def test_posix_adapter_preserves_other_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(atomic.ctypes, "get_errno", lambda: errno.EACCES)

    with pytest.raises(OSError) as caught:
        atomic._raise_posix_error(tmp_path / "destination")

    assert caught.value.errno == errno.EACCES


@pytest.mark.parametrize(
    ("result", "error_number", "exception"),
    [(1, 0, None), (0, 183, FileExistsError), (0, 5, OSError)],
)
def test_windows_adapter_maps_native_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: int,
    error_number: int,
    exception: type[BaseException] | None,
) -> None:
    library = _Library("MoveFileExW", result)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: library, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error_number, raising=False)
    monkeypatch.setattr(
        ctypes,
        "WinError",
        lambda value: OSError(value, "native failure"),
        raising=False,
    )

    if exception is None:
        atomic._rename_windows(tmp_path / "source", tmp_path / "destination")
    else:
        with pytest.raises(exception):
            atomic._rename_windows(tmp_path / "source", tmp_path / "destination")


def test_dispatches_to_platform_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    calls: list[str] = []
    monkeypatch.setattr(atomic, "_rename_windows", lambda *_: calls.append("windows"))
    monkeypatch.setattr(atomic, "_rename_macos", lambda *_: calls.append("macos"))
    monkeypatch.setattr(atomic, "_rename_linux", lambda *_: calls.append("linux"))

    monkeypatch.setattr(atomic.os, "name", "nt")
    atomic.rename_no_replace(source, destination)
    monkeypatch.setattr(atomic.os, "name", "posix")
    monkeypatch.setattr(atomic.sys, "platform", "darwin")
    atomic.rename_no_replace(source, destination)
    monkeypatch.setattr(atomic.sys, "platform", "linux")
    atomic.rename_no_replace(source, destination)

    assert calls == ["windows", "macos", "linux"]


def test_unknown_platform_is_explicitly_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(atomic.os, "name", "posix")
    monkeypatch.setattr(atomic.sys, "platform", "plan9")

    with pytest.raises(OSError) as caught:
        atomic.rename_no_replace(tmp_path / "source", tmp_path / "destination")

    assert caught.value.errno == errno.ENOTSUP
