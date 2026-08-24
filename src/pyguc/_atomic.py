"""Platform adapters for an atomic directory rename that never replaces."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path
from typing import Any


def rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory, failing if destination already exists."""

    if os.name == "nt":
        _rename_windows(source, destination)
    elif sys.platform == "darwin":
        _rename_macos(source, destination)
    elif sys.platform.startswith("linux"):
        _rename_linux(source, destination)
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unsupported", destination)


def _rename_linux(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        function = library.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOTSUP, "renameat2 is unavailable", destination) from error
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result:
        _raise_posix_error(destination)


def _rename_macos(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    function = library.renamex_np
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(os.fsencode(source), os.fsencode(destination), 0x00000004):
        _raise_posix_error(destination)


def _raise_posix_error(destination: Path) -> None:
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _rename_windows(source: Path, destination: Path) -> None:
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.MoveFileExW
    function.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    function.restype = wintypes.BOOL
    if function(str(source), str(destination), 0):
        return
    error_number = windows_ctypes.get_last_error()
    if error_number in {80, 183}:
        raise FileExistsError(error_number, "destination already exists", destination)
    raise windows_ctypes.WinError(error_number)
