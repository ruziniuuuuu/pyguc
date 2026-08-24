"""Atomic output bundle ownership."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ._atomic import rename_no_replace
from ._diagnostics import PygucError, error_report


class BundleTransaction:
    """Own a sibling staging directory and commit it with one rename."""

    __slots__ = ("_committed", "destination", "root")

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.root: Path | None = None
        self._committed = False

    def __enter__(self) -> BundleTransaction:
        if self.destination.exists() or self.destination.is_symlink():
            raise PygucError(
                "destination already exists",
                error_report(
                    "PG101",
                    f"destination already exists: {self.destination}",
                    suggestion="Choose a new destination directory; pyguc never overwrites output.",
                ),
            )

        parent = self.destination.parent
        if not parent.is_dir():
            raise PygucError(
                "invalid destination",
                error_report(
                    "PG102",
                    f"destination parent directory does not exist: {parent}",
                    suggestion="Create the parent directory before converting.",
                ),
            )

        self.root = Path(tempfile.mkdtemp(prefix=f".{self.destination.name}.pyguc-", dir=parent))
        (self.root / "assets").mkdir()
        return self

    def commit(self) -> None:
        root = self._root()
        try:
            rename_no_replace(root, self.destination)
        except FileExistsError as error:
            raise PygucError(
                "destination already exists",
                error_report(
                    "PG101",
                    f"destination appeared before commit: {self.destination}",
                    suggestion="Choose a new destination directory and retry the conversion.",
                ),
            ) from error
        except OSError as error:
            raise PygucError(
                "could not commit output bundle",
                error_report(
                    "PG103",
                    f"could not atomically commit output bundle: {error}",
                    suggestion=(
                        "Use a writable local filesystem that supports atomic no-replace rename."
                    ),
                ),
            ) from error

        self._committed = True

    def _root(self) -> Path:
        if self.root is None:
            raise RuntimeError("bundle transaction has not started")
        return self.root

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._committed and self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
