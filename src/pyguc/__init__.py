"""A strict, atomic glTF to OpenUSD converter."""

from __future__ import annotations

import logging
from importlib.metadata import version

from ._api import ConversionResult, convert, validate
from ._diagnostics import Diagnostic, PygucError, Severity, ValidationReport

__version__ = version("pyguc")

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ConversionResult",
    "Diagnostic",
    "PygucError",
    "Severity",
    "ValidationReport",
    "__version__",
    "convert",
    "validate",
]
