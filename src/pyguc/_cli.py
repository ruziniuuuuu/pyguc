"""Thin command-line adapter over pyguc's public interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from ._api import convert, validate
from ._diagnostics import Diagnostic, PygucError


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line adapter and return a process exit status."""

    parser = argparse.ArgumentParser(
        prog="pyguc",
        description="Validate glTF 2.0 or convert it to an OpenUSD bundle.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser("validate", help="validate one .gltf or .glb file")
    validate_parser.add_argument("source", type=Path)

    convert_parser = commands.add_parser("convert", help="create a new OpenUSD bundle")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("destination", type=Path)
    convert_parser.add_argument(
        "--format",
        choices=("usda", "usdc"),
        default="usdc",
        dest="output_format",
        help="USD layer format (default: usdc)",
    )
    arguments = parser.parse_args(argv)

    if arguments.command == "validate":
        report = validate(arguments.source)
        for diagnostic in report.diagnostics:
            _print_diagnostic(diagnostic)
        return 0 if report.is_valid else 1

    try:
        result = convert(
            arguments.source,
            arguments.destination,
            format=arguments.output_format,
        )
    except PygucError as error:
        for diagnostic in error.report.diagnostics:
            _print_diagnostic(diagnostic)
        return 1

    for diagnostic in result.report.diagnostics:
        _print_diagnostic(diagnostic)
    print(result.asset_path)
    return 0


def _print_diagnostic(diagnostic: Diagnostic) -> None:
    location = f" {diagnostic.pointer}" if diagnostic.pointer else ""
    print(
        f"{diagnostic.severity}: {diagnostic.code}{location}: {diagnostic.message}",
        file=sys.stderr,
    )
    if diagnostic.suggestion:
        print(f"  suggestion: {diagnostic.suggestion}", file=sys.stderr)
