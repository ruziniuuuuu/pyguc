"""Deep public interface for validation and conversion."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._bundle import BundleTransaction
from ._diagnostics import (
    PygucError,
    ValidationReport,
    error_report,
    merge_reports,
)
from ._gltf import load
from ._usd import author

_LOGGER = logging.getLogger("pyguc")


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Paths and validation report produced by a completed conversion."""

    bundle_path: Path
    asset_path: Path
    report: ValidationReport


def validate(source: str | os.PathLike[str]) -> ValidationReport:
    """Validate one local glTF or GLB without writing output.

    Expected input and resource failures are returned as diagnostics. Unexpected
    programming failures are not hidden.
    """

    try:
        asset = load(Path(source))
    except PygucError as error:
        return error.report
    return asset.report


def convert(
    source: str | os.PathLike[str],
    destination_bundle_dir: str | os.PathLike[str],
    *,
    format: Literal["usda", "usdc"] = "usdc",  # noqa: A002 - public interface spelling
) -> ConversionResult:
    """Convert one local glTF file to a new, self-contained OpenUSD bundle."""

    if not isinstance(format, str) or format not in {"usda", "usdc"}:
        raise PygucError(
            "invalid output format",
            error_report(
                "PG104",
                "format must be 'usda' or 'usdc'",
                suggestion="Pass format='usdc' for binary USD or format='usda' for text USD.",
            ),
        )
    source_path = Path(source)
    destination = Path(destination_bundle_dir)
    _LOGGER.info("loading glTF source %s", source_path)
    asset = load(source_path)
    source_report = asset.report
    _LOGGER.debug(
        "loaded %d scenes, %d nodes, %d meshes, and %d materials",
        len(asset.scenes),
        len(asset.nodes),
        len(asset.meshes),
        len(asset.materials),
    )
    try:
        with BundleTransaction(destination) as bundle:
            if bundle.root is None:
                raise RuntimeError("bundle transaction did not create a staging directory")
            _LOGGER.info("authoring OpenUSD bundle")
            output_report = author(asset, bundle.root, format)
            _LOGGER.info("committing validated bundle to %s", destination)
            bundle.commit()
    except PygucError as error:
        raise PygucError(str(error), merge_reports(source_report, error.report)) from error
    except OSError as error:
        raise PygucError(
            "conversion failed",
            merge_reports(
                source_report,
                error_report(
                    "PG500",
                    f"conversion I/O failed: {error}",
                    suggestion="Check filesystem permissions and available space, then retry.",
                ),
            ),
        ) from error
    report = merge_reports(source_report, output_report)
    return ConversionResult(
        bundle_path=destination,
        asset_path=destination / f"asset.{format}",
        report=report,
    )
