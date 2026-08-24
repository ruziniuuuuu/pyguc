from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from pyguc._cli import main


def test_cli_converts_and_prints_only_the_asset_path(
    tmp_path: Path, write_triangle_gltf, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _ = write_triangle_gltf()
    destination = tmp_path / "bundle"

    status = main(["convert", str(source), str(destination), "--format", "usda"])

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == f"{destination / 'asset.usda'}\n"
    assert captured.err == ""
    assert (destination / "asset.usda").is_file()


def test_cli_prints_diagnostics_and_returns_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "invalid.gltf"
    source.write_text('{"asset":{"version":"1.0"}}', encoding="utf-8")

    status = main(["convert", str(source), str(tmp_path / "bundle")])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "error: PG221 /asset/version: only glTF 2.0 is supported" in captured.err


def test_cli_prints_success_warnings_to_stderr(
    tmp_path: Path, write_triangle_gltf, capsys: pytest.CaptureFixture[str]
) -> None:
    source, document = write_triangle_gltf()
    document["asset"]["extras"] = {"ignored": True}
    source.write_text(json.dumps(document), encoding="utf-8")

    status = main(["convert", str(source), str(tmp_path / "bundle")])

    captured = capsys.readouterr()
    assert status == 0
    assert "warning: PG402: extras are ignored throughout the asset" in captured.err
    assert "suggestion:" in captured.err


def test_cli_validate_returns_report_without_creating_a_bundle(
    tmp_path: Path, write_triangle_gltf, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _ = write_triangle_gltf()

    status = main(["validate", str(source)])

    assert status == 0
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "bundle").exists()


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])

    assert caught.value.code == 0
    assert capsys.readouterr().out == f"pyguc {version('pyguc')}\n"


def test_python_module_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pyguc", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "usage: pyguc" in completed.stdout
