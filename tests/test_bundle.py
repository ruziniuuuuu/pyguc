from __future__ import annotations

from pathlib import Path

import pytest

import pyguc
import pyguc._api as api
import pyguc._bundle as bundle_module
import pyguc._usd as usd_module


def test_authoring_failure_removes_staging_directory(
    tmp_path: Path, write_triangle_gltf, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = write_triangle_gltf()
    destination = tmp_path / "bundle"

    def fail_after_partial_write(asset, root: Path, output_format: str) -> None:
        (root / f"asset.{output_format}").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected authoring failure")

    monkeypatch.setattr(api, "author", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="injected authoring failure"):
        pyguc.convert(source, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle.pyguc-*"))


def test_destination_parent_must_exist(tmp_path: Path, write_triangle_gltf) -> None:
    source, _ = write_triangle_gltf()

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "missing" / "bundle")

    assert caught.value.report.diagnostics[0].code == "PG102"


def test_destination_symlink_is_not_followed(tmp_path: Path, write_triangle_gltf) -> None:
    source, _ = write_triangle_gltf()
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "bundle"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, destination)

    assert caught.value.report.diagnostics[0].code == "PG101"
    assert not list(target.iterdir())


def test_destination_created_during_commit_is_never_replaced(
    tmp_path: Path, write_triangle_gltf, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = write_triangle_gltf()
    destination = tmp_path / "bundle"
    original = bundle_module.rename_no_replace

    def race(staging: Path, target: Path) -> None:
        target.mkdir()
        (target / "winner.txt").write_text("other process", encoding="utf-8")
        original(staging, target)

    monkeypatch.setattr(bundle_module, "rename_no_replace", race)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, destination)

    assert caught.value.report.errors[0].code == "PG101"
    assert (destination / "winner.txt").read_text(encoding="utf-8") == "other process"
    assert not list(tmp_path.glob(".bundle.pyguc-*"))


def test_usda_output_is_deterministic(tmp_path: Path, write_triangle_gltf) -> None:
    source, _ = write_triangle_gltf()

    first = pyguc.convert(source, tmp_path / "first", format="usda")
    second = pyguc.convert(source, tmp_path / "second", format="usda")

    assert first.asset_path.read_bytes() == second.asset_path.read_bytes()


def test_conversion_result_is_immutable(tmp_path: Path, write_triangle_gltf) -> None:
    source, _ = write_triangle_gltf()
    result = pyguc.convert(source, tmp_path / "bundle")

    with pytest.raises(AttributeError):
        result.asset_path = tmp_path / "other.usdc"  # type: ignore[misc]


@pytest.mark.parametrize("kind", ["warning", "error"])
def test_openusd_validation_severity_controls_conversion(
    tmp_path: Path,
    write_triangle_gltf,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    source, _ = write_triangle_gltf()
    destination = tmp_path / "bundle"

    class Issue:
        def HasNoError(self) -> bool:  # noqa: N802 - mirrors OpenUSD
            return False

        def GetIdentifier(self) -> str:  # noqa: N802 - mirrors OpenUSD
            return "test:issue"

        def GetMessage(self) -> str:  # noqa: N802 - mirrors OpenUSD
            return "injected validator issue"

        def GetType(self):  # noqa: N802 - mirrors OpenUSD
            if kind == "warning":
                return usd_module.UsdValidation.ValidationErrorType.Warn
            return usd_module.UsdValidation.ValidationErrorType.Error

    class Context:
        def __init__(self, validators) -> None:
            self.validators = validators

        def Validate(self, stage):  # noqa: N802 - mirrors OpenUSD
            assert stage
            return [Issue()]

    monkeypatch.setattr(usd_module.UsdValidation, "ValidationContext", Context)

    if kind == "warning":
        result = pyguc.convert(source, destination)
        assert [item.code for item in result.report.warnings] == ["PG404"]
        assert result.report.is_valid
    else:
        with pytest.raises(pyguc.PygucError) as caught:
            pyguc.convert(source, destination)
        assert [item.code for item in caught.value.report.errors] == ["PG505"]
        assert not destination.exists()
