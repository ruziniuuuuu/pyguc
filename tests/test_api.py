from __future__ import annotations

import json
import logging
from importlib.metadata import version
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade

import pyguc


def test_public_version_matches_package_metadata() -> None:
    assert pyguc.__version__ == version("pyguc")


@pytest.mark.parametrize(("source_format", "output_format"), [("gltf", "usda"), ("glb", "usdc")])
def test_convert_triangle_bundle(
    tmp_path: Path, write_triangle_gltf, source_format: str, output_format: str
) -> None:
    source, _ = write_triangle_gltf(glb=source_format == "glb")
    destination = tmp_path / "bundle"

    result = pyguc.convert(source, destination, format=output_format)

    assert result.bundle_path == destination
    assert result.asset_path == destination / f"asset.{output_format}"
    assert result.report.diagnostics == ()
    stage = Usd.Stage.Open(str(result.asset_path))
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/Asset")
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetPrimAtPath("/Asset/Scenes/scene_0000/node_0000/mesh/primitive_0000")
    mesh = UsdGeom.Mesh.Get(stage, "/Asset/Library/Meshes/mesh_0000/primitive_0000")
    assert mesh.GetFaceVertexCountsAttr().Get() == [3]
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
    bound, _relationship = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    assert bound.GetPath() == Sdf.Path("/Asset/Materials/material_0000")
    assert not stage.GetPrimAtPath("/Asset/Materials/material_default")
    assert stage.GetRootLayer().customLayerData == {}
    assert stage.GetDefaultPrim().GetCustomDataByKey("pyguc:outputVersion") == 1


def test_existing_destination_is_never_overwritten(tmp_path: Path, write_triangle_gltf) -> None:
    source, _ = write_triangle_gltf()
    destination = tmp_path / "bundle"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, destination)

    assert caught.value.report.diagnostics[0].code == "PG101"
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_failure_leaves_no_destination(tmp_path: Path) -> None:
    source = tmp_path / "invalid.gltf"
    source.write_text('{"asset":{"version":"1.0"}}', encoding="utf-8")
    destination = tmp_path / "bundle"

    with pytest.raises(pyguc.PygucError):
        pyguc.convert(source, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle.pyguc-*"))


def test_logging_is_opt_in_and_does_not_log_errors(
    tmp_path: Path, write_triangle_gltf, caplog: pytest.LogCaptureFixture
) -> None:
    source, _ = write_triangle_gltf()
    with caplog.at_level(logging.INFO, logger="pyguc"):
        pyguc.convert(source, tmp_path / "bundle")
    assert [record.levelname for record in caplog.records] == ["INFO", "INFO", "INFO"]
    assert "loading glTF" in caplog.records[0].message


def test_conversion_errors_are_not_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    source = tmp_path / "invalid.gltf"
    source.write_text('{"asset":{"version":"1.0"}}', encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="pyguc"), pytest.raises(pyguc.PygucError):
        pyguc.convert(source, tmp_path / "bundle")

    assert caplog.records == []


def test_invalid_format_fails_before_writing(tmp_path: Path, write_triangle_gltf) -> None:
    source, _ = write_triangle_gltf()
    with pytest.raises(pyguc.PygucError, match="format") as caught:
        pyguc.convert(source, tmp_path / "bundle", format="usd")  # type: ignore[arg-type]
    assert caught.value.report.diagnostics[0].code == "PG104"
    assert not (tmp_path / "bundle").exists()


def test_missing_material_uses_one_default_material(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    del document["meshes"][0]["primitives"][0]["material"]
    source.write_text(json.dumps(document), encoding="utf-8")

    result = pyguc.convert(source, tmp_path / "bundle")

    stage = Usd.Stage.Open(str(result.asset_path))
    default_material = UsdShade.Material.Get(stage, "/Asset/Materials/material_default")
    assert default_material
    mesh = UsdGeom.Mesh.Get(stage, "/Asset/Library/Meshes/mesh_0000/primitive_0000")
    bound, _relationship = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    assert bound.GetPath() == default_material.GetPath()
