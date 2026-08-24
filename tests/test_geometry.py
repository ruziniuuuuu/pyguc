from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom

import pyguc


def _rewrite(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (5, [0, 1, 2, 2, 1, 3]),
        (6, [0, 1, 2, 0, 2, 3]),
    ],
)
def test_strip_and_fan_are_authored_as_triangles(
    tmp_path: Path, write_triangle_gltf, mode: int, expected: list[int]
) -> None:
    source, document = write_triangle_gltf()
    positions = struct.pack("<12f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0)
    indices = struct.pack("<4H", 0, 1, 2, 3)
    binary = positions + indices
    (tmp_path / "mesh.bin").write_bytes(binary)
    document["buffers"][0]["byteLength"] = len(binary)
    document["bufferViews"] = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
        {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)},
    ]
    document["accessors"][0]["count"] = 4
    document["accessors"][1]["count"] = 4
    document["meshes"][0]["primitives"][0]["mode"] = mode
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle", format="usda")

    stage = Usd.Stage.Open(str(result.asset_path))
    mesh = UsdGeom.Mesh.Get(stage, "/Asset/Library/Meshes/mesh_0000/primitive_0000")
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3, 3]
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == expected


def test_nonindexed_triangle_gets_sequential_indices(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    del document["meshes"][0]["primitives"][0]["indices"]
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle")

    stage = Usd.Stage.Open(str(result.asset_path))
    mesh = UsdGeom.Mesh.Get(stage, "/Asset/Library/Meshes/mesh_0000/primitive_0000")
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2]


def test_gltf_matrix_translation_is_preserved(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    document["nodes"][0]["matrix"] = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        2.0,
        3.0,
        4.0,
        1.0,
    ]
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle", format="usda")

    stage = Usd.Stage.Open(str(result.asset_path))
    node = UsdGeom.Xformable.Get(stage, "/Asset/Library/Nodes/node_0000")
    matrix = node.GetLocalTransformation()
    assert Gf.IsClose(matrix.ExtractTranslation(), Gf.Vec3d(2.0, 3.0, 4.0), 1e-8)


def test_trs_order_matches_gltf(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    half_sqrt = 2**-0.5
    document["nodes"][0].update(
        translation=[1.0, 2.0, 3.0],
        rotation=[0.0, 0.0, half_sqrt, half_sqrt],
        scale=[2.0, 3.0, 4.0],
    )
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle", format="usda")

    stage = Usd.Stage.Open(str(result.asset_path))
    matrix = UsdGeom.Xformable.Get(stage, "/Asset/Library/Nodes/node_0000").GetLocalTransformation()
    assert Gf.IsClose(matrix.Transform(Gf.Vec3d(0.0)), Gf.Vec3d(1.0, 2.0, 3.0), 1e-6)
    assert Gf.IsClose(matrix.Transform(Gf.Vec3d(1.0, 0.0, 0.0)), Gf.Vec3d(1.0, 4.0, 3.0), 1e-6)


def test_data_uri_buffer_is_supported(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    binary = (tmp_path / "mesh.bin").read_bytes()
    document["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(
        binary
    ).decode("ascii")
    (tmp_path / "mesh.bin").unlink()
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle")

    assert result.asset_path.is_file()


def test_scenes_are_hidden_when_gltf_has_no_default_scene(
    tmp_path: Path, write_triangle_gltf
) -> None:
    source, document = write_triangle_gltf()
    del document["scene"]
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle")

    stage = Usd.Stage.Open(str(result.asset_path))
    scene = UsdGeom.Imageable.Get(stage, "/Asset/Scenes/scene_0000")
    assert scene.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible


def test_internal_references_are_not_instanceable(tmp_path: Path, write_triangle_gltf) -> None:
    source, _ = write_triangle_gltf()
    result = pyguc.convert(source, tmp_path / "bundle")

    stage = Usd.Stage.Open(str(result.asset_path))
    assert stage.GetPrimAtPath("/Asset/Scenes/scene_0000/node_0000").IsInstanceable() is False
    assert stage.GetPrimAtPath("/Asset/Library/Nodes/node_0000/mesh").IsInstanceable() is False
