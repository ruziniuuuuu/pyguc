from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pxr import Usd, UsdShade

import pyguc
from pyguc._gltf import decode_accessor, load


def _rewrite(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _codes(report: pyguc.ValidationReport) -> set[str]:
    return {item.code for item in report.diagnostics}


def _append_accessor(
    tmp_path: Path,
    document: dict[str, Any],
    values: bytes,
    *,
    component_type: int,
    count: int,
    value_type: str,
) -> int:
    path = tmp_path / "mesh.bin"
    binary = path.read_bytes()
    padding = b"\x00" * (-len(binary) % 4)
    offset = len(binary) + len(padding)
    binary += padding + values
    path.write_bytes(binary)
    document["buffers"][0]["byteLength"] = len(binary)
    document["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(values)})
    document["accessors"].append(
        {
            "bufferView": len(document["bufferViews"]) - 1,
            "componentType": component_type,
            "count": count,
            "type": value_type,
        }
    )
    return len(document["accessors"]) - 1


def test_position_requires_minimum_and_maximum(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    del document["accessors"][0]["min"]
    _rewrite(source, document)

    report = pyguc.validate(source)

    assert not report.is_valid
    assert "PG300" in _codes(report)


def test_position_bounds_must_match_decoded_values(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    document["accessors"][0]["max"] = [2.0, 2.0, 2.0]
    _rewrite(source, document)

    report = pyguc.validate(source)

    assert "PG340" in _codes(report)
    diagnostic = next(item for item in report.errors if item.code == "PG340")
    assert diagnostic.pointer == "/accessors/0/max"
    assert diagnostic.suggestion is not None


def test_attribute_and_index_targets_must_match_their_use(
    tmp_path: Path, write_triangle_gltf
) -> None:
    source, document = write_triangle_gltf()
    document["bufferViews"][0]["target"] = 34963
    document["bufferViews"][1]["target"] = 34962
    _rewrite(source, document)

    report = pyguc.validate(source)

    assert {"PG263", "PG269"} <= _codes(report)


def test_mat3_unsigned_byte_uses_four_byte_column_alignment(tmp_path: Path) -> None:
    values = bytes((1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0))
    (tmp_path / "matrix.bin").write_bytes(values)
    source = tmp_path / "matrix.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "matrix.bin", "byteLength": len(values)}],
                "bufferViews": [{"buffer": 0, "byteLength": len(values)}],
                "accessors": [{"bufferView": 0, "componentType": 5121, "count": 1, "type": "MAT3"}],
            }
        ),
        encoding="utf-8",
    )

    asset = load(source)

    np.testing.assert_array_equal(decode_accessor(asset, 0), [[1, 2, 3, 4, 5, 6, 7, 8, 9]])


def test_compact_mat3_unsigned_byte_layout_is_rejected(tmp_path: Path) -> None:
    values = bytes(range(9))
    (tmp_path / "matrix.bin").write_bytes(values)
    source = tmp_path / "matrix.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "matrix.bin", "byteLength": len(values)}],
                "bufferViews": [{"buffer": 0, "byteLength": len(values)}],
                "accessors": [{"bufferView": 0, "componentType": 5121, "count": 1, "type": "MAT3"}],
            }
        ),
        encoding="utf-8",
    )

    assert "PG251" in _codes(pyguc.validate(source))


def test_node_matrix_with_shear_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "shear.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "nodes": [
                    {
                        "matrix": [
                            1,
                            0,
                            0,
                            0,
                            0.5,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert "PG324" in _codes(pyguc.validate(source))


def test_glb_json_nul_padding_is_rejected(tmp_path: Path) -> None:
    json_chunk = b"{}\x00\x00"
    source = tmp_path / "nul-padded.glb"
    source.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_chunk))
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
    )

    assert "PG206" in _codes(pyguc.validate(source))


def test_alpha_cutoff_above_one_is_preserved(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    document["materials"][0].update(alphaMode="MASK", alphaCutoff=2.0)
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle", format="usda")

    stage = Usd.Stage.Open(str(result.asset_path))
    surface = UsdShade.Shader.Get(stage, "/Asset/Materials/material_0000/preview_surface")
    assert surface.GetInput("opacityThreshold").Get() == pytest.approx(2.0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["meshes"][0]["primitives"][0].update(mode=[]),
        lambda document: document["materials"][0].update(alphaMode=[]),
        lambda document: document.update(samplers=[{"wrapS": []}]),
        lambda document: document["accessors"][0].update(
            sparse={
                "count": 1,
                "indices": {"bufferView": 0, "componentType": []},
                "values": {"bufferView": 0},
            }
        ),
    ],
)
def test_unhashable_enum_values_produce_reports(
    tmp_path: Path, write_triangle_gltf, mutate
) -> None:
    source, document = write_triangle_gltf()
    mutate(document)
    _rewrite(source, document)

    report = pyguc.validate(source)

    assert not report.is_valid


def test_huge_numeric_value_produces_a_report(tmp_path: Path) -> None:
    source = tmp_path / "huge.gltf"
    source.write_text(
        '{"asset":{"version":"2.0"},"nodes":[{"translation":[1e10000,0,0]}]}',
        encoding="utf-8",
    )

    report = pyguc.validate(source)

    assert not report.is_valid
    assert "PG229" in _codes(report)


def test_nonfinite_accessor_value_is_rejected(tmp_path: Path, write_triangle_gltf) -> None:
    source, _ = write_triangle_gltf()
    binary = bytearray((tmp_path / "mesh.bin").read_bytes())
    struct.pack_into("<f", binary, 0, float("nan"))
    (tmp_path / "mesh.bin").write_bytes(binary)

    report = pyguc.validate(source)

    assert "PG342" in _codes(report)


def test_normal_and_tangent_constraints_are_reported(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    normal = _append_accessor(
        tmp_path,
        document,
        struct.pack("<9f", *(0.0, 0.0, 0.0) * 3),
        component_type=5126,
        count=3,
        value_type="VEC3",
    )
    tangent = _append_accessor(
        tmp_path,
        document,
        struct.pack("<12f", *(0.0, 0.0, 0.0, 0.0) * 3),
        component_type=5126,
        count=3,
        value_type="VEC4",
    )
    document["meshes"][0]["primitives"][0]["attributes"].update(NORMAL=normal, TANGENT=tangent)
    _rewrite(source, document)

    report = pyguc.validate(source)

    assert {"PG343", "PG344"} <= _codes(report)


def test_primitive_restart_index_is_rejected(tmp_path: Path, write_triangle_gltf) -> None:
    source, _ = write_triangle_gltf()
    binary = bytearray((tmp_path / "mesh.bin").read_bytes())
    struct.pack_into("<H", binary, 36, 65535)
    (tmp_path / "mesh.bin").write_bytes(binary)

    report = pyguc.validate(source)

    assert "PG341" in _codes(report)


def test_header_extension_relationships_are_validated(tmp_path: Path) -> None:
    source = tmp_path / "extensions.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0", "minVersion": "2.1"},
                "extensionsRequired": ["REQ"],
                "extensionsUsed": [1],
            }
        ),
        encoding="utf-8",
    )

    report = pyguc.validate(source)

    assert {"PG301", "PG222"} <= _codes(report)


def test_sparse_layout_constraints_are_validated(tmp_path: Path) -> None:
    data = bytes(range(16))
    (tmp_path / "data.bin").write_bytes(data)
    source = tmp_path / "sparse.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "data.bin", "byteLength": len(data)}],
                "bufferViews": [
                    {"buffer": 0, "byteLength": 4, "byteStride": 4},
                    {"buffer": 0, "byteOffset": 4, "byteLength": 4, "byteStride": 4},
                ],
                "accessors": [
                    {
                        "componentType": 5121,
                        "count": 1,
                        "type": "SCALAR",
                        "sparse": {
                            "count": 2,
                            "indices": {"bufferView": 0, "componentType": 5121},
                            "values": {"bufferView": 1},
                        },
                    },
                    {
                        "componentType": 5121,
                        "count": 1,
                        "type": "SCALAR",
                        "sparse": "bad",
                    },
                    {
                        "componentType": 5121,
                        "count": 1,
                        "type": "SCALAR",
                        "sparse": {"count": 0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = pyguc.validate(source)

    assert {"PG247", "PG248", "PG252", "PG253", "PG254"} <= _codes(report)


def test_material_and_image_edge_constraints(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    image_path = tmp_path / "pixel.png"
    Image.new("RGB", (1, 1), "white").save(image_path)
    document["images"] = [{"uri": "pixel.png", "mimeType": "image/jpeg"}]
    document["textures"] = [{"source": 0}]
    document["materials"][0].update(
        occlusionTexture={"index": 0, "strength": 2.0},
    )
    document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    _rewrite(source, document)

    report = pyguc.validate(source)

    assert {"PG260", "PG286"} <= _codes(report)


def test_material_texture_requires_matching_texcoord(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    Image.new("RGB", (1, 1), "white").save(tmp_path / "pixel.png")
    document["images"] = [{"uri": "pixel.png"}]
    document["textures"] = [{"source": 0}]
    document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    _rewrite(source, document)

    assert "PG259" in _codes(pyguc.validate(source))


def test_primitive_accessor_contracts_are_validated(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    document["accessors"][0]["type"] = "VEC2"
    document["accessors"][0]["min"] = [0, 0]
    document["accessors"][0]["max"] = [1, 1]
    document["accessors"][1].update(type="VEC2", normalized=True, count=2)
    document["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"] = 0
    _rewrite(source, document)

    report = pyguc.validate(source)

    assert {"PG295", "PG298", "PG299", "PG307"} <= _codes(report)


def test_node_graph_and_matrix_contracts_are_validated(tmp_path: Path) -> None:
    source = tmp_path / "nodes.gltf"
    perspective = [1, 0, 0, 0.5, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "nodes": [
                    {"children": [2]},
                    {"children": [2]},
                    {},
                    {"matrix": perspective, "translation": [0, 0, 0]},
                ],
                "scenes": [{"nodes": [0, 0, 2]}],
            }
        ),
        encoding="utf-8",
    )

    report = pyguc.validate(source)

    assert {"PG322", "PG324", "PG332", "PG334", "PG336"} <= _codes(report)


def test_numeric_overflow_and_accessor_bound_shapes_are_reports(tmp_path: Path) -> None:
    source = tmp_path / "numbers.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "nodes": [{"translation": [10**4000, 0, 0]}],
                "accessors": [
                    {
                        "componentType": 5126,
                        "count": 1,
                        "type": "SCALAR",
                        "sparse": {
                            "count": 1,
                            "indices": {"bufferView": 0, "componentType": 5121},
                            "values": {"bufferView": 0},
                        },
                        "min": "bad",
                        "max": [10**4000],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = pyguc.validate(source)

    assert {"PG229", "PG261"} <= _codes(report)
