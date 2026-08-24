from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

import pyguc
import pyguc._gltf as gltf_module


def _rewrite(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _codes(error: pyguc.PygucError) -> set[str]:
    return {item.code for item in error.report.diagnostics}


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"glTF", "PG207"),
        (struct.pack("<4sII", b"BAD!", 2, 12), "PG208"),
        (struct.pack("<4sII", b"glTF", 1, 12), "PG208"),
        (struct.pack("<4sII", b"glTF", 2, 99), "PG209"),
        (struct.pack("<4sII", b"glTF", 2, 12), "PG215"),
        (
            struct.pack("<4sII", b"glTF", 2, 23) + struct.pack("<II", 3, 0x4E4F534A) + b"{} ",
            "PG216",
        ),
    ],
)
def test_malformed_glb_headers_are_rejected(tmp_path: Path, data: bytes, code: str) -> None:
    source = tmp_path / "bad.glb"
    source.write_bytes(data)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert code in _codes(caught.value)


def _glb(*chunks: tuple[int, bytes]) -> bytes:
    body = b"".join(struct.pack("<II", len(data), kind) + data for kind, data in chunks)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (struct.pack("<4sII", b"glTF", 2, 16) + b"1234", "PG210"),
        (_glb((0x4E4F534A, b"{}  "))[:-1], "PG209"),
        (_glb((0x004E4942, b"")), "PG212"),
        (_glb((0x4E4F534A, b"{}  "), (0x4E4F534A, b"{}  ")), "PG213"),
        (
            _glb(
                (0x4E4F534A, b'{"asset":{"version":"2.0"}} '),
                (0x004E4942, b""),
                (0x004E4942, b""),
            ),
            "PG214",
        ),
    ],
)
def test_glb_chunk_structure_errors(tmp_path: Path, data: bytes, code: str) -> None:
    source = tmp_path / "chunks.glb"
    source.write_bytes(data)

    report = pyguc.validate(source)

    assert code in {item.code for item in report.errors}


def test_glb_chunk_cannot_exceed_file(tmp_path: Path) -> None:
    body = struct.pack("<II", 4, 0x4E4F534A)
    source = tmp_path / "chunk.glb"
    source.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)

    assert "PG211" in {item.code for item in pyguc.validate(source).errors}


def test_glb_json_and_bin_have_independent_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = json.dumps({"asset": {"version": "2.0"}}, separators=(",", ":")).encode()
    document += b" " * (-len(document) % 4)
    source = tmp_path / "empty-bin.glb"
    source.write_bytes(_glb((0x4E4F534A, document), (0x004E4942, b"")))

    assert "PG228" in {item.code for item in pyguc.validate(source).errors}

    monkeypatch.setattr(gltf_module, "_MAX_JSON_BYTES", 1)
    assert "PG204" in {item.code for item in pyguc.validate(source).errors}


@pytest.mark.parametrize(
    ("name", "content", "code"),
    [
        ("asset.obj", b"{}", "PG201"),
        ("missing.gltf", None, "PG202"),
        ("array.gltf", b"[]", "PG205"),
    ],
)
def test_source_shape_failures_are_reports(
    tmp_path: Path, name: str, content: bytes | None, code: str
) -> None:
    source = tmp_path / name
    if content is not None:
        source.write_bytes(content)

    assert code in {item.code for item in pyguc.validate(source).errors}


def test_truncated_buffer_and_view_report_both_errors(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    (tmp_path / "mesh.bin").write_bytes(b"short")
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert {"PG226", "PG227", "PG232"} <= _codes(caught.value)


def test_out_of_range_primitive_index_is_rejected(tmp_path: Path, write_triangle_gltf) -> None:
    source, _document = write_triangle_gltf()
    binary = bytearray((tmp_path / "mesh.bin").read_bytes())
    struct.pack_into("<3H", binary, 36, 0, 1, 9)
    (tmp_path / "mesh.bin").write_bytes(binary)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert "PG258" in _codes(caught.value)


def test_attribute_count_mismatch_is_rejected_regardless_of_member_order(
    tmp_path: Path, write_triangle_gltf
) -> None:
    source, document = write_triangle_gltf()
    document["accessors"].append(
        {"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"}
    )
    document["meshes"][0]["primitives"][0]["attributes"] = {"NORMAL": 2, "POSITION": 0}
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert "PG297" in _codes(caught.value)


@pytest.mark.parametrize(
    ("material_change", "code"),
    [
        ({"alphaMode": "MASK", "alphaCutoff": -0.1}, "PG285"),
        ({"pbrMetallicRoughness": {"metallicFactor": 1.1}}, "PG284"),
        ({"pbrMetallicRoughness": {"roughnessFactor": -0.1}}, "PG284"),
        ({"emissiveFactor": [-1.0, 0.0, 0.0]}, "PG287"),
    ],
)
def test_out_of_range_material_values_are_rejected(
    tmp_path: Path, write_triangle_gltf, material_change: dict, code: str
) -> None:
    source, document = write_triangle_gltf()
    document["materials"][0].update(material_change)
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert code in _codes(caught.value)


@pytest.mark.parametrize(
    "camera",
    [
        {"type": "perspective", "perspective": {"yfov": 0.0, "znear": 0.1}},
        {
            "type": "perspective",
            "perspective": {"yfov": 1.0, "aspectRatio": 0.0, "znear": 0.1},
        },
        {
            "type": "orthographic",
            "orthographic": {"xmag": 0.0, "ymag": 1.0, "znear": 0.1, "zfar": 1.0},
        },
        {
            "type": "orthographic",
            "orthographic": {"xmag": 1.0, "ymag": 1.0, "znear": 2.0, "zfar": 1.0},
        },
    ],
)
def test_invalid_camera_values_are_rejected(
    tmp_path: Path, write_triangle_gltf, camera: dict
) -> None:
    source, document = write_triangle_gltf()
    document["cameras"] = [camera]
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert _codes(caught.value) & {"PG314", "PG315"}


def test_duplicate_child_is_rejected(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    document["nodes"].append({})
    document["nodes"][0]["children"] = [1, 1]
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert "PG335" in _codes(caught.value)


def test_malformed_object_shapes_are_collected_without_crashing(tmp_path: Path) -> None:
    source = tmp_path / "shapes.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": [],
                "extensionsRequired": ["REQ", "REQ"],
                "extensionsUsed": ["OTHER", "OTHER"],
                "metadata": {
                    "bad": {"extensions": []},
                    "undeclared": {"extensions": {"VENDOR_payload": {}}},
                },
                "animations": [{}],
                "skins": [{}],
                "buffers": [None, {"byteLength": 0}, {"byteLength": "bad", "uri": 1}],
                "bufferViews": [
                    None,
                    {
                        "buffer": 99,
                        "byteLength": 0,
                        "byteStride": 3,
                        "target": [],
                    },
                ],
                "accessors": [
                    None,
                    {
                        "bufferView": 99,
                        "byteOffset": -1,
                        "componentType": [],
                        "count": 0,
                        "type": [],
                        "normalized": 1,
                        "min": [2],
                        "max": [1],
                        "sparse": [],
                    },
                ],
                "images": [None, {}, {"uri": "x.png", "bufferView": 0}],
                "samplers": [None, {"minFilter": 1, "wrapS": False}],
                "textures": [None, {"source": 99, "sampler": 99}],
                "materials": [
                    None,
                    {
                        "pbrMetallicRoughness": [],
                        "normalTexture": [],
                        "occlusionTexture": [],
                        "emissiveTexture": [],
                        "doubleSided": 0,
                        "alphaCutoff": 0.5,
                    },
                ],
                "meshes": [
                    None,
                    {"weights": [], "primitives": []},
                    {
                        "primitives": [
                            None,
                            {"targets": [{}], "mode": [], "attributes": []},
                        ]
                    },
                ],
                "cameras": [None, {}],
                "nodes": [
                    None,
                    {"skin": 0, "children": "bad", "rotation": [0, 0, 0, 0]},
                ],
                "scenes": [None, {"nodes": "bad"}],
                "scene": 99,
            }
        ),
        encoding="utf-8",
    )

    report = pyguc.validate(source)

    assert not report.is_valid
    assert {
        "PG220",
        "PG223",
        "PG230",
        "PG240",
        "PG260",
        "PG270",
        "PG272",
        "PG280",
        "PG290",
        "PG310",
        "PG320",
        "PG330",
    } <= {item.code for item in report.errors}
