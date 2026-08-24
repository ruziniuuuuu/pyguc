from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

import pyguc
from tests.conftest import _base_document


def _rewrite(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _append_accessor(
    tmp_path: Path,
    document: dict[str, Any],
    values: bytes,
    *,
    component_type: int,
    count: int,
    value_type: str,
) -> int:
    binary_path = tmp_path / "mesh.bin"
    binary = binary_path.read_bytes()
    padding = b"\x00" * (-len(binary) % 4)
    offset = len(binary) + len(padding)
    binary += padding + values
    binary_path.write_bytes(binary)
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


def _add_png_texture(tmp_path: Path, document: dict[str, Any]) -> bytes:
    image_path = tmp_path / "pixel.png"
    Image.new("RGBA", (2, 1), (64, 128, 255, 192)).save(image_path)
    uv_index = _append_accessor(
        tmp_path,
        document,
        struct.pack("<6f", 0.0, 0.0, 1.0, 0.25, 0.5, 1.0),
        component_type=5126,
        count=3,
        value_type="VEC2",
    )
    document["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = uv_index
    document["images"] = [{"uri": "pixel.png"}, {"uri": "pixel.png"}]
    document["samplers"] = [{"wrapS": 33648, "wrapT": 33071}]
    document["textures"] = [
        {"source": 0, "sampler": 0},
        {"source": 1, "sampler": 0},
    ]
    return image_path.read_bytes()


def test_texture_material_network_and_content_dedup(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    image_bytes = _add_png_texture(tmp_path, document)
    material = document["materials"][0]
    material.update(
        alphaMode="MASK",
        alphaCutoff=0.25,
        doubleSided=True,
        emissiveFactor=[0.5, 0.25, 0.125],
        emissiveTexture={"index": 0},
        occlusionTexture={"index": 1, "strength": 0.75},
    )
    material["pbrMetallicRoughness"].update(
        baseColorTexture={"index": 0}, metallicRoughnessTexture={"index": 1}
    )
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle", format="usda")

    digest = hashlib.sha256(image_bytes).hexdigest()
    assets = list((result.bundle_path / "assets").iterdir())
    assert assets == [result.bundle_path / "assets" / f"{digest}.png"]
    assert assets[0].read_bytes() == image_bytes
    stage = Usd.Stage.Open(str(result.asset_path))
    material_path = "/Asset/Materials/material_0000"
    surface = UsdShade.Shader.Get(stage, f"{material_path}/preview_surface")
    assert surface.GetIdAttr().Get() == "UsdPreviewSurface"
    assert surface.GetInput("diffuseColor").GetAttr().GetConnections()
    assert surface.GetInput("opacityThreshold").Get() == pytest.approx(0.25)
    texture = UsdShade.Shader.Get(stage, f"{material_path}/base_color_texture")
    assert texture.GetInput("file").Get().path == f"assets/{digest}.png"
    assert texture.GetInput("wrapS").Get() == "mirror"
    assert texture.GetInput("wrapT").Get() == "clamp"
    mesh = UsdGeom.Mesh.Get(stage, "/Asset/Library/Meshes/mesh_0000/primitive_0000")
    assert mesh.GetDoubleSidedAttr().Get() is True
    uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()
    np.testing.assert_allclose(uv, [[0.0, 1.0], [1.0, 0.75], [0.5, 0.0]])
    shader_ids = {
        UsdShade.Shader(prim).GetIdAttr().Get()
        for prim in stage.Traverse()
        if prim.IsA(UsdShade.Shader)
    }
    assert shader_ids == {"UsdPreviewSurface", "UsdUVTexture", "UsdPrimvarReader_float2"}
    moved = tmp_path / "moved-bundle"
    result.bundle_path.rename(moved)
    moved_stage = Usd.Stage.Open(str(moved / "asset.usda"))
    moved_texture = UsdShade.Shader.Get(
        moved_stage, f"{material_path}/base_color_texture"
    ).GetInput("file")
    assert Path(moved_texture.Get().resolvedPath).is_relative_to(moved)


def test_data_uri_image_is_extracted_without_reencoding(
    tmp_path: Path, write_triangle_gltf
) -> None:
    source, document = write_triangle_gltf()
    image_bytes = _add_png_texture(tmp_path, document)
    data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    document["images"] = [{"uri": data_uri}, {"uri": data_uri}]
    document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle")

    extracted = list((result.bundle_path / "assets").iterdir())
    assert len(extracted) == 1
    assert extracted[0].suffix == ".png"
    assert extracted[0].read_bytes() == image_bytes


def test_jpeg_keeps_its_bytes_and_extension(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    _add_png_texture(tmp_path, document)
    jpeg_path = tmp_path / "pixel.jpg"
    Image.new("RGB", (2, 1), (64, 128, 255)).save(jpeg_path, format="JPEG")
    jpeg_bytes = jpeg_path.read_bytes()
    document["images"] = [{"uri": "pixel.jpg"}, {"uri": "pixel.jpg"}]
    document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle")

    extracted = list((result.bundle_path / "assets").iterdir())
    assert len(extracted) == 1
    assert extracted[0].suffix == ".jpg"
    assert extracted[0].read_bytes() == jpeg_bytes


def test_glb_buffer_view_image_is_supported(tmp_path: Path) -> None:
    document, binary = _base_document(None)
    image_stream = io.BytesIO()
    Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(image_stream, format="PNG")
    image_bytes = image_stream.getvalue()
    uv_bytes = struct.pack("<6f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    binary += b"\x00" * (-len(binary) % 4)
    uv_offset = len(binary)
    binary += uv_bytes
    image_offset = len(binary)
    binary += image_bytes
    document["buffers"][0]["byteLength"] = len(binary)
    document["bufferViews"].extend(
        [
            {"buffer": 0, "byteOffset": uv_offset, "byteLength": len(uv_bytes)},
            {"buffer": 0, "byteOffset": image_offset, "byteLength": len(image_bytes)},
        ]
    )
    document["accessors"].append(
        {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"}
    )
    document["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = 2
    document["images"] = [{"bufferView": 3, "mimeType": "image/png"}]
    document["textures"] = [{"source": 0}]
    document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    json_bytes = json.dumps(document, separators=(",", ":")).encode()
    json_chunk = json_bytes + b" " * (-len(json_bytes) % 4)
    bin_chunk = binary + b"\x00" * (-len(binary) % 4)
    length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    source = tmp_path / "embedded.glb"
    source.write_bytes(
        struct.pack("<4sII", b"glTF", 2, length)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(bin_chunk), 0x004E4942)
        + bin_chunk
    )

    result = pyguc.convert(source, tmp_path / "bundle")

    extracted = list((result.bundle_path / "assets").iterdir())
    assert len(extracted) == 1
    assert extracted[0].read_bytes() == image_bytes


def test_normal_map_authors_tangent_frame(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    _add_png_texture(tmp_path, document)
    normal_index = _append_accessor(
        tmp_path,
        document,
        struct.pack("<9f", *(0.0, 0.0, 1.0) * 3),
        component_type=5126,
        count=3,
        value_type="VEC3",
    )
    tangent_index = _append_accessor(
        tmp_path,
        document,
        struct.pack("<12f", *(1.0, 0.0, 0.0, 1.0) * 3),
        component_type=5126,
        count=3,
        value_type="VEC4",
    )
    attributes = document["meshes"][0]["primitives"][0]["attributes"]
    attributes.update(NORMAL=normal_index, TANGENT=tangent_index)
    document["materials"][0]["normalTexture"] = {"index": 0}
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle")

    stage = Usd.Stage.Open(str(result.asset_path))
    mesh = UsdGeom.Mesh.Get(stage, "/Asset/Library/Meshes/mesh_0000/primitive_0000")
    primvars = UsdGeom.PrimvarsAPI(mesh)
    np.testing.assert_allclose(primvars.GetPrimvar("tangents").Get(), [[1, 0, 0]] * 3)
    np.testing.assert_allclose(primvars.GetPrimvar("bitangents").Get(), [[0, 1, 0]] * 3)
    normal_input = UsdShade.Shader.Get(
        stage, "/Asset/Materials/material_0000/preview_surface"
    ).GetInput("normal")
    assert normal_input.GetAttr().GetConnections()


def test_normal_map_without_tangents_is_rejected(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    _add_png_texture(tmp_path, document)
    document["materials"][0]["normalTexture"] = {"index": 0}
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert "PG309" in {item.code for item in caught.value.report.diagnostics}


def test_nondefault_normal_scale_is_rejected(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    _add_png_texture(tmp_path, document)
    document["materials"][0]["normalTexture"] = {"index": 0, "scale": 0.5}
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert "PG309" in {item.code for item in caught.value.report.diagnostics}


def test_all_scenes_cameras_and_unreferenced_nodes_are_preserved(
    tmp_path: Path, write_triangle_gltf
) -> None:
    source, document = write_triangle_gltf()
    document["cameras"] = [
        {
            "name": "Perspective camera",
            "type": "perspective",
            "perspective": {"yfov": 1.0, "aspectRatio": 1.5, "znear": 0.1, "zfar": 100.0},
        },
        {
            "name": "Orthographic camera",
            "type": "orthographic",
            "orthographic": {"xmag": 2.0, "ymag": 1.0, "znear": 0.1, "zfar": 10.0},
        },
    ]
    document["nodes"] = [
        {"name": "Moved mesh", "mesh": 0, "translation": [1.0, 2.0, 3.0]},
        {"camera": 0},
        {"name": "Unused camera", "camera": 1},
        {"name": "Unused mesh", "mesh": 0},
    ]
    document["scenes"] = [
        {"name": "Main", "nodes": [0, 1]},
        {"name": "Alternate", "nodes": [3]},
    ]
    _rewrite(source, document)

    result = pyguc.convert(source, tmp_path / "bundle", format="usda")

    stage = Usd.Stage.Open(str(result.asset_path))
    assert stage.GetPrimAtPath("/Asset/Scenes/scene_0000/node_0000/mesh")
    assert stage.GetPrimAtPath("/Asset/Scenes/scene_0001/node_0003/mesh")
    assert stage.GetPrimAtPath("/Asset/Library/Nodes/node_0002/camera")
    assert (
        UsdGeom.Imageable.Get(stage, "/Asset/Scenes/scene_0000").GetVisibilityAttr().Get()
        == UsdGeom.Tokens.inherited
    )
    assert (
        UsdGeom.Imageable.Get(stage, "/Asset/Scenes/scene_0001").GetVisibilityAttr().Get()
        == UsdGeom.Tokens.invisible
    )
    assert (
        UsdGeom.Camera.Get(stage, "/Asset/Library/Cameras/camera_0000").GetProjectionAttr().Get()
        == UsdGeom.Tokens.perspective
    )
    assert (
        UsdGeom.Camera.Get(stage, "/Asset/Library/Cameras/camera_0001").GetProjectionAttr().Get()
        == UsdGeom.Tokens.orthographic
    )
