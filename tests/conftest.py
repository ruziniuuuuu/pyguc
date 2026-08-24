from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def _base_document(buffer_uri: str | None = "mesh.bin") -> tuple[dict[str, Any], bytes]:
    binary = struct.pack(
        "<9f3H",
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
        1,
        2,
    )
    buffer: dict[str, Any] = {"byteLength": len(binary)}
    if buffer_uri is not None:
        buffer["uri"] = buffer_uri
    document = {
        "asset": {"version": "2.0", "generator": "pyguc tests"},
        "buffers": [buffer],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "materials": [
            {
                "name": "Blue paint",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.1, 0.2, 0.8, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.4,
                },
            }
        ],
        "meshes": [
            {
                "name": "Triangle",
                "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}],
            }
        ],
        "nodes": [{"name": "Root triangle", "mesh": 0}],
        "scenes": [{"name": "Main scene", "nodes": [0]}],
        "scene": 0,
    }
    return document, binary


@pytest.fixture
def write_triangle_gltf(tmp_path: Path) -> Callable[..., tuple[Path, dict[str, Any]]]:
    def write(*, glb: bool = False) -> tuple[Path, dict[str, Any]]:
        document, binary = _base_document(None if glb else "mesh.bin")
        if glb:
            encoded = json.dumps(document, separators=(",", ":")).encode()
            json_chunk = encoded + b" " * (-len(encoded) % 4)
            bin_chunk = binary + b"\x00" * (-len(binary) % 4)
            length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
            path = tmp_path / "triangle.glb"
            path.write_bytes(
                struct.pack("<4sII", b"glTF", 2, length)
                + struct.pack("<II", len(json_chunk), 0x4E4F534A)
                + json_chunk
                + struct.pack("<II", len(bin_chunk), 0x004E4942)
                + bin_chunk
            )
        else:
            path = tmp_path / "triangle.gltf"
            path.write_text(json.dumps(document), encoding="utf-8")
            (tmp_path / "mesh.bin").write_bytes(binary)
        return path, document

    return write
