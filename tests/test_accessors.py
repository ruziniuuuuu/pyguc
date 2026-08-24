from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyguc import PygucError
from pyguc._gltf import decode_accessor, load, triangulate


def _load_accessor(
    tmp_path: Path, data: bytes, views: list[dict[str, Any]], accessors: list[dict[str, Any]]
):
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "data.bin", "byteLength": len(data)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    source = tmp_path / "accessor.gltf"
    source.write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "data.bin").write_bytes(data)
    return load(source)


def test_interleaved_accessor_stride(tmp_path: Path) -> None:
    data = struct.pack("<8f", 1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0, 99.0)
    asset = _load_accessor(
        tmp_path,
        data,
        [{"buffer": 0, "byteLength": len(data), "byteStride": 16}],
        [{"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"}],
    )

    np.testing.assert_array_equal(
        decode_accessor(asset, 0), np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    )


def test_normalized_unsigned_accessor(tmp_path: Path) -> None:
    data = bytes([0, 255, 128, 64])
    asset = _load_accessor(
        tmp_path,
        data,
        [{"buffer": 0, "byteLength": 4}],
        [
            {
                "bufferView": 0,
                "componentType": 5121,
                "normalized": True,
                "count": 2,
                "type": "VEC2",
            }
        ],
    )

    np.testing.assert_allclose(
        decode_accessor(asset, 0), [[0.0, 1.0], [128 / 255, 64 / 255]], rtol=1e-6
    )


def test_normalized_signed_accessor_clamps_minimum(tmp_path: Path) -> None:
    data = struct.pack("<4b", -128, 127, -64, 0)
    asset = _load_accessor(
        tmp_path,
        data,
        [{"buffer": 0, "byteLength": len(data)}],
        [
            {
                "bufferView": 0,
                "componentType": 5120,
                "normalized": True,
                "count": 2,
                "type": "VEC2",
            }
        ],
    )

    np.testing.assert_allclose(decode_accessor(asset, 0), [[-1.0, 1.0], [-64 / 127, 0.0]])


def test_sparse_accessor(tmp_path: Path) -> None:
    data = bytes([0, 2]) + struct.pack("<2H", 5, 9)
    asset = _load_accessor(
        tmp_path,
        data,
        [
            {"buffer": 0, "byteOffset": 0, "byteLength": 2},
            {"buffer": 0, "byteOffset": 2, "byteLength": 4},
        ],
        [
            {
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
                "sparse": {
                    "count": 2,
                    "indices": {"bufferView": 0, "componentType": 5121},
                    "values": {"bufferView": 1},
                },
            }
        ],
    )

    np.testing.assert_array_equal(decode_accessor(asset, 0), [5, 0, 9])


def test_triangle_strip_and_fan_winding() -> None:
    np.testing.assert_array_equal(
        triangulate(5, np.asarray([0, 1, 2, 3, 4])),
        [[0, 1, 2], [2, 1, 3], [2, 3, 4]],
    )
    np.testing.assert_array_equal(
        triangulate(6, np.asarray([0, 1, 2, 3, 4])),
        [[0, 1, 2], [0, 2, 3], [0, 3, 4]],
    )
    assert triangulate(5, np.asarray([0, 1])).shape == (0, 3)


def test_sparse_indices_must_be_strictly_increasing(tmp_path: Path) -> None:
    data = bytes([1, 1]) + struct.pack("<2H", 5, 9)
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "data.bin", "byteLength": len(data)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 2},
            {"buffer": 0, "byteOffset": 2, "byteLength": 4},
        ],
        "accessors": [
            {
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
                "sparse": {
                    "count": 2,
                    "indices": {"bufferView": 0, "componentType": 5121},
                    "values": {"bufferView": 1},
                },
            }
        ],
    }
    source = tmp_path / "sparse.gltf"
    source.write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "data.bin").write_bytes(data)

    with pytest.raises(PygucError) as caught:
        load(source)

    assert "PG255" in {item.code for item in caught.value.report.errors}


@given(st.integers(min_value=3, max_value=100))
def test_triangle_modes_always_produce_n_minus_two_faces(count: int) -> None:
    indices = np.arange(count)
    assert triangulate(5, indices).shape == (count - 2, 3)
    assert triangulate(6, indices).shape == (count - 2, 3)
