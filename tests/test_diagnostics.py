from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

import pyguc
import pyguc._resource as resource_module
from pyguc._gltf import load


def _rewrite(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _codes(error: pyguc.PygucError) -> set[str]:
    return {diagnostic.code for diagnostic in error.report.diagnostics}


def test_duplicate_json_member_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.gltf"
    source.write_text('{"asset":{"version":"2.0","version":"2.0"}}', encoding="utf-8")

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert _codes(caught.value) == {"PG206"}


@pytest.mark.parametrize(
    "uri",
    [
        "../secret.bin",
        "https://example.com/mesh.bin",
        "/absolute/mesh.bin",
        "subdir\\mesh.bin",
        "mesh.bin?query=yes",
        "mesh.bin#fragment",
    ],
)
def test_unsafe_buffer_uri_is_rejected(tmp_path: Path, write_triangle_gltf, uri: str) -> None:
    source, document = write_triangle_gltf()
    document["buffers"][0]["uri"] = uri
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert "PG224" in _codes(caught.value)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation is not always permitted")
def test_symlinked_resource_cannot_escape_source_directory(
    tmp_path: Path, write_triangle_gltf
) -> None:
    source, document = write_triangle_gltf()
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes((tmp_path / "mesh.bin").read_bytes())
    (tmp_path / "escape.bin").symlink_to(outside)
    document["buffers"][0]["uri"] = "escape.bin"
    _rewrite(source, document)

    try:
        with pytest.raises(pyguc.PygucError) as caught:
            pyguc.convert(source, tmp_path / "bundle")
        assert "PG224" in _codes(caught.value)
    finally:
        outside.unlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX dir-fd race regression")
def test_resource_replaced_by_symlink_during_open_is_rejected(
    tmp_path: Path, write_triangle_gltf, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = write_triangle_gltf()
    mesh = tmp_path / "mesh.bin"
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(mesh.read_bytes())
    original = resource_module._open_relative
    raced = False

    def race(base_fd: int, parts: tuple[str, ...]) -> int:
        nonlocal raced
        if parts == ("mesh.bin",) and not raced:
            raced = True
            mesh.unlink()
            mesh.symlink_to(outside)
        return original(base_fd, parts)

    monkeypatch.setattr(resource_module, "_open_relative", race)
    try:
        report = pyguc.validate(source)
    finally:
        outside.unlink()

    assert raced
    assert not report.is_valid
    assert "PG224" in {item.code for item in report.errors}


def test_optional_extension_and_extras_are_structured_warnings(
    tmp_path: Path, write_triangle_gltf, caplog: pytest.LogCaptureFixture
) -> None:
    source, document = write_triangle_gltf()
    document["extensionsUsed"] = ["VENDOR_optional"]
    document["asset"]["extras"] = {"ignored": True}
    _rewrite(source, document)

    with caplog.at_level("WARNING", logger="pyguc"):
        result = pyguc.convert(source, tmp_path / "bundle")

    assert [item.code for item in result.report.diagnostics] == ["PG402", "PG401"]
    assert all(item.suggestion for item in result.report.warnings)
    assert caplog.records == []


@pytest.mark.parametrize(
    "mutation, code",
    [
        (lambda doc: doc.update(extensionsRequired=["KHR_draco_mesh_compression"]), "PG301"),
        (lambda doc: doc.update(animations=[{}]), "PG302"),
        (lambda doc: doc.update(skins=[{}]), "PG302"),
        (lambda doc: doc["meshes"][0]["primitives"][0].update(mode=1), "PG305"),
        (
            lambda doc: doc["meshes"][0]["primitives"][0]["attributes"].update(COLOR_0=0),
            "PG306",
        ),
        (lambda doc: doc.update(samplers=[{"minFilter": 9729}]), "PG303"),
    ],
)
def test_unsupported_features_have_stable_codes(
    tmp_path: Path, write_triangle_gltf, mutation, code: str
) -> None:
    source, document = write_triangle_gltf()
    mutation(document)
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert code in _codes(caught.value)


def test_independent_validation_errors_are_collected(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    document["asset"]["version"] = "1.0"
    document["animations"] = [{}]
    document["buffers"][0]["uri"] = "../outside.bin"
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert {"PG221", "PG302", "PG224"} <= _codes(caught.value)


def test_cyclic_node_graph_is_rejected(tmp_path: Path, write_triangle_gltf) -> None:
    source, document = write_triangle_gltf()
    document["nodes"][0]["children"] = [0]
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert "PG333" in _codes(caught.value)


def test_corrupt_image_is_rejected_even_when_unreferenced(
    tmp_path: Path, write_triangle_gltf
) -> None:
    source, document = write_triangle_gltf()
    (tmp_path / "broken.png").write_bytes(b"not a png")
    document["images"] = [{"uri": "broken.png"}]
    _rewrite(source, document)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert "PG260" in _codes(caught.value)


def test_decompression_bomb_warning_becomes_a_diagnostic(
    tmp_path: Path, write_triangle_gltf, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, document = write_triangle_gltf()
    Image.new("RGB", (2, 1), "white").save(tmp_path / "large.png")
    document["images"] = [{"uri": "large.png"}]
    _rewrite(source, document)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    with pytest.raises(pyguc.PygucError) as caught:
        pyguc.convert(source, tmp_path / "bundle")

    assert "PG260" in _codes(caught.value)


def test_warning_limit_does_not_turn_valid_input_into_an_error(tmp_path: Path) -> None:
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": [f"VENDOR_{index}" for index in range(101)],
    }
    source = tmp_path / "warnings.gltf"
    source.write_text(json.dumps(document), encoding="utf-8")

    asset = load(source)

    assert len(asset.report.diagnostics) == 101
    assert asset.report.diagnostics[0].code == "PG002"
    assert asset.report.diagnostics[0].severity is pyguc.Severity.WARNING


def test_deep_node_hierarchy_does_not_use_python_recursion(tmp_path: Path) -> None:
    node_count = 1_500
    nodes = [{"children": [index + 1]} for index in range(node_count - 1)] + [{}]
    source = tmp_path / "deep.gltf"
    source.write_text(
        json.dumps({"asset": {"version": "2.0"}, "nodes": nodes, "scenes": [{"nodes": [0]}]}),
        encoding="utf-8",
    )

    asset = load(source)

    assert len(asset.nodes) == node_count
