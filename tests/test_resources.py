from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from PIL import Image

import pyguc
import pyguc._resource as resource_module
from pyguc._resource import Budget, ResourceReader


def _codes(report: pyguc.ValidationReport) -> set[str]:
    return {item.code for item in report.diagnostics}


def test_resource_cache_charges_repeated_uri_once(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.gltf"
    source.write_bytes(b"{}")
    (tmp_path / "shared.bin").write_bytes(b"1234")
    (tmp_path / "other.bin").write_bytes(b"x")
    monkeypatch.setattr(resource_module, "_MAX_TOTAL_RESOURCE_BYTES", 6)
    budget = Budget()

    with ResourceReader(source, budget) as resources:
        assert resources.read_source() == b"{}"
        first = resources.read_uri("shared.bin", {"application/octet-stream"})
        second = resources.read_uri("shared.bin", {"application/octet-stream"})
        assert first is second
        try:
            resources.read_uri("other.bin", {"application/octet-stream"})
        except ValueError as error:
            assert "cumulative resource limit" in str(error)
        else:
            raise AssertionError("distinct resource must consume the remaining budget")


def test_json_node_budget_has_stable_diagnostic(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "nodes.gltf"
    source.write_text('{"asset":{"version":"2.0"}}', encoding="utf-8")
    monkeypatch.setattr(resource_module, "_MAX_JSON_NODES", 2)

    report = pyguc.validate(source)

    assert "PG217" in _codes(report)


def test_accessor_value_budget_has_stable_diagnostic(
    tmp_path: Path, write_triangle_gltf, monkeypatch
) -> None:
    source, _ = write_triangle_gltf()
    monkeypatch.setattr(resource_module, "_MAX_ACCESSOR_VALUES", 8)

    report = pyguc.validate(source)

    assert "PG245" in _codes(report)


def test_image_pixel_budget_has_stable_diagnostic(
    tmp_path: Path, write_triangle_gltf, monkeypatch
) -> None:
    source, document = write_triangle_gltf()
    Image.new("RGB", (2, 1), "white").save(tmp_path / "two.png")
    document["images"] = [{"uri": "two.png"}]
    source.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(resource_module, "_MAX_IMAGE_PIXELS", 1)

    report = pyguc.validate(source)

    assert "PG260" in _codes(report)


def test_output_prim_budget_has_stable_diagnostic(
    tmp_path: Path, write_triangle_gltf, monkeypatch
) -> None:
    source, _ = write_triangle_gltf()
    monkeypatch.setattr(resource_module, "_MAX_OUTPUT_PRIMS", 1)

    report = pyguc.validate(source)

    assert "PG346" in _codes(report)


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("data:application/octet-stream,AAAA", "base64"),
        ("data:text/plain;base64,QQ==", "MIME type"),
        ("data:application/octet-stream;base64,not!base64", "invalid base64"),
    ],
)
def test_invalid_data_uris_are_rejected(tmp_path: Path, uri: str, message: str) -> None:
    source = tmp_path / "source.gltf"
    source.write_bytes(b"{}")

    with (
        ResourceReader(source, Budget()) as resources,
        pytest.raises(ValueError, match=message),
    ):
        resources.read_uri(uri, {"application/octet-stream"})


def test_data_uri_obeys_single_resource_limit(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.gltf"
    source.write_bytes(b"{}")
    monkeypatch.setattr(resource_module, "_MAX_SINGLE_RESOURCE_BYTES", 1)

    with (
        ResourceReader(source, Budget()) as resources,
        pytest.raises(ValueError, match="512 MiB"),
    ):
        resources.read_uri(
            "data:application/octet-stream;base64,QUI=",
            {"application/octet-stream"},
        )


def test_nested_resource_is_read_through_directory_handles(tmp_path: Path) -> None:
    source = tmp_path / "source.gltf"
    source.write_bytes(b"{}")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "data.bin").write_bytes(b"data")

    with ResourceReader(source, Budget()) as resources:
        data, mime = resources.read_uri("nested/data.bin", {"application/octet-stream"})

    assert data == b"data"
    assert mime is None


def test_closed_reader_rejects_local_resource(tmp_path: Path) -> None:
    source = tmp_path / "source.gltf"
    source.write_bytes(b"{}")
    (tmp_path / "data.bin").write_bytes(b"data")
    resources = ResourceReader(source, Budget())
    resources.__exit__(None, None, None)

    with pytest.raises(RuntimeError, match="closed"):
        resources.read_uri("data.bin", {"application/octet-stream"})


def test_empty_relative_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        resource_module._open_relative(0, ())


def test_bounded_reader_rejects_directory_and_oversize_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"1234")
    fd = os.open(path, os.O_RDONLY)
    try:
        directory = os.stat_result((stat.S_IFDIR, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        with monkeypatch.context() as patch:
            patch.setattr(resource_module.os, "fstat", lambda _fd: directory)
            with pytest.raises(ValueError, match="regular file"):
                resource_module._read_bounded(fd)
    finally:
        os.close(fd)

    fd = os.open(path, os.O_RDONLY)
    monkeypatch.setattr(resource_module, "_MAX_SINGLE_RESOURCE_BYTES", 3)
    try:
        with pytest.raises(ValueError, match="512 MiB"):
            resource_module._read_bounded(fd)
    finally:
        os.close(fd)


def test_bounded_reader_detects_growth(monkeypatch) -> None:
    metadata = os.stat_result((stat.S_IFREG, 0, 0, 0, 0, 0, 1, 0, 0, 0))
    monkeypatch.setattr(resource_module, "_MAX_SINGLE_RESOURCE_BYTES", 3)
    monkeypatch.setattr(resource_module.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(resource_module.os, "read", lambda _fd, _count: b"1234")

    with pytest.raises(ValueError, match="grew beyond"):
        resource_module._read_bounded(99)


@pytest.mark.skipif(os.name == "nt", reason="POSIX dir-fd constructor branch")
def test_source_open_failure_closes_base_handle(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.gltf"
    source.write_bytes(b"{}")
    original_close = resource_module.os.close
    closed: list[int] = []

    def close(fd: int) -> None:
        closed.append(fd)
        original_close(fd)

    monkeypatch.setattr(
        resource_module,
        "_open_relative",
        lambda *_: (_ for _ in ()).throw(OSError()),
    )
    monkeypatch.setattr(resource_module.os, "close", close)

    with pytest.raises(OSError):
        ResourceReader(source, Budget())

    assert len(closed) == 1


def test_windows_handle_check_accepts_only_source_directory(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.gltf"
    source.write_bytes(b"{}")
    local = tmp_path / "data.bin"
    local.write_bytes(b"data")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside")
    monkeypatch.setattr(resource_module.os, "name", "nt")
    monkeypatch.setattr(resource_module, "_windows_final_path", lambda _fd: local.resolve())
    try:
        with ResourceReader(source, Budget()) as resources:
            assert resources.read_uri("data.bin", {"application/octet-stream"})[0] == b"data"

        monkeypatch.setattr(resource_module, "_windows_final_path", lambda _fd: outside.resolve())
        with (
            ResourceReader(source, Budget()) as resources,
            pytest.raises(ValueError, match="outside"),
        ):
            resources.read_uri("data.bin", {"application/octet-stream"})
    finally:
        outside.unlink()
