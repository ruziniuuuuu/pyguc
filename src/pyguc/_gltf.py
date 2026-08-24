"""Load, validate, and decode the supported glTF 2.0 profile."""

from __future__ import annotations

import io
import json
import math
import struct
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Never, cast

import numpy as np
from PIL import Image

from ._diagnostics import DiagnosticBag, PygucError, ValidationReport, error_report
from ._resource import Budget, ResourceReader

_MAX_JSON_BYTES = 64 * 1024 * 1024
_COMPONENT_DTYPES: dict[int, np.dtype[Any]] = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass(frozen=True, slots=True)
class BufferView:
    buffer: int
    byte_offset: int
    byte_length: int
    byte_stride: int | None
    target: int | None


@dataclass(frozen=True, slots=True)
class SparseAccessor:
    count: int
    indices_buffer_view: int
    indices_byte_offset: int
    indices_component_type: int
    values_buffer_view: int
    values_byte_offset: int


@dataclass(frozen=True, slots=True)
class Accessor:
    buffer_view: int | None
    byte_offset: int
    component_type: int
    normalized: bool
    count: int
    value_type: str
    sparse: SparseAccessor | None
    minimum: tuple[float, ...] | None
    maximum: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class Sampler:
    wrap_s: int
    wrap_t: int


@dataclass(frozen=True, slots=True)
class ImageAsset:
    data: bytes
    mime_type: str
    name: str | None


@dataclass(frozen=True, slots=True)
class Texture:
    source: int
    sampler: int | None


@dataclass(frozen=True, slots=True)
class TextureInfo:
    index: int
    texcoord: int


@dataclass(frozen=True, slots=True)
class NormalTextureInfo:
    index: int
    texcoord: int
    scale: float


@dataclass(frozen=True, slots=True)
class OcclusionTextureInfo:
    index: int
    texcoord: int
    strength: float


@dataclass(frozen=True, slots=True)
class Material:
    name: str | None
    base_color_factor: tuple[float, float, float, float]
    base_color_texture: TextureInfo | None
    metallic_factor: float
    roughness_factor: float
    metallic_roughness_texture: TextureInfo | None
    normal_texture: NormalTextureInfo | None
    occlusion_texture: OcclusionTextureInfo | None
    emissive_texture: TextureInfo | None
    emissive_factor: tuple[float, float, float]
    alpha_mode: str
    alpha_cutoff: float
    double_sided: bool


@dataclass(frozen=True, slots=True)
class Primitive:
    attributes: Mapping[str, int]
    indices: int | None
    material: int | None
    mode: int


@dataclass(frozen=True, slots=True)
class Mesh:
    name: str | None
    primitives: tuple[Primitive, ...]


@dataclass(frozen=True, slots=True)
class Camera:
    name: str | None
    kind: str
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Node:
    name: str | None
    children: tuple[int, ...]
    mesh: int | None
    camera: int | None
    matrix: tuple[float, ...] | None
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Scene:
    name: str | None
    nodes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GltfAsset:
    source: Path
    buffers: tuple[bytes, ...]
    buffer_views: tuple[BufferView, ...]
    accessors: tuple[Accessor, ...]
    images: tuple[ImageAsset, ...]
    samplers: tuple[Sampler, ...]
    textures: tuple[Texture, ...]
    materials: tuple[Material, ...]
    meshes: tuple[Mesh, ...]
    cameras: tuple[Camera, ...]
    nodes: tuple[Node, ...]
    scenes: tuple[Scene, ...]
    default_scene: int | None
    generator: str | None
    copyright: str | None
    report: ValidationReport


def load(source: Path) -> GltfAsset:
    """Read and validate one local glTF or GLB file."""

    source = source.expanduser()
    if source.suffix.lower() not in {".gltf", ".glb"}:
        _resource_failure("PG201", "source must end in .gltf or .glb", source)
    budget = Budget()
    try:
        resources = ResourceReader(source, budget)
    except FileNotFoundError:
        _resource_failure("PG202", "source is not a file", source)
    except (OSError, ValueError) as error:
        _resource_failure("PG203", f"cannot safely open source: {error}", source)
    with resources:
        source = resources.source
        try:
            raw = resources.read_source()
        except (OSError, ValueError) as error:
            _resource_failure("PG203", f"cannot safely read source: {error}", source)
        if source.suffix.lower() == ".glb":
            document, binary_chunk = _parse_glb(raw, source)
        else:
            if len(raw) > _MAX_JSON_BYTES:
                _resource_failure("PG204", "glTF JSON exceeds 64 MiB", source)
            document = _parse_json(raw, source)
            binary_chunk = None

        if not isinstance(document, dict):
            _resource_failure("PG205", "glTF root must be a JSON object", source)
        try:
            budget.check_json_nodes(document)
        except ValueError as error:
            _resource_failure("PG217", str(error), source)

        bag = DiagnosticBag()
        used_extensions = _validate_header(document, bag)
        _scan_ignored_members(document, used_extensions, bag)

        buffers = _load_buffers(document, resources, binary_chunk, bag)
        buffer_views = _load_buffer_views(document, buffers, bag)
        accessors = _load_accessors(document, buffer_views, buffers, budget, bag)
        images = _load_images(document, resources, buffer_views, buffers, budget, bag)
        samplers = _load_samplers(document, bag)
        textures = _load_textures(document, images, samplers, bag)
        materials = _load_materials(document, textures, bag)
        meshes = _load_meshes(document, accessors, buffer_views, materials, bag)
        cameras = _load_cameras(document, bag)
        nodes = _load_nodes(document, meshes, cameras, bag)
        scenes, default_scene = _load_scenes(document, nodes, bag)
        _validate_node_graph(nodes, scenes, bag)
        if not bag.errors:
            try:
                budget.charge_output_prims(
                    _estimate_output_prims(meshes, materials, cameras, nodes, scenes)
                )
            except ValueError as error:
                bag.error(
                    "PG346",
                    str(error),
                    "",
                    "Reduce repeated scenes, node depth, or object counts before conversion.",
                )

    bag.raise_if_errors()
    asset_object = document.get("asset")
    asset_metadata = asset_object if isinstance(asset_object, dict) else {}
    asset = GltfAsset(
        source=source,
        buffers=buffers,
        buffer_views=buffer_views,
        accessors=accessors,
        images=images,
        samplers=samplers,
        textures=textures,
        materials=materials,
        meshes=meshes,
        cameras=cameras,
        nodes=nodes,
        scenes=scenes,
        default_scene=default_scene,
        generator=_optional_string(asset_metadata, "generator"),
        copyright=_optional_string(asset_metadata, "copyright"),
        report=bag.report,
    )
    _validate_decoded_data(asset, bag)
    bag.raise_if_errors()
    return replace(asset, report=bag.report)


def decode_accessor(asset: GltfAsset, index: int) -> np.ndarray[Any, Any]:
    """Decode an accessor into an owned native-endian NumPy array."""

    return _decode_accessor(asset, index, normalize=True)


def _decode_accessor(asset: GltfAsset, index: int, *, normalize: bool) -> np.ndarray[Any, Any]:
    """Decode an accessor, optionally applying glTF integer normalization."""

    accessor = asset.accessors[index]
    dtype = _COMPONENT_DTYPES[accessor.component_type]
    components = _TYPE_COMPONENTS[accessor.value_type]
    shape = (accessor.count,) if components == 1 else (accessor.count, components)
    result = np.zeros(shape, dtype=dtype)
    item_bytes, component_offsets = _element_layout(accessor.value_type, dtype)

    if accessor.buffer_view is not None:
        view = asset.buffer_views[accessor.buffer_view]
        stride = view.byte_stride or item_bytes
        offset = view.byte_offset + accessor.byte_offset
        result = _read_elements(
            asset.buffers[view.buffer],
            dtype,
            accessor.count,
            components,
            offset,
            stride,
            component_offsets,
        )

    if accessor.sparse is not None:
        sparse = accessor.sparse
        index_view = asset.buffer_views[sparse.indices_buffer_view]
        index_dtype = _COMPONENT_DTYPES[sparse.indices_component_type]
        sparse_indices = np.frombuffer(
            asset.buffers[index_view.buffer],
            dtype=index_dtype,
            count=sparse.count,
            offset=index_view.byte_offset + sparse.indices_byte_offset,
        )
        value_view = asset.buffer_views[sparse.values_buffer_view]
        sparse_values = _read_elements(
            asset.buffers[value_view.buffer],
            dtype,
            sparse.count,
            components,
            value_view.byte_offset + sparse.values_byte_offset,
            item_bytes,
            component_offsets,
        )
        result[sparse_indices] = sparse_values

    if normalize and accessor.normalized:
        if accessor.component_type in {5120, 5122}:
            divisor = float(np.iinfo(dtype).max)
            result = np.maximum(result.astype(np.float32) / divisor, -1.0)
        else:
            result = result.astype(np.float32) / float(np.iinfo(dtype).max)
    return np.asarray(result, dtype=result.dtype.newbyteorder("="))


def _read_elements(
    data: bytes,
    dtype: np.dtype[Any],
    count: int,
    components: int,
    offset: int,
    stride: int,
    component_offsets: tuple[int, ...],
) -> np.ndarray[Any, Any]:
    shape = (count,) if components == 1 else (count, components)
    if component_offsets == tuple(range(0, components * dtype.itemsize, dtype.itemsize)):
        return np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=data,
            offset=offset,
            strides=(stride,) if components == 1 else (stride, dtype.itemsize),
        ).copy()
    result = np.empty((count, components), dtype=dtype)
    for component, component_offset in enumerate(component_offsets):
        result[:, component] = np.ndarray(
            shape=(count,),
            dtype=dtype,
            buffer=data,
            offset=offset + component_offset,
            strides=(stride,),
        )
    return result


def _element_layout(value_type: str, dtype: np.dtype[Any]) -> tuple[int, tuple[int, ...]]:
    components = _TYPE_COMPONENTS[value_type]
    if not value_type.startswith("MAT"):
        return dtype.itemsize * components, tuple(
            range(0, dtype.itemsize * components, dtype.itemsize)
        )
    dimension = int(value_type[-1])
    column_bytes = dimension * dtype.itemsize
    padded_column_bytes = (column_bytes + 3) // 4 * 4
    offsets = tuple(
        column * padded_column_bytes + row * dtype.itemsize
        for column in range(dimension)
        for row in range(dimension)
    )
    return padded_column_bytes * dimension, offsets


def triangulate(mode: int, indices: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Normalize TRIANGLES, TRIANGLE_STRIP, or TRIANGLE_FAN to triangle indices."""

    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if mode == 4:
        return values.reshape((-1, 3))
    if values.size < 3:
        return np.empty((0, 3), dtype=np.int64)
    if mode == 5:
        triangles = []
        for offset in range(values.size - 2):
            if offset % 2:
                triangles.append((values[offset + 1], values[offset], values[offset + 2]))
            else:
                triangles.append((values[offset], values[offset + 1], values[offset + 2]))
        return np.asarray(triangles, dtype=np.int64)
    return np.asarray([(values[0], values[i], values[i + 1]) for i in range(1, values.size - 1)])


def _validate_decoded_data(asset: GltfAsset, bag: DiagnosticBag) -> None:
    decoded: list[np.ndarray[Any, Any]] = []
    for index in range(len(asset.accessors)):
        accessor = asset.accessors[index]
        try:
            raw_value = _decode_accessor(asset, index, normalize=False)
            value = decode_accessor(asset, index) if accessor.normalized else raw_value
        except (IndexError, ValueError, OverflowError) as error:
            bag.error("PG256", f"accessor cannot be decoded: {error}", f"/accessors/{index}")
            raw_value = np.empty(0)
            value = np.empty(0)
        if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
            bag.error(
                "PG342",
                "accessor contains a non-finite value",
                f"/accessors/{index}",
                "Replace NaN and infinity values with finite data, then recompute accessor bounds.",
            )
        if raw_value.size and np.all(np.isfinite(raw_value)):
            actual_minimum = np.min(raw_value, axis=0).reshape(-1)
            actual_maximum = np.max(raw_value, axis=0).reshape(-1)
            _validate_declared_bound(
                accessor.minimum, actual_minimum, "min", index, raw_value.dtype, bag
            )
            _validate_declared_bound(
                accessor.maximum, actual_maximum, "max", index, raw_value.dtype, bag
            )
        decoded.append(value)

    for mesh_index, mesh in enumerate(asset.meshes):
        for primitive_index, primitive in enumerate(mesh.primitives):
            pointer = f"/meshes/{mesh_index}/primitives/{primitive_index}"
            position_index = primitive.attributes.get("POSITION")
            if position_index is None or position_index >= len(decoded):
                continue
            point_count = len(decoded[position_index])
            if primitive.indices is None:
                indices = np.arange(point_count, dtype=np.int64)
            elif primitive.indices < len(decoded):
                indices = decoded[primitive.indices].reshape(-1)
            else:
                continue
            if indices.size and int(np.max(indices)) >= point_count:
                bag.error("PG258", "primitive index exceeds POSITION count", f"{pointer}/indices")
            if primitive.indices is not None and primitive.indices < len(asset.accessors):
                index_accessor = asset.accessors[primitive.indices]
                restart_value = np.iinfo(_COMPONENT_DTYPES[index_accessor.component_type]).max
                if np.any(indices == restart_value):
                    bag.error(
                        "PG341",
                        "primitive indices contain a forbidden restart value",
                        f"{pointer}/indices",
                        "Remove the maximum component value from the index accessor.",
                    )
            normal_index = primitive.attributes.get("NORMAL")
            if normal_index is not None and normal_index < len(decoded):
                normal_lengths = np.linalg.norm(decoded[normal_index].astype(np.float64), axis=1)
                if not np.allclose(normal_lengths, 1.0, rtol=1e-4, atol=1e-4):
                    bag.error(
                        "PG343",
                        "NORMAL values must have unit length",
                        pointer,
                        "Normalize every NORMAL xyz vector and update its accessor metadata.",
                    )
            tangent_index = primitive.attributes.get("TANGENT")
            if tangent_index is not None and tangent_index < len(decoded):
                tangents = decoded[tangent_index]
                if not np.allclose(
                    np.linalg.norm(tangents[:, :3].astype(np.float64), axis=1),
                    1.0,
                    rtol=1e-4,
                    atol=1e-4,
                ) or not np.all(np.isin(tangents[:, 3], (-1.0, 1.0))):
                    bag.error(
                        "PG344",
                        "TANGENT xyz values must have unit length and w must be -1 or 1",
                        pointer,
                        (
                            "Regenerate normalized tangent xyz values and set tangent w "
                            "to exactly -1 or 1."
                        ),
                    )
            material = (
                asset.materials[primitive.material]
                if primitive.material is not None and primitive.material < len(asset.materials)
                else None
            )
            if (
                material is not None
                and material.normal_texture is not None
                and ("NORMAL" not in primitive.attributes or "TANGENT" not in primitive.attributes)
            ):
                bag.error(
                    "PG309",
                    "normalTexture requires NORMAL and TANGENT attributes",
                    pointer,
                )
            texture_infos = []
            if material is not None:
                texture_infos = [
                    material.base_color_texture,
                    material.metallic_roughness_texture,
                    material.normal_texture,
                    material.occlusion_texture,
                    material.emissive_texture,
                ]
            for info in (item for item in texture_infos if item is not None):
                if f"TEXCOORD_{info.texcoord}" not in primitive.attributes:
                    bag.error(
                        "PG259",
                        f"material requires missing TEXCOORD_{info.texcoord}",
                        pointer,
                    )


def _validate_declared_bound(
    declared: tuple[float, ...] | None,
    actual: np.ndarray[Any, Any],
    member: str,
    accessor_index: int,
    dtype: np.dtype[Any],
    bag: DiagnosticBag,
) -> None:
    if declared is None:
        return
    expected = np.asarray(declared, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    matches = (
        np.allclose(expected, observed, rtol=1e-6, atol=1e-7)
        if np.issubdtype(dtype, np.floating)
        else np.array_equal(expected, observed)
    )
    if not matches:
        bag.error(
            "PG340",
            f"accessor {member} does not match decoded values",
            f"/accessors/{accessor_index}/{member}",
            (
                f"Recompute accessor {member} from the decoded values, or remove it when "
                "the accessor is not POSITION."
            ),
        )


def _parse_json(raw: bytes, source: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        _resource_failure("PG206", f"invalid glTF JSON: {error}", source)


def _parse_glb(raw: bytes, source: Path) -> tuple[dict[str, Any], bytes | None]:
    if len(raw) < 12:
        _resource_failure("PG207", "GLB header is truncated", source)
    magic, version, declared_length = struct.unpack_from("<4sII", raw)
    if magic != b"glTF" or version != 2:
        _resource_failure("PG208", "GLB must use the glTF 2.0 header", source)
    if declared_length != len(raw):
        _resource_failure("PG209", "GLB declared length does not match the file", source)

    offset = 12
    json_chunk: bytes | None = None
    binary_chunk: bytes | None = None
    first = True
    while offset < len(raw):
        if offset + 8 > len(raw):
            _resource_failure("PG210", "GLB chunk header is truncated", source)
        length, kind = struct.unpack_from("<II", raw, offset)
        if length % 4:
            _resource_failure("PG216", "GLB chunk length must be four-byte aligned", source)
        offset += 8
        end = offset + length
        if end > len(raw):
            _resource_failure("PG211", "GLB chunk exceeds the file", source)
        chunk = raw[offset:end]
        offset = end
        if first and kind != 0x4E4F534A:
            _resource_failure("PG212", "GLB first chunk must be JSON", source)
        first = False
        if kind == 0x4E4F534A:
            if json_chunk is not None:
                _resource_failure("PG213", "GLB contains multiple JSON chunks", source)
            json_chunk = chunk.rstrip(b" ")
        elif kind == 0x004E4942:
            if binary_chunk is not None:
                _resource_failure("PG214", "GLB contains multiple BIN chunks", source)
            binary_chunk = chunk
    if json_chunk is None:
        _resource_failure("PG215", "GLB has no JSON chunk", source)
    assert json_chunk is not None
    if len(json_chunk) > _MAX_JSON_BYTES:
        _resource_failure("PG204", "GLB JSON exceeds 64 MiB", source)
    return _parse_json(json_chunk, source), binary_chunk


def _validate_header(document: dict[str, Any], bag: DiagnosticBag) -> set[str]:
    asset = document.get("asset")
    if not isinstance(asset, dict):
        bag.error("PG220", "asset must be an object", "/asset")
    elif asset.get("version") != "2.0":
        bag.error("PG221", "only glTF 2.0 is supported", "/asset/version")
    elif "minVersion" in asset and asset["minVersion"] != "2.0":
        bag.error("PG301", "asset minVersion is unsupported", "/asset/minVersion")

    required_extensions: set[str] = set()
    required = document.get("extensionsRequired", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        bag.error("PG222", "extensionsRequired must be an array of strings", "/extensionsRequired")
    else:
        required_extensions = set(required)
        if len(required_extensions) != len(required):
            bag.error(
                "PG264", "extensionsRequired must not contain duplicates", "/extensionsRequired"
            )
        if required:
            bag.error(
                "PG301",
                "required glTF extensions are unsupported",
                "/extensionsRequired",
                (
                    "Export the asset without required extensions or provide valid "
                    "core glTF fallback data."
                ),
            )
    used = document.get("extensionsUsed", [])
    used_extensions: set[str] = set()
    if isinstance(used, list) and all(isinstance(item, str) for item in used):
        used_extensions = set(used)
        if len(used_extensions) != len(used):
            bag.error("PG265", "extensionsUsed must not contain duplicates", "/extensionsUsed")
        if not required_extensions <= used_extensions:
            bag.error(
                "PG266",
                "extensionsRequired must be a subset of extensionsUsed",
                "/extensionsRequired",
            )
        for extension in sorted(used_extensions):
            index = used.index(extension)
            bag.warning(
                "PG401",
                f"optional extension {extension!r} is ignored; core fallback is used",
                f"/extensionsUsed/{index}",
                (
                    "Remove the extension after baking its effect into core glTF data "
                    "if exact conversion is required."
                ),
            )
    elif used is not None:
        bag.error("PG222", "extensionsUsed must be an array of strings", "/extensionsUsed")

    for member, message in (
        ("animations", "animations are unsupported"),
        ("skins", "skins are unsupported"),
    ):
        if document.get(member):
            bag.error("PG302", message, f"/{member}")
    return used_extensions


def _scan_ignored_members(
    value: Any, used_extensions: set[str], bag: DiagnosticBag, pointer: str = ""
) -> None:
    pending = [(value, pointer)]
    extras_seen = False
    payload_extensions: set[str] = set()
    while pending:
        current, current_pointer = pending.pop()
        children: list[tuple[Any, str]] = []
        if isinstance(current, dict):
            for key, child in current.items():
                child_pointer = f"{current_pointer}/{_escape_pointer(key)}"
                if key == "extras":
                    extras_seen = True
                elif key == "extensions":
                    if not isinstance(child, dict):
                        bag.error("PG267", "extensions must be an object", child_pointer)
                    else:
                        payload_extensions.update(child)
                else:
                    children.append((child, child_pointer))
        elif isinstance(current, list):
            children.extend(
                (child, f"{current_pointer}/{index}") for index, child in enumerate(current)
            )
        pending.extend(reversed(children))
    if extras_seen:
        bag.warning(
            "PG402",
            "extras are ignored throughout the asset",
            "",
            (
                "Remove extras if they are accidental; pyguc does not preserve "
                "application-specific metadata."
            ),
        )
    undeclared = payload_extensions - used_extensions
    for extension in sorted(undeclared):
        bag.error(
            "PG268",
            f"extension payload {extension!r} is not declared in extensionsUsed",
            "",
            f"Add {extension!r} to extensionsUsed or remove its extension payload.",
        )


def _load_buffers(
    document: dict[str, Any],
    resources: ResourceReader,
    binary_chunk: bytes | None,
    bag: DiagnosticBag,
) -> tuple[bytes, ...]:
    result: list[bytes] = []
    objects = _array(document, "buffers", bag)
    for index, value in enumerate(objects):
        pointer = f"/buffers/{index}"
        if not isinstance(value, dict):
            bag.error("PG223", "buffer must be an object", pointer)
            result.append(b"")
            continue
        length = _nonnegative_int(value.get("byteLength"), bag, f"{pointer}/byteLength")
        if length == 0:
            bag.error(
                "PG229",
                "buffer byteLength must be positive",
                f"{pointer}/byteLength",
                "Remove the empty buffer and its references, or populate it with data.",
            )
        uri = value.get("uri")
        try:
            if uri is None:
                if index != 0 or binary_chunk is None:
                    raise ValueError("a URI-less buffer requires the GLB BIN chunk")
                data = binary_chunk
            elif isinstance(uri, str):
                data, _ = resources.read_uri(
                    uri,
                    allowed_data_types={"application/octet-stream", "application/gltf-buffer"},
                )
            else:
                raise ValueError("buffer URI must be a string")
        except (OSError, ValueError) as error:
            bag.error("PG224", f"cannot read buffer: {error}", f"{pointer}/uri")
            data = b""
        if length is not None and len(data) < length:
            bag.error("PG226", "buffer is shorter than byteLength", pointer)
        if length is not None and binary_chunk is None and len(data) != length:
            bag.error("PG227", "buffer length does not match byteLength", pointer)
        result.append(data[:length] if length is not None else data)
    if binary_chunk is not None and not objects:
        bag.error("PG228", "GLB BIN chunk has no buffer declaration", "/buffers")
    return tuple(result)


def _load_buffer_views(
    document: dict[str, Any], buffers: tuple[bytes, ...], bag: DiagnosticBag
) -> tuple[BufferView, ...]:
    result: list[BufferView] = []
    for index, value in enumerate(_array(document, "bufferViews", bag)):
        pointer = f"/bufferViews/{index}"
        if not isinstance(value, dict):
            bag.error("PG230", "bufferView must be an object", pointer)
            result.append(BufferView(0, 0, 0, None, None))
            continue
        buffer_index = _index(value.get("buffer"), len(buffers), bag, f"{pointer}/buffer")
        offset = _optional_nonnegative_int(value.get("byteOffset"), 0, bag, f"{pointer}/byteOffset")
        length = _nonnegative_int(value.get("byteLength"), bag, f"{pointer}/byteLength") or 0
        if length == 0:
            bag.error(
                "PG229",
                "bufferView byteLength must be positive",
                f"{pointer}/byteLength",
                "Remove the empty bufferView and its references, or give it non-empty data.",
            )
        stride_value = value.get("byteStride")
        stride = None
        if stride_value is not None:
            stride = _nonnegative_int(stride_value, bag, f"{pointer}/byteStride")
            if stride is not None and (not 4 <= stride <= 252 or stride % 4):
                bag.error(
                    "PG231",
                    "byteStride must be a multiple of four between 4 and 252",
                    f"{pointer}/byteStride",
                )
        target = value.get("target")
        if target is not None and (
            not isinstance(target, int) or isinstance(target, bool) or target not in {34962, 34963}
        ):
            bag.error("PG233", "invalid bufferView target", f"{pointer}/target")
            target = None
        if buffer_index is not None and offset + length > len(buffers[buffer_index]):
            bag.error("PG232", "bufferView exceeds its buffer", pointer)
        result.append(BufferView(buffer_index or 0, offset, length, stride, target))
    return tuple(result)


def _load_accessors(
    document: dict[str, Any],
    views: tuple[BufferView, ...],
    buffers: tuple[bytes, ...],
    budget: Budget,
    bag: DiagnosticBag,
) -> tuple[Accessor, ...]:
    result: list[Accessor] = []
    for index, value in enumerate(_array(document, "accessors", bag)):
        pointer = f"/accessors/{index}"
        if not isinstance(value, dict):
            bag.error("PG240", "accessor must be an object", pointer)
            result.append(Accessor(None, 0, 5126, False, 0, "SCALAR", None, None, None))
            continue
        view_index = None
        if "bufferView" in value:
            view_index = _index(value["bufferView"], len(views), bag, f"{pointer}/bufferView")
        offset = _optional_nonnegative_int(value.get("byteOffset"), 0, bag, f"{pointer}/byteOffset")
        component_type = value.get("componentType")
        if (
            not isinstance(component_type, int)
            or isinstance(component_type, bool)
            or component_type not in _COMPONENT_DTYPES
        ):
            bag.error("PG241", "invalid accessor componentType", f"{pointer}/componentType")
            component_type = 5126
        count = _nonnegative_int(value.get("count"), bag, f"{pointer}/count") or 0
        if count == 0:
            bag.error("PG229", "accessor count must be positive", f"{pointer}/count")
        value_type = value.get("type")
        if not isinstance(value_type, str) or value_type not in _TYPE_COMPONENTS:
            bag.error("PG242", "invalid accessor type", f"{pointer}/type")
            value_type = "SCALAR"
        normalized = value.get("normalized", False)
        if not isinstance(normalized, bool):
            bag.error("PG243", "normalized must be boolean", f"{pointer}/normalized")
            normalized = False
        if normalized and component_type not in {5120, 5121, 5122, 5123}:
            bag.error("PG244", "componentType cannot be normalized", pointer)
        try:
            budget.charge_accessor_values(count * _TYPE_COMPONENTS[value_type])
        except ValueError as error:
            bag.error(
                "PG245",
                str(error),
                pointer,
                "Reduce accessor counts or split the asset into smaller independent inputs.",
            )
        sparse = _load_sparse(value.get("sparse"), views, pointer, bag)
        if view_index is None and sparse is None:
            bag.error("PG246", "accessor requires bufferView or sparse data", pointer)
        component_count = _TYPE_COMPONENTS[value_type]
        minimum = _accessor_bound(value.get("min"), component_count, bag, f"{pointer}/min")
        maximum = _accessor_bound(value.get("max"), component_count, bag, f"{pointer}/max")
        if (
            minimum is not None
            and maximum is not None
            and any(lower > upper for lower, upper in zip(minimum, maximum, strict=True))
        ):
            bag.error(
                "PG262",
                "accessor min must not exceed max",
                pointer,
                "Recompute the accessor min and max metadata from the decoded values.",
            )
        accessor = Accessor(
            view_index,
            offset,
            component_type,
            normalized,
            count,
            value_type,
            sparse,
            minimum,
            maximum,
        )
        _validate_accessor_bounds(accessor, views, buffers, pointer, bag)
        result.append(accessor)
    return tuple(result)


def _load_sparse(
    value: Any, views: tuple[BufferView, ...], pointer: str, bag: DiagnosticBag
) -> SparseAccessor | None:
    if value is None:
        return None
    sparse_pointer = f"{pointer}/sparse"
    if not isinstance(value, dict):
        bag.error("PG247", "sparse must be an object", sparse_pointer)
        return None
    count = _nonnegative_int(value.get("count"), bag, f"{sparse_pointer}/count") or 0
    if count == 0:
        bag.error("PG229", "sparse count must be positive", f"{sparse_pointer}/count")
    indices = value.get("indices")
    values = value.get("values")
    if not isinstance(indices, dict) or not isinstance(values, dict):
        bag.error("PG248", "sparse indices and values must be objects", sparse_pointer)
        return None
    index_view = _index(
        indices.get("bufferView"), len(views), bag, f"{sparse_pointer}/indices/bufferView"
    )
    index_type = indices.get("componentType")
    if (
        not isinstance(index_type, int)
        or isinstance(index_type, bool)
        or index_type not in {5121, 5123, 5125}
    ):
        bag.error(
            "PG249",
            "sparse index componentType must be unsigned",
            f"{sparse_pointer}/indices/componentType",
        )
        index_type = 5121
    value_view = _index(
        values.get("bufferView"), len(views), bag, f"{sparse_pointer}/values/bufferView"
    )
    return SparseAccessor(
        count,
        index_view or 0,
        _optional_nonnegative_int(
            indices.get("byteOffset"), 0, bag, f"{sparse_pointer}/indices/byteOffset"
        ),
        index_type,
        value_view or 0,
        _optional_nonnegative_int(
            values.get("byteOffset"), 0, bag, f"{sparse_pointer}/values/byteOffset"
        ),
    )


def _validate_accessor_bounds(
    accessor: Accessor,
    views: tuple[BufferView, ...],
    buffers: tuple[bytes, ...],
    pointer: str,
    bag: DiagnosticBag,
) -> None:
    dtype = _COMPONENT_DTYPES[accessor.component_type]
    item_bytes, _component_offsets = _element_layout(accessor.value_type, dtype)
    if accessor.buffer_view is not None and accessor.buffer_view < len(views):
        view = views[accessor.buffer_view]
        stride = view.byte_stride or item_bytes
        if stride < item_bytes or stride % dtype.itemsize:
            bag.error("PG250", "bufferView stride is incompatible with accessor", pointer)
        required = accessor.byte_offset
        if accessor.count:
            required += (accessor.count - 1) * stride + item_bytes
        if (
            view.byte_offset + accessor.byte_offset
        ) % dtype.itemsize or required > view.byte_length:
            bag.error("PG251", "accessor exceeds or misaligns its bufferView", pointer)
    sparse = accessor.sparse
    if sparse is not None:
        if sparse.count > accessor.count:
            bag.error("PG252", "sparse count exceeds accessor count", f"{pointer}/sparse/count")
        if sparse.indices_buffer_view < len(views):
            view = views[sparse.indices_buffer_view]
            if view.byte_stride is not None:
                bag.error("PG253", "sparse bufferViews cannot have byteStride", f"{pointer}/sparse")
            byte_count = sparse.count * _COMPONENT_DTYPES[sparse.indices_component_type].itemsize
            if (view.byte_offset + sparse.indices_byte_offset) % _COMPONENT_DTYPES[
                sparse.indices_component_type
            ].itemsize or sparse.indices_byte_offset + byte_count > view.byte_length:
                bag.error("PG253", "sparse indices exceed their bufferView", f"{pointer}/sparse")
        if sparse.values_buffer_view < len(views):
            view = views[sparse.values_buffer_view]
            if view.byte_stride is not None:
                bag.error("PG254", "sparse bufferViews cannot have byteStride", f"{pointer}/sparse")
            byte_count = sparse.count * item_bytes
            if (
                view.byte_offset + sparse.values_byte_offset
            ) % dtype.itemsize or sparse.values_byte_offset + byte_count > view.byte_length:
                bag.error("PG254", "sparse values exceed their bufferView", f"{pointer}/sparse")
        if not bag.errors:
            try:
                index_view = views[sparse.indices_buffer_view]
                sparse_indices = np.frombuffer(
                    buffers[index_view.buffer],
                    dtype=_COMPONENT_DTYPES[sparse.indices_component_type],
                    count=sparse.count,
                    offset=index_view.byte_offset + sparse.indices_byte_offset,
                )
                if sparse_indices.size and (
                    int(sparse_indices[-1]) >= accessor.count
                    or np.any(sparse_indices[1:] <= sparse_indices[:-1])
                ):
                    bag.error(
                        "PG255",
                        "sparse indices must be strictly increasing and in range",
                        f"{pointer}/sparse/indices",
                    )
            except (ValueError, IndexError):
                bag.error("PG253", "cannot decode sparse indices", f"{pointer}/sparse/indices")


def _load_images(
    document: dict[str, Any],
    resources: ResourceReader,
    views: tuple[BufferView, ...],
    buffers: tuple[bytes, ...],
    budget: Budget,
    bag: DiagnosticBag,
) -> tuple[ImageAsset, ...]:
    result: list[ImageAsset] = []
    for index, value in enumerate(_array(document, "images", bag)):
        pointer = f"/images/{index}"
        data = b""
        declared_mime: str | None = None
        name = None
        try:
            if not isinstance(value, dict):
                raise ValueError("image must be an object")
            name = _optional_string(value, "name")
            declared_mime = _optional_string(value, "mimeType")
            uri = value.get("uri")
            buffer_view = value.get("bufferView")
            if isinstance(uri, str) and buffer_view is None:
                data, uri_mime = resources.read_uri(uri, {"image/png", "image/jpeg"})
                declared_mime = declared_mime or uri_mime
            elif uri is None and buffer_view is not None:
                view_index = _checked_index(buffer_view, len(views))
                view = views[view_index]
                data = buffers[view.buffer][view.byte_offset : view.byte_offset + view.byte_length]
                if declared_mime not in {"image/png", "image/jpeg"}:
                    raise ValueError("bufferView image requires image/png or image/jpeg mimeType")
            else:
                raise ValueError("image must define exactly one of uri or bufferView")
            actual_mime = _validate_image(data, budget)
            if declared_mime is not None and declared_mime != actual_mime:
                raise ValueError("declared image MIME type does not match the bytes")
            declared_mime = actual_mime
        except (
            OSError,
            ValueError,
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
        ) as error:
            bag.error("PG260", f"cannot read image: {error}", pointer)
            declared_mime = declared_mime or "image/png"
        result.append(ImageAsset(data, declared_mime, name))
    return tuple(result)


def _validate_image(data: bytes, budget: Budget) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            budget.charge_image_pixels(image.width * image.height)
            image.verify()
            image_format = image.format
    if image_format == "PNG":
        return "image/png"
    if image_format == "JPEG":
        return "image/jpeg"
    raise ValueError("only PNG and JPEG images are supported")


def _load_samplers(document: dict[str, Any], bag: DiagnosticBag) -> tuple[Sampler, ...]:
    result: list[Sampler] = []
    valid_wrap = {33071, 33648, 10497}
    for index, value in enumerate(_array(document, "samplers", bag)):
        pointer = f"/samplers/{index}"
        if not isinstance(value, dict):
            bag.error("PG270", "sampler must be an object", pointer)
            value = {}
        if "minFilter" in value or "magFilter" in value:
            bag.error(
                "PG303",
                "explicit sampler filters are unsupported",
                pointer,
                "Remove minFilter and magFilter to use the default filtering behavior.",
            )
        wrap_s = value.get("wrapS", 10497)
        wrap_t = value.get("wrapT", 10497)
        if (
            not isinstance(wrap_s, int)
            or isinstance(wrap_s, bool)
            or wrap_s not in valid_wrap
            or not isinstance(wrap_t, int)
            or isinstance(wrap_t, bool)
            or wrap_t not in valid_wrap
        ):
            bag.error("PG271", "invalid sampler wrap mode", pointer)
            wrap_s = wrap_t = 10497
        result.append(Sampler(wrap_s, wrap_t))
    return tuple(result)


def _load_textures(
    document: dict[str, Any],
    images: tuple[ImageAsset, ...],
    samplers: tuple[Sampler, ...],
    bag: DiagnosticBag,
) -> tuple[Texture, ...]:
    result: list[Texture] = []
    for index, value in enumerate(_array(document, "textures", bag)):
        pointer = f"/textures/{index}"
        if not isinstance(value, dict):
            bag.error("PG272", "texture must be an object", pointer)
            value = {}
        source = _index(value.get("source"), len(images), bag, f"{pointer}/source")
        sampler = None
        if "sampler" in value:
            sampler = _index(value["sampler"], len(samplers), bag, f"{pointer}/sampler")
        result.append(Texture(source or 0, sampler))
    return tuple(result)


def _texture_info(
    value: Any, texture_count: int, pointer: str, bag: DiagnosticBag
) -> TextureInfo | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        bag.error("PG273", "texture info must be an object", pointer)
        return None
    index = _index(value.get("index"), texture_count, bag, f"{pointer}/index")
    texcoord = _optional_nonnegative_int(value.get("texCoord"), 0, bag, f"{pointer}/texCoord")
    return TextureInfo(index or 0, texcoord) if index is not None else None


def _load_materials(
    document: dict[str, Any], textures: tuple[Texture, ...], bag: DiagnosticBag
) -> tuple[Material, ...]:
    result: list[Material] = []
    for index, value in enumerate(_array(document, "materials", bag)):
        pointer = f"/materials/{index}"
        if not isinstance(value, dict):
            bag.error("PG280", "material must be an object", pointer)
            value = {}
        pbr = value.get("pbrMetallicRoughness", {})
        if not isinstance(pbr, dict):
            bag.error(
                "PG281", "pbrMetallicRoughness must be an object", f"{pointer}/pbrMetallicRoughness"
            )
            pbr = {}
        base = _float_tuple(
            pbr.get("baseColorFactor"),
            4,
            (1.0, 1.0, 1.0, 1.0),
            bag,
            f"{pointer}/pbrMetallicRoughness/baseColorFactor",
        )
        emissive = _float_tuple(
            value.get("emissiveFactor"), 3, (0.0, 0.0, 0.0), bag, f"{pointer}/emissiveFactor"
        )
        normal_base = _texture_info(
            value.get("normalTexture"), len(textures), f"{pointer}/normalTexture", bag
        )
        normal = None
        if normal_base is not None:
            normal_object = value["normalTexture"]
            normal_scale = _finite_float(
                normal_object.get("scale", 1.0),
                1.0,
                bag,
                f"{pointer}/normalTexture/scale",
            )
            if normal_scale != 1.0:
                bag.error(
                    "PG309",
                    "non-default normalTexture scale cannot be represented by UsdPreviewSurface",
                    f"{pointer}/normalTexture/scale",
                )
            normal = NormalTextureInfo(
                normal_base.index,
                normal_base.texcoord,
                normal_scale,
            )
        occlusion_base = _texture_info(
            value.get("occlusionTexture"), len(textures), f"{pointer}/occlusionTexture", bag
        )
        occlusion = None
        if occlusion_base is not None:
            occlusion_object = value["occlusionTexture"]
            occlusion = OcclusionTextureInfo(
                occlusion_base.index,
                occlusion_base.texcoord,
                _finite_float(
                    occlusion_object.get("strength", 1.0),
                    1.0,
                    bag,
                    f"{pointer}/occlusionTexture/strength",
                ),
            )
        alpha_mode = value.get("alphaMode", "OPAQUE")
        if not isinstance(alpha_mode, str) or alpha_mode not in {"OPAQUE", "MASK", "BLEND"}:
            bag.error("PG282", "invalid alphaMode", f"{pointer}/alphaMode")
            alpha_mode = "OPAQUE"
        double_sided = value.get("doubleSided", False)
        if not isinstance(double_sided, bool):
            bag.error("PG283", "doubleSided must be boolean", f"{pointer}/doubleSided")
            double_sided = False
        metallic = _finite_float(
            pbr.get("metallicFactor", 1.0),
            1.0,
            bag,
            f"{pointer}/pbrMetallicRoughness/metallicFactor",
        )
        roughness = _finite_float(
            pbr.get("roughnessFactor", 1.0),
            1.0,
            bag,
            f"{pointer}/pbrMetallicRoughness/roughnessFactor",
        )
        alpha_cutoff = _finite_float(
            value.get("alphaCutoff", 0.5), 0.5, bag, f"{pointer}/alphaCutoff"
        )
        unit_factors = [
            *[
                (
                    component,
                    f"{pointer}/pbrMetallicRoughness/baseColorFactor/{component_index}",
                )
                for component_index, component in enumerate(base)
            ],
            (metallic, f"{pointer}/pbrMetallicRoughness/metallicFactor"),
            (roughness, f"{pointer}/pbrMetallicRoughness/roughnessFactor"),
        ]
        for factor, factor_pointer in unit_factors:
            if not 0.0 <= factor <= 1.0:
                bag.error("PG284", "material factor must be between 0 and 1", factor_pointer)
        if occlusion is not None and not 0.0 <= occlusion.strength <= 1.0:
            bag.error(
                "PG286",
                "occlusion strength must be between 0 and 1",
                f"{pointer}/occlusionTexture/strength",
            )
        if alpha_cutoff < 0.0:
            bag.error(
                "PG285",
                "alphaCutoff must be non-negative",
                f"{pointer}/alphaCutoff",
                (
                    "Use zero or a positive cutoff; values above one are valid and "
                    "render fully transparent."
                ),
            )
        if "alphaCutoff" in value and "alphaMode" not in value:
            bag.error(
                "PG288",
                "alphaCutoff requires alphaMode to be explicitly defined",
                f"{pointer}/alphaCutoff",
                "Add alphaMode or remove alphaCutoff.",
            )
        for component_index, component in enumerate(emissive):
            if not 0.0 <= component <= 1.0:
                bag.error(
                    "PG287",
                    "emissive factor must be between 0 and 1",
                    f"{pointer}/emissiveFactor/{component_index}",
                    (
                        "Clamp the core emissive factor to [0, 1] or use a supported "
                        "emissive-strength extension."
                    ),
                )
        result.append(
            Material(
                _optional_string(value, "name"),
                cast(tuple[float, float, float, float], base),
                _texture_info(
                    pbr.get("baseColorTexture"),
                    len(textures),
                    f"{pointer}/pbrMetallicRoughness/baseColorTexture",
                    bag,
                ),
                metallic,
                roughness,
                _texture_info(
                    pbr.get("metallicRoughnessTexture"),
                    len(textures),
                    f"{pointer}/pbrMetallicRoughness/metallicRoughnessTexture",
                    bag,
                ),
                normal,
                occlusion,
                _texture_info(
                    value.get("emissiveTexture"), len(textures), f"{pointer}/emissiveTexture", bag
                ),
                cast(tuple[float, float, float], emissive),
                alpha_mode,
                alpha_cutoff,
                double_sided,
            )
        )
    return tuple(result)


def _load_meshes(
    document: dict[str, Any],
    accessors: tuple[Accessor, ...],
    views: tuple[BufferView, ...],
    materials: tuple[Material, ...],
    bag: DiagnosticBag,
) -> tuple[Mesh, ...]:
    result: list[Mesh] = []
    for mesh_index, value in enumerate(_array(document, "meshes", bag)):
        pointer = f"/meshes/{mesh_index}"
        if not isinstance(value, dict):
            bag.error("PG290", "mesh must be an object", pointer)
            value = {}
        if "weights" in value:
            bag.error("PG304", "morph weights are unsupported", f"{pointer}/weights")
        primitive_values = value.get("primitives")
        if not isinstance(primitive_values, list) or not primitive_values:
            bag.error("PG291", "mesh primitives must be a non-empty array", f"{pointer}/primitives")
            primitive_values = []
        primitives: list[Primitive] = []
        for primitive_index, primitive in enumerate(primitive_values):
            primitive_pointer = f"{pointer}/primitives/{primitive_index}"
            if not isinstance(primitive, dict):
                bag.error("PG292", "primitive must be an object", primitive_pointer)
                continue
            if primitive.get("targets"):
                bag.error("PG304", "morph targets are unsupported", f"{primitive_pointer}/targets")
            mode = primitive.get("mode", 4)
            if not isinstance(mode, int) or isinstance(mode, bool) or mode not in {4, 5, 6}:
                bag.error(
                    "PG305",
                    "only triangle primitive modes are supported",
                    f"{primitive_pointer}/mode",
                )
                mode = 4
            attributes_value = primitive.get("attributes")
            attributes: dict[str, int] = {}
            if not isinstance(attributes_value, dict):
                bag.error(
                    "PG293",
                    "primitive attributes must be an object",
                    f"{primitive_pointer}/attributes",
                )
            else:
                for semantic, accessor_value in attributes_value.items():
                    semantic_pointer = f"{primitive_pointer}/attributes/{_escape_pointer(semantic)}"
                    if semantic.startswith("COLOR_"):
                        bag.error("PG306", "vertex colors are unsupported", semantic_pointer)
                    elif semantic not in {
                        "POSITION",
                        "NORMAL",
                        "TANGENT",
                    } and not _is_texcoord_semantic(semantic):
                        bag.error(
                            "PG307", f"attribute {semantic!r} is unsupported", semantic_pointer
                        )
                    accessor_index = _index(accessor_value, len(accessors), bag, semantic_pointer)
                    if accessor_index is not None:
                        attributes[semantic] = accessor_index
            if "POSITION" not in attributes:
                bag.error("PG294", "primitive requires POSITION", f"{primitive_pointer}/attributes")
            indices = None
            if "indices" in primitive:
                indices = _index(
                    primitive["indices"], len(accessors), bag, f"{primitive_pointer}/indices"
                )
            material = None
            if "material" in primitive:
                material = _index(
                    primitive["material"], len(materials), bag, f"{primitive_pointer}/material"
                )
            _validate_primitive_accessors(
                attributes, indices, accessors, views, mode, primitive_pointer, bag
            )
            primitives.append(Primitive(attributes, indices, material, mode))
        result.append(Mesh(_optional_string(value, "name"), tuple(primitives)))
    return tuple(result)


def _validate_primitive_accessors(
    attributes: Mapping[str, int],
    indices: int | None,
    accessors: tuple[Accessor, ...],
    views: tuple[BufferView, ...],
    mode: int,
    pointer: str,
    bag: DiagnosticBag,
) -> None:
    position_index = attributes.get("POSITION")
    position_count = (
        accessors[position_index].count
        if position_index is not None and position_index < len(accessors)
        else None
    )
    for semantic, index in attributes.items():
        if index >= len(accessors):
            continue
        accessor = accessors[index]
        expected = (
            "VEC2"
            if semantic.startswith("TEXCOORD_")
            else "VEC4"
            if semantic == "TANGENT"
            else "VEC3"
        )
        if accessor.value_type != expected:
            bag.error("PG295", f"{semantic} accessor must be {expected}", pointer)
        if semantic in {"POSITION", "NORMAL", "TANGENT"} and accessor.component_type != 5126:
            bag.error("PG296", f"{semantic} accessor must use FLOAT", pointer)
        if semantic == "POSITION" and (accessor.minimum is None or accessor.maximum is None):
            bag.error(
                "PG300",
                "POSITION accessor requires min and max bounds",
                f"{pointer}/attributes/POSITION",
                "Compute min and max from the POSITION accessor values and store both properties.",
            )
        if semantic.startswith("TEXCOORD_") and not (
            accessor.component_type == 5126
            or (accessor.component_type in {5121, 5123} and accessor.normalized)
        ):
            bag.error(
                "PG296",
                f"{semantic} must use FLOAT or normalized unsigned integers",
                pointer,
            )
        if position_count is not None and accessor.count != position_count:
            bag.error("PG297", "primitive attribute counts must match", pointer)
        if accessor.buffer_view is not None and accessor.buffer_view < len(views):
            target = views[accessor.buffer_view].target
            if target is not None and target != 34962:
                bag.error(
                    "PG263",
                    "vertex attribute bufferView target must be ARRAY_BUFFER",
                    f"{pointer}/attributes/{_escape_pointer(semantic)}",
                    "Set the referenced bufferView target to 34962 or omit target.",
                )
    count = position_count or 0
    if indices is not None and indices < len(accessors):
        accessor = accessors[indices]
        if accessor.value_type != "SCALAR" or accessor.component_type not in {5121, 5123, 5125}:
            bag.error("PG298", "indices must be unsigned SCALAR values", f"{pointer}/indices")
        if accessor.normalized:
            bag.error("PG298", "indices cannot be normalized", f"{pointer}/indices")
        if accessor.buffer_view is not None and accessor.buffer_view < len(views):
            target = views[accessor.buffer_view].target
            if target is not None and target != 34963:
                bag.error(
                    "PG269",
                    "index bufferView target must be ELEMENT_ARRAY_BUFFER",
                    f"{pointer}/indices",
                    "Set the referenced bufferView target to 34963 or omit target.",
                )
        count = accessor.count
    if mode == 4 and count % 3:
        bag.error("PG299", "TRIANGLES index or vertex count must be divisible by three", pointer)
    if count < 3:
        bag.error("PG299", "triangle primitives require at least three vertices", pointer)


def _load_cameras(document: dict[str, Any], bag: DiagnosticBag) -> tuple[Camera, ...]:
    result: list[Camera] = []
    for index, value in enumerate(_array(document, "cameras", bag)):
        pointer = f"/cameras/{index}"
        if not isinstance(value, dict):
            bag.error("PG310", "camera must be an object", pointer)
            value = {}
        kind = value.get("type")
        if not isinstance(kind, str) or kind not in {"perspective", "orthographic"}:
            bag.error("PG311", "camera type must be perspective or orthographic", f"{pointer}/type")
            kind = "perspective"
        body = value.get(kind)
        if not isinstance(body, dict):
            bag.error("PG312", f"camera requires a {kind} object", f"{pointer}/{kind}")
            body = {}
        required = ("yfov", "znear") if kind == "perspective" else ("xmag", "ymag", "znear", "zfar")
        values: dict[str, float] = {}
        for member in required:
            if member not in body:
                bag.error("PG313", f"camera requires {member}", f"{pointer}/{kind}/{member}")
            values[member] = _finite_float(
                body.get(member, 1.0), 1.0, bag, f"{pointer}/{kind}/{member}"
            )
        for member in ("aspectRatio", "zfar"):
            if member in body:
                values[member] = _finite_float(body[member], 1.0, bag, f"{pointer}/{kind}/{member}")
        if values.get("znear", 0.0) <= 0 or values.get("zfar", math.inf) <= values.get(
            "znear", 0.0
        ):
            bag.error("PG314", "camera clipping range is invalid", f"{pointer}/{kind}")
        if kind == "perspective" and not 0 < values.get("yfov", 0.0) < math.pi:
            bag.error(
                "PG315", "perspective yfov must be between 0 and pi", f"{pointer}/perspective/yfov"
            )
        if kind == "perspective" and values.get("aspectRatio", 1.0) <= 0:
            bag.error(
                "PG315",
                "perspective aspectRatio must be positive",
                f"{pointer}/perspective/aspectRatio",
            )
        if kind == "orthographic" and (
            values.get("xmag", 0.0) <= 0 or values.get("ymag", 0.0) <= 0
        ):
            bag.error(
                "PG315",
                "orthographic magnification must be positive",
                f"{pointer}/orthographic",
            )
        result.append(Camera(_optional_string(value, "name"), kind, values))
    return tuple(result)


def _load_nodes(
    document: dict[str, Any],
    meshes: tuple[Mesh, ...],
    cameras: tuple[Camera, ...],
    bag: DiagnosticBag,
) -> tuple[Node, ...]:
    values = _array(document, "nodes", bag)
    result: list[Node] = []
    for index, value in enumerate(values):
        pointer = f"/nodes/{index}"
        if not isinstance(value, dict):
            bag.error("PG320", "node must be an object", pointer)
            value = {}
        if "skin" in value or "weights" in value:
            bag.error("PG308", "skinned or morphed nodes are unsupported", pointer)
        children_values = value.get("children", [])
        children: list[int] = []
        if not isinstance(children_values, list):
            bag.error("PG321", "node children must be an array", f"{pointer}/children")
        else:
            seen_children: set[int] = set()
            for child_index, child in enumerate(children_values):
                checked = _index(child, len(values), bag, f"{pointer}/children/{child_index}")
                if checked is not None:
                    if checked in seen_children:
                        bag.error(
                            "PG335",
                            "node children must not contain duplicates",
                            f"{pointer}/children/{child_index}",
                        )
                    seen_children.add(checked)
                    children.append(checked)
        mesh = (
            _index(value["mesh"], len(meshes), bag, f"{pointer}/mesh") if "mesh" in value else None
        )
        camera = (
            _index(value["camera"], len(cameras), bag, f"{pointer}/camera")
            if "camera" in value
            else None
        )
        matrix = None
        if "matrix" in value:
            matrix = _float_tuple(
                value["matrix"],
                16,
                tuple(float(i % 5 == 0) for i in range(16)),
                bag,
                f"{pointer}/matrix",
            )
            if any(member in value for member in ("translation", "rotation", "scale")):
                bag.error("PG322", "matrix cannot be combined with TRS properties", pointer)
            if not _matrix_is_trs(matrix):
                bag.error(
                    "PG324",
                    "node matrix must be decomposable into translation, rotation, and scale",
                    f"{pointer}/matrix",
                    "Remove perspective or shear terms and export the node as a TRS transform.",
                )
        translation = _float_tuple(
            value.get("translation"), 3, (0.0, 0.0, 0.0), bag, f"{pointer}/translation"
        )
        rotation = _float_tuple(
            value.get("rotation"), 4, (0.0, 0.0, 0.0, 1.0), bag, f"{pointer}/rotation"
        )
        scale = _float_tuple(value.get("scale"), 3, (1.0, 1.0, 1.0), bag, f"{pointer}/scale")
        quaternion_length = math.sqrt(sum(component * component for component in rotation))
        if not math.isclose(quaternion_length, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            bag.error("PG323", "node rotation quaternion must be normalized", f"{pointer}/rotation")
        result.append(
            Node(
                _optional_string(value, "name"),
                tuple(children),
                mesh,
                camera,
                matrix,
                cast(tuple[float, float, float], translation),
                cast(tuple[float, float, float, float], rotation),
                cast(tuple[float, float, float], scale),
            )
        )
    return tuple(result)


def _matrix_is_trs(value: tuple[float, ...]) -> bool:
    matrix = np.asarray(value, dtype=np.float64).reshape((4, 4), order="F")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-8):
        return False
    basis = matrix[:3, :3]
    gram = basis.T @ basis
    off_diagonal = gram - np.diag(np.diag(gram))
    scale = max(float(np.max(np.abs(gram))), 1.0)
    return bool(np.all(np.abs(off_diagonal) <= scale * 1e-8))


def _load_scenes(
    document: dict[str, Any], nodes: tuple[Node, ...], bag: DiagnosticBag
) -> tuple[tuple[Scene, ...], int | None]:
    result: list[Scene] = []
    for index, value in enumerate(_array(document, "scenes", bag)):
        pointer = f"/scenes/{index}"
        if not isinstance(value, dict):
            bag.error("PG330", "scene must be an object", pointer)
            value = {}
        roots: list[int] = []
        node_values = value.get("nodes", [])
        if not isinstance(node_values, list):
            bag.error("PG331", "scene nodes must be an array", f"{pointer}/nodes")
        else:
            seen_roots: set[int] = set()
            for root_index, root in enumerate(node_values):
                checked = _index(root, len(nodes), bag, f"{pointer}/nodes/{root_index}")
                if checked is not None:
                    if checked in seen_roots:
                        bag.error(
                            "PG336",
                            "scene nodes must not contain duplicates",
                            f"{pointer}/nodes/{root_index}",
                        )
                    seen_roots.add(checked)
                    roots.append(checked)
        result.append(Scene(_optional_string(value, "name"), tuple(roots)))
    default_scene = None
    if "scene" in document:
        default_scene = _index(document["scene"], len(result), bag, "/scene")
    return tuple(result), default_scene


def _validate_node_graph(
    nodes: tuple[Node, ...], scenes: tuple[Scene, ...], bag: DiagnosticBag
) -> None:
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        for child in node.children:
            previous = parents.setdefault(child, parent_index)
            if previous != parent_index:
                bag.error(
                    "PG332", "node has more than one parent", f"/nodes/{parent_index}/children"
                )
    colors = [0] * len(nodes)
    for start in range(len(nodes)):
        if colors[start] != 0:
            continue
        pending = [(start, False)]
        while pending:
            index, exiting = pending.pop()
            if exiting:
                colors[index] = 2
                continue
            if colors[index] == 2:
                continue
            if colors[index] == 1:
                bag.error("PG333", "node hierarchy contains a cycle", f"/nodes/{index}")
                continue
            colors[index] = 1
            pending.append((index, True))
            for child in reversed(nodes[index].children):
                if colors[child] == 1:
                    bag.error("PG333", "node hierarchy contains a cycle", f"/nodes/{child}")
                elif colors[child] == 0:
                    pending.append((child, False))
    for scene_index, scene in enumerate(scenes):
        for root in scene.nodes:
            if root in parents:
                bag.error(
                    "PG334", "scene root is also a child node", f"/scenes/{scene_index}/nodes"
                )


def _estimate_output_prims(
    meshes: tuple[Mesh, ...],
    materials: tuple[Material, ...],
    cameras: tuple[Camera, ...],
    nodes: tuple[Node, ...],
    scenes: tuple[Scene, ...],
) -> int:
    mesh_costs = [1 + len(mesh.primitives) for mesh in meshes]
    subtree_costs = [0] * len(nodes)
    pending = [(index, False) for index in range(len(nodes))]
    while pending:
        index, exiting = pending.pop()
        node = nodes[index]
        if exiting:
            cost = 1
            if node.mesh is not None:
                cost += mesh_costs[node.mesh]
            if node.camera is not None:
                cost += 1
            cost += sum(subtree_costs[child] for child in node.children)
            subtree_costs[index] = cost
        elif subtree_costs[index] == 0:
            pending.append((index, True))
            pending.extend((child, False) for child in node.children)
    authored = (
        7
        + sum(mesh_costs)
        + len(cameras)
        + len(nodes)
        + sum(
            len(node.children) + (node.mesh is not None) + (node.camera is not None)
            for node in nodes
        )
        + len(scenes)
        + len(materials) * 12
    )
    composed = sum(subtree_costs[root] for scene in scenes for root in scene.nodes)
    return authored + composed


def _resource_failure(code: str, message: str, path: Path) -> Never:
    raise PygucError(
        message,
        error_report(
            code,
            f"{message}: {path}",
            suggestion="Check that the source path names a readable local .gltf or .glb file.",
        ),
    )


def _array(document: dict[str, Any], member: str, bag: DiagnosticBag) -> list[Any]:
    value = document.get(member, [])
    if not isinstance(value, list):
        bag.error("PG229", f"{member} must be an array", f"/{member}")
        return []
    return value


def _index(value: Any, length: int, bag: DiagnosticBag, pointer: str) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < length:
        bag.error("PG229", f"index must be between 0 and {max(length - 1, 0)}", pointer)
        return None
    return value


def _checked_index(value: Any, length: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < length:
        raise ValueError("index is out of range")
    return value


def _nonnegative_int(value: Any, bag: DiagnosticBag, pointer: str) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        bag.error("PG229", "value must be a non-negative integer", pointer)
        return None
    return value


def _optional_nonnegative_int(value: Any, default: int, bag: DiagnosticBag, pointer: str) -> int:
    if value is None:
        return default
    checked = _nonnegative_int(value, bag, pointer)
    return default if checked is None else checked


def _finite_float(value: Any, default: float, bag: DiagnosticBag, pointer: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        bag.error("PG229", "value must be a finite number", pointer)
        return default
    try:
        converted = float(value)
    except OverflowError:
        bag.error("PG229", "value must be a finite number", pointer)
        return default
    if not math.isfinite(converted):
        bag.error("PG229", "value must be a finite number", pointer)
        return default
    return converted


def _float_tuple(
    value: Any,
    length: int,
    default: tuple[float, ...],
    bag: DiagnosticBag,
    pointer: str,
) -> tuple[float, ...]:
    if value is None:
        return default
    if not isinstance(value, list) or len(value) != length:
        bag.error("PG229", f"value must be an array of {length} numbers", pointer)
        return default
    return tuple(
        _finite_float(component, default[index], bag, f"{pointer}/{index}")
        for index, component in enumerate(value)
    )


def _accessor_bound(
    value: Any, component_count: int, bag: DiagnosticBag, pointer: str
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != component_count:
        bag.error(
            "PG261",
            f"accessor bound must be an array of {component_count} finite numbers",
            pointer,
            "Recompute the accessor min and max metadata from the decoded values.",
        )
        return None
    result: list[float] = []
    for index, component in enumerate(value):
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            bag.error(
                "PG261", "accessor bound must contain only finite numbers", f"{pointer}/{index}"
            )
            return None
        try:
            converted = float(component)
        except OverflowError:
            bag.error(
                "PG261", "accessor bound must contain only finite numbers", f"{pointer}/{index}"
            )
            return None
        if not math.isfinite(converted):
            bag.error(
                "PG261", "accessor bound must contain only finite numbers", f"{pointer}/{index}"
            )
            return None
        result.append(converted)
    return tuple(result)


def _optional_string(value: dict[str, Any], member: str) -> str | None:
    result = value.get(member)
    return result if isinstance(result, str) else None


def _is_texcoord_semantic(value: str) -> bool:
    suffix = value.removeprefix("TEXCOORD_") if value.startswith("TEXCOORD_") else ""
    return bool(suffix) and suffix.isascii() and suffix.isdecimal() and str(int(suffix)) == suffix


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
