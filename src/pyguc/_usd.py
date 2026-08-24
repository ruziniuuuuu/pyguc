"""Author and reopen-validate one self-contained OpenUSD bundle."""

from __future__ import annotations

import hashlib
import importlib
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any, Never

import numpy as np

from ._diagnostics import (
    DiagnosticBag,
    PygucError,
    ValidationReport,
    error_report,
)
from ._gltf import (
    GltfAsset,
    Material,
    NormalTextureInfo,
    OcclusionTextureInfo,
    TextureInfo,
    decode_accessor,
    triangulate,
)

# OpenUSD's extension modules do not publish typing information. Keeping these
# imports dynamic confines that untyped boundary to this module while ty checks
# the Python-owned parser and public API normally.
Gf: Any = importlib.import_module("pxr.Gf")
Kind: Any = importlib.import_module("pxr.Kind")
Sdf: Any = importlib.import_module("pxr.Sdf")
Tf: Any = importlib.import_module("pxr.Tf")
Usd: Any = importlib.import_module("pxr.Usd")
UsdGeom: Any = importlib.import_module("pxr.UsdGeom")
UsdShade: Any = importlib.import_module("pxr.UsdShade")
UsdUtils: Any = importlib.import_module("pxr.UsdUtils")
UsdValidation: Any = importlib.import_module("pxr.UsdValidation")
Vt: Any = importlib.import_module("pxr.Vt")


def author(asset: GltfAsset, bundle_root: Path, output_format: str) -> ValidationReport:
    """Author and validate one bundle, translating expected OpenUSD failures."""

    try:
        return _author(asset, bundle_root, output_format)
    except PygucError:
        raise
    except Tf.ErrorException as error:
        _conversion_failure(
            "PG508",
            f"OpenUSD failed while authoring the bundle: {error}",
            suggestion=(
                "Retry with a valid supported asset; report this diagnostic if the "
                "failure persists."
            ),
        )


def _author(asset: GltfAsset, bundle_root: Path, output_format: str) -> ValidationReport:
    """Author the fixed USD hierarchy, then close, reopen, and validate it."""

    output_path = bundle_root / f"asset.{output_format}"
    stage = Usd.Stage.CreateNew(str(output_path))
    if stage is None:
        _conversion_failure("PG501", "OpenUSD could not create the output layer", output_path)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset_prim = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(asset_prim)
    asset_prim.SetKind(Kind.Tokens.component)
    asset_prim.SetCustomDataByKey("pyguc:outputVersion", 1)
    asset_prim.SetCustomDataByKey("pyguc:version", version("pyguc"))
    asset_prim.SetCustomDataByKey("pyguc:sourceFile", asset.source.name)
    if asset.generator:
        asset_prim.SetCustomDataByKey("gltf:generator", asset.generator)
    if asset.copyright:
        asset_prim.SetCustomDataByKey("gltf:copyright", asset.copyright)

    UsdGeom.Scope.Define(stage, "/Asset/Scenes")
    library = UsdGeom.Scope.Define(stage, "/Asset/Library")
    UsdGeom.Imageable(library.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdGeom.Scope.Define(stage, "/Asset/Library/Nodes")
    UsdGeom.Scope.Define(stage, "/Asset/Library/Meshes")
    UsdGeom.Scope.Define(stage, "/Asset/Library/Cameras")
    UsdGeom.Scope.Define(stage, "/Asset/Materials")

    image_paths = _write_images(asset, bundle_root)
    usd_materials = [
        _author_material(stage, asset, material, index, image_paths)
        for index, material in enumerate(asset.materials)
    ]
    needs_default_material = any(
        primitive.material is None for mesh in asset.meshes for primitive in mesh.primitives
    )
    default_material = (
        _author_material(stage, asset, None, None, image_paths) if needs_default_material else None
    )

    for index, mesh in enumerate(asset.meshes):
        mesh_root = UsdGeom.Xform.Define(stage, f"/Asset/Library/Meshes/mesh_{index:04d}")
        _set_display_name(mesh_root.GetPrim(), mesh.name)
        mesh_root.GetPrim().SetCustomDataByKey("gltf:index", index)
        for primitive_index, primitive in enumerate(mesh.primitives):
            mesh_schema = UsdGeom.Mesh.Define(
                stage,
                f"/Asset/Library/Meshes/mesh_{index:04d}/primitive_{primitive_index:04d}",
            )
            _author_primitive(mesh_schema, asset, primitive)
            material = (
                usd_materials[primitive.material]
                if primitive.material is not None
                else default_material
            )
            if material is None:
                raise RuntimeError("default material was not authored")
            UsdShade.MaterialBindingAPI.Apply(mesh_schema.GetPrim()).Bind(material)
            material_model = (
                asset.materials[primitive.material] if primitive.material is not None else None
            )
            mesh_schema.CreateDoubleSidedAttr().Set(
                material_model.double_sided if material_model is not None else False
            )
            mesh_schema.GetPrim().SetCustomDataByKey("gltf:primitiveIndex", primitive_index)

    for index, camera in enumerate(asset.cameras):
        camera_schema = UsdGeom.Camera.Define(stage, f"/Asset/Library/Cameras/camera_{index:04d}")
        _set_display_name(camera_schema.GetPrim(), camera.name)
        camera_schema.GetPrim().SetCustomDataByKey("gltf:index", index)
        _author_camera(camera_schema, camera.kind, camera.values)

    usd_nodes = []
    for index, node in enumerate(asset.nodes):
        node_schema = UsdGeom.Xform.Define(stage, f"/Asset/Library/Nodes/node_{index:04d}")
        node_prim = node_schema.GetPrim()
        usd_nodes.append(node_prim)
        _set_display_name(node_prim, node.name)
        node_prim.SetCustomDataByKey("gltf:index", index)
        _author_transform(node_schema, node)
    for node_prim, node in zip(usd_nodes, asset.nodes, strict=True):
        if node.mesh is not None:
            _add_internal_reference(
                stage,
                f"{node_prim.GetPath()}/mesh",
                f"/Asset/Library/Meshes/mesh_{node.mesh:04d}",
            )
        if node.camera is not None:
            _add_internal_reference(
                stage,
                f"{node_prim.GetPath()}/camera",
                f"/Asset/Library/Cameras/camera_{node.camera:04d}",
            )
        for child in node.children:
            _add_internal_reference(
                stage,
                f"{node_prim.GetPath()}/node_{child:04d}",
                f"/Asset/Library/Nodes/node_{child:04d}",
            )

    for index, scene in enumerate(asset.scenes):
        scene_schema = UsdGeom.Xform.Define(stage, f"/Asset/Scenes/scene_{index:04d}")
        scene_prim = scene_schema.GetPrim()
        _set_display_name(scene_prim, scene.name)
        scene_prim.SetCustomDataByKey("gltf:index", index)
        visibility = (
            UsdGeom.Tokens.inherited
            if asset.default_scene is not None and index == asset.default_scene
            else UsdGeom.Tokens.invisible
        )
        scene_schema.CreateVisibilityAttr().Set(visibility)
        for node in scene.nodes:
            _add_internal_reference(
                stage,
                f"{scene_prim.GetPath()}/node_{node:04d}",
                f"/Asset/Library/Nodes/node_{node:04d}",
            )

    if not stage.GetRootLayer().Save():
        _conversion_failure("PG502", "OpenUSD could not save the output layer", output_path)
    stage = None
    return _validate_output(output_path, bundle_root)


def _write_images(asset: GltfAsset, root: Path) -> tuple[str, ...]:
    result: list[str] = []
    for image in asset.images:
        digest = hashlib.sha256(image.data).hexdigest()
        extension = ".png" if image.mime_type == "image/png" else ".jpg"
        relative = f"assets/{digest}{extension}"
        path = root / relative
        if not path.exists():
            path.write_bytes(image.data)
        result.append(relative)
    return tuple(result)


def _author_primitive(mesh: Any, asset: GltfAsset, primitive: Any) -> None:
    points = np.asarray(decode_accessor(asset, primitive.attributes["POSITION"]), dtype=np.float32)
    point_count = len(points)
    source_indices = (
        np.arange(point_count, dtype=np.int64)
        if primitive.indices is None
        else decode_accessor(asset, primitive.indices).astype(np.int64, copy=False)
    )
    triangles = triangulate(primitive.mode, source_indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr().Set(triangles.reshape(-1).tolist())
    if point_count:
        minimum = np.min(points, axis=0).tolist()
        maximum = np.max(points, axis=0).tolist()
        extent = Vt.Vec3fArray([Gf.Vec3f(*minimum), Gf.Vec3f(*maximum)])
        mesh.CreateExtentAttr().Set(extent)

    normal_index = primitive.attributes.get("NORMAL")
    if normal_index is not None:
        normals = np.asarray(decode_accessor(asset, normal_index), dtype=np.float32)
        mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    else:
        face_normals = _flat_normals(points, triangles)
        mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(face_normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

    primvars = UsdGeom.PrimvarsAPI(mesh)
    for semantic, accessor_index in sorted(primitive.attributes.items()):
        if not semantic.startswith("TEXCOORD_"):
            continue
        set_index = int(semantic.removeprefix("TEXCOORD_"))
        uv = np.asarray(decode_accessor(asset, accessor_index), dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        name = "st" if set_index == 0 else f"st{set_index}"
        primvar = primvars.CreatePrimvar(
            name, Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        primvar.Set(Vt.Vec2fArray.FromNumpy(uv))

    tangent_index = primitive.attributes.get("TANGENT")
    if tangent_index is not None and normal_index is not None:
        tangent4 = np.asarray(decode_accessor(asset, tangent_index), dtype=np.float32)
        normals = np.asarray(decode_accessor(asset, normal_index), dtype=np.float32)
        tangents = tangent4[:, :3]
        bitangents = np.cross(normals, tangents) * tangent4[:, 3:4]
        tangent_primvar = primvars.CreatePrimvar(
            "tangents", Sdf.ValueTypeNames.Vector3fArray, UsdGeom.Tokens.vertex
        )
        tangent_primvar.Set(Vt.Vec3fArray.FromNumpy(tangents))
        bitangent_primvar = primvars.CreatePrimvar(
            "bitangents", Sdf.ValueTypeNames.Vector3fArray, UsdGeom.Tokens.vertex
        )
        bitangent_primvar.Set(Vt.Vec3fArray.FromNumpy(bitangents))


def _flat_normals(
    points: np.ndarray[Any, Any], triangles: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if not len(triangles):
        return np.empty((0, 3), dtype=np.float32)
    edge_a = points[triangles[:, 1]] - points[triangles[:, 0]]
    edge_b = points[triangles[:, 2]] - points[triangles[:, 0]]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = lengths[:, 0] <= np.finfo(np.float32).eps
    lengths[degenerate] = 1.0
    normals = normals / lengths
    normals[degenerate] = (0.0, 0.0, 1.0)
    return np.repeat(normals.astype(np.float32), 3, axis=0)


def _author_material(
    stage: Any,
    asset: GltfAsset,
    model: Material | None,
    index: int | None,
    image_paths: tuple[str, ...],
) -> Any:
    suffix = "default" if index is None else f"{index:04d}"
    material = UsdShade.Material.Define(stage, f"/Asset/Materials/material_{suffix}")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/preview_surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface_output)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    if model is None:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        material.GetPrim().SetDisplayName("Default glTF material")
        return material

    _set_display_name(material.GetPrim(), model.name)
    material.GetPrim().SetCustomDataByKey("gltf:index", index)
    material.GetPrim().SetCustomDataByKey("gltf:alphaMode", model.alpha_mode)
    material.GetPrim().SetCustomDataByKey("gltf:doubleSided", model.double_sided)

    uv_readers: dict[int, Any] = {}

    def texture_shader(
        label: str,
        info: TextureInfo | NormalTextureInfo | OcclusionTextureInfo,
        *,
        color_space: str,
        scale: tuple[float, float, float, float],
        bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> Any:
        texture = asset.textures[info.index]
        image_path = image_paths[texture.source]
        texture_node = UsdShade.Shader.Define(stage, f"{material.GetPath()}/{label}_texture")
        texture_node.CreateIdAttr("UsdUVTexture")
        texture_node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(image_path))
        texture_node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        texture_node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(scale)
        texture_node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(bias)
        sampler = asset.samplers[texture.sampler] if texture.sampler is not None else None
        wrap_s = sampler.wrap_s if sampler is not None else 10497
        wrap_t = sampler.wrap_t if sampler is not None else 10497
        texture_node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(_wrap_token(wrap_s))
        texture_node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(_wrap_token(wrap_t))
        reader = uv_readers.get(info.texcoord)
        if reader is None:
            reader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/uv_{info.texcoord}")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(
                "st" if info.texcoord == 0 else f"st{info.texcoord}"
            )
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            uv_readers[info.texcoord] = reader
        texture_node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.GetOutput("result")
        )
        texture_node.CreateOutput("r", Sdf.ValueTypeNames.Float)
        texture_node.CreateOutput("g", Sdf.ValueTypeNames.Float)
        texture_node.CreateOutput("b", Sdf.ValueTypeNames.Float)
        texture_node.CreateOutput("a", Sdf.ValueTypeNames.Float)
        texture_node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        return texture_node

    base = model.base_color_factor
    if model.base_color_texture is None:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(base[:3])
        if model.alpha_mode != "OPAQUE":
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(base[3])
    else:
        base_texture = texture_shader(
            "base_color",
            model.base_color_texture,
            color_space="sRGB",
            scale=base,
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            base_texture.GetOutput("rgb")
        )
        if model.alpha_mode != "OPAQUE":
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
                base_texture.GetOutput("a")
            )
    if model.alpha_mode == "MASK":
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(model.alpha_cutoff)

    if model.metallic_roughness_texture is None:
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(model.metallic_factor)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(model.roughness_factor)
    else:
        packed = texture_shader(
            "metallic_roughness",
            model.metallic_roughness_texture,
            color_space="raw",
            scale=(1.0, model.roughness_factor, model.metallic_factor, 1.0),
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
            packed.GetOutput("g")
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
            packed.GetOutput("b")
        )

    if model.emissive_texture is None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(model.emissive_factor)
    else:
        emissive = texture_shader(
            "emissive",
            model.emissive_texture,
            color_space="sRGB",
            scale=(*model.emissive_factor, 1.0),
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            emissive.GetOutput("rgb")
        )

    if model.occlusion_texture is not None:
        strength = model.occlusion_texture.strength
        occlusion = texture_shader(
            "occlusion",
            model.occlusion_texture,
            color_space="raw",
            scale=(strength, strength, strength, 1.0),
            bias=(1.0 - strength, 1.0 - strength, 1.0 - strength, 0.0),
        )
        shader.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
            occlusion.GetOutput("r")
        )

    if model.normal_texture is not None:
        normal = texture_shader(
            "normal",
            model.normal_texture,
            color_space="raw",
            scale=(2.0, 2.0, 2.0, 1.0),
            bias=(-1.0, -1.0, -1.0, 0.0),
        )
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normal.GetOutput("rgb")
        )
    return material


def _author_transform(xform: Any, node: Any) -> None:
    if node.matrix is not None:
        # glTF's column-major serialization maps directly to Gf's row-vector
        # in-memory convention; transposing here would lose translation.
        xform.AddTransformOp().Set(Gf.Matrix4d(*node.matrix))
        return
    if node.translation != (0.0, 0.0, 0.0):
        xform.AddTranslateOp().Set(Gf.Vec3d(*node.translation))
    if node.rotation != (0.0, 0.0, 0.0, 1.0):
        x, y, z, w = node.rotation
        xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    if node.scale != (1.0, 1.0, 1.0):
        xform.AddScaleOp().Set(Gf.Vec3f(*node.scale))


def _author_camera(camera: Any, kind: str, values: Any) -> None:
    gf_camera = Gf.Camera()
    if kind == "perspective":
        gf_camera.SetPerspectiveFromAspectRatioAndFieldOfView(
            values.get("aspectRatio", 1.0),
            math.degrees(values["yfov"]),
            Gf.Camera.FOVVertical,
        )
        gf_camera.clippingRange = Gf.Range1f(values["znear"], values.get("zfar", 1_000_000.0))
    else:
        gf_camera.SetOrthographicFromAspectRatioAndSize(
            values["xmag"] / values["ymag"], values["ymag"] * 2.0, Gf.Camera.FOVVertical
        )
        gf_camera.clippingRange = Gf.Range1f(values["znear"], values["zfar"])
    camera.SetFromCamera(gf_camera)
    camera.GetPrim().RemoveProperty("xformOp:transform")
    camera.GetPrim().RemoveProperty("xformOpOrder")


def _add_internal_reference(stage: Any, path: str, target: str) -> None:
    prim = stage.DefinePrim(path, "Xform")
    prim.SetInstanceable(False)
    prim.GetReferences().AddInternalReference(target)


def _validate_output(output_path: Path, bundle_root: Path) -> ValidationReport:
    stage = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadAll)
    if stage is None:
        _conversion_failure("PG503", "OpenUSD could not reopen the generated layer", output_path)
    default_prim = stage.GetDefaultPrim()
    if not default_prim or default_prim.GetPath() != Sdf.Path("/Asset"):
        _conversion_failure("PG504", "generated USD has an invalid default prim", output_path)

    registry = UsdValidation.ValidationRegistry()
    selected_names: list[str] = []
    for metadata in registry.GetAllValidatorMetadata():
        keywords = {str(value) for value in metadata.GetKeywords()}
        if keywords & {"UsdCoreValidators", "UsdGeomValidators", "UsdShadeValidators"}:
            selected_names.append(str(metadata.name))
    validators = registry.GetOrLoadValidatorsByName(selected_names)
    bag = DiagnosticBag()
    for issue in UsdValidation.ValidationContext(validators).Validate(stage):
        if issue.HasNoError():
            continue
        identifier = str(issue.GetIdentifier())
        message = f"{identifier}: {issue.GetMessage()}"
        if issue.GetType() == UsdValidation.ValidationErrorType.Error:
            bag.error(
                "PG505",
                message,
                "",
                "Report this generated-USD validation failure with the source asset.",
            )
        elif issue.GetType() == UsdValidation.ValidationErrorType.Warn:
            bag.warning(
                "PG404",
                message,
                "",
                "Inspect the generated USD warning before using the bundle in production.",
            )
    if not bag.report.is_valid:
        raise PygucError("generated USD failed OpenUSD validation", bag.report)

    root = bundle_root.resolve(strict=True)
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset:
                value = attribute.Get()
                if isinstance(value, Sdf.AssetPath) and value.path:
                    _validate_asset_path(value.path, root, attribute.GetPath())
            elif attribute.GetTypeName() == Sdf.ValueTypeNames.AssetArray:
                for value in attribute.Get() or []:
                    if value.path:
                        _validate_asset_path(value.path, root, attribute.GetPath())

    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(str(output_path.resolve(strict=True)))
    )
    if unresolved:
        for value in unresolved:
            bag.error(
                "PG506",
                f"unresolved USD dependency: {value}",
                "",
                "Ensure every authored dependency exists inside the output bundle.",
            )
        raise PygucError("generated USD has unresolved dependencies", bag.report)
    return bag.report


def _validate_asset_path(value: str, root: Path, pointer: Any) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        _conversion_failure("PG507", "generated asset path is not bundle-relative", pointer)
    try:
        resolved = (root / path).resolve(strict=True)
    except OSError:
        _conversion_failure("PG506", "generated asset path does not resolve", pointer)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        _conversion_failure("PG507", "generated asset path escapes the bundle", pointer)


def _wrap_token(value: int) -> str:
    return {33071: "clamp", 33648: "mirror", 10497: "repeat"}[value]


def _set_display_name(prim: Any, name: str | None) -> None:
    if name:
        prim.SetDisplayName(name)


def _conversion_failure(
    code: str,
    message: str,
    context: Any | None = None,
    suggestion: str | None = None,
) -> Never:
    rendered = f"{message}: {context}" if context is not None else message
    raise PygucError(message, error_report(code, rendered, suggestion=suggestion))
