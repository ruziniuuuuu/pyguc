# Supported features

The current release supports:

- local `.gltf` and `.glb` input;
- `.usda` and `.usdc` output bundles;
- every glTF scene, including unreferenced nodes and meshes in a library;
- indexed and non-indexed triangles, triangle strips, and triangle fans;
- accessor offsets, strides, normalized integers, and sparse accessors;
- aligned matrix accessors and validated `POSITION` bounds;
- flat normal generation when normals are absent;
- perspective and orthographic cameras;
- core metallic-roughness materials authored as `UsdPreviewSurface`;
- PNG and JPEG textures, `TEXCOORD_n`, and core wrap modes;
- alpha modes, normal maps with tangents, occlusion, emissive, and double-sided materials.

The supported runtime is CPython 3.12-3.14 with `usd-core` 26.8.x and NumPy 2.x
on Linux x86-64, macOS universal2, and Windows x86-64.

Intentionally unsupported:

- animation, skins, morph targets, and time samples;
- points, lines, lights, vertex colors, and glTF extensions;
- explicit sampler minification or magnification filters;
- MaterialX, USDZ, Draco, meshopt, KTX, and WebP;
- non-default `normalTexture.scale` (it has no compliant `UsdPreviewSurface` representation);
- network, absolute, or parent-directory resource URIs;
- public plugin API;
- automated Storm rendering tests (the `usd-core` wheels do not include imaging).

An optional extension is ignored with one warning per extension only when valid
core fallback data exists. All `extras` members produce one aggregate warning.
A required extension is always an error. Warnings include repair guidance and
never consume the independent error-diagnostic budget.

Input is treated as untrusted: network and escaping resource paths are blocked,
resources are opened through anchored handles, and fixed limits bound individual
and cumulative bytes, JSON nodes, accessor values, decoded image pixels, and
estimated composed USD prims.
