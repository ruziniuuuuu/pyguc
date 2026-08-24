# pyguc

`pyguc` converts local glTF 2.0 assets into relocatable OpenUSD bundles. It is
strict by design: unsupported or ambiguous content fails with a precise
diagnostic instead of producing a surprising asset.

The current release supports static triangle geometry, cameras, textures, and
core PBR materials authored as `UsdPreviewSurface`.
