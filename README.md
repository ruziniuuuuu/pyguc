# pyguc

[![PyPI version](https://img.shields.io/pypi/v/pyguc.svg)](https://pypi.org/project/pyguc/)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://ruziniuuuuu.github.io/pyguc/)

`pyguc` is a small, strict glTF 2.0 to OpenUSD converter. It turns a local
`.gltf` or `.glb` file into a self-contained `.usdc` or `.usda` asset bundle.

```console
pyguc convert model.glb model-usd
pyguc validate model.glb
```

```python
from pathlib import Path

import pyguc

result = pyguc.convert(Path("model.glb"), Path("model-usd"))
print(result.asset_path)
```

The destination directory must not exist. Conversion is atomic and never
replaces a destination that appears concurrently; a failure does not leave a
partial destination behind.

The current release targets static triangle assets and `UsdPreviewSurface`.
See the [user guide](https://ruziniuuuuu.github.io/pyguc/) for supported
features, diagnostics, and known limitations.

## Development

```console
uv sync
uv run prek install
uv run prek run --all-files
uv run pytest
mdbook serve --open
```

Releases use Conventional Commits and Release Please. Merging a release pull
request creates the version tag, GitHub release, signed distributions, and PyPI
publication through Trusted Publishing.

## Acknowledgements

Thanks to [guc](https://github.com/pablode/guc) and its contributors for their
work on glTF-to-USD conversion.

Licensed under Apache-2.0.
