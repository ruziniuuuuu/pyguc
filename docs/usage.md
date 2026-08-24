# Usage

Install `pyguc` in an environment supported by the `usd-core` package:

```console
uv add pyguc
```

## Command line

Convert to the default binary `.usdc` format:

```console
pyguc convert chair.glb chair-usd
```

Select text `.usda` output when needed:

```console
pyguc convert chair.gltf chair-usd --format usda
```

Validate without writing output:

```console
pyguc validate chair.glb
```

The command prints the generated USD path to standard output. Diagnostics are
printed to standard error, and failures return a non-zero status.

## Python

The same conversion is available as a library call:

```python
from pathlib import Path

import pyguc

result = pyguc.convert(
    Path("chair.glb"),
    Path("chair-usd"),
    format="usdc",
)

print(result.asset_path)  # chair-usd/asset.usdc
print(result.report.warnings)  # warnings collected during conversion
```

Validation returns expected input failures instead of raising them:

```python
report = pyguc.validate("chair.glb")
if not report.is_valid:
    for diagnostic in report.errors:
        print(diagnostic.code, diagnostic.pointer, diagnostic.suggestion)
```

The destination directory must not exist. `pyguc` first writes a sibling
staging directory, reopens and validates the USD, then commits it with an atomic
no-replace operation. Existing output, including one created by another process
during conversion, is never overwritten.

```text
chair-usd/
├── asset.usdc
└── assets/
    └── <sha256>.png
```

Move or rename the whole directory; all authored asset paths stay relative to
the bundle.

## Logging

`pyguc` uses the standard `pyguc` logger and does not configure handlers. An
application can opt in to phase-level messages:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("pyguc").setLevel(logging.INFO)
```

Warnings are returned as structured diagnostics and are not logged. Expected
conversion failures raise one `PygucError` carrying a `ValidationReport`; they
are not logged a second time.
