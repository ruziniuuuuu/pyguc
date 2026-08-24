# Troubleshooting

## The destination already exists

Choose a new directory or remove the old one yourself. `pyguc` never overwrites
output.

## A conversion failed

Catch `pyguc.PygucError` and inspect its `report`. Each diagnostic has a stable
`code`, readable `message`, severity, JSON-pointer-style `pointer`, and an
optional repair `suggestion`.

```python
import pyguc

try:
    pyguc.convert(source, destination)
except pyguc.PygucError as error:
    for diagnostic in error.report.diagnostics:
        print(
            diagnostic.code,
            diagnostic.pointer,
            diagnostic.message,
            diagnostic.suggestion,
        )
```

The destination is not created after a failed conversion.

Use `pyguc validate SOURCE` or `pyguc.validate(source)` when invalid input is an
expected outcome. Unexpected programming errors are deliberately not converted
into diagnostics.

## OpenUSD cannot be imported

Use CPython 3.12-3.14 on a platform supported by `usd-core` 26.8.x. Embedded DCC
Python environments are outside the supported runtime.
