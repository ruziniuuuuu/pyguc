# Repository guidance

- Support CPython 3.12-3.14 and keep the public API intentionally small.
- Prefer clear Python over abstractions.
- Keep conversion deterministic, strict, and atomic.
- Never configure the root logger; use `logging.getLogger("pyguc")`.
- Preserve stable diagnostic codes and JSON-pointer-style locations.
- Treat every glTF and URI as untrusted input.
- Do not read outside the source directory or write outside the bundle.
- Keep private modules private; avoid plugin registries and compatibility shims.
- Add tests for every behavior and every fixed failure mode.
- Run `uv run prek run --all-files` and `uv run pytest` before handoff.
- Format and lint with Ruff; type-check with ty.
- Use `uv` for environments, dependencies, locking, and command execution.
- Keep user docs concise, English-only, and synchronized with behavior.
- Document unsupported features rather than silently ignoring them.
