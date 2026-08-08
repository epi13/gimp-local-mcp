# Contributing

GIMP Local MCP is early-stage. Small, focused changes are easiest to review.

## Development setup

```bash
python -m pip install -e '.[dev]'
ruff format src tests
ruff check src tests
pytest -q
```

Do not make the ordinary suite depend on GIMP. Use the deterministic Script-Fu TCP fixture for transport behavior and mark live-GIMP tests with `integration`.

## Design expectations

- Keep GIMP responsible for image manipulation and rendering.
- Put protocol/framing logic in `gimp/protocol.py` and `gimp/transport.py`.
- Add structured serializers instead of concatenating untrusted Scheme.
- Prefer a reusable service/PDB operation over many near-identical MCP wrappers.
- Keep MCP stdout clean; use the logging module for diagnostics.
- Preserve user files, metadata choices, and undo history.
- Do not expose raw Scheme, shell execution, Python eval, or cloud image processing.
- Add unit tests for policy and serialization changes, and live tests only when they skip cleanly.

## Pull requests

Describe the user-facing behavior, safety implications, tests run, and any live-GIMP validation. Keep README claims aligned with the actual implementation.
