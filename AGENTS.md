# AGENTS.md

## Project invariants

- This is local-first automation. GIMP owns pixels, rendering, loading, saving, and exporting.
- Never add OpenAI image generation, cloud image editing, remote image processing, or replacement-pixel generation.
- Never expose raw Script-Fu as a normal MCP tool. `GimpService.evaluate()` is internal only.
- MCP stdout is protocol traffic. Diagnostics belong on stderr through `logging`.
- Script-Fu framing and socket behavior belong in `gimp/protocol.py` and `gimp/transport.py`, not in MCP tools.
- PDB discovery and structured invocation are the path toward broad GIMP coverage.
- High-level tools should compose the service, serializers, transport, and PDB gateway rather than duplicate them.
- Preserve user files, explicit metadata choices, and GIMP undo history. Never overwrite by default.
- Use stable GIMP object IDs in MCP-facing state and do not invent properties GIMP has not exposed.
- Keep semantic vision optional and local: worker commands come from trusted configuration, image
  snapshots use temporary duplicate GIMP images, and mask artifacts use bounded lossless PNGs.
- Keep semantic segmentation, alpha refinement/matting, provider confidence, and ground-truth
  quality evidence as separate concepts; do not claim segmentation accuracy without ground truth.
- User-requested semantic extraction should leave persistent, clearly named GIMP layers. Automated
  integration fixtures clean up; an explicit final benchmark intentionally keeps its useful result.
- Derive background/remainder masks from the exact complement of the accepted foreground union.
  Report independently prompted overlap instead of silently assigning ownership.

## Working safely

- Inspect the repository and `git status` before editing.
- Use `apply_patch` for source edits.
- Run `ruff format src tests`, `ruff check src tests`, and `pytest -q` before handoff.
- Use deterministic fake Script-Fu servers for ordinary tests. Mark live GIMP tests with `integration` and skip cleanly when unavailable.
- For source changes involving control flow, reachability, calls, data flow, validation, error handling, cleanup, or state transitions, run real Joern analysis before and after the edit. Record focused queries and compare snapshots.
- Do not commit generated CPGs, caches, or local image files.
