# Changelog

## Unreleased

- Added a reusable structured layer-mask gateway with safe mask inspection, creation, attachment,
  and enable/disable controls. Added bounded non-destructive `isolate_subject` support using a
  border-seeded high-key contiguous-color fallback, baseline/refined mask proxy metrics, and
  deterministic live synthetic-mask coverage. Native foreground extraction was probed on GIMP
  3.2.0 and is documented as unavailable through this Script-Fu bridge.
- Hardened `export_image` to refuse the unsafe `gimp-file-save` fallback when the tested GIMP
  Script-Fu bridge lacks a dedicated export binding.
- Added a bounded current-document context snapshot, recursive layer/group trees, GIMP 3
  multi-layer selection inspection/control, parent/image ownership checks, and reusable group-layer
  creation. Live tests preserve the pre-existing user image and use a separate temporary image for
  nested-group mutation coverage.
- Made the Forge project verifier suppress generated Python bytecode so checks do not mutate the
  candidate content scope.
- Added a reusable structured non-destructive drawable-filter gateway with Gaussian blur and
  brightness/contrast MCP tools, bounded GEGL parameters, filter identity/state inspection, and
  live GIMP integration coverage.
- Added a committed MNCS Forge project configuration and Provider Protocol checks that distinguish
  semantic PASS, FAIL, and unavailable/ skipped UNKNOWN results.
- Recorded live compatibility findings for GIMP 3.2.0: documented drawable-filter bindings work,
  while legacy Script-Fu PDB query helper bindings are not exposed in the tested environment.
- Added typed PDB parameter/return metadata models and explicit available, partial,
  unavailable, and malformed introspection states.
- Added conservative named-argument validation when a trusted metadata adapter reports
  complete names and requiredness; the default Script-Fu path continues to report its
  metadata limitation instead of guessing signatures.

## 0.1.0 — initial foundation

- Added an official MCP Python SDK 2.x stdio server.
- Added reconnectable GIMP Script-Fu TCP transport and documented framing.
- Added bounded Scheme parsing and safe structured serialization.
- Added image, layer, transform, selection, adjustment, undo, file-safety, and PDB tools.
- Added localhost-first configuration and `doctor` diagnostics.
- Added deterministic fake-server tests and documentation of live-GIMP limitations.
