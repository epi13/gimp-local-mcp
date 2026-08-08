# Changelog

## Unreleased

- Added a reusable Torch vision runtime with measured CPU, full-CUDA, and true Accelerate
  sequential-CPU-offload placement; dynamic free-VRAM reserve/cap policy; probed dtype selection;
  allocator/RSS/hook residency evidence; and bounded AUTO CUDA-OOM recovery.
- Migrated CLIPSeg from its fixed 1.5 GiB threshold to the shared runtime and added a same-input
  CPU/full-CUDA/sequential benchmark command. CUDA readiness now requires a real matmul, exposing
  incompatible wheels that merely report CUDA available.
- Replaced the SAM 3 diagnostic stub with an offline official Meta image adapter for tested text and
  box grounding, multiple masks/scores/boxes, explicit gated checkpoint setup, and honest binary
  mask/refinement capability boundaries. Normal operation never downloads weights.
- Added the real offline-by-default CLIPSeg worker in a separately managed vision environment,
  explicit checkpoint download, conservative CPU/CUDA selection, typed OOM errors, and provider
  readiness metadata for Torch, CUDA, device, VRAM, checkpoint, model load, and self-test state.
- Added `separate_subject_to_layers` and `separate_concepts_to_layers`. They leave persistent named
  subject/concept and complementary background layers, preserve and hide the unchanged source,
  bound concepts/instances/layer counts, report overlaps conservatively, and roll back GIMP state
  when creation or mask import fails.
- Added exact soft-mask complement/union algebra, overlap statistics, deterministic multi-instance
  and rollback tests, a disposable live GIMP decomposition test, and Forge claims that distinguish
  fake bridge correctness, real provider readiness, actual inference, live layering, and unknown
  semantic quality.
- Expanded `doctor` to distinguish NVIDIA driver detection from provider Torch/CUDA visibility and
  from checkpoint/model/self-test/text-segmentation readiness.
- Added an optional local semantic-vision subsystem with typed capability/request/candidate/result
  models, bounded JSONL worker protocol, strict lossless PNG mask artifacts, resident trusted
  worker client, and an explicit provider-alpha refinement boundary. The core package remains
  Python-only and does not install SAM, PyTorch, CUDA, Transformers, checkpoints, or cloud clients.
- Added safe duplicate-image GIMP snapshots and a temporary RGBA alpha bridge into the existing
  layer-mask gateway. Added `vision_status`, `segment_subject`, and `isolate_subject_vision`; auto
  subject isolation prefers a capable local semantic provider and otherwise retains the explicit
  high-key heuristic fallback. Added deterministic fake-worker tests and live synthetic bridge
  coverage. The SAM 3 worker reports unavailable until a separately installed audited adapter is
  configured.
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
