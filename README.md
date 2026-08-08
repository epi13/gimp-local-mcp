# GIMP Local MCP

GIMP Local MCP is a local-first [Model Context Protocol](https://modelcontextprotocol.io/) server for controlling GIMP 3 through an LLM. The LLM chooses an operation through structured MCP calls; GIMP owns the image manipulation, rendering, file loading, and exporting. Images and files remain on the user’s machine.

This v0.1 implementation favors a small, composable architecture over hundreds of one-off wrappers.

## Philosophy

- GIMP owns pixels. This project does not generate replacement pixels or call cloud image services.
- Routine edits use ergonomic tools with stable image and layer IDs.
- The live GIMP PDB is the long-term path to broad coverage, with structured invocation rather than raw Scheme input.
- File writes are explicit and never overwrite existing targets by default.
- The bridge preserves GIMP’s undo history and groups multi-step operations when the operation needs multiple PDB calls.

## Architecture

```text
Natural language
      |
      v
MCP host (Codex, Claude, custom client)
      |
      | MCP stdio / JSON-RPC
      v
gimp-local-mcp
  ergonomic tools ---> GimpService ---> safe Scheme serializer
                              |                  |
                              +--> PDB catalog/invoker
                              +--> structured GEGL filter gateway
                              +--> VisionClient ---> trusted local JSONL worker
                                                   |
                                                   v
                                    Script-Fu TCP framing (127.0.0.1:10008)
                                                   |
                                                   v
                                             GIMP 3 Script-Fu server
```

The official MCP Python SDK is used for the server. Script-Fu socket and framing logic is isolated in `gimp/transport.py` and `gimp/protocol.py`; tool functions do not manipulate sockets directly.

## Requirements

- Python 3.10 or newer
- GIMP 3.x
- A running GIMP 3 Script-Fu server

GIMP is not required for the ordinary unit test suite.

## GIMP 3 setup

1. Start GIMP 3.
2. Open **Filters → Development → Script-Fu → Start Server**.
3. Keep the listener on `127.0.0.1` and port `10008` unless you have a specific reason to change it.
4. Start the MCP server after the Script-Fu server is listening.

GIMP’s Script-Fu server accepts a three-byte `G`/16-bit-length request frame and returns a four-byte `G`/error/16-bit-length response frame. GIMP allows one Script-Fu client at a time; this server maintains one reconnectable client connection.

## Installation

From a checkout:

```bash
python -m pip install -e '.[dev]'
```

For a runtime-only installation:

```bash
python -m pip install .
```

## Run the server

```bash
gimp-local-mcp serve
```

With no subcommand, `gimp-local-mcp` also starts the stdio MCP server. Stdout is reserved for MCP protocol traffic; diagnostics go to stderr.

Check the local connection independently:

```bash
gimp-local-mcp doctor
```

`doctor` prints Python details, the configured endpoint, local-only status, Script-Fu reachability, GIMP version, and open-image count. A connection refusal is expected until GIMP’s Script-Fu server is started.

## Codex configuration

Add a project-local MCP entry in the Codex configuration used for this checkout. The exact location depends on the Codex client version; do not modify global configuration automatically.

```toml
[mcp_servers.gimp-local-mcp]
command = "gimp-local-mcp"
args = ["serve"]
```

If the command is not on the client’s PATH, use the absolute path to the virtual-environment executable or launch it with Python:

```toml
[mcp_servers.gimp-local-mcp]
command = "python"
args = ["-m", "gimp_local_mcp.cli", "serve"]
```

## Configuration

Defaults work with a standard local GIMP setup:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `GIMP_MCP_HOST` | `127.0.0.1` | Script-Fu host |
| `GIMP_MCP_PORT` | `10008` | Script-Fu port |
| `GIMP_MCP_TIMEOUT` | `10` | TCP operation timeout in seconds |
| `GIMP_MCP_LOG_LEVEL` | `INFO` | stderr log level |
| `GIMP_MCP_MAX_RESPONSE_BYTES` | `65535` | bounded response body size |
| `GIMP_MCP_ALLOW_REMOTE` | `false` | explicit opt-in for non-loopback hosts |
| `GIMP_MCP_VISION_PROVIDER` | `none` | optional trusted local provider name, such as `clipseg` |
| `GIMP_MCP_VISION_COMMAND` | unset | trusted local worker command; never an MCP argument |
| `GIMP_MCP_VISION_TIMEOUT` | `60` | local worker response timeout in seconds, capped at 600 |
| `GIMP_MCP_VISION_DEVICE` | `auto` | provider device policy: `auto`, `cpu`, or `cuda` |

Remote connections are rejected unless `GIMP_MCP_ALLOW_REMOTE=true` is set. Localhost-only operation is the secure default.

## MCP tools

The current server registers 57 tools across these groups:

- Session: `gimp_status`, `gimp_capabilities`, `list_open_images`, `get_active_image`, `get_current_context`, `get_image_info`
- Files and images: `open_image`, `create_image`, `save_xcf`, `export_image`, `close_image`
- Layers: `list_layers`, `get_layer_tree`, `get_layer_info`, `get_selected_layers`, `set_selected_layers`, `create_layer`, `create_layer_group`, `duplicate_layer`, `rename_layer`, `delete_layer`, `set_layer_visibility`, `set_layer_opacity`, `set_layer_mode`, `move_layer`, `merge_down`
- Transforms: `resize_image`, `resize_canvas`, `crop_image`, `rotate_image`, `flip_image`
- Selection: `select_all`, `select_none`, `invert_selection`, `select_rectangle`, `select_ellipse`, `select_layer_alpha`
- Masks and subject isolation: `get_layer_mask_info`, `create_layer_mask`, `set_layer_mask_enabled`, `isolate_subject`, `isolate_subject_vision`, `separate_subject_to_layers`, `separate_concepts_to_layers`
- Optional local vision: `vision_status`, `segment_subject`
- Adjustments and undo: `brightness_contrast`, `hue_saturation`, `desaturate`, `undo`, `redo`
- Non-destructive filters: `apply_gaussian_blur_filter`, `apply_brightness_contrast_filter`, `list_drawable_filters`
- PDB: `search_pdb`, `describe_pdb_procedure`, `invoke_pdb_procedure`

`invoke_pdb_procedure` accepts JSON-compatible structured values, including `{ "scheme_symbol": "RGB" }` for a GIMP enum. It does not accept Scheme source, Python, shell commands, or arbitrary evaluation. Runtime PDB counts and documentation are used when available. Procedure descriptions now include a bounded typed-metadata state (`available`, `partial`, `unavailable`, or `malformed`) plus argument/return records when a trusted structured adapter reports them. The default Script-Fu TCP adapter reports argument metadata as unavailable because Script-Fu does not expose a stable `GimpProcedure`/`GParamSpec` representation; no signatures or types are guessed. Named-argument validation is performed only when trustworthy names are actually available.

GIMP 3 also exposes non-destructive drawable filters through special Script-Fu bindings rather than ordinary PDB procedures. The filter gateway uses the documented `gimp-drawable-append-new-filter` binding with structured named GEGL properties, then reads the actual filter ID and state back through GIMP. The current high-level slices are Gaussian blur and brightness/contrast; they preserve the existing destructive adjustment tool names.

Layer masks use a reusable gateway over the GIMP bindings `gimp-layer-create-mask`,
`gimp-layer-add-mask`, `gimp-layer-get-mask`, and the mask state accessors. Existing masks are
reported and never replaced by the public creation tool. `isolate_subject` duplicates the
explicit source layer, measures the perimeter, and seeds a contiguous-color background selection
from either light perimeter samples or all perimeter samples when high-key evidence is strong.
It inverts that selection into a mask and leaves the source layer intact. The
operation returns baseline and refined observable mask proxies: border transparency, transparent
and opaque proportions, partial-alpha proportion, edge-transition samples, a sampled retained
bounding box, and retention relative to the baseline's confident samples. These are not semantic
accuracy scores.

### Optional local semantic vision

Semantic vision is deliberately out of process. `VisionClient` speaks a bounded JSONL protocol to
a trusted command configured by the operator; the core installation does not install PyTorch,
CUDA, Transformers, SAM, or model weights. A worker returns structured candidates and lossless
grayscale PNG mask artifacts. Diagnostics go to worker stderr, while protocol traffic remains
stdout-only.

CLIPSeg through `tools/vision/clipseg_worker.py` remains the lightweight fallback. It supports
local free-text concept masks and fits CPU-only systems much better than SAM 3. It loads
checkpoints with `local_files_only=True` during ordinary operation; downloading is a separate
explicit setup step. `tools/vision/sam3_worker.py` is now an adapter for the official Meta SAM 3
image API. It supports the official text and positive-box grounding paths, multiple scored masks,
and boxes only after a real checkpoint self-test succeeds. Point prompting belongs to SAM 3's
separate interactive predictor and is not claimed by this adapter. No normal request downloads a
checkpoint.

Create a separate provider environment; this does not change `pip install .`:

```bash
python3.13 -m venv .venv-vision
.venv-vision/bin/python -m pip install --upgrade pip
.venv-vision/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-vision/bin/python -m pip install transformers accelerate pillow psutil
.venv-vision/bin/python tools/vision/clipseg_worker.py --download-model

export GIMP_MCP_VISION_PROVIDER=clipseg
export GIMP_MCP_VISION_COMMAND="$PWD/.venv-vision/bin/python $PWD/tools/vision/clipseg_worker.py"
export GIMP_MCP_VISION_DEVICE=auto
export GIMP_MCP_VISION_OFFLOAD=auto
export GIMP_MCP_VISION_GPU_RESERVE_MIB=256
export GIMP_MCP_VISION_DTYPE=auto
export GIMP_MCP_VISION_TIMEOUT=120
gimp-local-mcp doctor
```

CLIPSeg converts its coarse semantic logits into a soft mask with conservative defaults of `0.2`
for the probability threshold and `2.0` for the sigmoid slope. Operators can tune those generic
controls with `GIMP_MCP_CLIPSEG_MASK_THRESHOLD` (`0.01`–`0.99`) and
`GIMP_MCP_CLIPSEG_MASK_SLOPE` (`0.25`–`8.0`). Higher thresholds reject weaker surrounding
regions; higher slopes make the transition firmer. These settings refine a semantic mask but do
not turn it into a fur/hair alpha matte.

Device and offload are separate policies:

- `GIMP_MCP_VISION_DEVICE=auto|cpu|cuda` chooses the execution device.
- `GIMP_MCP_VISION_OFFLOAD=auto|none|sequential-cpu` controls weight placement.
- `GIMP_MCP_VISION_GPU_RESERVE_MIB` preserves dynamic headroom from the free VRAM measured at
  worker startup. `GIMP_MCP_VISION_MAX_VRAM_MIB` can impose an optional planning cap.
- `GIMP_MCP_VISION_DTYPE=auto|float32|float16|bfloat16` accepts reduced precision only after a real
  execution probe. AUTO remains FP32 on Pascal-class GPUs without Tensor Cores.

AUTO first executes an actual CUDA matmul; discovery alone is insufficient. It subtracts the
configured reserve from current free VRAM and uses full CUDA only when estimated model storage plus
workspace fits. Otherwise it uses Hugging Face Accelerate's `cpu_offload` when the provider supports
it, or CPU. Sequential CPU offload keeps primary parameter storage in **system RAM**, attaches
layer-wise execution hooks, moves parameters to CUDA as their modules execute, and removes them
after use. It lowers persistent VRAM residency but does not lower system-RAM requirements. Disk
offload is a separate slower tradeoff and is intentionally not the default.

AUTO has a bounded OOM path: full CUDA may retry sequential offload, then CPU; explicit device and
offload selections fail instead of silently violating operator intent. Vision completes before any
document mutation, so provider OOM cannot leave partial GIMP edits. `doctor` reports the placement
reason, CUDA kernel probe and compiled architectures, startup free VRAM, reserve/budget, Torch
allocated/reserved peaks, process RSS, and evidence from Accelerate hooks and persistent parameter
devices. Torch allocator peaks and `nvidia-smi` process usage are different measurements.

For an explicit official SAM 3 setup, install Meta's package according to its prerequisites, obtain
access to the gated checkpoint, authenticate outside MCP, and run:

```bash
.venv-sam3/bin/python tools/vision/sam3_worker.py --download-checkpoint
export GIMP_MCP_SAM3_CHECKPOINT=/absolute/path/printed/by/setup/sam3.1_multiplex.pt
export GIMP_MCP_VISION_PROVIDER=sam3
export GIMP_MCP_VISION_COMMAND="$PWD/.venv-sam3/bin/python $PWD/tools/vision/sam3_worker.py"
```

The worker passes `load_from_HF=False` to the official builder during normal operation and reports
unavailable if the explicit file is absent. Use the bounded comparison command for one provider and
input across placements:

```bash
python tools/vision/provider_check.py \
  --benchmark cpu,full-cuda,sequential-cpu-offload,auto
```

`segment_subject(image_id, layer_id, prompt="red fox")` snapshots a duplicate of the current
GIMP image to a uniquely named temporary PNG, invokes the local worker, validates candidate
artifacts, and removes the snapshot and artifacts after returning bounded summaries. It does not
mutate the document. `isolate_subject_vision` uses the same bridge, duplicates the explicit source
layer, imports the mask through a temporary RGBA layer/selection, and attaches it through the
existing layer-mask gateway. `isolate_subject(strategy="auto", prompt="red fox")` prefers a
capable semantic worker and falls back to the existing high-key contiguous-color strategy when no
provider is available. Explicit `high-key-background` and `border-color` retain the heuristic.

`separate_subject_to_layers(image_id, layer_id, prompt="red fox")` leaves a persistent group with
`Subject — red fox` and `Background`, hides but does not modify the original source, and selects the
subject layer. The background mask is the exact 8-bit complement of the subject/foreground-union
artifact; both layers use duplicates of the source. `separate_concepts_to_layers` accepts up to
eight concepts, preserves distinct provider instances when returned, bounds the total generated
layers to 24, reports mask overlap without assigning ownership, and derives the remainder from the
soft-alpha foreground union. Provider failures happen before GIMP mutation; mask-import failures
roll back generated layers, the group, source visibility, and selected-layer state.

GIMP 3.2 stores imported partial mask samples in its image-space encoding. In live validation,
complementary 128/127 artifact values read back near 188/187; after decoding to linear alpha their
sum was 1.0 within 0.005. Binary endpoints remained exact. This is reported as bridge behavior and
is not described as learned matting.

The current refinement boundary is `IdentityMaskRefiner`: provider output is retained, but this is
not learned alpha matting. CLIPSeg emits a coarse semantic probability mask; SAM 3 emits binary
segmentation masks. Semantic/object discovery, segmentation, optional future edge/alpha refinement,
artifact validation, and GIMP mask creation remain distinct stages. Coverage, border transparency,
partial-alpha, edge-transition, and sampled bounding-box values are observable proxies only. The
fox-on-snow benchmark has no ground-truth alpha matte, so confidence and proxy metrics must not be
presented as accuracy.

On the prior 1280×960 fox benchmark, the cached CLIPSeg model reported about 603 MB of parameter
storage and completed CPU inference in 0.49 seconds after a 12.66-second cold load. The new
placement comparison used the same deterministic 32×32 image/prompt on the P620 with PyTorch
2.7.1+cu126 and 128 MiB reserve. CPU/full-CUDA/sequential inference took 0.272/0.178/0.307 seconds.
Full CUDA peaked at 660,646,912 allocated and 687,865,856 reserved bytes; sequential offload peaked
at 131,516,416 allocated and 146,800,640 reserved bytes, an 80.1% lower allocated peak. The offload
run had 233 hooks, 602,990,984 meta/CPU-backed parameter bytes, zero persistent CUDA parameter
bytes, and `sequential_offload_verified=true`. Full CUDA retained all 602,990,984 parameter bytes on
CUDA. Both CUDA placements produced identical masks; compared with CPU, mean absolute difference
was 0.00098 on the 8-bit mask and maximum difference was one level. Sequential RSS was higher
(about 1.71 GiB) because offload trades VRAM for system RAM. These are one-machine observations,
not performance guarantees.

For document navigation, start with `get_current_context`. It returns the open image IDs, a current-image snapshot when one can be established, the resolution source, and all selected layer IDs. GIMP 3 uses multi-layer selection, so `get_selected_layers` returns a list rather than inventing a single active layer. `get_layer_tree` recursively reports groups and children with stable IDs, parent IDs, positions, and bounded recursion/item limits. The Script-Fu server does not always expose a default-display context; when exactly one image is open, the service reports `single-open-image` as an explicit fallback, and it reports multiple open images as ambiguous instead of guessing.

## Example requests

- “Open this image and crop it to 16:9 around the subject.”
- “Duplicate the background layer, desaturate it, and reduce its opacity to 40%.”
- “Resize this to 2048 pixels wide while preserving aspect ratio.”
- “Export a JPEG copy at this path without overwriting anything.”
- “What layers are currently in this document?”
- “Separate the red fox from the background and leave both on editable layers.”
- “Separate the fox, trees, and snowbank into layers and report any overlap.”
- “Find a GIMP procedure capable of applying Gaussian blur and describe its arguments.”

The final request is supported through PDB search and description; the current description reports procedure counts, documentation, and the explicit argument-metadata capability state. Rich argument records remain unavailable through the default Script-Fu bridge until a live, trustworthy adapter is available.

## Security model

See [SECURITY.md](SECURITY.md). In brief:

- the MCP server is stdio-first and connects to loopback by default;
- no shell, Python eval, raw Script-Fu MCP tool, or cloud image API exists;
- procedure names and structured values are validated before serialization;
- strings and paths are escaped as Scheme literals;
- output paths must be explicit, normalized, and non-overwriting unless requested;
- closing a dirty image requires an explicit discard choice.

## Limitations

- Live validation in this iteration used GIMP 3.2.0 on the default loopback endpoint
  `127.0.0.1:10008`. Other GIMP 3 versions still need compatibility validation.
- On the tested GIMP 3.2.0 Script-Fu environment, the legacy `gimp-pdb-query` and `gimp-pdb-proc-exists` helper bindings are unavailable. The typed PDB adapter therefore retains its explicit unavailable fallback; the filter bindings are validated independently. A future bridge adapter should use a supported GIMP-side PDB or GObject-introspection channel rather than guessing signatures.
- Export metadata behavior uses GIMP’s configured defaults in v0.1; no hidden metadata is added or removed.
- The tested GIMP 3.2.0 Script-Fu bridge does not expose `gimp-file-export`; `export_image`
  refuses to fall back to `gimp-file-save`, which could change the open document's associated
  file. Use GIMP's own export UI until a supported export binding is available.
- The initial adjustment tools call stable legacy PDB adjustment procedures that GIMP 3.2 marks deprecated in favor of non-destructive filters. Only brightness/contrast and Gaussian blur have non-destructive high-level slices so far.
- Native foreground extraction is unavailable through the tested GIMP 3.2.0 Script-Fu bridge.
  CLIPSeg is genuinely text-prompted but produces a coarse 352×352 semantic probability mask;
  upsampling can leave halos, soften fur, omit thin whiskers, or retain nearby regions. It does not
  separate instances or discover/language-label every object. Without a capable worker, `auto`
  retains the bounded high-key fallback.
- The tested GIMP 3.2.0 bridge safely supports duplicate-image PNG snapshots through
  `gimp-image-duplicate` plus `gimp-file-save` on the duplicate. The unsafe save fallback remains
  prohibited for user images. The mask import bridge preserves partial alpha observed through
  GIMP's selection-to-mask path, but GIMP image-space mask encoding changes the raw sampled byte
  values for partial alpha. This is not a learned hair/fur-matting engine.
- Independently prompted concept masks can overlap. The default and currently supported policy is
  `report`: layers retain their masks, overlap statistics are returned, and remainder background
  uses their soft union. No arbitrary first-concept ownership is imposed.
- The Quadro P620 probe found compute capability 6.1 and 2 GiB total VRAM, with about 1.06 GiB free
  under KDE/Wayland during this iteration. The official PyTorch 2.10+cu128 wheel discovered CUDA but
  omitted `sm_61`, so its real kernel probe failed with `no kernel image`. PyTorch 2.7.1+cu126
  includes `sm_60` and completed real FP32, FP16, and BF16 matmuls. Wheel architecture support must
  be checked for this Pascal GPU; a newer version number is not evidence of compatibility.
- Official SAM 3.1 requires gated checkpoint access. The adapter and explicit setup path are
  implemented, but this run had no authenticated Hugging Face account or local checkpoint, so no
  SAM inference or offload claim is made. Import success alone is not provider support.
- `get_active_image` uses GIMP’s default display when the Script-Fu context provides one. If that helper is unavailable and exactly one image is open, it returns that image with the same documented single-image fallback used by `get_current_context`; multiple open images remain ambiguous.
- Selected-layer control and recursive group inspection were validated against GIMP 3.2.0. The bridge rejects empty selection vectors, validates layer/image ownership, and reads selection state back after setting it.
- Multi-call layer creation and duplication are grouped into one GIMP undo step. Additional composite operations should adopt the same internal helper as they are added.

## Roadmap

1. Validate the high-level vertical slices against additional GIMP 3.x releases and platforms.
2. Add a supported GIMP-side structured PDB metadata adapter for argument names, types, defaults, and enum choices; keep the explicit unavailable fallback when the bridge cannot provide it.
3. Add structured non-destructive GEGL filter operations for levels, curves, and hue/saturation.
4. Validate document context and multi-layer selection semantics against additional GIMP 3.x releases and multi-window setups.
5. Add explicit export metadata policies and more file-format option models.
6. Add safe, persistent capability caching with GIMP version invalidation.
7. Validate official SAM 3.1 text, box, multi-instance, activation-memory, and Accelerate hook
   behavior with an authorized local checkpoint; add provider-specific preload hooks only if the
   measured module access pattern requires them.
8. Add a local alpha-matting provider behind `MaskRefiner` when a supported lightweight runtime is
   available; keep semantic segmentation and matting evidence separate.
9. Implement automatic object proposals with stable unlabeled candidate IDs, boxes, areas, and
   previews before adding any semantic labels not produced by a model.

## Development

```bash
ruff format src tests
ruff check src tests
pytest -q
```

Live tests are marked `integration` and skip when `127.0.0.1:10008` is unavailable. See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) for repository invariants.

## Forge development evidence

The committed `mncs-forge.toml` declares a project-owned Provider Protocol 0.1 provider in
`tools/forge_provider.py`. It parses fixed Ruff, pytest, pip, and live-GIMP checks into explicit
PASS/FAIL/UNKNOWN responses. In particular, a skipped live integration test is UNKNOWN. Forge
runtime ledgers and `.mncs-forge/` state remain local and are intentionally not committed. The
provider disables Python bytecode writes in its subprocesses so normal checks do not mutate the
Forge candidate scope with generated `__pycache__` files.
