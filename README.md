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
| `GIMP_MCP_VISION_PROVIDER` | `none` | optional trusted local provider name, such as `sam3` |
| `GIMP_MCP_VISION_COMMAND` | unset | trusted local worker command; never an MCP argument |
| `GIMP_MCP_VISION_TIMEOUT` | `60` | local worker response timeout in seconds, capped at 600 |

Remote connections are rejected unless `GIMP_MCP_ALLOW_REMOTE=true` is set. Localhost-only operation is the secure default.

## MCP tools

The current server registers 55 tools across these groups:

- Session: `gimp_status`, `gimp_capabilities`, `list_open_images`, `get_active_image`, `get_current_context`, `get_image_info`
- Files and images: `open_image`, `create_image`, `save_xcf`, `export_image`, `close_image`
- Layers: `list_layers`, `get_layer_tree`, `get_layer_info`, `get_selected_layers`, `set_selected_layers`, `create_layer`, `create_layer_group`, `duplicate_layer`, `rename_layer`, `delete_layer`, `set_layer_visibility`, `set_layer_opacity`, `set_layer_mode`, `move_layer`, `merge_down`
- Transforms: `resize_image`, `resize_canvas`, `crop_image`, `rotate_image`, `flip_image`
- Selection: `select_all`, `select_none`, `invert_selection`, `select_rectangle`, `select_ellipse`, `select_layer_alpha`
- Masks and subject isolation: `get_layer_mask_info`, `create_layer_mask`, `set_layer_mask_enabled`, `isolate_subject`, `isolate_subject_vision`
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

The preferred future provider is Meta SAM 3, represented by `tools/vision/sam3_worker.py`. It
reports `unavailable` unless a separately installed, audited SAM 3 environment and adapter are
supplied. No weights are downloaded by normal MCP operation, and no image is sent to hosted
inference. `vision_status` reports this state honestly.

`segment_subject(image_id, layer_id, prompt="red fox")` snapshots a duplicate of the current
GIMP image to a uniquely named temporary PNG, invokes the local worker, validates candidate
artifacts, and removes the snapshot and artifacts after returning bounded summaries. It does not
mutate the document. `isolate_subject_vision` uses the same bridge, duplicates the explicit source
layer, imports the mask through a temporary RGBA layer/selection, and attaches it through the
existing layer-mask gateway. `isolate_subject(strategy="auto", prompt="red fox")` prefers a
capable semantic worker and falls back to the existing high-key contiguous-color strategy when no
provider is available. Explicit `high-key-background` and `border-color` retain the heuristic.

The current refinement boundary is `IdentityMaskRefiner`: provider alpha is retained, but this is
not learned alpha matting. Coverage, border transparency, partial-alpha, edge-transition, and
sampled bounding-box values are observable proxies only. The fox-on-snow benchmark has no
ground-truth alpha matte, so confidence and proxy metrics must not be presented as accuracy.

For document navigation, start with `get_current_context`. It returns the open image IDs, a current-image snapshot when one can be established, the resolution source, and all selected layer IDs. GIMP 3 uses multi-layer selection, so `get_selected_layers` returns a list rather than inventing a single active layer. `get_layer_tree` recursively reports groups and children with stable IDs, parent IDs, positions, and bounded recursion/item limits. The Script-Fu server does not always expose a default-display context; when exactly one image is open, the service reports `single-open-image` as an explicit fallback, and it reports multiple open images as ambiguous instead of guessing.

## Example requests

- “Open this image and crop it to 16:9 around the subject.”
- “Duplicate the background layer, desaturate it, and reduce its opacity to 40%.”
- “Resize this to 2048 pixels wide while preserving aspect ratio.”
- “Export a JPEG copy at this path without overwriting anything.”
- “What layers are currently in this document?”
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

- Live validation in this iteration used a loopback GIMP 3.2.0 Script-Fu server at `127.0.0.1:10009` (the repository default remains `10008`). Other GIMP 3 versions still need compatibility validation.
- On the tested GIMP 3.2.0 Script-Fu environment, the legacy `gimp-pdb-query` and `gimp-pdb-proc-exists` helper bindings are unavailable. The typed PDB adapter therefore retains its explicit unavailable fallback; the filter bindings are validated independently. A future bridge adapter should use a supported GIMP-side PDB or GObject-introspection channel rather than guessing signatures.
- Export metadata behavior uses GIMP’s configured defaults in v0.1; no hidden metadata is added or removed.
- The tested GIMP 3.2.0 Script-Fu bridge does not expose `gimp-file-export`; `export_image`
  refuses to fall back to `gimp-file-save`, which could change the open document's associated
  file. Use GIMP's own export UI until a supported export binding is available.
- The initial adjustment tools call stable legacy PDB adjustment procedures that GIMP 3.2 marks deprecated in favor of non-destructive filters. Only brightness/contrast and Gaussian blur have non-destructive high-level slices so far.
- Native foreground extraction was probed and is unavailable through the tested GIMP 3.2.0
  Script-Fu bridge. The local semantic architecture is available, but SAM 3 is optional and is
  not claimed as live unless a configured worker reports usable text segmentation. Without a
  capable worker, `auto` uses the bounded high-key contiguous-color fallback. Snow, foliage, and
  subjects with similar light colors can require manual refinement; the operation refuses an
  existing selection or source mask and rejects pathological masks.
- The tested GIMP 3.2.0 bridge safely supports duplicate-image PNG snapshots through
  `gimp-image-duplicate` plus `gimp-file-save` on the duplicate. The unsafe save fallback remains
  prohibited for user images. The mask import bridge preserves partial alpha observed through
  GIMP's selection-to-mask path, but is not a learned hair-matting engine.
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
7. Install and audit a separate SAM 3 worker adapter, then demonstrate genuine local text-prompted
   segmentation on the fox benchmark without adding ML dependencies to the core package.
8. Add a local alpha-matting provider behind `MaskRefiner` when a supported lightweight runtime is
   available; keep semantic segmentation and matting evidence separate.

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
