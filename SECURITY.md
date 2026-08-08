# Security model

GIMP Local MCP is intentionally powerful local automation. Treat the MCP host and its users as trusted actors, and keep the GIMP Script-Fu listener local.

## Defaults and boundaries

- The default endpoint is `127.0.0.1:10008`.
- Non-loopback hosts are rejected unless `GIMP_MCP_ALLOW_REMOTE=true` is explicitly set.
- The MCP server uses stdio. Normal logs go to stderr so stdout remains MCP protocol traffic.
- There is no shell-command tool, arbitrary Python execution, raw Scheme execution tool, image upload, cloud editing, or image-generation integration.
- Optional vision workers are a separate local trust boundary. Their command is read only from
  trusted process configuration (`GIMP_MCP_VISION_COMMAND`), never from MCP request data, and is
  launched without a shell. The worker protocol is bounded JSONL; diagnostics go to stderr and
  mask artifacts must be lossless PNGs in a service-owned temporary directory.
- Vision snapshots duplicate the requested GIMP image and save only the duplicate. Normal MCP
  operation never downloads model weights and never sends user pixels to hosted inference.
- The CLIPSeg worker loads only locally cached checkpoints during normal JSONL operation. Its
  `--download-model` mode is an explicit operator setup action and is not reachable from MCP tool
  arguments. Device and model identifiers remain trusted environment configuration.
- The structured PDB gateway validates procedure identifiers and serializes values. User strings cannot become Scheme syntax.
- Request and response bodies are bounded by the Script-Fu protocol’s 16-bit length and the configured response limit.

## File safety

File-writing tools require an explicit path, normalize it, require an existing parent directory, and refuse to overwrite an existing file unless `overwrite=true`. Metadata is not silently stripped or replaced. Closing an image with dirty state requires `discard=true`.

## Trust assumptions

GIMP’s Script-Fu server has no authentication in this bridge. A process that can access the listener can ask GIMP to manipulate files and images available to that GIMP session. Use loopback, OS account isolation, and normal filesystem permissions. Do not expose port 10008 to a network or container boundary without understanding that risk.

The MCP host can request destructive image operations through the intended tools. Review client permissions and prompts accordingly.

CLIPSeg, SAM 3, CUDA, model checkpoints, and separately installed adapters remain operator-managed
dependencies outside the core package. Review upstream licenses, checkpoint provenance, cache and
environment permissions, and process isolation. A provider OOM is a typed failure and decomposition
validates all masks before document mutation. Provider confidence or activation is not proof of
segmentation quality; this project does not claim a ground-truth result for the fox benchmark.

## Reporting issues

Do not include private images, credentials, or filesystem contents in a report. For suspected vulnerabilities, open a private security report through the project’s hosting service when available; otherwise contact the maintainers before public disclosure.
