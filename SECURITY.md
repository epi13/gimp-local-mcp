# Security model

GIMP Local MCP is intentionally powerful local automation. Treat the MCP host and its users as trusted actors, and keep the GIMP Script-Fu listener local.

## Defaults and boundaries

- The default endpoint is `127.0.0.1:10008`.
- Non-loopback hosts are rejected unless `GIMP_MCP_ALLOW_REMOTE=true` is explicitly set.
- The MCP server uses stdio. Normal logs go to stderr so stdout remains MCP protocol traffic.
- There is no shell-command tool, arbitrary Python execution, raw Scheme execution tool, image upload, cloud editing, or image-generation integration.
- The structured PDB gateway validates procedure identifiers and serializes values. User strings cannot become Scheme syntax.
- Request and response bodies are bounded by the Script-Fu protocol’s 16-bit length and the configured response limit.

## File safety

File-writing tools require an explicit path, normalize it, require an existing parent directory, and refuse to overwrite an existing file unless `overwrite=true`. Metadata is not silently stripped or replaced. Closing an image with dirty state requires `discard=true`.

## Trust assumptions

GIMP’s Script-Fu server has no authentication in this bridge. A process that can access the listener can ask GIMP to manipulate files and images available to that GIMP session. Use loopback, OS account isolation, and normal filesystem permissions. Do not expose port 10008 to a network or container boundary without understanding that risk.

The MCP host can request destructive image operations through the intended tools. Review client permissions and prompts accordingly.

## Reporting issues

Do not include private images, credentials, or filesystem contents in a report. For suspected vulnerabilities, open a private security report through the project’s hosting service when available; otherwise contact the maintainers before public disclosure.
