"""Command-line entry points for serving and diagnostics."""

from __future__ import annotations

import argparse
import logging
import platform
import sys

from . import __version__
from .config import Config
from .errors import GimpMcpError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gimp-local-mcp", description="Local-first MCP server for GIMP 3"
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="run the MCP server over stdio")
    subparsers.add_parser("doctor", help="check Python configuration and GIMP reachability")
    return parser


def _doctor() -> int:
    try:
        config = Config.from_env()
        config.validate()
        print(f"Python: {platform.python_version()} ({platform.python_implementation()})")
        print(
            f"Configured GIMP Script-Fu: {config.host}:{config.port} (timeout {config.timeout:g}s)"
        )
        if config.allow_remote:
            print("Remote host opt-in: enabled (GIMP_MCP_ALLOW_REMOTE=true)")
        else:
            print("Remote host opt-in: disabled; localhost-only policy active")
        from .service import GimpService

        service = GimpService(config)
        try:
            status = service.status()
        finally:
            service.close()
        print(
            f"Script-Fu: reachable; GIMP {status['gimp_version']}; "
            f"open images {status['open_image_count']}"
        )
        return 0
    except (GimpMcpError, OSError) as exc:
        print(f"Script-Fu: unavailable or misconfigured: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command or "serve"
    if command == "doctor":
        return _doctor()
    logging.basicConfig(
        level=Config.from_env().log_level,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from .server import run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
