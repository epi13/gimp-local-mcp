#!/usr/bin/env python3
"""Bounded Provider Protocol 0.1 checks for GIMP Local MCP development."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROVIDER = {
    "id": "gimp-local-mcp-project-checks",
    "name": "GIMP Local MCP project checks",
    "identity": "gimp-local-mcp-project-checks-v1",
    "version": "0.1",
}
METHODS = ["ruff-format", "ruff-check", "unit-tests", "live-gimp", "pip-check"]
_MAX_OUTPUT = 65536
_COMMANDS = {
    "ruff-format": ["ruff", "format", "--check", "src", "tests"],
    "ruff-check": ["ruff", "check", "src", "tests"],
    "unit-tests": ["pytest", "-q"],
    "live-gimp": ["pytest", "-m", "integration", "-q"],
    "pip-check": ["python", "-m", "pip", "check"],
}


def _response(
    request: dict[str, Any],
    status: str,
    summary: str,
    *,
    limitations: list[str] | None = None,
    unsupported: list[str] | None = None,
    witnesses: list[object] | None = None,
) -> dict[str, object]:
    return {
        "protocol_version": "0.1",
        "type": "analysis_response",
        "request_id": request.get("request_id", "invalid"),
        "provider": PROVIDER,
        "status": status,
        "summary": summary,
        "witnesses": witnesses or [],
        "limitations": limitations or [],
        "extensions": {
            "unsupported_constructs": unsupported or [],
            "mncs_forge": {
                "assumptions": ["the fixed command is the declared repository check"],
                "dependency_envelope": {
                    "paths": ["src", "tests", "pyproject.toml"],
                    "complete": False,
                },
            },
        },
    }


def _run(method: str) -> tuple[int, str, str] | None:
    try:
        result = subprocess.run(
            _COMMANDS[method],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return result.returncode, result.stdout[:_MAX_OUTPUT], result.stderr[:_MAX_OUTPUT]


def _check_result(
    method: str, result: tuple[int, str, str] | None, request: dict[str, Any]
) -> dict[str, object]:
    if result is None:
        return _response(
            request,
            "UNKNOWN",
            f"{method} could not be executed in the configured environment",
            limitations=["the fixed check executable was unavailable or timed out"],
        )
    returncode, stdout, stderr = result
    command = " ".join(_COMMANDS[method])
    witness = [{"command": _COMMANDS[method], "returncode": returncode}]
    if method == "ruff-format":
        passed = (
            returncode == 0
            and re.fullmatch(r"\d+ files? already formatted", stdout.strip()) is not None
            and not stderr.strip()
        )
        summary = "Ruff formatting verification completed without findings."
    elif method == "ruff-check":
        passed = returncode == 0 and stdout.strip() == "All checks passed!" and not stderr.strip()
        summary = "Ruff lint verification completed without findings."
    elif method == "unit-tests":
        passed_count = re.search(r"(\d+) passed", stdout)
        passed = returncode == 0 and passed_count is not None and " failed" not in stdout
        summary = "The repository unit test suite completed without failures."
    elif method == "live-gimp":
        passed_count = re.search(r"(\d+) passed", stdout)
        skipped = "skipped" in stdout.lower()
        passed = returncode == 0 and passed_count is not None and not skipped
        summary = (
            "The configured local GIMP Script-Fu server responded and the bounded live "
            "integration workflow completed successfully."
        )
        if returncode == 0 and skipped:
            return _response(
                request,
                "UNKNOWN",
                "The live GIMP integration workflow was skipped because the server was "
                "unavailable.",
                limitations=["a skipped live integration test is not execution evidence"],
                witnesses=witness,
            )
    else:
        passed = (
            returncode == 0 and stdout.strip() == "No broken requirements found." and not stderr
        )
        summary = "Python dependency verification found no broken requirements."
    if passed:
        return _response(request, "PASS", summary, witnesses=witness)
    return _response(
        request,
        "FAIL" if returncode != 0 else "UNKNOWN",
        f"The fixed check did not establish its claim: {command}",
        limitations=[stderr.strip() or stdout.strip() or "the check output was not decisive"],
        witnesses=witness,
    )


def handle(request: dict[str, Any]) -> dict[str, object]:
    if request.get("protocol_version") != "0.1":
        return {
            "protocol_version": "0.1",
            "type": "error",
            "request_id": "invalid",
            "provider": PROVIDER,
            "extensions": {},
            "error": "unsupported protocol",
        }
    if request.get("type") == "capabilities":
        return {
            "protocol_version": "0.1",
            "type": "capabilities",
            "request_id": request.get("request_id", "invalid"),
            "provider": PROVIDER,
            "analyses": METHODS,
            "statuses": ["PASS", "FAIL", "UNKNOWN"],
            "cancellation": False,
            "health_checks": False,
            "extensions": {
                "supported_constructs": [
                    "fixed-command-semantic-parsing",
                    "live-gimp-availability",
                ],
                "unsupported_constructs": ["arbitrary-commands", "whole-project-proof"],
                "limitations": [
                    "development evidence only; commands use the ambient project environment"
                ],
            },
        }
    if request.get("type") != "analysis_request":
        return _response(
            request,
            "UNKNOWN",
            "the provider request type is unsupported",
            unsupported=["unsupported-request-type"],
        )
    method = request.get("analysis")
    if not isinstance(method, str) or method not in METHODS:
        return _response(
            request,
            "UNKNOWN",
            "the requested project check is unsupported",
            unsupported=["unsupported-check"],
        )
    return _check_result(method, _run(method), request)


def main() -> int:
    line = sys.stdin.readline()
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        request = {"request_id": "invalid"}
    print(json.dumps(handle(request), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
