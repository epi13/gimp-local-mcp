from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gimp_local_mcp_forge_provider", Path(__file__).parents[1] / "tools" / "forge_provider.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_check_result = _MODULE._check_result
handle = _MODULE.handle


def request(method: str) -> dict[str, object]:
    return {
        "protocol_version": "0.1",
        "type": "analysis_request",
        "analysis": method,
        "request_id": "test-1",
    }


def test_forge_provider_parses_successful_fixed_checks() -> None:
    assert (
        _check_result(
            "ruff-format", (0, "21 files already formatted\n", ""), request("ruff-format")
        )["status"]
        == "PASS"
    )
    assert (
        _check_result("ruff-check", (0, "All checks passed!\n", ""), request("ruff-check"))[
            "status"
        ]
        == "PASS"
    )
    assert (
        _check_result(
            "unit-tests", (0, "35 passed, 1 skipped in 1.0s\n", ""), request("unit-tests")
        )["status"]
        == "PASS"
    )
    assert (
        _check_result(
            "vision-unit-tests", (0, "17 passed in 0.2s\n", ""), request("vision-unit-tests")
        )["status"]
        == "PASS"
    )
    assert (
        _check_result(
            "layer-decomposition-tests",
            (0, "6 passed in 0.2s\n", ""),
            request("layer-decomposition-tests"),
        )["status"]
        == "PASS"
    )


def test_forge_provider_distinguishes_live_skip_from_pass() -> None:
    result = _check_result("live-gimp", (0, "1 skipped in 0.1s\n", ""), request("live-gimp"))
    assert result["status"] == "UNKNOWN"
    result = _check_result(
        "live-subject-isolation", (0, "1 passed in 0.1s\n", ""), request("live-subject-isolation")
    )
    assert result["status"] == "PASS"


def test_forge_provider_reports_failures_and_rejects_unbounded_methods() -> None:
    result = _check_result("unit-tests", (1, "2 failed in 1.0s\n", ""), request("unit-tests"))
    assert result["status"] == "FAIL"
    unsupported = handle(
        {
            "protocol_version": "0.1",
            "type": "analysis_request",
            "analysis": "execute-shell",
            "request_id": "test-2",
        }
    )
    assert unsupported["status"] == "UNKNOWN"


def test_forge_separates_provider_readiness_inference_and_quality() -> None:
    readiness = _check_result(
        "vision-provider-status",
        (0, '{"provider_available":true}\n', ""),
        request("vision-provider-status"),
    )
    inference = _check_result(
        "semantic-inference",
        (0, '{"semantic_inference":true}\n', ""),
        request("semantic-inference"),
    )
    unavailable = _check_result(
        "vision-provider-status",
        (2, '{"provider_available":false,"reason":"missing"}\n', ""),
        request("vision-provider-status"),
    )
    quality = handle(request("semantic-quality-uncertainty"))
    assert readiness["status"] == "PASS"
    assert inference["status"] == "PASS"
    assert unavailable["status"] == "UNKNOWN"
    assert quality["status"] == "UNKNOWN"
