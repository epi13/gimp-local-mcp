"""Command-line entry points for serving and diagnostics."""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
import subprocess
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
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            print("NVIDIA driver: not detected (nvidia-smi is unavailable)")
        else:
            probe = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                print(f"NVIDIA driver: detected; {probe.stdout.strip()}")
            else:
                print("NVIDIA driver: nvidia-smi failed; CUDA is not established")
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
            vision = service.vision_status()
            status = service.status()
        finally:
            service.close()
        print(
            f"Script-Fu: reachable; GIMP {status['gimp_version']}; "
            f"open images {status['open_image_count']}"
        )
        print(
            f"Vision: {vision['provider']} "
            f"({'available' if vision['available'] else 'unavailable'})"
            + (f"; {vision['reason']}" if vision.get("reason") else "")
        )
        print(
            "Provider runtime: "
            f"torch={vision.get('torch_version') or 'not reported'}; "
            f"torch CUDA={vision.get('torch_cuda_version') or 'none'}; "
            f"CUDA available={vision.get('cuda_available')}; "
            f"device={vision.get('device') or 'none'}; "
            f"GPU={vision.get('gpu_name') or 'none'}; "
            f"compute capability={vision.get('compute_capability') or 'none'}"
        )
        print(
            "CUDA execution: "
            f"kernel smoke test={vision.get('cuda_kernel_smoke_test_success')}; "
            f"error={vision.get('cuda_kernel_smoke_test_error') or 'none'}; "
            f"Torch architectures={vision.get('torch_supported_architectures') or []}; "
            f"FP16={vision.get('float16_execution_success')}; "
            f"BF16={vision.get('bfloat16_execution_success')}"
        )
        print(
            "Provider placement: "
            f"mode={vision.get('execution_mode') or 'none'}; "
            f"execution device={vision.get('execution_device') or 'none'}; "
            f"offload={vision.get('offload_mode') or 'none'}; "
            f"dtype={vision.get('dtype') or 'none'}; "
            f"reason={vision.get('placement_reason') or 'not reported'}"
        )
        print(
            "Provider memory: "
            f"free at startup={vision.get('free_vram_bytes_at_startup')}; "
            f"reserve={vision.get('configured_gpu_reserve_bytes')}; "
            f"budget={vision.get('effective_gpu_budget_bytes')}; "
            f"model={vision.get('model_storage_bytes')}; "
            f"peak allocated={vision.get('peak_cuda_allocated_bytes')}; "
            f"peak reserved={vision.get('peak_cuda_reserved_bytes')}"
        )
        print(
            "Sequential offload evidence: "
            f"verified={vision.get('sequential_offload_verified')}; "
            f"hooks={vision.get('sequential_offload_hook_count')}; "
            f"CPU-backed/meta parameters={vision.get('offloaded_meta_parameter_bytes')}; "
            f"persistent CUDA parameters={vision.get('persistent_cuda_parameter_bytes')}"
        )
        print(
            "Provider readiness: "
            f"checkpoint={vision.get('checkpoint_available')}; "
            f"model load={vision.get('model_load_success')}; "
            f"test inference={vision.get('test_inference_success')}; "
            f"text segmentation={vision.get('text_segmentation')}"
        )
        return 0
    except (GimpMcpError, OSError, subprocess.SubprocessError) as exc:
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
