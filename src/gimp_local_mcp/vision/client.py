"""Trusted local stdio JSONL vision client."""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Sequence

from ..config import Config
from .models import SegmentationRequest, SegmentationResult, VisionCapabilities
from .protocol import (
    VisionProtocolError,
    VisionUnavailableError,
    VisionWorkerError,
    capabilities_request,
    decode_capabilities,
    decode_message,
    decode_segmentation,
    encode_message,
    segmentation_request,
)

logger = logging.getLogger(__name__)


class VisionClient:
    """One resident trusted worker; commands never come from MCP request data."""

    def __init__(
        self, config: Config | None = None, *, command: Sequence[str] | None = None
    ) -> None:
        self.config = config or Config.from_env()
        self._command = tuple(command) if command is not None else self.config.vision_command
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: list[str] = []
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self._command) and self.config.vision_provider != "none"

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def capabilities(self) -> VisionCapabilities:
        if not self.configured:
            return VisionCapabilities(
                "none", False, reason="no trusted local vision provider is configured"
            )
        try:
            return decode_capabilities(self._round_trip(capabilities_request()))
        except (VisionProtocolError, VisionWorkerError, VisionUnavailableError, OSError) as exc:
            return VisionCapabilities(
                self.config.vision_provider,
                False,
                reason=str(exc)[:1024],
            )

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        if not self.configured:
            raise VisionUnavailableError("no trusted local vision provider is configured")
        if not request.image_path.is_file():
            raise VisionProtocolError("vision input snapshot does not exist")
        return decode_segmentation(
            self._round_trip(segmentation_request(request), timeout=self.config.vision_timeout)
        )

    def _start(self) -> subprocess.Popen[bytes]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not self._command:
            raise VisionUnavailableError("no trusted local vision provider command is configured")
        if len(self._command) > 32 or any(
            not isinstance(part, str) or not part or len(part) > 4096 for part in self._command
        ):
            raise VisionProtocolError("trusted vision command is malformed")
        try:
            process = subprocess.Popen(
                list(self._command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
        except (OSError, ValueError) as exc:
            raise VisionUnavailableError(
                f"vision provider could not start: {type(exc).__name__}"
            ) from exc
        self._process = process
        self._stderr_tail = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(process,), daemon=True
        )
        self._stderr_thread.start()
        return process

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        if process.stderr is None:
            return
        try:
            for line in iter(process.stderr.readline, b""):
                decoded = line.decode("utf-8", "replace").strip()
                if decoded:
                    self._stderr_tail.append(decoded[:512])
                    del self._stderr_tail[:-16]
        except OSError:
            return

    def _round_trip(
        self, message: dict[str, object], *, timeout: float | None = None
    ) -> dict[str, object]:
        with self._lock:
            process = self._start()
            if process.stdin is None or process.stdout is None:
                raise VisionWorkerError("vision worker pipes are unavailable")
            try:
                process.stdin.write(encode_message(message))
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise VisionWorkerError("vision worker stdin closed") from exc
            response: list[bytes] = []

            def read_response() -> None:
                try:
                    response.append(process.stdout.readline())
                except OSError:
                    response.append(b"")

            reader = threading.Thread(target=read_response, daemon=True)
            reader.start()
            reader.join(timeout if timeout is not None else self.config.vision_timeout)
            if reader.is_alive():
                self.close()
                raise VisionWorkerError("vision worker response timed out")
            if not response or not response[0]:
                diagnostic = "; ".join(self._stderr_tail[-3:])
                self.close()
                suffix = f": {diagnostic}" if diagnostic else ""
                raise VisionWorkerError(f"vision worker exited without a response{suffix}")
            return decode_message(response[0])
