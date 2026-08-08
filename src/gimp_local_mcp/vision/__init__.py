"""Optional local semantic-vision support.

The core package intentionally contains no machine-learning runtime.  Vision
providers communicate through the small, local JSONL protocol in this package.
"""

from .client import VisionClient
from .models import (
    BoundingBox,
    MaskArtifact,
    SegmentationCandidate,
    SegmentationRequest,
    SegmentationResult,
    VisionCapabilities,
)

__all__ = [
    "BoundingBox",
    "MaskArtifact",
    "SegmentationCandidate",
    "SegmentationRequest",
    "SegmentationResult",
    "VisionCapabilities",
    "VisionClient",
]
