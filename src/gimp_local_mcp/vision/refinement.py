"""Explicit semantic-mask refinement boundary.

This iteration preserves provider alpha and does not pretend that feathering is
learned alpha matting.  A future local matting engine can implement the same
interface without changing the GIMP-facing API.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import MaskArtifact


@dataclass(frozen=True, slots=True)
class RefinementResult:
    mask: MaskArtifact
    strategy: str
    warnings: tuple[str, ...] = ()


class MaskRefiner:
    """Interface for safe semantic-mask refinement."""

    def refine(self, mask: MaskArtifact) -> RefinementResult:
        raise NotImplementedError


class IdentityMaskRefiner(MaskRefiner):
    """Keep the provider's soft mask unchanged; no fake matting is performed."""

    def refine(self, mask: MaskArtifact) -> RefinementResult:
        return RefinementResult(mask, "provider-alpha-preserved")
