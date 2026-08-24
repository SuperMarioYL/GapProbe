"""gapprobe/verdict.py — data models + verdict assembler for cross-layer attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LayerName = Literal["chat_template", "sampling"]


@dataclass
class LayerVerdict:
    """One layer's diagnostic result."""

    layer: LayerName
    divergence_score: float  # 0..1, distance from known-good
    est_capability_loss_pct: float  # 0..100, calibrated per layer
    evidence: list[str] = field(default_factory=list)

    @property
    def is_misconfigured(self) -> bool:
        return self.divergence_score > 0.3


@dataclass
class AttributionVerdict:
    """The cross-layer attribution verdict — the product's core primitive."""

    model: str
    endpoint: str
    layers: list[LayerVerdict] = field(default_factory=list)
    dominant_layer: str = ""
    total_estimated_loss_pct: float = 0.0
    one_liner: str = ""
    suggested_fix: str = ""

    def render(self) -> str:
        """Render the screenshot-able one-line verdict."""
        if not self.one_liner:
            self._assemble()
        return self.one_liner

    def _assemble(self) -> None:
        if not self.layers:
            self.one_liner = f"{self.model}: no probes ran — check endpoint {self.endpoint}"
            return
        self.dominant_layer = max(self.layers, key=lambda v: v.est_capability_loss_pct).layer
        self.total_estimated_loss_pct = round(
            sum(v.est_capability_loss_pct for v in self.layers), 1
        )
        dom = next(v for v in self.layers if v.layer == self.dominant_layer)
        detail = dom.evidence[0] if dom.evidence else "unspecified divergence"
        self.one_liner = (
            f"{self.model}: ~{int(self.total_estimated_loss_pct)}% capability loss "
            f"— dominant layer={self.dominant_layer} ({detail}). "
            f"Fix: {self.suggested_fix}"
        )


def assemble_verdict(
    model: str,
    endpoint: str,
    layer_verdicts: list[LayerVerdict],
    suggested_fix: str = "",
) -> AttributionVerdict:
    """Assemble layer verdicts into a cross-layer attribution verdict."""
    v = AttributionVerdict(
        model=model,
        endpoint=endpoint,
        layers=layer_verdicts,
        suggested_fix=suggested_fix,
    )
    v._assemble()
    return v
