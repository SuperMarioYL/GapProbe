"""gapprobe/layers/chat_template.py — chat-template layer attribution probe."""

from __future__ import annotations

from gapprobe.probe import ProbeEngine, ProbeResult, all_transport_errors
from gapprobe.registry import ModelEntry, calibrate_loss
from gapprobe.verdict import LayerVerdict, transport_failure_verdict


class ChatTemplateLayer:
    """Probes the chat-template layer: detects mis-applied templates."""

    LAYER_NAME = "chat_template"

    def run(self, engine: ProbeEngine, entry: ModelEntry, model_key: str) -> LayerVerdict:
        probes = entry.chat_template_probes
        results: list[ProbeResult] = engine.run_probes(probes)
        if all_transport_errors(results):
            return transport_failure_verdict(self.LAYER_NAME, len(results))
        return self._build_verdict(results, model_key)

    def _build_verdict(
        self, results: list[ProbeResult], model_key: str
    ) -> LayerVerdict:
        total = len(results)
        failed = [r for r in results if not r.success]
        divergence = len(failed) / total if total else 0.0
        loss_pct = calibrate_loss(self.LAYER_NAME, divergence, model_key)
        evidence = [
            f"{r.probe_id}: {r.detail or 'passed'}"
            + (f" — resp='{r.response_text[:60]}...'" if not r.success and r.response_text else "")
            for r in failed
        ]
        if not evidence:
            evidence = ["all chat-template probes passed — template appears correctly applied"]
        return LayerVerdict(
            layer=self.LAYER_NAME,
            divergence_score=round(divergence, 3),
            est_capability_loss_pct=loss_pct,
            evidence=evidence,
        )
