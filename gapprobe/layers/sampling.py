"""gapprobe/layers/sampling.py — sampling-layer attribution probe."""

from __future__ import annotations

from gapprobe.probe import ProbeEngine, ProbeResult
from gapprobe.registry import ModelEntry, SamplerDefaults, calibrate_loss
from gapprobe.verdict import LayerVerdict

_SAMPLER_PARAMS = ("temperature", "top_p", "repeat_penalty")


def diff_sampler_defaults(
    served: dict[str, float],
    canonical_defaults: SamplerDefaults,
    tol: float = 0.05,
) -> tuple[list[str], int]:
    """Diff served sampler defaults against the canonical registry entry.

    Returns (mismatch evidence lines, number of params compared). Values within
    `tol` of canonical count as matching — servers round-trip floats (llama.cpp
    /props reports 0.8 as 0.800000011920929). Only params present in `served`
    are compared.
    """
    canonical = {
        "temperature": canonical_defaults.temperature,
        "top_p": canonical_defaults.top_p,
        "repeat_penalty": canonical_defaults.repeat_penalty,
    }
    mismatches: list[str] = []
    checked = 0
    for name in _SAMPLER_PARAMS:
        if name not in served:
            continue
        checked += 1
        served_value = served[name]
        canonical_value = canonical[name]
        if abs(served_value - canonical_value) > tol:
            mismatches.append(
                f"served {name}={served_value:g} differs from canonical {canonical_value:g}"
            )
    return mismatches, checked


class SamplingLayer:
    """Probes the sampling layer: detects mis-set temperature/top_p/repeat_penalty."""

    LAYER_NAME = "sampling"

    def run(self, engine: ProbeEngine, entry: ModelEntry, model_key: str) -> LayerVerdict:
        results: list[ProbeResult] = engine.run_probes(entry.sampling_probes)
        served = engine.fetch_served_defaults()
        served_evidence = None
        if served is not None:
            served_evidence = diff_sampler_defaults(served, entry.canonical_sampler)
        return self._build_verdict(results, model_key, served_evidence)

    def _build_verdict(
        self,
        results: list[ProbeResult],
        model_key: str,
        served_evidence: tuple[list[str], int] | None = None,
    ) -> LayerVerdict:
        # (mismatch lines, params compared) from /props, when the server
        # exposed its default sampler; None = behavioral probes only.
        total = len(results)
        failed = [r for r in results if not r.success]
        misses = len(failed)
        denom = total
        evidence = [
            f"{r.probe_id}: {r.detail or 'passed'}"
            + (f" — resp='{r.response_text[:60]}'..." if not r.success and r.response_text else "")
            for r in failed
        ]
        if served_evidence is not None:
            mismatches, checked = served_evidence
            denom += checked
            misses += len(mismatches)
            if mismatches:
                # Named-param evidence first: it points straight at the bad setting.
                evidence = mismatches + evidence
            else:
                evidence.insert(0, "served sampler defaults match the canonical registry")
        if not evidence:
            evidence = ["all sampling probes passed — sampler appears correctly configured"]
        divergence = misses / denom if denom else 0.0
        loss_pct = calibrate_loss(self.LAYER_NAME, divergence, model_key)
        return LayerVerdict(
            layer=self.LAYER_NAME,
            divergence_score=round(divergence, 3),
            est_capability_loss_pct=loss_pct,
            evidence=evidence,
        )
