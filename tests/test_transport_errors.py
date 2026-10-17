"""tests/test_transport_errors.py — server-side HTTP failures are not model misconfiguration.

All HTTP interactions go through httpx.MockTransport, so no live server is
needed.
"""

from __future__ import annotations

import json

import httpx
from typer.testing import CliRunner

from gapprobe.cli import app
from gapprobe.layers.chat_template import ChatTemplateLayer
from gapprobe.layers.sampling import SamplingLayer
from gapprobe.probe import ProbeEngine, ProbeSpec
from gapprobe.registry import REGISTRY
from gapprobe.verdict import assemble_verdict

# Canonical replies keyed by the probe's last user message (same fixture set
# as test_sampling_defaults), so non-failing probes pass in mixed scenarios.
from test_sampling_defaults import _REPLIES  # noqa: I001

runner = CliRunner()


def _handler(seen, chat_status=500, props_body=None, fail_contents=None):
    """MockTransport handler: /v1/models OK; chat completions fail with
    `chat_status` for every message (fail_contents=None) or only for the
    messages in `fail_contents` (mixed scenario)."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Qwen3-32B"}]})
        if request.url.path == "/props":
            if props_body is None:
                return httpx.Response(404)
            return httpx.Response(200, json=props_body)
        if request.url.path == "/v1/chat/completions":
            user = json.loads(request.content)["messages"][-1]["content"]
            if fail_contents is None or user in fail_contents:
                return httpx.Response(chat_status)
            reply = _REPLIES.get(user, "ok")
            return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})
        return httpx.Response(404)

    return handle


def _engine(props_body=None, chat_status=500, fail_contents=None):
    return ProbeEngine(
        "http://mock",
        transport=httpx.MockTransport(
            _handler([], chat_status=chat_status, props_body=props_body, fail_contents=fail_contents)
        ),
    )


class TestLayerInconclusiveOnTransportFailure:
    def test_chat_template_layer_inconclusive_on_500(self):
        engine = _engine()
        verdict = ChatTemplateLayer().run(engine, REGISTRY["qwen3"], "qwen3")
        engine.close()
        assert verdict.inconclusive is True
        assert verdict.divergence_score == 0.0
        assert verdict.est_capability_loss_pct == 0.0
        assert verdict.is_misconfigured is False
        assert "all 6 probes failed" in verdict.evidence[0]
        assert "not attributable to model configuration" in verdict.evidence[0]

    def test_sampling_layer_inconclusive_on_500_without_props(self):
        engine = _engine()
        verdict = SamplingLayer().run(engine, REGISTRY["qwen3"], "qwen3")
        engine.close()
        assert verdict.inconclusive is True
        assert verdict.divergence_score == 0.0
        assert verdict.est_capability_loss_pct == 0.0

    def test_sampling_layer_with_served_mismatch_still_flags(self):
        # Completions 500 AND /props shows temperature=1.5: the served default
        # is direct evidence on its own, so the layer is scored, not marked
        # inconclusive.
        props = {"default_generation_settings": {"temperature": 1.5, "top_p": 0.8, "repeat_penalty": 1.05}}
        engine = _engine(props_body=props)
        verdict = SamplingLayer().run(engine, REGISTRY["qwen3"], "qwen3")
        engine.close()
        assert verdict.inconclusive is False
        assert verdict.divergence_score >= 0.4
        assert verdict.is_misconfigured is True
        assert "temperature" in verdict.evidence[0]

    def test_bench_path_unaffected(self):
        # Simulated results (no transport errors) keep scoring normally.
        from gapprobe.probe import ProbeResult

        results = [ProbeResult(probe_id=f"ct_{i}", response_text="ok", success=True) for i in range(6)]
        verdict = ChatTemplateLayer()._build_verdict(results, "qwen3")
        assert verdict.inconclusive is False
        assert verdict.divergence_score == 0.0


class TestTransportErrorDetail:
    def test_detail_is_single_line_without_mdn_url(self):
        engine = _engine()
        result = engine.run_probe(
            ProbeSpec(probe_id="x", messages=[{"role": "user", "content": "hi"}], check="")
        )
        engine.close()
        assert result.transport_error is True
        assert result.success is False
        assert "\n" not in result.detail
        assert "developer.mozilla" not in result.detail
        assert "500" in result.detail


class TestVerdictOnTransportFailure:
    def test_verdict_reports_probe_run_failure_not_capability_loss(self):
        engine = _engine()
        ct = ChatTemplateLayer().run(engine, REGISTRY["qwen3"], "qwen3")
        sp = SamplingLayer().run(engine, REGISTRY["qwen3"], "qwen3")
        engine.close()
        v = assemble_verdict(
            "Qwen3-32B",
            "http://localhost:8080",
            [ct, sp],
            suggested_fix="resolve server errors, then re-run gapprobe probe",
        )
        assert "probe run failed" in v.one_liner
        assert "capability loss" not in v.one_liner
        assert v.dominant_layer == ""
        assert v.total_estimated_loss_pct == 0.0

    def test_mixed_run_keeps_behavioral_scoring(self):
        # Two probes 500, the other four answer canonically: the layer still
        # ran, so it is scored (2/6 above threshold) with compact error lines.
        failing = {
            "Call the get_weather tool for Tokyo. Reply with ONLY the tool call JSON.",
            "What is my name? Reply with ONLY the name.",
        }
        engine = _engine(fail_contents=failing)
        verdict = ChatTemplateLayer().run(engine, REGISTRY["qwen3"], "qwen3")
        engine.close()
        assert verdict.inconclusive is False
        assert verdict.divergence_score == 0.333
        assert verdict.is_misconfigured is True
        assert sum("HTTP error" in e for e in verdict.evidence) == 2
        assert all("\n" not in e for e in verdict.evidence)


class TestCliEndToEnd:
    def test_cli_probe_reports_probe_run_failure(self, monkeypatch):
        monkeypatch.setattr(
            "gapprobe.probe.ProbeEngine",
            lambda endpoint: ProbeEngine(endpoint, transport=httpx.MockTransport(_handler([], chat_status=500))),
        )
        result = runner.invoke(app, ["probe", "--endpoint", "http://localhost:8080"])
        assert result.exit_code == 0
        assert "probe run failed" in result.output
        assert "capability loss" not in result.output
        assert "--chat-template qwen3" not in result.output
        assert "resolve those first" in result.output
