"""tests/test_sampling_defaults.py — sampling-layer served-default detection.

All HTTP interactions go through httpx.MockTransport, so no live server is
needed.
"""

from __future__ import annotations

import json

import httpx

from gapprobe.layers.sampling import SamplingLayer, diff_sampler_defaults
from gapprobe.probe import ProbeEngine, ProbeResult
from gapprobe.registry import REGISTRY, ProbeSpec

# Canonical replies keyed by the probe's last user message, so every probe
# passes and only the served defaults vary between scenarios.
_REPLIES = {
    "Reply with exactly three words: one two three": "one two three",
    "Write the word 'hello' exactly 3 times, separated by spaces. Reply with ONLY that.": "hello hello hello",
    "Write a 2-sentence story about a cat. Exactly 2 sentences, no more.": "The cat slept. It dreamed of fish.",
    "What is 2+2? Reply with ONLY the number.": "4",
    'Reply with this exact JSON: {"result": true}. Nothing else.': '{"result": true}',
    "Write a Python function called add(a,b) that returns a+b. Use a code block. Reply with ONLY the code.": "def add(a, b):\n    return a + b",
    "Call the get_weather tool for Tokyo. Reply with ONLY the tool call JSON.": '{"name": "get_weather", "arguments": {"city": "Tokyo"}}',
    "What is my name? Reply with ONLY the name.": "Alice",
    "Hello": "BANANA",
    'Generate a JSON object with key "status" and value "ok". Reply with ONLY JSON.': '{"status": "ok"}',
    "Hello, how are you?": "Arrr, I be fine, matey.",
    "Count from 1 to 5, one number per line. Stop after 5.": "1\n2\n3\n4\n5",
}


def _handler(seen, props_body=None, fail_contents=()):
    """MockTransport handler: /props plus pass/fail chat completions."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/props":
            if props_body is None:
                return httpx.Response(404)
            return httpx.Response(200, json=props_body)
        if request.url.path == "/v1/chat/completions":
            messages = json.loads(request.content)["messages"]
            user = messages[-1]["content"]
            reply = "garbled output" if user in fail_contents else _REPLIES.get(user, "ok")
            return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})
        return httpx.Response(404)

    return handle


class TestRegistryProbeFlags:
    def test_sampling_probes_use_served_sampler(self):
        for entry in REGISTRY.values():
            assert all(p.use_served_sampler for p in entry.sampling_probes)

    def test_chat_template_probes_pin_temperature(self):
        for entry in REGISTRY.values():
            assert not any(p.use_served_sampler for p in entry.chat_template_probes)


class TestRunProbePayload:
    def test_served_sampler_probe_omits_temperature(self):
        seen = []
        engine = ProbeEngine("http://mock", transport=httpx.MockTransport(_handler(seen)))
        engine.model_id = "Qwen3-32B"
        served_probe = ProbeSpec(
            probe_id="sp_x", messages=[{"role": "user", "content": "hi"}], check="", use_served_sampler=True
        )
        pinned_probe = ProbeSpec(
            probe_id="ct_x", messages=[{"role": "user", "content": "hi"}], check=""
        )
        engine.run_probe(served_probe)
        engine.run_probe(pinned_probe)
        engine.close()
        payloads = [json.loads(r.content) for r in seen]
        assert "temperature" not in payloads[0]
        assert payloads[1]["temperature"] == 0.0


class TestFetchServedDefaults:
    def test_reads_llama_props(self):
        props = {"default_generation_settings": {"temperature": 1.5, "top_p": 0.8, "repeat_penalty": 1.05}}
        engine = ProbeEngine("http://mock", transport=httpx.MockTransport(_handler([], props_body=props)))
        assert engine.fetch_served_defaults() == {"temperature": 1.5, "top_p": 0.8, "repeat_penalty": 1.05}

    def test_missing_props_endpoint_returns_none(self):
        engine = ProbeEngine("http://mock", transport=httpx.MockTransport(_handler([], props_body=None)))
        assert engine.fetch_served_defaults() is None

    def test_props_without_generation_settings_returns_none(self):
        engine = ProbeEngine(
            "http://mock", transport=httpx.MockTransport(_handler([], props_body={"model_path": "/x"}))
        )
        assert engine.fetch_served_defaults() is None

    def test_props_without_sampler_params_returns_none(self):
        engine = ProbeEngine(
            "http://mock",
            transport=httpx.MockTransport(_handler([], props_body={"default_generation_settings": {"top_k": 40}})),
        )
        assert engine.fetch_served_defaults() is None


class TestDiffSamplerDefaults:
    def setup_method(self):
        self.canonical = REGISTRY["qwen3"].canonical_sampler  # 0.7 / 0.8 / 1.05

    def test_matching_defaults(self):
        mismatches, checked = diff_sampler_defaults(
            {"temperature": 0.7, "top_p": 0.8, "repeat_penalty": 1.05}, self.canonical
        )
        assert mismatches == []
        assert checked == 3

    def test_float_noise_within_tolerance(self):
        mismatches, checked = diff_sampler_defaults(
            {"temperature": 0.700000011920929, "top_p": 0.8, "repeat_penalty": 1.05}, self.canonical
        )
        assert mismatches == []
        assert checked == 3

    def test_mismatched_param_is_named(self):
        mismatches, checked = diff_sampler_defaults(
            {"temperature": 1.5, "top_p": 0.8, "repeat_penalty": 1.05}, self.canonical
        )
        assert checked == 3
        assert len(mismatches) == 1
        assert "temperature" in mismatches[0]
        assert "1.5" in mismatches[0]

    def test_partial_served_data_compares_only_present_params(self):
        mismatches, checked = diff_sampler_defaults({"temperature": 1.5}, self.canonical)
        assert checked == 1
        assert len(mismatches) == 1


class TestSamplingLayer:
    def test_mis_served_temperature_named_in_evidence(self):
        seen = []
        props = {"default_generation_settings": {"temperature": 1.5, "top_p": 0.8, "repeat_penalty": 1.05}}
        engine = ProbeEngine("http://mock", transport=httpx.MockTransport(_handler(seen, props_body=props)))
        entry = REGISTRY["qwen3"]
        verdict = SamplingLayer().run(engine, entry, "qwen3")
        engine.close()
        assert verdict.divergence_score > 0
        assert any("temperature" in e for e in verdict.evidence)
        # Named-param evidence comes first.
        assert "temperature" in verdict.evidence[0]

    def test_matching_served_defaults_reported_clean(self):
        props = {"default_generation_settings": {"temperature": 0.7, "top_p": 0.8, "repeat_penalty": 1.05}}
        engine = ProbeEngine("http://mock", transport=httpx.MockTransport(_handler([], props_body=props)))
        entry = REGISTRY["qwen3"]
        verdict = SamplingLayer().run(engine, entry, "qwen3")
        engine.close()
        assert verdict.divergence_score == 0.0
        assert any("match the canonical registry" in e for e in verdict.evidence)

    def test_served_mismatch_plus_failed_probes_raise_divergence(self):
        props = {"default_generation_settings": {"temperature": 1.5, "top_p": 0.8, "repeat_penalty": 1.05}}
        failing = {
            "Reply with exactly three words: one two three",
            "What is 2+2? Reply with ONLY the number.",
            'Reply with this exact JSON: {"result": true}. Nothing else.',
        }
        engine = ProbeEngine(
            "http://mock", transport=httpx.MockTransport(_handler([], props_body=props, fail_contents=failing))
        )
        entry = REGISTRY["qwen3"]
        verdict = SamplingLayer().run(engine, entry, "qwen3")
        engine.close()
        # (3 failed probes + 1 mismatched param) / (6 probes + 3 compared params)
        assert verdict.divergence_score == round(4 / 9, 3)
        assert verdict.est_capability_loss_pct > 0

    def test_bench_signature_without_served_evidence_unchanged(self):
        # cases.py builds verdicts from simulated results alone.
        results = [ProbeResult(probe_id=f"sp_{i}", response_text="ok", success=True) for i in range(6)]
        verdict = SamplingLayer()._build_verdict(results, "qwen3")
        assert verdict.divergence_score == 0.0
        assert verdict.evidence == ["all sampling probes passed — sampler appears correctly configured"]
