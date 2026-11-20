"""tests/test_protocol_errors.py — completion payloads that are not usable text.

An OpenAI-compatible server may legally answer with "content": null (a
tool-calling reply), with content as a list of text parts, or with a 200 error
envelope that carries no choice at all. None of those is a model-behavior
signal, and none may crash the run or be scored as misconfiguration.

All HTTP interactions go through httpx.MockTransport, so no live server is
needed.
"""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from gapprobe.cli import app
from gapprobe.layers.chat_template import ChatTemplateLayer
from gapprobe.probe import ProbeEngine, ProbeSpec
from gapprobe.registry import REGISTRY

runner = CliRunner()

_PROBE = ProbeSpec(
    probe_id="ct_tool_call_format",
    messages=[{"role": "user", "content": "hi"}],
    check="",
    expected_substring='"name"',
)


def _engine(chat_body: dict, models: list[dict] | None = None) -> ProbeEngine:
    """Engine whose /v1/models serves `models` and whose completions reply with
    `chat_body`."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            served = models if models is not None else [{"id": "Qwen3-32B"}]
            return httpx.Response(200, json={"data": served})
        if request.url.path == "/props":
            return httpx.Response(404)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=chat_body)
        return httpx.Response(404)

    return ProbeEngine("http://mock", transport=httpx.MockTransport(handle))


class TestContentNormalisation:
    def test_null_content_with_tool_calls_is_a_protocol_failure(self):
        # OpenAI-legal tool-calling reply: content is null, tool_calls carries
        # the answer. v0.3.0 raised AttributeError out of run_probe.
        engine = _engine(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                            ],
                        }
                    }
                ]
            }
        )
        result = engine.run_probe(_PROBE)
        engine.close()
        assert result.transport_error is True
        assert result.success is False
        assert "no usable" in result.detail

    def test_parts_content_is_concatenated_and_checked(self):
        engine = _engine(
            {"choices": [{"message": {"content": [{"type": "text", "text": '{"name": "ok"}'}]}}]}
        )
        result = engine.run_probe(_PROBE)
        engine.close()
        assert result.transport_error is False
        assert result.success is True
        assert result.response_text == '{"name": "ok"}'

    def test_error_envelope_is_not_scored_as_misconfiguration(self):
        # A 200 body without choices (gateway / OOM error envelope) used to
        # become one failed behavioral probe: divergence 1.0, loss 47.0%.
        engine = _engine({"error": {"message": "context length exceeded"}})
        verdict = ChatTemplateLayer().run(engine, REGISTRY["qwen3"], "qwen3")
        engine.close()
        assert verdict.inconclusive is True
        assert verdict.divergence_score == 0.0
        assert verdict.est_capability_loss_pct == 0.0
        assert verdict.is_misconfigured is False

    def test_malformed_response_detail_is_single_line(self):
        engine = _engine({"error": {"message": "context length exceeded"}})
        result = engine.run_probe(_PROBE)
        engine.close()
        assert "\n" not in result.detail


class TestMultiModelDetection:
    def test_supported_model_is_found_after_an_unrelated_one(self):
        engine = _engine(
            {"choices": [{"message": {"content": "ok"}}]},
            models=[{"id": "llama-3-70b"}, {"id": "Qwen3-32B"}],
        )
        assert engine.detect_model() == "qwen3"
        assert engine.model_id == "Qwen3-32B"
        engine.close()

    def test_no_supported_model_lists_what_was_served(self):
        engine = _engine(
            {"choices": [{"message": {"content": "ok"}}]},
            models=[{"id": "llama-3-70b"}, {"id": "mistral-7b"}],
        )
        with pytest.raises(RuntimeError) as excinfo:
            engine.detect_model()
        engine.close()
        message = str(excinfo.value)
        assert "llama-3-70b" in message
        assert "Qwen3" in message and "DeepSeek" in message
        assert "v0.1" not in message


class TestCliEndToEnd:
    def test_null_content_server_reports_probe_run_failure(self, monkeypatch):
        monkeypatch.setattr(
            "gapprobe.probe.ProbeEngine",
            lambda endpoint: _engine({"choices": [{"message": {"content": None}}]}),
        )
        result = runner.invoke(app, ["probe", "--endpoint", "http://localhost:8080"])
        assert result.exit_code == 0
        assert "probe run failed" in result.output
        assert "capability loss" not in result.output
        assert "Traceback" not in result.output
