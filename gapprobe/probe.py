"""gapprobe/probe.py — HTTP probe engine against a local OpenAI-compatible server."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from gapprobe.registry import ProbeSpec, match_model, get_entry


@dataclass
class ProbeResult:
    """Result of running a single behavioral probe."""

    probe_id: str
    response_text: str
    success: bool  # did the probe check pass?
    detail: str = ""


class ProbeEngine:
    """Sends behavioral probes to a local OpenAI-compatible server."""

    def __init__(self, endpoint: str, timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.model_key: str | None = None
        self.model_id: str = ""

    def detect_model(self) -> str:
        """GET /v1/models, match against the registry. Returns the registry key."""
        resp = self.client.get(f"{self.endpoint}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if not models:
            raise RuntimeError("No models returned from /v1/models")
        self.model_id = models[0].get("id", "")
        self.model_key = match_model(self.model_id)
        if self.model_key is None:
            raise RuntimeError(
                f"Model '{self.model_id}' not in registry. "
                f"v0.1 supports Qwen3 + DeepSeek only."
            )
        return self.model_key

    def run_probe(self, probe: ProbeSpec) -> ProbeResult:
        """Send a single behavioral probe and check the response."""
        payload = {
            "model": self.model_id,
            "messages": probe.messages,
            "max_tokens": 128,
            "temperature": 0.0,  # deterministic for probe consistency
        }
        try:
            resp = self.client.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as exc:
            return ProbeResult(
                probe_id=probe.probe_id,
                response_text="",
                success=False,
                detail=f"HTTP error: {exc}",
            )

        text = text.strip()
        success = True
        detail = ""
        if probe.expected_substring and probe.expected_substring.lower() not in text.lower():
            success = False
            detail = f"expected '{probe.expected_substring}' not found in response"
        if probe.expected_not_substring and probe.expected_not_substring.lower() in text.lower():
            success = False
            detail = f"unexpected '{probe.expected_not_substring}' found in response"

        return ProbeResult(
            probe_id=probe.probe_id,
            response_text=text[:200],
            success=success,
            detail=detail,
        )

    def run_probes(self, probes: list[ProbeSpec]) -> list[ProbeResult]:
        """Run a list of probes with throttling."""
        results = []
        for probe in probes:
            results.append(self.run_probe(probe))
            time.sleep(0.3)  # gentle throttle
        return results

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
