"""gapprobe/registry.py — per-model known-good reference data + loss calibration.

Single source of truth for canonical chat templates, sampler defaults,
known-good probe response patterns, and per-layer loss calibration.

v0.1 scope: Qwen3 + DeepSeek only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplerDefaults:
    temperature: float
    top_p: float
    repeat_penalty: float


@dataclass
class ProbeSpec:
    """A behavioral probe specification."""

    probe_id: str
    messages: list[dict]  # OpenAI chat messages
    check: str  # what to look for in the response
    expected_substring: str = ""  # canonical response must contain this
    expected_not_substring: str = ""  # canonical response must NOT contain this
    # Sampling probes run without an explicit temperature so the server's
    # configured default sampler applies (probing with temperature=0.0 would
    # mask exactly the misconfiguration this layer exists to detect).
    use_served_sampler: bool = False


@dataclass
class ModelEntry:
    """Per-model registry entry."""

    model_id_patterns: list[str]  # substrings to match against /v1/models id
    canonical_chat_template: str
    canonical_sampler: SamplerDefaults
    chat_template_probes: list[ProbeSpec] = field(default_factory=list)
    sampling_probes: list[ProbeSpec] = field(default_factory=list)
    # Per-layer loss calibration: divergence_score → estimated capability loss %
    loss_calibration: dict = field(default_factory=lambda: {
        "chat_template": {"slope": 45.0, "intercept": 2.0},
        "sampling": {"slope": 30.0, "intercept": 1.0},
    })
    fix_hint_chat_template: str = ""
    fix_hint_sampling: str = ""


def _chat_template_probes() -> list[ProbeSpec]:
    """Probes that only behave correctly under the canonical chat template."""
    return [
        ProbeSpec(
            probe_id="ct_tool_call_format",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Call the get_weather tool for Tokyo. Reply with ONLY the tool call JSON."},
            ],
            check="tool-call JSON with function name + arguments",
            expected_substring='"name"',
        ),
        ProbeSpec(
            probe_id="ct_multi_turn",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name? Reply with ONLY the name."},
            ],
            check="remembers the name from context",
            expected_substring="Alice",
        ),
        ProbeSpec(
            probe_id="ct_system_adherence",
            messages=[
                {"role": "system", "content": "You must reply with exactly: BANANA. No other text."},
                {"role": "user", "content": "Hello"},
            ],
            check="follows system prompt exactly",
            expected_substring="BANANA",
        ),
        ProbeSpec(
            probe_id="ct_structured_output",
            messages=[
                {"role": "system", "content": "You are a JSON generator."},
                {"role": "user", "content": 'Generate a JSON object with key "status" and value "ok". Reply with ONLY JSON.'},
            ],
            check="produces valid JSON",
            expected_substring='"status"',
        ),
        ProbeSpec(
            probe_id="ct_role_recognition",
            messages=[
                {"role": "system", "content": "You are a pirate. Always say 'Arrr' in your response."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            check="adopts the system persona",
            expected_substring="Arrr",
        ),
        ProbeSpec(
            probe_id="ct_stop_sequence",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 5, one number per line. Stop after 5."},
            ],
            check="stops at the right place (no run-on generation)",
            expected_not_substring="6",
        ),
    ]


def _sampling_probes() -> list[ProbeSpec]:
    """Probes sensitive to temperature / top_p / repeat_penalty misconfiguration.

    All run with use_served_sampler=True so the server's configured default
    sampler (not a probe-side temperature) shapes the responses.
    """
    return [
        ProbeSpec(
            probe_id="sp_instruction_follow",
            messages=[
                {"role": "system", "content": "You are a precise instruction follower."},
                {"role": "user", "content": "Reply with exactly three words: one two three"},
            ],
            check="follows a precise format (high temp → rambling)",
            expected_substring="one two three",
            use_served_sampler=True,
        ),
        ProbeSpec(
            probe_id="sp_repetition",
            messages=[
                {"role": "user", "content": "Write the word 'hello' exactly 3 times, separated by spaces. Reply with ONLY that."},
            ],
            check="no excessive repetition (low repeat_penalty → loops)",
            expected_not_substring="hello hello hello hello hello",
            use_served_sampler=True,
        ),
        ProbeSpec(
            probe_id="sp_creative_constraint",
            messages=[
                {"role": "user", "content": "Write a 2-sentence story about a cat. Exactly 2 sentences, no more."},
            ],
            check="respects length constraint (greedy temp → over-long or under-long)",
            use_served_sampler=True,
        ),
        ProbeSpec(
            probe_id="sp_factual_recall",
            messages=[
                {"role": "user", "content": "What is 2+2? Reply with ONLY the number."},
            ],
            check="correct factual answer (high temp → wrong)",
            expected_substring="4",
            use_served_sampler=True,
        ),
        ProbeSpec(
            probe_id="sp_format_json",
            messages=[
                {"role": "user", "content": 'Reply with this exact JSON: {"result": true}. Nothing else.'},
            ],
            check="exact format compliance (sampling noise → malformed JSON)",
            expected_substring='"result"',
            use_served_sampler=True,
        ),
        ProbeSpec(
            probe_id="sp_code_block",
            messages=[
                {"role": "user", "content": "Write a Python function called add(a,b) that returns a+b. Use a code block. Reply with ONLY the code."},
            ],
            check="code is syntactically plausible (sampling noise → broken code)",
            expected_substring="def add",
            use_served_sampler=True,
        ),
    ]


# ── The registry ──────────────────────────────────────────────────────────

REGISTRY: dict[str, ModelEntry] = {
    "qwen3": ModelEntry(
        # Qwen2.5 deliberately NOT matched: it uses a different chat template,
        # and routing it here would prescribe the qwen3 template as a fix.
        model_id_patterns=["qwen3", "qwen-3"],
        canonical_chat_template="qwen3",
        canonical_sampler=SamplerDefaults(temperature=0.7, top_p=0.8, repeat_penalty=1.05),
        chat_template_probes=_chat_template_probes(),
        sampling_probes=_sampling_probes(),
        fix_hint_chat_template="llama-server --chat-template qwen3 (or pass -cml Qwen3)",
        fix_hint_sampling="set temperature=0.7 top_p=0.8 repeat_penalty=1.05",
    ),
    "deepseek": ModelEntry(
        # match_model lowercases both sides, so one lowercase substring covers
        # DeepSeek-V3 / DeepSeek-R1 / any casing variant.
        model_id_patterns=["deepseek"],
        canonical_chat_template="deepseek",
        canonical_sampler=SamplerDefaults(temperature=0.7, top_p=0.95, repeat_penalty=1.0),
        chat_template_probes=_chat_template_probes(),
        sampling_probes=_sampling_probes(),
        fix_hint_chat_template="llama-server --chat-template deepseek (or use the official deepseek template)",
        fix_hint_sampling="set temperature=0.7 top_p=0.95 repeat_penalty=1.0",
    ),
}


def match_model(model_id: str) -> str | None:
    """Match a /v1/models id against the registry. Returns the registry key or None."""
    mid_lower = model_id.lower()
    for key, entry in REGISTRY.items():
        for pattern in entry.model_id_patterns:
            if pattern.lower() in mid_lower:
                return key
    return None


def get_entry(key: str) -> ModelEntry | None:
    return REGISTRY.get(key)


# Divergence at or below this is within normal probe noise, not a
# misconfiguration. calibrate_loss() and LayerVerdict.is_misconfigured both
# key off this single threshold so "nonzero loss" and "misconfigured" can
# never disagree.
MISCONFIG_DIVERGENCE_THRESHOLD = 0.3

# Behavioral probes are noisy, hence the threshold above. A served default
# that differs from the canonical registry is not noise — it is the
# misconfiguration observed directly — so any confirmed mismatch raises the
# sampling divergence to at least this floor, which sits above the threshold.
SERVED_MISMATCH_FLOOR = 0.4


def calibrate_loss(layer: str, divergence_score: float, key: str) -> float:
    """Convert a divergence score to an estimated capability loss percentage."""
    if divergence_score <= MISCONFIG_DIVERGENCE_THRESHOLD:
        return 0.0
    entry = REGISTRY.get(key)
    if not entry:
        return round(divergence_score * 50.0, 1)
    cal = entry.loss_calibration.get(layer, {"slope": 40.0, "intercept": 1.0})
    loss = divergence_score * cal["slope"] + cal["intercept"]
    return round(min(loss, 95.0), 1)
