"""tests/test_attribution.py — validate 2-layer attribution beats random guess."""

import pytest

from gapprobe.cases import run_bench, BENCH_CASES
from gapprobe.verdict import LayerVerdict, assemble_verdict
from gapprobe.registry import calibrate_loss, match_model


class TestBench:
    def test_bench_beats_random_guess(self):
        correct, total, details = run_bench()
        pct = correct / total
        assert pct > 0.5, f"Attribution accuracy {pct:.0%} does not beat random guess (50%)"

    def test_bench_has_20_cases(self):
        assert len(BENCH_CASES) == 20

    def test_bench_has_10_misconfig_10_clean(self):
        misconfig = [c for c in BENCH_CASES if c.expected_dominant]
        clean = [c for c in BENCH_CASES if not c.expected_dominant]
        assert len(misconfig) == 10
        assert len(clean) == 10

    def test_misconfig_cases_all_predicted_correctly(self):
        correct, _, details = run_bench()
        # All 10 misconfig cases should be correctly attributed
        misconfig_details = [d for d in details if d[2] != "none"]
        misconfig_correct = [d for d in misconfig_details if d[1] == d[2]]
        assert len(misconfig_correct) == 10, (
            f"Only {len(misconfig_correct)}/10 misconfig cases correctly attributed"
        )

    def test_clean_cases_report_no_misconfig(self):
        correct, _, details = run_bench()
        clean_details = [d for d in details if d[2] == "none"]
        clean_correct = [d for d in clean_details if d[1] == "none"]
        assert len(clean_correct) == 10, (
            f"Only {len(clean_correct)}/10 clean cases correctly reported no misconfig"
        )


class TestVerdictAssembly:
    def test_clean_case_zero_loss(self):
        v = assemble_verdict("test-model", "bench://", [
            LayerVerdict("chat_template", 0.0, 0.0, ["all passed"]),
            LayerVerdict("sampling", 0.0, 0.0, ["all passed"]),
        ])
        assert v.total_estimated_loss_pct == 0.0

    def test_misconfig_dominant_layer(self):
        v = assemble_verdict("test-model", "bench://", [
            LayerVerdict("chat_template", 0.5, 24.5, ["ChatML missing"]),
            LayerVerdict("sampling", 0.0, 0.0, ["all passed"]),
        ])
        assert v.dominant_layer == "chat_template"
        assert v.total_estimated_loss_pct == 24.5

    def test_one_liner_contains_model_and_layer(self):
        v = assemble_verdict("qwen3-27b", "http://localhost:8080", [
            LayerVerdict("chat_template", 0.5, 24.5, ["ChatML not applied"]),
            LayerVerdict("sampling", 0.0, 0.0, []),
        ], suggested_fix="llama-server --chat-template qwen3")
        assert "qwen3-27b" in v.one_liner
        assert "chat_template" in v.one_liner
        assert "qwen3" in v.one_liner


class TestRegistry:
    def test_match_qwen3(self):
        assert match_model("Qwen3-27B-Instruct") == "qwen3"

    def test_match_deepseek(self):
        assert match_model("deepseek-v3") == "deepseek"

    def test_match_unknown(self):
        assert match_model("llama3-70b") is None

    def test_calibrate_zero_divergence(self):
        assert calibrate_loss("chat_template", 0.0, "qwen3") == 0.0

    def test_calibrate_nonzero_divergence(self):
        loss = calibrate_loss("chat_template", 0.5, "qwen3")
        assert loss > 0
        assert loss <= 95.0
