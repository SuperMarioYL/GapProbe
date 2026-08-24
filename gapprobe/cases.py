"""gapprobe/cases.py — 20-case bench for validating 2-layer attribution.

10 mis-set cases (each has a known dominant misconfiguration layer) +
10 correctly-set cases (both layers clean). The bench asserts that the
attribution verdict assembler correctly identifies the dominant layer
in >50% of the mis-set cases (beats random guess).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gapprobe.probe import ProbeResult
from gapprobe.registry import ProbeSpec


@dataclass
class BenchCase:
    """One bench case with simulated probe results."""

    case_id: str
    model_key: str
    description: str
    ct_results: list[ProbeResult]  # chat-template probe results
    sp_results: list[ProbeResult]  # sampling probe results
    expected_dominant: str  # "chat_template" | "sampling" | "" (correctly-set)


def _make_results(
    probe_ids: list[str], fail_mask: list[bool]
) -> list[ProbeResult]:
    """Build simulated ProbeResults from a pass/fail mask.

    fail_mask: True = this probe FAILS, False = this probe PASSES.
    """
    return [
        ProbeResult(
            probe_id=pid,
            response_text="ok" if not fail else "wrong",
            success=not fail,
            detail="" if not fail else "simulated failure",
        )
        for pid, fail in zip(probe_ids, fail_mask)
    ]


_CT_IDS = [
    "ct_tool_call_format",
    "ct_multi_turn",
    "ct_system_adherence",
    "ct_structured_output",
    "ct_role_recognition",
    "ct_stop_sequence",
]
_SP_IDS = [
    "sp_instruction_follow",
    "sp_repetition",
    "sp_creative_constraint",
    "sp_factual_recall",
    "sp_format_json",
    "sp_code_block",
]


def _build_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []

    # ── 10 mis-set cases ──────────────────────────────────────────────
    # Chat-template mis-set (4 cases): 3-4 of 6 CT probes fail, SP all pass
    for i, fails in enumerate(
        [
            [True, True, True, False, False, False],
            [False, True, True, True, False, False],
            [True, False, True, False, True, False],
            [True, True, False, True, True, False],
        ]
    ):
        cases.append(
            BenchCase(
                case_id=f"mis_ct_{i+1}",
                model_key="qwen3" if i % 2 == 0 else "deepseek",
                description=f"chat-template misconfiguration variant {i+1}",
                ct_results=_make_results(_CT_IDS, fails),
                sp_results=_make_results(_SP_IDS, [False] * 6),
                expected_dominant="chat_template",
            )
        )

    # Sampling mis-set (4 cases): SP 3-4 fail, CT all pass
    for i, fails in enumerate(
        [
            [True, True, True, False, False, False],
            [False, True, False, True, False, True],
            [True, False, True, False, True, False],
            [False, True, True, True, False, False],
        ]
    ):
        cases.append(
            BenchCase(
                case_id=f"mis_sp_{i+1}",
                model_key="deepseek" if i % 2 == 0 else "qwen3",
                description=f"sampling misconfiguration variant {i+1}",
                ct_results=_make_results(_CT_IDS, [False] * 6),
                sp_results=_make_results(_SP_IDS, fails),
                expected_dominant="sampling",
            )
        )

    # Both-layers mis-set (2 cases): both CT and SP have failures, one dominant
    cases.append(
        BenchCase(
            case_id="mis_both_ct_dom",
            model_key="qwen3",
            description="both layers mis-set, chat-template dominant",
            ct_results=_make_results(_CT_IDS, [True, True, True, True, False, False]),
            sp_results=_make_results(_SP_IDS, [True, False, False, False, False, False]),
            expected_dominant="chat_template",
        )
    )
    cases.append(
        BenchCase(
            case_id="mis_both_sp_dom",
            model_key="deepseek",
            description="both layers mis-set, sampling dominant",
            ct_results=_make_results(_CT_IDS, [True, False, False, False, False, False]),
            sp_results=_make_results(_SP_IDS, [True, True, True, True, False, False]),
            expected_dominant="sampling",
        )
    )

    # ── 10 correctly-set cases ────────────────────────────────────────
    for i in range(10):
        model = "qwen3" if i % 2 == 0 else "deepseek"
        cases.append(
            BenchCase(
                case_id=f"clean_{i+1}",
                model_key=model,
                description=f"correctly configured {model}",
                ct_results=_make_results(_CT_IDS, [False] * 6),
                sp_results=_make_results(_SP_IDS, [False] * 6),
                expected_dominant="",  # no misconfiguration
            )
        )

    return cases


BENCH_CASES: list[BenchCase] = _build_cases()


def run_bench() -> tuple[int, int, list[tuple[str, str, str]]]:
    """Run the 20-case bench. Returns (correct, total, details)."""
    from gapprobe.layers.chat_template import ChatTemplateLayer
    from gapprobe.layers.sampling import SamplingLayer
    from gapprobe.verdict import assemble_verdict
    from gapprobe.registry import get_entry

    ct_layer = ChatTemplateLayer()
    sp_layer = SamplingLayer()
    correct = 0
    details: list[tuple[str, str, str]] = []

    for case in BENCH_CASES:
        entry = get_entry(case.model_key)
        if not entry:
            continue
        ct_verdict = ct_layer._build_verdict(case.ct_results, case.model_key)
        sp_verdict = sp_layer._build_verdict(case.sp_results, case.model_key)
        verdict = assemble_verdict(
            model=case.model_key,
            endpoint="bench://simulated",
            layer_verdicts=[ct_verdict, sp_verdict],
        )
        predicted = verdict.dominant_layer if verdict.total_estimated_loss_pct > 0 else ""
        # For correctly-set cases, expected dominant is "" (no misconfig)
        if case.expected_dominant == "":
            if predicted == "":
                correct += 1
                details.append((case.case_id, predicted or "none", "none"))
            else:
                details.append((case.case_id, predicted, "none"))
        else:
            if predicted == case.expected_dominant:
                correct += 1
                details.append((case.case_id, predicted, case.expected_dominant))
            else:
                details.append((case.case_id, predicted, case.expected_dominant))

    return correct, len(BENCH_CASES), details
