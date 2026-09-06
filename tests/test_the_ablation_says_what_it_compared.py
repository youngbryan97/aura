"""The boundary travels with the numbers, because the numbers travel further.

An external review traced the "full architecture versus raw model" recall
ablation and found it is one model against itself with and without assembled
context. That is a real experiment about memory. It is not the Aura organism
against the same model and compute without Aura, and the downstream artifact
writes the FULL condition's scores under the key ``aura_scores`` — which
reads as the organism to anyone who did not open the harness.

Nothing here changes what the harness measures. It makes the artifact say
what was compared, in the same place as the result, the way the budget-parity
report now travels inside BASELINES.json.
"""

from __future__ import annotations

from core.evaluation.ablation_harness import (
    FULL,
    PROMPTED,
    RAW,
    WHAT_FULL_MEANS,
    AblationHarness,
    AblationTask,
)


def _tasks() -> list[AblationTask]:
    return [
        AblationTask(
            task_id="t1",
            family="recall",
            turns=["The codeword is orbit-9.", "What was the codeword?"],
            answer_key="orbit-9",
        )
    ]


def _responder(condition, task, turn_index, history):
    """Only the stateful condition can see the earlier turn."""
    return "orbit-9" if condition == FULL else "I do not know"


def test_the_report_says_what_full_meant():
    harness = AblationHarness(conditions=(RAW, PROMPTED, FULL), bootstrap_iterations=64)
    report = harness.report(_responder, _tasks())
    assert report["claim_boundary"] == WHAT_FULL_MEANS


def test_the_other_report_path_says_it_too():
    """A live runner feeds scores back through report_from_results."""
    harness = AblationHarness(conditions=(RAW, PROMPTED, FULL), bootstrap_iterations=64)
    results = harness.run(_responder, _tasks())
    report = harness.report_from_results(results)
    assert report["claim_boundary"] == WHAT_FULL_MEANS


def test_the_boundary_denies_the_bigger_claim_out_loud():
    """A boundary that only describes what WAS done leaves the rest to inference."""
    said = WHAT_FULL_MEANS.lower()
    assert "assembled context" in said
    assert "not the" in said and "whole organism" in said
    assert "matched substrate" in said


def test_a_verdict_still_comes_out():
    """The boundary is added beside the result, not instead of it."""
    harness = AblationHarness(conditions=(RAW, PROMPTED, FULL), bootstrap_iterations=64)
    report = harness.report(_responder, _tasks())
    assert "architecture_beats_stateless" in report["verdict"]
