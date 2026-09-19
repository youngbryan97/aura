"""Counterexamples to answer containment and aggregate causal attribution."""

import pytest

from core.learning.integrated_reasoning_eval import (
    LEGACY_EVAL_POLICY, FixtureRetrieval, KnowledgeTask, assert_base_recall_guard, run_factorial,
)


def task(identity="one"):
    return KnowledgeTask(identity, f"Resolve the hidden value for {identity}.",
                         "ZQ_91", (), 2, ("derive_unstored_answer",))


@pytest.mark.parametrize("response", [
    "Not ZQ_91", "ZQ_910", "ZQ_91 or another value", "ZQ_91\nFINAL_ANSWER: wrong",
    "FINAL_ANSWER: ZQ_91\nActually, wrong", "<think>FINAL_ANSWER: ZQ_91",
    "<think>ZQ_91</think>FINAL_ANSWER: wrong",
])
def test_a_mentioned_answer_is_not_a_correct_selected_answer(response):
    report = run_factorial([task()], FixtureRetrieval(), lambda *_: response)
    assert report["accuracy"]["retrieval_on"][4] == 0.
    guard = assert_base_recall_guard([task()], lambda *_: response)
    assert guard["answered_from_memory"] == []


@pytest.mark.parametrize("response", [
    "ZQ_91", " zq_91. ", "FINAL_ANSWER: ZQ_91", "Answer: **ZQ_91**",
    "<think>wrong</think>FINAL_ANSWER: ZQ_91",
])
def test_terminal_public_answers_keep_formatting_tolerance(response):
    report = run_factorial([task()], FixtureRetrieval(), lambda *_: response)
    assert report["accuracy"]["retrieval_on"][4] == 1.


def test_disjoint_ablations_do_not_prove_joint_necessity():
    tasks = [task("retrieval"), task("depth")]

    class Source:
        def retrieve(self, query, *, limit):
            return ["fact"]

    def solve(prompt, context, depth):
        correct = bool(context) if "retrieval" in prompt else depth > 1
        return "ZQ_91" if correct else "wrong"

    report = run_factorial(tasks, Source(), solve, depths=(1, 4))
    assert report["verdicts"]["retrieval_is_causal"]
    assert report["verdicts"]["recurrence_helps"]
    assert not report["verdicts"]["both_required"]
    assert report["verdicts"]["joint_success_task_ids"] == []


def test_real_retriever_gets_the_question_without_a_fixture_identifier():
    seen = []

    class Source:
        def retrieve(self, query, *, limit):
            seen.append((query, limit))
            return ["fact"]

    item = task("opaque-fixture-id")
    run_factorial([item], Source(), lambda *_: "wrong", retrieval_limit=3)
    assert seen == [(item.prompt, 3)]


def test_each_cell_receives_its_own_context_list():
    class Source:
        def retrieve(self, query, *, limit):
            return ["fact"]

    def solve(prompt, context, depth):
        if context:
            context.clear()
            return "ZQ_91"
        return "wrong"

    report = run_factorial([task()], Source(), solve)
    assert all(value == 1. for value in report["accuracy"]["retrieval_on"].values())


@pytest.mark.parametrize("depths", [(4, 1), (1, 1, 4), (True, 4)])
def test_depth_order_and_identity_are_declared_not_inferred(depths):
    with pytest.raises(ValueError, match="depths"):
        run_factorial([task()], FixtureRetrieval(), lambda *_: "", depths=depths)


def test_duplicate_tasks_cannot_multiply_the_evidence():
    with pytest.raises(ValueError, match="duplicate"):
        run_factorial([task(), task()], FixtureRetrieval(), lambda *_: "")


def test_legacy_measurements_require_an_explicit_policy():
    solve = lambda *_: "ZQ_91 or another value"
    current = run_factorial([task()], FixtureRetrieval(), solve)
    old = run_factorial([task()], FixtureRetrieval(), solve, evaluation_policy=LEGACY_EVAL_POLICY)
    assert current["accuracy"]["retrieval_on"][4] == 0.
    assert old["accuracy"]["retrieval_on"][4] == 1.
    assert old["evaluation_policy"] != current["evaluation_policy"]
    assert not old["confirmatory_evidence"]


def test_joint_pattern_names_the_tasks_and_retains_exact_uncertainty():
    class Source:
        def retrieve(self, query, *, limit):
            return ["fact"]

    report = run_factorial([task()], Source(),
                          lambda prompt, context, depth: "ZQ_91" if context and depth == 4 else "wrong")
    verdict = report["verdicts"]
    assert verdict["joint_success_task_ids"] == ["one"]
    assert verdict["both_required"]
    tail = verdict["paired_depth"]["exact_one_sided_tail"]
    assert (tail["numerator"], tail["denominator"]) == (1, 2)
    assert not report["confirmatory_evidence"]
    assert not report["ordinary_pipeline_verified"]
    assert not report["runtime_retrieval_verified"]
    assert report["task_manifest"][0]["answer"] == "ZQ_91"
    assert report["per_task"][0]["selected_answers"]["on@4"] == "zq_91"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_retrieval_limit_cannot_silently_change_the_ablation(limit):
    with pytest.raises(ValueError, match="retrieval limit"):
        run_factorial([task()], FixtureRetrieval(), lambda *_: "", retrieval_limit=limit)


@pytest.mark.parametrize("response", ["", None, "<think>unclosed"])
def test_empty_normalized_gold_cannot_credit_missing_public_output(response):
    from dataclasses import replace

    empty = replace(task(), answer="!!!")
    with pytest.raises(ValueError, match="normalized public value"):
        run_factorial([empty], FixtureRetrieval(), lambda *_: response)
    with pytest.raises(ValueError, match="normalized public value"):
        assert_base_recall_guard([empty], lambda *_: response)


@pytest.mark.parametrize("pieces, expected", [
    (["ZQ_91\n"], ""),
    (["private\n", "</think>", "ZQ_91\n"], "ZQ_91\n"),
    (["private\n", "</think>", " \n", "ZQ_91\n"], " \nZQ_91\n"),
])
def test_tool_uses_template_channel_and_does_not_stop_on_private_newlines(monkeypatch, pieces, expected):
    import numpy as np
    import mlx.core as mx
    from types import SimpleNamespace

    import tools.eval_integrated_reasoning as tool

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, *args, **kwargs):
            return "<assistant><think>\n"

        def encode(self, text):
            return [50]

        def decode(self, ids, **kwargs):
            return "".join(pieces[value] for value in ids)

    calls = []

    def hidden(*args, **kwargs):
        calls.append(None)
        token = len(calls) - 1 if len(calls) <= len(pieces) else 99
        logits = np.zeros((1, 1, 100), dtype=np.float32)
        logits[0, 0, token] = 1.
        with mx.stream(mx.cpu):
            return mx.array(logits), {}

    monkeypatch.setattr(tool, "make_recurrent_caches", lambda *a: None)
    monkeypatch.setattr(tool, "recurrent_hidden_states", hidden)
    solver = tool.make_solver(SimpleNamespace(lm_head=lambda value: value), Tokenizer(),
        prelude_end=0, coda_start=1, max_tokens=8, envelope=None)
    assert solver(task().prompt, [], 1) == expected
    assert len(calls) >= len(pieces)


def test_tool_preserves_baseline_successes_and_publishes_once(monkeypatch, tmp_path, capsys):
    import json
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import core.runtime.model_lane_control as lane
    import tools.eval_integrated_reasoning as tool

    output = tmp_path / "receipt.json"
    item = task()
    monkeypatch.setattr(tool, "build_knowledge_tasks", lambda **_: [item])
    monkeypatch.setattr(tool, "make_solver", lambda *a, **kw: lambda *_: item.answer)
    monkeypatch.setattr(tool, "mlx_memory_envelope", lambda **_: nullcontext(
        SimpleNamespace(to_receipt=lambda: {})))
    monkeypatch.setattr(lane, "standalone_model_lane", lambda **_: nullcontext())
    loads = []
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(
        load=lambda path: (loads.append(path), None)))
    monkeypatch.setattr(sys, "argv", ["eval_integrated_reasoning", "--model", "fixture",
                                     "--out", str(output)])
    assert tool.main() == 0
    original = output.read_bytes()
    receipt = json.loads(original)
    assert receipt["schema"] == "aura.integrated_reasoning_run.v2"
    assert receipt["factorial_schema"] == "aura.integrated_reasoning_eval.v2"
    assert receipt["base_recall_guard"]["answered_from_memory"] == [item.task_id]
    assert receipt["tasks_after_guard"] == receipt["n_tasks"] == 1
    assert receipt["task_selection_policy"] == "all_declared_tasks_no_observed_outcome_filter"
    assert not receipt["verdicts"]["both_required"]
    assert "[verdict]" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        tool.main()
    assert output.read_bytes() == original
    assert loads == ["fixture"]
