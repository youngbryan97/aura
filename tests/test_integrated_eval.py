"""Contract tests: the integrated evaluation measures what it claims.

The structural properties are the test surface: answers exist ONLY in
context items (so context-off is capped at chance and above-chance means
leakage), both arms carry distractors (so the delta cannot be mere
presence-of-context), ingress must FIRE (receipted slots) or the run voids
itself, and generation/grading are deterministic.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from core.learning.integrated_eval_tasks import (
    context_for_arm,
    generate_tasks,
    grade,
)

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "integrated_rlc_eval.py"
_SPEC = importlib.util.spec_from_file_location("integrated_rlc_eval", _TOOL_PATH)
integrated = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(integrated)


# ── Task structure ──────────────────────────────────────────────────────


def test_generation_is_deterministic_and_answer_never_leaks():
    first = generate_tasks(count=8, seed=3, hops=2)
    second = generate_tasks(count=8, seed=3, hops=2)
    assert [t.prompt for t in first] == [t.prompt for t in second]
    assert [t.answer for t in first] == [t.answer for t in second]
    for task in first:
        assert task.answer.lower() not in task.prompt.lower()
        context_text = " ".join(item["text"] for item in task.context_items)
        assert task.answer in context_text, "the facts must carry the answer"


def test_two_hop_tasks_require_integration_not_lookup():
    task = generate_tasks(count=1, seed=5, hops=2)[0]
    assert len(task.context_items) == 2
    # The subject entity appears in the question and in the FIRST fact; the
    # answer code rides only the SECOND fact — combining them is the task.
    assert task.answer in task.context_items[1]["text"]
    assert task.answer not in task.context_items[0]["text"]


def test_both_arms_carry_distractors():
    task = generate_tasks(count=1, seed=9, hops=1)[0]
    on = context_for_arm(task, with_facts=True)
    off = context_for_arm(task, with_facts=False)
    assert len(off) == len(task.distractor_items) >= 1
    assert len(on) == len(off) + len(task.context_items)
    on_text = " ".join(item["text"] for item in on)
    off_text = " ".join(item["text"] for item in off)
    assert task.answer in on_text
    assert task.answer not in off_text


def test_grading_contract():
    task = generate_tasks(count=1, seed=1, hops=1)[0]
    assert grade(task, f"FINAL_ANSWER: {task.answer}") == "correct"
    assert grade(task, f"FINAL_ANSWER: {task.answer.lower()}.") == "correct"
    assert grade(task, f"the code is {task.answer} I believe") == "correct_lenient"
    assert grade(task, "FINAL_ANSWER: ZZ-000") == "incorrect"
    assert grade(task, "") == "unparseable"
    assert grade(task, "no idea") == "incorrect_lenient"


def test_generator_bounds():
    with pytest.raises(ValueError):
        generate_tasks(count=0, seed=1)
    with pytest.raises(ValueError):
        generate_tasks(count=4, seed=1, hops=3)


# ── Harness mechanics on the tiny model ─────────────────────────────────


class _StubTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [(hash(word) % 96) + 1 for word in str(text).split()][:80]

    def decode(self, tokens):
        return " ".join(f"tok{int(t)}" for t in tokens)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return self.encode(" ".join(str(m.get("content", "")) for m in messages))


def test_paired_eval_runs_and_receipts_ingress(monkeypatch, tmp_path):
    args_model_dir = tmp_path / "tiny"
    args_model_dir.mkdir()
    (args_model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    model_args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(model_args)
    mx.eval(model.parameters())

    import mlx_lm

    monkeypatch.setattr(mlx_lm, "load", lambda _p: (model, _StubTokenizer()))
    args = argparse.Namespace(
        model=str(args_model_dir),
        count=2,
        seed=7,
        hops=2,
        n_slots=6,
        max_steps=2,
        budget_layer_apps=2_000_000,
        max_seconds=600.0,
        out="",
    )
    receipt = integrated.run_eval(args)
    assert receipt["schema"] == integrated.INTEGRATED_EVAL_SCHEMA
    assert receipt["tasks"] == 2
    assert receipt["context_on"]["n"] == 2
    assert receipt["context_off"]["n"] == 2
    # Ingress FIRED: every context-on episode receipted memory-seeded slots.
    assert receipt["ingress_fired_everywhere"] is True, receipt["rows"]
    on_rows = [r for r in receipt["rows"] if r["arm"] == "context_on"]
    assert all("memory" in r["seeded_sources"] for r in on_rows)
    # A random tiny model cannot know the codes: leakage must not fire.
    assert receipt["leakage_suspected"] is False
    assert receipt["valid"] is True


def test_the_budget_is_spent_by_the_run_and_not_by_the_import():
    """`--max-seconds` measures the run, from the moment the run starts.

    It used to be measured from a module-level `_START` set at import. Run as
    a script those are the same instant. Imported into a process that goes on
    doing other things — a test chunk that loaded this module a quarter of an
    hour before it called it — the budget was already spent before the first
    task, and a 29-second eval raised TimeoutError.
    """
    import inspect

    source = inspect.getsource(integrated.run_eval)
    assert "started = time.time()" in source, (
        "run_eval has to take its own timestamp; a module-level one is spent "
        "by whatever the process did between import and call"
    )
    assert "time.time() - started" in source
    assert not hasattr(integrated, "_START"), (
        "a module-scope start time is the defect this replaced"
    )
