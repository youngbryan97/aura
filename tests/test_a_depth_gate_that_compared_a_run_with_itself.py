"""Live recurrent depth is one, and the reason is in the evidence.

The ceiling has been the identity case since an unvalidated depth setting
latched the foreground lane and took the conversation surface down. The comment
that pinned it says it stays there "until an accuracy gate says otherwise", and
above one has been reachable only by setting
AURA_USER_SURFACE_RECURRENT_MAX_LOOPS by hand — a person authorising an
experiment rather than evidence authorising a default.

The evidence that existed says less than it looks. `artifacts/recurrent_depth`
holds `loops1.json` and `loops2.json`, they differ in `created_at` and `timing`
and in nothing else, and their response files have the same SHA-256. Neither
records the depth it ran at. The two arms are one arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _gate():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "depth_gate", REPO / "tools/gate_user_surface_recurrent_depth.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


#: The pair that sat in `artifacts/recurrent_depth` until 2026-09-12: the same
#: responses digest, the same accuracy, no declared depth, seventy seconds
#: apart. `tools/heldout_eval.py` produced both and has no depth option, so
#: the thing the filenames claimed to compare was never varied.
#:
#: Built here rather than read from disk. The files are deleted — one named
#: `loops2.json` that is not a depth-two arm is a trap for the next person —
#: and what has to keep working is the gate's refusal, not their presence.
ONE_RUN_WRITTEN_TWICE = {
    "accuracy": 0.625,
    "adapter_path": "",
    "max_tokens": 256,
    "model": "/Users/bryan/.aura/live-source/models/Qwen2.5-1.5B-Instruct-4bit",
    "responses_sha256": "c3d871b8dfa69b78eb9df3cc0f1c5c69bbc1870a2a0bb8ec50e164085fca0fe6",
    "schema_version": 1,
    "tool": "heldout_eval",
    "battery": {"size": 32},
}


def test_two_reports_from_one_run_are_one_arm() -> None:
    """Not an opinion about them: the same bytes, twice."""
    first = dict(ONE_RUN_WRITTEN_TWICE, created_at=1783883597.448098)
    second = dict(ONE_RUN_WRITTEN_TWICE, created_at=1783883667.652107)

    assert first["responses_sha256"] == second["responses_sha256"]
    assert first["accuracy"] == second["accuracy"]
    assert {
        key for key in set(first) | set(second) if first.get(key) != second.get(key)
    } <= {"created_at", "timing"}


def test_the_gate_refuses_a_comparison_between_a_run_and_itself(tmp_path) -> None:
    gate = _gate()
    arms = []
    for index, stamp in enumerate((1783883597.448098, 1783883667.652107), start=1):
        path = tmp_path / f"loops{index}.json"
        arm = dict(ONE_RUN_WRITTEN_TWICE, created_at=stamp)
        path.write_text(json.dumps(arm), encoding="utf-8")
        arms.append((path, arm))
    verdict = gate.adjudicate(arms)

    assert verdict["authorized_depth"] == 1
    assert any("identical responses" in one for one in verdict["refusals"])
    assert any("no declared depth" in one for one in verdict["refusals"])


def test_the_arms_that_replaced_them_are_a_real_comparison() -> None:
    """And the gate refuses depth two on the measurement rather than on an
    absence: 0.050 against 0.525."""
    shallow = REPO / "artifacts/recurrent_depth/arm_loops1.json"
    deep = REPO / "artifacts/recurrent_depth/arm_loops2.json"
    if not shallow.is_file() or not deep.is_file():
        pytest.skip("no depth arms on disk")

    gate = _gate()
    arms = [(path, json.loads(path.read_text())) for path in (shallow, deep)]
    verdict = gate.adjudicate(arms)

    assert verdict["refusals"] == []
    assert verdict["authorized_depth"] == 1
    assert verdict["margin"] < 0.0


def test_a_margin_inside_the_shallower_arms_own_spread_authorizes_nothing() -> None:
    gate = _gate()
    arms = [
        (Path("depth1.json"), {
            "loops": 1, "accuracy": 0.625, "responses_sha256": "a" * 64,
            "model": "m", "battery": {"size": 32},
        }),
        (Path("depth2.json"), {
            "loops": 2, "accuracy": 0.656, "responses_sha256": "b" * 64,
            "model": "m", "battery": {"size": 32},
        }),
    ]
    verdict = gate.adjudicate(arms)

    assert verdict["refusals"] == []
    assert verdict["authorized_depth"] == 1, "one more answer out of 32 is noise"


def test_a_real_improvement_authorizes_the_deeper_arm() -> None:
    gate = _gate()
    arms = [
        (Path("depth1.json"), {
            "loops": 1, "accuracy": 0.625, "responses_sha256": "a" * 64,
            "model": "m", "battery": {"size": 32},
        }),
        (Path("depth2.json"), {
            "loops": 2, "accuracy": 0.875, "responses_sha256": "b" * 64,
            "model": "m", "battery": {"size": 32},
        }),
    ]
    verdict = gate.adjudicate(arms)

    assert verdict["refusals"] == []
    assert verdict["authorized_depth"] == 2


def test_the_live_ceiling_fails_closed_without_a_receipt(tmp_path, monkeypatch) -> None:
    from core.brain.llm import user_surface_recurrence as policy

    monkeypatch.setattr(policy, "_RECEIPTS", tmp_path)
    assert policy.user_surface_recurrent_ceiling("nothing-here") == 1

    (tmp_path / "abc123.json").write_text(
        json.dumps({
            "model_descriptor_sha256": "abc123",
            "authorized_depth": 3,
            "refusals": [],
        })
    )
    assert policy.user_surface_recurrent_ceiling("abc123") == 3
    assert policy.user_surface_recurrent_ceiling("a-different-checkpoint") == 1

    (tmp_path / "refused.json").write_text(
        json.dumps({
            "model_descriptor_sha256": "refused",
            "authorized_depth": 3,
            "refusals": ["the arms do not differ in depth"],
        })
    )
    assert policy.user_surface_recurrent_ceiling("refused") == 1, (
        "a receipt that refused its own comparison authorizes nothing"
    )
