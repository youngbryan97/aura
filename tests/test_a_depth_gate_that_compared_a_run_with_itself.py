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


def test_the_two_depth_arms_on_disk_are_one_arm() -> None:
    """Not an opinion about them: the same bytes, twice."""
    first = json.loads((REPO / "artifacts/recurrent_depth/loops1.json").read_text())
    second = json.loads((REPO / "artifacts/recurrent_depth/loops2.json").read_text())

    assert first["responses_sha256"] == second["responses_sha256"]
    assert first["accuracy"] == second["accuracy"]
    assert {
        key for key in set(first) | set(second) if first.get(key) != second.get(key)
    } <= {"created_at", "timing"}


def test_the_gate_refuses_a_comparison_between_a_run_and_itself() -> None:
    gate = _gate()
    arms = [
        (path, json.loads(path.read_text()))
        for path in (
            REPO / "artifacts/recurrent_depth/loops1.json",
            REPO / "artifacts/recurrent_depth/loops2.json",
        )
    ]
    verdict = gate.adjudicate(arms)

    assert verdict["authorized_depth"] == 1
    assert any("identical responses" in one for one in verdict["refusals"])
    assert any("no declared depth" in one for one in verdict["refusals"])


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
