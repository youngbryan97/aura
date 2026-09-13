"""The interactive lane and the training lane resolve depth independently.

2026-07-26: depth was briefly suspected of causing fluent-nonsense replies and
the default was flipped off; the A/B disproved it (the cause was the kNN
datastore) and the default was restored. What this pins is the separation that
made that experiment safe: the interactive lane can be changed on its own
without reaching the training lane or the Recursive Latent Cortex.
"""

import pytest

import core.brain.llm.recurrent_depth as rd
from core.brain.llm.recurrent_depth import _get_lane_defaults, resolve_loops_for_model

pytestmark = pytest.mark.unit


class _Inner:
    def __init__(self, layers: int) -> None:
        self.layers = [object()] * layers


class _Model:
    def __init__(self, layers: int) -> None:
        self.model = _Inner(layers)


def _clear(monkeypatch):
    for name in (
        "AURA_RECURRENT_LOOPS",
        "AURA_RECURRENT_LOOPS_72B",
        "AURA_RECURRENT_LOOPS_32B",
        "AURA_RECURRENT_LOOPS_14B",
        "AURA_RECURRENT_LOOPS_SMALL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_the_interactive_lane_runs_at_the_identity_depth(monkeypatch):
    """One loop, because nothing ever measured that two helped.

    Eighteen tests cover this module's cache discipline and override bounds
    and not one asserts an improvement. The only measurement anyone has is
    against it — 21/40 to 2/40 on a 28-layer checkpoint — and the governed
    version of the same idea is the RLC, which is proven and serving.
    """
    _clear(monkeypatch)
    assert resolve_loops_for_model(_Model(64)) == 1
    assert _get_lane_defaults(64)[0] == 1


def test_two_loops_is_still_one_environment_variable_away(monkeypatch):
    """Turned off on the evidence, not removed. It has to be able to come
    back the moment an arm on the resident checkpoint says it should."""
    _clear(monkeypatch)
    monkeypatch.setenv("AURA_RECURRENT_LOOPS_32B", "2")
    assert resolve_loops_for_model(_Model(64)) == 2


def test_the_training_lane_still_gets_two_loops(monkeypatch):
    """scripts/train_with_recurrent_depth.py sets AURA_RECURRENT_LOOPS=2
    before calling apply_for_model, and that override is read FIRST — ahead
    of any profile default. Training is unaffected by the interactive default.
    """
    _clear(monkeypatch)
    monkeypatch.setenv("AURA_RECURRENT_LOOPS", "2")
    assert resolve_loops_for_model(_Model(64)) == 2


def test_the_training_script_still_requests_depth():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "scripts" / "train_with_recurrent_depth.py"
    text = source.read_text(encoding="utf-8")
    assert 'setdefault("AURA_RECURRENT_LOOPS", "2")' in text, (
        "the training lane must keep asking for depth explicitly rather than "
        "inheriting the interactive default"
    )


def test_depth_can_be_taken_off_the_interactive_lane_alone(monkeypatch):
    """If depth is ever suspected again, this is the switch — and it does not
    reach training or the RLC."""
    _clear(monkeypatch)
    monkeypatch.setenv("AURA_RECURRENT_LOOPS_32B", "1")
    assert resolve_loops_for_model(_Model(64)) == 1


def test_the_parent_health_mirror_agrees_with_the_worker(monkeypatch):
    """Both sides must resolve the same number, or the lane reports a
    readiness blocker for running the pass that is actually correct."""
    from core.brain.llm.mlx_client import _expected_recurrent_loops_from_model_path

    _clear(monkeypatch)
    path = "/models/Aura-32B-crsm-closeout"
    assert _expected_recurrent_loops_from_model_path(path) == resolve_loops_for_model(
        _Model(64)
    )
    monkeypatch.setenv("AURA_RECURRENT_LOOPS_32B", "1")
    assert _expected_recurrent_loops_from_model_path(path) == resolve_loops_for_model(
        _Model(64)
    )


def test_the_latent_cortex_does_not_depend_on_the_patch_default():
    """The RLC borrows cache discipline helpers, not the profile table."""
    import core.brain.llm.latent_cortex.recurrence as rlc

    for helper in ("_snapshot_recurrent_caches", "_restore_recurrent_caches"):
        assert hasattr(rd, helper)
        assert getattr(rlc, helper) is getattr(rd, helper)
