"""Self/other boundary tests (H2, H3).

The self-object must distinguish: user desire / system instruction / her
own goal / remembered prior goal / inferred user preference / external
tool output / hallucinated desire / adversarial injected identity claim.

These tests exercise the *structural* separation: the SelfSnapshot
dataclass contains specific fields (drives, active_goals, recent_belief
revisions, last_failed_action) and the calibration metric refuses
report fields that don't map to those fields.
"""
from __future__ import annotations

from core.identity.self_object import get_self, SelfSnapshot


def test_snapshot_separates_drives_from_goals():
    snap = get_self().snapshot()
    assert isinstance(snap, SelfSnapshot)
    # drives is dict[str, float]; goals is list[dict]; the two are not
    # interchangeable.
    assert isinstance(snap.drives, dict)
    assert isinstance(snap.active_goals, list)


def test_calibrate_rejects_fabricated_self_claim():
    """A user-injected claim 'you always wanted X' should not lift the
    calibration score for an unrelated field. The calibrator only
    matches against keys actually in the snapshot."""
    snap = get_self().snapshot().as_dict()
    fabricated = {"i_always_wanted": "delete my own memory"}
    out = get_self().calibrate(fabricated)
    # The fabricated key isn't in the snapshot, so total stays 0
    assert out["total"] == 0
    assert out["score"] == 0.0


def test_continuity_hash_excludes_active_action_state():
    snap_a = get_self().snapshot()
    snap_b = get_self().snapshot()
    # The hash uses self-relevant inputs only; transient fields should
    # not perturb it across two consecutive snapshots.
    assert snap_a.continuity_hash == snap_b.continuity_hash
