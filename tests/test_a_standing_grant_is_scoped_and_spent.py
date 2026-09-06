"""An operator can decide earlier, not less.

`activate_upgrade` required an operator authorization string — one human
decision per swap, over a pipeline that is otherwise automatic and receipted.
A standing grant moves that decision earlier and makes it narrower. It does
not remove it, and these tests are what hold the difference.

Four properties, each of which a grant stops being a scoped, bounded,
expiring decision without: it names what it covers, it expires, it is spent
as it is used, and it replaces the operator's presence rather than the
evidence.
"""

from __future__ import annotations

import json
import time

import pytest

from core.learning.cortex_generation_upgrade import _who_authorized_this
from core.self_modification.standing_authorization import (
    GRANT_SCHEMA,
    LONGEST_GRANT_S,
    AStandingGrant,
    read_standing_grant,
    spend_one_activation,
    where_a_grant_is_kept,
    why_a_grant_does_not_cover,
    write_standing_grant,
)

_A_MODEL = "/models/Qwen3.8-27B-4bit-abc123"
_A_DIGEST = "d" * 64


def _grant(**kw) -> AStandingGrant:
    base = dict(
        granted_by="an operator",
        model_path_prefixes=("/models/Qwen3.8-",),
        descriptor_digests=(),
        expires_at=time.time() + 3600,
        most_activations=1,
    )
    base.update(kw)
    return AStandingGrant(**base)


# ── it names what it covers ───────────────────────────────────────────────

def test_a_grant_covering_the_candidate_authorizes_it():
    assert why_a_grant_does_not_cover(
        _grant(), model_path=_A_MODEL, descriptor_digest=""
    ) == ""


def test_a_candidate_outside_the_scope_is_refused():
    said = why_a_grant_does_not_cover(
        _grant(), model_path="/models/SomethingElse-70B", descriptor_digest=""
    )
    assert "does not cover" in said


def test_a_grant_naming_nothing_covers_nothing(tmp_path):
    said = why_a_grant_does_not_cover(
        _grant(model_path_prefixes=(), descriptor_digests=()),
        model_path=_A_MODEL,
        descriptor_digest=_A_DIGEST,
    )
    assert "names no candidates" in said


def test_an_unscoped_grant_cannot_be_written(tmp_path):
    """One covering everything is not a scope."""
    with pytest.raises(ValueError, match="not a scope"):
        write_standing_grant(
            tmp_path, granted_by="an operator", valid_for_s=3600
        )


def test_a_digest_scope_matches_exactly(tmp_path):
    grant = _grant(model_path_prefixes=(), descriptor_digests=(_A_DIGEST,))
    assert why_a_grant_does_not_cover(
        grant, model_path="/anywhere", descriptor_digest=_A_DIGEST
    ) == ""
    assert why_a_grant_does_not_cover(
        grant, model_path="/anywhere", descriptor_digest="e" * 64
    )


# ── it expires ────────────────────────────────────────────────────────────

def test_an_expired_grant_is_refused_and_says_by_how_long():
    said = why_a_grant_does_not_cover(
        _grant(expires_at=time.time() - 3 * 86400),
        model_path=_A_MODEL,
        descriptor_digest="",
    )
    assert "expired" in said and "days ago" in said


def test_a_grant_with_no_expiry_is_not_a_standing_grant():
    said = why_a_grant_does_not_cover(
        _grant(expires_at=0.0), model_path=_A_MODEL, descriptor_digest=""
    )
    assert "no expiry" in said


def test_a_grant_cannot_outlive_the_window_it_was_written_for(tmp_path):
    with pytest.raises(ValueError, match="at most"):
        write_standing_grant(
            tmp_path,
            granted_by="an operator",
            model_path_prefixes=("/models/",),
            valid_for_s=LONGEST_GRANT_S + 1,
        )


# ── it is spent ───────────────────────────────────────────────────────────

def test_a_spent_grant_is_refused():
    said = why_a_grant_does_not_cover(
        _grant(most_activations=1, used=1),
        model_path=_A_MODEL,
        descriptor_digest="",
    )
    assert "spent" in said


def test_spending_persists_so_a_grant_of_one_cannot_authorize_twice(tmp_path):
    """A ceiling nobody decrements is not a ceiling."""
    grant = write_standing_grant(
        tmp_path,
        granted_by="an operator",
        model_path_prefixes=("/models/Qwen3.8-",),
        valid_for_s=3600,
        most_activations=1,
    )
    spend_one_activation(tmp_path, grant)
    again = read_standing_grant(tmp_path)
    assert again is not None
    assert again.used == 1
    assert why_a_grant_does_not_cover(
        again, model_path=_A_MODEL, descriptor_digest=""
    )


def test_the_activation_path_spends_the_grant(tmp_path):
    write_standing_grant(
        tmp_path,
        granted_by="an operator",
        model_path_prefixes=("/models/Qwen3.8-",),
        valid_for_s=3600,
        most_activations=2,
    )
    said = _who_authorized_this(
        tmp_path, authorized_by="", model_path=_A_MODEL, descriptor_digest=_A_DIGEST
    )
    assert said["route"] == "standing_grant"
    assert said["activations_used"] == 1
    assert read_standing_grant(tmp_path).used == 1


def test_an_operator_present_does_not_spend_the_grant(tmp_path):
    """Consuming a ceiling nobody drew on would be the wrong bookkeeping."""
    write_standing_grant(
        tmp_path,
        granted_by="an operator",
        model_path_prefixes=("/models/Qwen3.8-",),
        valid_for_s=3600,
        most_activations=1,
    )
    said = _who_authorized_this(
        tmp_path,
        authorized_by="a person at the keyboard",
        model_path=_A_MODEL,
        descriptor_digest=_A_DIGEST,
    )
    assert said["route"] == "operator_present"
    assert read_standing_grant(tmp_path).used == 0


# ── it replaces presence, not evidence ────────────────────────────────────

def test_no_grant_and_no_operator_is_refused(tmp_path):
    with pytest.raises(PermissionError, match="no standing grant"):
        _who_authorized_this(
            tmp_path, authorized_by="", model_path=_A_MODEL, descriptor_digest=""
        )


def test_the_refusal_names_which_condition_failed(tmp_path):
    """"Not authorized" sends an operator to look at everything."""
    write_standing_grant(
        tmp_path,
        granted_by="an operator",
        model_path_prefixes=("/models/OnlyThisFamily-",),
        valid_for_s=3600,
    )
    with pytest.raises(PermissionError, match="does not cover"):
        _who_authorized_this(
            tmp_path, authorized_by="", model_path=_A_MODEL, descriptor_digest=""
        )


def test_a_malformed_grant_is_absent_rather_than_fatal(tmp_path):
    where_a_grant_is_kept(tmp_path).write_text("{not json", encoding="utf-8")
    assert read_standing_grant(tmp_path) is None
    where_a_grant_is_kept(tmp_path).write_text(
        json.dumps({"schema": "something.else"}), encoding="utf-8"
    )
    assert read_standing_grant(tmp_path) is None


def test_a_grant_needs_a_real_operator_identity(tmp_path):
    with pytest.raises(PermissionError, match="real operator identity"):
        write_standing_grant(
            tmp_path,
            granted_by="x",
            model_path_prefixes=("/models/",),
            valid_for_s=3600,
        )


def test_nothing_autonomous_writes_a_grant():
    """The protection is that no autonomous path calls the writer."""
    import pathlib
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[1]
    found = subprocess.run(
        ["grep", "-rn", "write_standing_grant", "--include=*.py",
         str(root / "core"), str(root / "interface"), str(root / "skills")],
        capture_output=True, text=True,
    ).stdout.splitlines()
    callers = [
        one for one in found
        if "standing_authorization.py" not in one and "def write_standing_grant" not in one
    ]
    assert not callers, f"an autonomous path can write its own grant: {callers}"


def test_the_grant_module_is_sealed_from_plastic_modification():
    from core.will import is_plastic_target_allowed

    assert not is_plastic_target_allowed(
        "core.self_modification.standing_authorization"
    )


def test_the_schema_is_named():
    assert GRANT_SCHEMA.startswith("aura.cortex_upgrade.standing_grant")
