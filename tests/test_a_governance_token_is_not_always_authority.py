"""`require_governance` returns a token on three paths. One is a grant.

    * a real governed scope             — authority
    * `degraded_mode` / domain `degraded`   — the runtime is not up yet, so
      the boundary could not be applied
    * `VIOLATION` / domain `ungoverned`     — the call was made OUTSIDE a
      governed context and the bypass was recorded

The last two are records of ABSENCE. Returning them as tokens is deliberate:
boot has to proceed, and a recorded violation is more useful than a silent
one. The defect is downstream — `GovernanceToken.valid` is True for all
three (they have a receipt_id and have not expired), so a sink gating on
`valid`, or merely on `token is not None`, accepts the two objects that
exist to record that governance did not happen. That converts the absence of
the security boundary into permission, which is the one thing the boundary
exists to prevent.

`authorizes` is the question a sink means. `subprocess_gateway` was already
reaching for it by hand and getting it half right: it checked
`domain == "degraded"` and so missed `ungoverned` entirely.
"""
from __future__ import annotations

import contextlib
import re
from pathlib import Path

from core.governance_context import GovernanceToken

ROOT = Path(__file__).resolve().parents[1]


def _token(receipt: str, domain: str, ttl: float = 300.0) -> GovernanceToken:
    return GovernanceToken(
        receipt_id=receipt, domain=domain, source="test", ttl=ttl
    )


# A half-built engine has enough state for the guard clause and not enough
# for the shaping that follows it. Only the errors a missing attribute can
# raise are absorbed here — anything else is a real failure and must surface.
_STUB_SHAPING_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError)


def _filter_through_the_guard(engine) -> None:
    from core.brain import personality_engine as pe

    with contextlib.suppress(*_STUB_SHAPING_ERRORS):
        pe.PersonalityEngine.filter_response(engine, "hello", user_facing=True)


# ─────────────────────────── the three tokens are distinguishable


def test_a_real_token_authorizes():
    assert _token("receipt-1", "file_write").authorizes is True


def test_a_degraded_token_does_not_authorize():
    """Boot mode. The boundary could not be applied — that is not consent."""
    token = _token("degraded_mode", "degraded")

    assert token.valid is True, "it is still a well-formed, unexpired token"
    assert token.authorizes is False


def test_a_violation_token_does_not_authorize():
    """A recorded bypass is a record OF the bypass, not permission for it."""
    token = _token("VIOLATION", "ungoverned", ttl=1.0)

    assert token.authorizes is False


def test_validity_and_authority_are_different_questions():
    """The whole defect in one assertion: `valid` cannot separate them."""
    degraded = _token("degraded_mode", "degraded")
    violation = _token("VIOLATION", "ungoverned")

    assert degraded.valid == violation.valid is True
    assert degraded.authorizes == violation.authorizes is False


def test_an_expired_token_authorizes_nothing():
    import time

    token = _token("receipt-1", "file_write", ttl=0.01)
    time.sleep(0.05)

    assert token.expired is True
    assert token.authorizes is False


def test_an_empty_receipt_authorizes_nothing():
    assert _token("", "file_write").authorizes is False


# ──────────────────── the sink that was getting it half right


def test_the_subprocess_gateway_catches_both_non_authority_tokens():
    """It checked `domain == "degraded"` and missed `ungoverned`.

    Both record the absence of the boundary; only one was caught.
    """
    from tests.source_contract import family_text_at

    # the privilege check was lifted into `subprocess_privilege`
    source = family_text_at(ROOT / "core" / "runtime" / "subprocess_gateway.py")

    assert 'getattr(token, "domain", "") == "degraded"' not in source, (
        "the gateway hand-rolls a degraded-only check again, which lets an "
        "ungoverned VIOLATION token through"
    )
    assert 'not getattr(token, "authorizes", False)' in source


# ──────────────────────────── the guard on the whole class


def test_no_sink_treats_a_bare_governance_token_as_permission():
    """The structural half, so the next one is caught.

    A sink that captures the token and then branches on `token is None`, or
    on a hand-written domain comparison, is asking a question that cannot
    tell a grant from a record of absence. `authorizes` is the one that can.
    """
    # `token is None` alone is fine as a null check when it is ALSO asking
    # about authority; what this catches is a domain string comparison
    # standing in for the property.
    hand_rolled = re.compile(r'\.domain\s*(?:==|!=)\s*["\'](?:degraded|ungoverned)["\']')
    offenders: list[str] = []
    for path in (ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        rel = str(path.relative_to(ROOT))
        if rel.endswith("governance_context.py"):
            # The module that DEFINES the distinction may name the domains.
            continue
        for line_no, line in enumerate(
            path.read_text("utf-8", errors="ignore").splitlines(), 1
        ):
            if line.lstrip().startswith("#"):
                continue
            if hand_rolled.search(line):
                offenders.append(f"{rel}:{line_no}")

    assert not offenders, (
        f"these compare a governance token's domain by hand instead of "
        f"asking `token.authorizes`: {offenders}. A domain check written at "
        "one call site drifts from the set the module actually maintains."
    )


def test_the_non_authority_sets_are_named_once():
    """Two copies of this list diverge, and the copy is the one that goes
    stale — the same argument the contract module makes about thresholds."""
    from core.governance_context import (
        _NON_AUTHORITY_DOMAINS,
        _NON_AUTHORITY_RECEIPTS,
    )

    assert "degraded" in _NON_AUTHORITY_DOMAINS
    assert "ungoverned" in _NON_AUTHORITY_DOMAINS
    assert "degraded_mode" in _NON_AUTHORITY_RECEIPTS
    assert "VIOLATION" in _NON_AUTHORITY_RECEIPTS


# ───────── a tampered identity seal produced a working engine


def test_the_seal_verdict_is_state_not_just_a_log_line():
    """`if not self._verify_cryptographic_seal(): logger.critical(...)` — and
    then construction finished. A tampered identity kernel produced a fully
    working PersonalityEngine, indistinguishable from a verified one to
    every caller. The seal exists to detect exactly that, and the detection
    reached a log and stopped."""
    source = (ROOT / "core" / "brain" / "personality_engine.py").read_text("utf-8")

    assert "self.identity_verified = bool(self._verify_cryptographic_seal())" in source, (
        "the seal result is discarded again instead of being recorded on the "
        "engine"
    )


def test_an_unverified_engine_records_it_when_it_filters(monkeypatch):
    """The filter must not present its output as identity-filtered when the
    engine backing it was never attested."""
    from core.brain import personality_engine as pe

    engine = pe.PersonalityEngine.__new__(pe.PersonalityEngine)
    engine.identity_verified = False
    engine._unverified_filter_reported = False

    recorded: list[dict] = []
    monkeypatch.setattr(
        pe,
        "_record_personality_degradation",
        lambda exc, **kw: recorded.append(kw),
    )

    _filter_through_the_guard(engine)

    assert recorded, "an unverified engine filtered a reply and said nothing"
    assert recorded[0].get("severity") == "critical"


def test_the_report_fires_once_per_engine(monkeypatch):
    """A per-turn critical record for a standing condition is noise that
    buries the next real one."""
    from core.brain import personality_engine as pe

    engine = pe.PersonalityEngine.__new__(pe.PersonalityEngine)
    engine.identity_verified = False
    engine._unverified_filter_reported = False

    recorded: list[dict] = []
    monkeypatch.setattr(
        pe,
        "_record_personality_degradation",
        lambda exc, **kw: recorded.append(kw),
    )

    for _ in range(3):
        _filter_through_the_guard(engine)

    assert len(recorded) == 1


def test_a_verified_engine_records_nothing(monkeypatch):
    from core.brain import personality_engine as pe

    engine = pe.PersonalityEngine.__new__(pe.PersonalityEngine)
    engine.identity_verified = True

    recorded: list[dict] = []
    monkeypatch.setattr(
        pe,
        "_record_personality_degradation",
        lambda exc, **kw: recorded.append(kw),
    )

    _filter_through_the_guard(engine)

    assert not recorded


# ──────── an approval nobody can look up later


def test_an_approval_without_a_receipt_is_marked_on_the_goal():
    """`if receipt_id: goal["will_receipt"] = receipt_id` — and then
    `return True` regardless.

    An approval carrying no receipt proceeded with the key simply ABSENT,
    which reads downstream exactly like an action that was never authorized.
    An approval nobody can look up later is indistinguishable from no
    approval at all once the logs roll.
    """
    source = (ROOT / "core" / "volition.py").read_text("utf-8")

    assert 'goal["will_unaudited"] = True' in source, (
        "an approved action with no receipt leaves will_receipt absent "
        "again, so the audit gap is invisible to anything reading the goal"
    )
    assert 'goal["will_receipt"] = ""' in source, (
        "the key should be present and empty rather than missing — a missing "
        "key and an unaudited approval must not look the same"
    )


def test_the_action_still_proceeds_on_a_real_approval():
    """The Will DID approve. Discarding a real approval over missing
    bookkeeping would be its own failure."""
    import ast

    source = (ROOT / "core" / "volition.py").read_text("utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if 'goal["will_unaudited"] = True' not in rendered:
            continue
        # The branch records and marks; it must not return False.
        assert "return False" not in rendered, (
            "a missing receipt now suppresses an action the Will approved"
        )
        break
    else:
        raise AssertionError("the unaudited-approval branch was not found")


# ───── "starting empty" and "is empty" were the same observable state


def _library(tmp_path, payload: str | None):
    from core.brain.llm.latent_cortex.schedules import ScheduleLibrary

    path = tmp_path / "schedules.json"
    if payload is not None:
        path.write_text(payload, encoding="utf-8")
    return ScheduleLibrary(path)


def test_a_corrupt_schedule_library_is_distinguishable_from_an_empty_one(tmp_path):
    """A silent loss of every measured schedule looked like a fresh install.

    Downstream status and schedule selection could not tell a library that
    legitimately holds nothing from one that is corrupt, unreadable by
    permission, or written under a schema this build does not accept.
    """
    corrupt = _library(tmp_path, "{ not json")

    assert corrupt.load_error, "an unreadable library reports nothing wrong"
    assert "JSONDecodeError" in corrupt.load_error


def test_a_schema_mismatch_is_reported_rather_than_silently_empty(tmp_path):
    import json

    mismatched = _library(
        tmp_path, json.dumps({"version": -1, "revision": 1, "records": []})
    )

    assert "schema version" in mismatched.load_error


def test_an_absent_library_is_not_an_error(tmp_path):
    """A first run has nothing to load, and that is a fact about the
    library rather than about the reader."""
    assert _library(tmp_path, None).load_error == ""


def test_a_valid_library_clears_the_error_and_keeps_its_revision(tmp_path):
    import json

    from core.brain.llm.latent_cortex.schedules import (
        SCHEDULE_LIBRARY_SCHEMA_VERSION,
    )

    library = _library(
        tmp_path,
        json.dumps(
            {"version": SCHEDULE_LIBRARY_SCHEMA_VERSION, "revision": 3, "records": []}
        ),
    )

    assert library.load_error == ""
    assert library._revision == 3, (
        "a successful load must keep the on-disk revision; starting at 0 is "
        "what makes a later save look stale"
    )


# ──── an empty override set meant two different things


def test_a_failed_override_compilation_says_so_rather_than_returning_nothing(
    monkeypatch,
):
    """`{}` from a failed compilation was indistinguishable from `{}`
    meaning "no overrides were needed", so substrate-driven sampling and
    voice constraints vanished with nothing the caller could see."""
    from core.brain.llm import runtime_wiring as rw

    class _Boom:
        def get_generation_params_for(self, **_kw):
            raise RuntimeError("substrate voice engine is down")

    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: _Boom(),
    )

    overrides = rw.derive_substrate_generation_overrides(
        runtime_state=object(), objective="say something", origin="user", is_background=False
    )

    assert overrides.get("substrate_generation_source", "").startswith("unavailable:"), (
        f"a failed override compilation returned {overrides!r}, which reads "
        "exactly like 'no overrides were needed'"
    )


def test_the_failure_marker_reaches_the_request():
    """It rides the field the SUCCESS path already uses, so it arrives by a
    route that exists rather than a new one."""
    from core.brain.llm.llm_router import IntelligentLLMRouter

    kwargs: dict = {}
    IntelligentLLMRouter._apply_substrate_generation_overrides(
        kwargs, {"substrate_generation_source": "unavailable:RuntimeError"}
    )

    assert kwargs["substrate_generation_source"] == "unavailable:RuntimeError"


def test_no_sampling_values_are_invented_on_failure(monkeypatch):
    """The caller's defaults still apply — that is the honest outcome. The
    fix states it; it must not fabricate a temperature."""
    from core.brain.llm import runtime_wiring as rw

    class _Boom:
        def get_generation_params_for(self, **_kw):
            raise RuntimeError("down")

    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: _Boom(),
    )

    overrides = rw.derive_substrate_generation_overrides(
        runtime_state=object(), objective="x", origin="user", is_background=False
    )

    assert set(overrides) == {"substrate_generation_source"}


def test_a_background_turn_still_needs_no_overrides():
    """The fix must not turn "not applicable" into "failed"."""
    from core.brain.llm.runtime_wiring import derive_substrate_generation_overrides

    assert (
        derive_substrate_generation_overrides(
            runtime_state=None, objective="x", origin="s", is_background=True
        )
        == {}
    )


# ─── the person asked for five bullets, got three, and was not told


def test_an_unmet_stated_requirement_is_disclosed():
    """`_DELIVERABLE_RESIDUAL_SURFACE_REASONS` rightly delivers a draft that
    misses a formatting requirement — destroying the turn leaves the person
    with nothing. But a stated requirement is the PERSON'S instruction, not
    our judgement about thinness, and delivering silently means they cannot
    tell a shortfall from a decision."""
    from core.brain.llm.mlx_worker import _requirement_shortfall_note

    note = _requirement_shortfall_note(["missing_requested_list_count"])

    assert "list items you asked for" in note
    assert note.startswith("\n\n(")


def test_several_shortfalls_read_as_one_sentence():
    from core.brain.llm.mlx_worker import _requirement_shortfall_note

    note = _requirement_shortfall_note(
        ["missing_requested_word_count", "missing_requested_followup_question"]
    )

    assert note.count("(") == 1
    assert " or " in note


def test_a_thinness_residual_discloses_nothing():
    """Thinness is our judgement about the draft, not an instruction the
    person gave. Apologising for it would bury the answer."""
    from core.brain.llm.mlx_worker import _requirement_shortfall_note

    assert _requirement_shortfall_note(["too_thin_for_user_turn"]) == ""
    assert _requirement_shortfall_note(["reply_abandons_thread"]) == ""


def test_every_shortfall_reason_has_a_human_label():
    """A reason with no label discloses nothing and silently rejoins the
    class this fix exists to close."""
    from core.brain.llm.mlx_worker import (
        _REQUIREMENT_SHORTFALL_LABELS,
        _REQUIREMENT_SHORTFALL_REASONS,
    )

    unlabelled = _REQUIREMENT_SHORTFALL_REASONS - set(_REQUIREMENT_SHORTFALL_LABELS)

    assert not unlabelled, f"these shortfalls would be delivered silently: {unlabelled}"


def test_the_shortfall_reasons_are_actually_deliverable():
    """If one of these were NOT in the deliverable set, the turn would be
    destroyed and the disclosure would never run."""
    from core.brain.llm.mlx_worker import (
        _DELIVERABLE_RESIDUAL_SURFACE_REASONS,
        _REQUIREMENT_SHORTFALL_REASONS,
    )

    assert _REQUIREMENT_SHORTFALL_REASONS <= _DELIVERABLE_RESIDUAL_SURFACE_REASONS
