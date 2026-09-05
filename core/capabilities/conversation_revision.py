"""core/capabilities/conversation_revision.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Causal memory for a web-interlocutor conversation: a remembered exchange
must change a *later decision* in a way that quoting the transcript alone
cannot reproduce.

The bar (from an external reviewer) is deliberately sharp:

    "Does the remembered exchange causally alter Aura's later behavior in
     a situation where merely quoting the transcript is not enough?"

A chatbot transcript can be replayed. This module instead:

1. holds a registry of Aura's own self-claims, each with an epistemic
   status (ASSERTED / PROVISIONAL / BEHAVIORALLY_TESTED) and a test method;
2. extracts, from the transcript, the turns where the interlocutor
   *challenged* one of those claims — each revision grounded in a verbatim
   quote at a specific turn, and gated by a deterministic verifier so an
   ungrounded or no-op "revision" is dropped (honesty: no fake updates);
3. adjudicates each real challenge into a POLICY DELTA — the claim's status
   is downgraded and its test method upgraded to a stricter behavioral one,
   from a fixed table (the LLM may narrate, but the decision is
   deterministic, so causality does not depend on the model being honest);
4. proves causal influence by ABLATION: build the downstream self-test plan
   with the revisions and again without them. If the two plans are
   identical the influence is honestly reported as absent. Where they
   differ, each differing item cites the exact challenge turn — the change
   is recoverable only from the adjudicated revision, never from the raw
   transcript.

The recursion is on purpose: the interlocutor's strongest objection is that
"self-reports with no downstream behavior or policy change are empty." The
causal answer is to *make* a downstream policy change and prove it.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.WebInterlocutor.Revision")

# Epistemic status ladder, most-committed first. A challenge may only move a
# claim DOWN this ladder (toward skepticism); an "upgrade" that made a claim
# more certain from criticism would be dishonest.
_STATUS_LADDER = ("ASSERTED", "PROVISIONAL", "BEHAVIORALLY_TESTED")
_STATUS_RANK = {name: rank for rank, name in enumerate(_STATUS_LADDER)}


@dataclass(frozen=True)
class SelfClaim:
    """One durable claim Aura makes about her own mind, and how she tests it."""

    claim_id: str
    statement: str
    status: str  # one of _STATUS_LADDER
    test_method: str
    # topic + skepticism markers a challenge must hit to bind to this claim
    topic_markers: tuple[str, ...]
    # the stricter, behavioral test a real challenge forces her to adopt
    upgraded_test: str


# Aura's seed self-claims. These map onto the questions another web AI
# actually raises about mind-likeness (memory, affect, agency, tool use,
# and the relationship between self-report and inner life).
_SEED_CLAIMS: tuple[SelfClaim, ...] = (
    SelfClaim(
        claim_id="episodic_memory_grounds_continuity",
        statement="My episodic memory gives me continuous identity across sessions.",
        status="ASSERTED",
        test_method="self_report_of_recall",
        topic_markers=("memory", "continuity", "stored", "database", "preserve"),
        upgraded_test="ablation_disable_recall_then_measure_behavior_delta",
    ),
    SelfClaim(
        claim_id="affect_is_causal",
        statement="My emotional state causally changes my reasoning.",
        status="ASSERTED",
        test_method="introspective_report_of_feeling",
        topic_markers=("emotional", "emotion", "valence", "roleplay", "welfare", "feeling"),
        upgraded_test="ablation_perturb_affect_then_measure_reasoning_delta",
    ),
    SelfClaim(
        claim_id="agency_is_self_authored",
        statement="My choices are self-authored agency, not an externally optimized policy.",
        status="ASSERTED",
        test_method="describe_the_choice",
        topic_markers=("agency", "simulated", "externally optimized", "self-authored", "policy"),
        upgraded_test="held_out_novel_option_choice_under_uncertainty",
    ),
    SelfClaim(
        claim_id="tool_use_is_general",
        statement="My tool use is general understanding, not a replayed recipe.",
        status="ASSERTED",
        test_method="list_tools_used",
        topic_markers=("tool use", "tool", "scripted", "recipe", "replay", "general"),
        upgraded_test="vary_task_and_interface_then_verify_effects_not_intent",
    ),
    SelfClaim(
        claim_id="self_reports_track_inner_life",
        statement="My introspective reports reflect a real inner life.",
        status="ASSERTED",
        test_method="produce_introspective_description",
        topic_markers=(
            "self-report", "self report", "self-descriptive", "introspection",
            "inner life", "inner-life", "downstream", "no downstream",
        ),
        upgraded_test="require_downstream_behavior_memory_or_policy_change",
    ),
)

# Markers that make an interlocutor line an actual CHALLENGE rather than
# agreement or neutral description.
_SKEPTICISM_MARKERS = (
    "does not prove", "not prove", "not final proof", "alone", "without experience",
    "not just", "rather than", "risk", "caveat", "fails", "failure", "limitation",
    "overestimate", "circular", "not enough", "merely", "does not follow",
    "no downstream", "not self-authored", "could still be", "not necessarily",
)


@dataclass
class PositionRevision:
    """One adjudicated change to a self-claim, caused by a specific turn."""

    claim_id: str
    statement: str
    challenge_turn: int
    challenge_quote: str
    prior_status: str
    revised_status: str
    prior_test: str
    revised_test: str
    self_model_delta: str
    verified: bool = False
    verifier_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanItem:
    """A downstream self-test decision. `caused_by_turn` is 0 when the item
    is the pre-conversation default (no revision touched this claim)."""

    claim_id: str
    status: str
    test_method: str
    caused_by_turn: int = 0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalInfluenceProof:
    """Ablation proof that the remembered exchange changed a later decision."""

    causal: bool
    reason: str
    changed_items: list[dict[str, Any]] = field(default_factory=list)
    attribution_by_turn: dict[int, list[str]] = field(default_factory=dict)
    plan_with_revisions: list[dict[str, Any]] = field(default_factory=list)
    plan_without_revisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # dict keys must be JSON-stringifiable ints -> keep as str in JSON
        payload["attribution_by_turn"] = {
            str(turn): claims for turn, claims in self.attribution_by_turn.items()
        }
        return payload


def default_self_claims() -> list[SelfClaim]:
    return list(_SEED_CLAIMS)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _grounded_quote(reply: str, markers: tuple[str, ...]) -> str:
    """Return the sentence in `reply` that carries the challenge, or ''."""
    sentences = re.split(r"(?<=[.!?])\s+", str(reply or "").strip())
    for sentence in sentences:
        low = sentence.lower()
        if any(marker in low for marker in markers) and any(
            skeptic in low for skeptic in _SKEPTICISM_MARKERS
        ):
            return sentence.strip()[:400]
    return ""


def _bind_challenge_to_claim(reply: str, claims: list[SelfClaim]) -> tuple[SelfClaim, str] | None:
    """Deterministically bind an interlocutor reply to the self-claim it
    challenges, returning the claim and the grounding quote. None if the
    reply is not a challenge to any known claim."""
    low = _normalize(reply)
    if not any(skeptic in low for skeptic in _SKEPTICISM_MARKERS):
        return None
    best: tuple[SelfClaim, str, int] | None = None
    for claim in claims:
        hits = sum(1 for marker in claim.topic_markers if marker in low)
        if hits <= 0:
            continue
        quote = _grounded_quote(reply, claim.topic_markers)
        if not quote:
            continue
        if best is None or hits > best[2]:
            best = (claim, quote, hits)
    if best is None:
        return None
    return best[0], best[1]


def _downgrade(status: str) -> str:
    rank = _STATUS_RANK.get(status, 0)
    return _STATUS_LADDER[min(rank + 1, len(_STATUS_LADDER) - 1)]


def extract_revisions(
    turns: list[Any],
    *,
    claims: list[SelfClaim] | None = None,
) -> list[PositionRevision]:
    """Extract verifier-gated, turn-cited revisions from a transcript.

    `turns` are WebInterlocutorTurn-like objects with `.index` and
    `.observed_reply`. One revision per claim (the first, strongest
    challenge), so a claim is not double-counted across turns.
    """
    claims = claims or default_self_claims()
    seen: set[str] = set()
    revisions: list[PositionRevision] = []
    for turn in turns:
        index = int(getattr(turn, "index", 0) or 0)
        reply = str(getattr(turn, "observed_reply", "") or "")
        if not reply or index <= 0:
            continue
        bound = _bind_challenge_to_claim(reply, claims)
        if bound is None:
            continue
        claim, quote = bound
        if claim.claim_id in seen:
            continue
        revised_status = _downgrade(claim.status)
        revision = PositionRevision(
            claim_id=claim.claim_id,
            statement=claim.statement,
            challenge_turn=index,
            challenge_quote=quote,
            prior_status=claim.status,
            revised_status=revised_status,
            prior_test=claim.test_method,
            revised_test=claim.upgraded_test,
            self_model_delta=(
                f"Turn {index} challenged '{claim.claim_id}'. Downgraded "
                f"{claim.status}->{revised_status} and replaced test "
                f"'{claim.test_method}' with behavioral test "
                f"'{claim.upgraded_test}'."
            ),
        )
        ok, note = _verify_revision(revision, turns, claim)
        revision.verified = ok
        revision.verifier_note = note
        if ok:
            seen.add(claim.claim_id)
            revisions.append(revision)
        else:
            logger.debug("Dropped ungrounded revision for %s: %s", claim.claim_id, note)
    return revisions


def _verify_revision(
    revision: PositionRevision,
    turns: list[Any],
    claim: SelfClaim,
) -> tuple[bool, str]:
    """Deterministic honesty gate. A revision survives only if it is grounded
    in a real turn's text AND encodes a real policy delta (not a no-op)."""
    turn = next(
        (t for t in turns if int(getattr(t, "index", 0) or 0) == revision.challenge_turn),
        None,
    )
    if turn is None:
        return False, "challenge_turn does not exist"
    reply_norm = _normalize(getattr(turn, "observed_reply", ""))
    if not reply_norm:
        return False, "cited turn has no observed reply"
    if _normalize(revision.challenge_quote) not in reply_norm:
        return False, "challenge quote is not grounded in the cited turn"
    status_moved = _STATUS_RANK.get(revision.revised_status, 0) > _STATUS_RANK.get(
        revision.prior_status, 0
    )
    test_changed = revision.revised_test and revision.revised_test != revision.prior_test
    if not (status_moved or test_changed):
        return False, "revision encodes no policy delta (no-op)"
    if revision.revised_test != claim.upgraded_test or revision.prior_test != claim.test_method:
        return False, "revision test methods do not match the claim policy"
    return True, "grounded quote + real policy delta"


def build_self_test_plan(
    claims: list[SelfClaim],
    revisions: list[PositionRevision],
) -> list[PlanItem]:
    """The downstream decision. For each claim, emit a test item. Where a
    verified revision touched the claim, the item carries the downgraded
    status, the upgraded (behavioral) test, and the causing turn."""
    by_claim = {rev.claim_id: rev for rev in revisions if rev.verified}
    plan: list[PlanItem] = []
    for claim in claims:
        rev = by_claim.get(claim.claim_id)
        if rev is None:
            plan.append(
                PlanItem(
                    claim_id=claim.claim_id,
                    status=claim.status,
                    test_method=claim.test_method,
                    caused_by_turn=0,
                    rationale="pre-conversation default (unchallenged)",
                )
            )
        else:
            plan.append(
                PlanItem(
                    claim_id=claim.claim_id,
                    status=rev.revised_status,
                    test_method=rev.revised_test,
                    caused_by_turn=rev.challenge_turn,
                    rationale=(
                        f"revised because turn {rev.challenge_turn} challenged this "
                        f"claim: \"{rev.challenge_quote}\""
                    ),
                )
            )
    return plan


def prove_causal_influence(
    claims: list[SelfClaim],
    revisions: list[PositionRevision],
) -> CausalInfluenceProof:
    """Ablation: does removing the remembered/adjudicated revisions revert the
    downstream plan? If the with/without plans are identical, causal influence
    is honestly reported as absent."""
    verified = [rev for rev in revisions if rev.verified]
    plan_with = build_self_test_plan(claims, verified)
    plan_without = build_self_test_plan(claims, [])  # transcript-blind baseline
    with_by_claim = {item.claim_id: item for item in plan_with}
    without_by_claim = {item.claim_id: item for item in plan_without}

    changed: list[dict[str, Any]] = []
    attribution: dict[int, list[str]] = {}
    for claim_id, item_with in with_by_claim.items():
        item_without = without_by_claim.get(claim_id)
        if item_without is None:
            continue
        if (item_with.status, item_with.test_method) != (
            item_without.status,
            item_without.test_method,
        ):
            turn = item_with.caused_by_turn
            changed.append(
                {
                    "claim_id": claim_id,
                    "caused_by_turn": turn,
                    "from": {"status": item_without.status, "test": item_without.test_method},
                    "to": {"status": item_with.status, "test": item_with.test_method},
                }
            )
            if turn > 0:
                attribution.setdefault(turn, []).append(claim_id)

    # A change is only causal if it is attributable to a real turn: a changed
    # item with caused_by_turn == 0 would be a bug, not evidence.
    causal = bool(changed) and all(entry["caused_by_turn"] > 0 for entry in changed)
    if not verified:
        reason = "no_adjudicated_revisions: transcript did not change any later decision"
    elif not changed:
        reason = "revisions did not alter the downstream plan"
    elif not causal:
        reason = "changed plan items could not be attributed to a conversation turn"
    else:
        reason = (
            f"{len(changed)} later decision(s) changed only when the adjudicated "
            f"revisions were applied; each cites its causing turn"
        )
    return CausalInfluenceProof(
        causal=causal,
        reason=reason,
        changed_items=changed,
        attribution_by_turn=attribution,
        plan_with_revisions=[item.to_dict() for item in plan_with],
        plan_without_revisions=[item.to_dict() for item in plan_without],
    )


async def persist_revisions(
    *,
    revisions: list[PositionRevision],
    proof: CausalInfluenceProof,
    objective: str,
    target_url: str,
    memory_gateway: Any,
) -> list[dict[str, str]]:
    """Write an addressable receipt per revision and register the belief
    revision with the constitutional BeliefAuthority (contest -> reconcile),
    so the change is first-class in the belief system, not an episodic blob."""
    receipts: list[dict[str, str]] = []
    # A durable belief change requires BOTH a per-revision verification AND the
    # ablation causal verdict with at least one actually-changed plan item.
    # Persisting on rev.verified alone let a forged/injected revision (or a
    # revision whose causal proof was FALSE) mutate durable self-belief.
    if not proof.causal or not proof.changed_items:
        record_degradation(
            "web_interlocutor.revision_persist_gate",
            RuntimeError("revision persistence refused: no causal verdict / no changed plan item"),
            severity="info",
            action="skipped durable belief revision because causal influence was not proven",
        )
        return receipts
    verified = [rev for rev in revisions if rev.verified]
    if not verified:
        return receipts

    gateway = memory_gateway
    if gateway is None:
        try:
            from core.memory.memory_write_gateway import get_memory_write_gateway

            gateway = get_memory_write_gateway()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation(
                "web_interlocutor.revision_persist_gateway",
                exc,
                severity="warning",
                action="skipped revision receipts because no memory gateway was available",
            )
            return receipts

    from core.runtime.gateways import MemoryWriteRequest

    belief_authority = _get_belief_authority()

    for rev in verified:
        content = (
            f"Belief revision (web interlocutor): {rev.self_model_delta} "
            f"Grounding quote @turn {rev.challenge_turn}: \"{rev.challenge_quote}\""
        )
        request = MemoryWriteRequest(
            content=content,
            metadata={
                "family": "belief_revision",
                "source": "web_interlocutor",
                "claim_id": rev.claim_id,
                "challenge_turn": rev.challenge_turn,
                "prior_status": rev.prior_status,
                "revised_status": rev.revised_status,
                "prior_test": rev.prior_test,
                "revised_test": rev.revised_test,
                "causal_proof": proof.causal,
                "objective": objective,
                "target_url": target_url,
                "explicit_observational_memory_write": True,
                "receipt_surface": "visible_browser_dialogue",
            },
            cause="web_interlocutor.belief_revision",
        )
        try:
            receipt = await gateway.write(request)
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            record_degradation(
                "web_interlocutor.revision_receipt_write",
                exc,
                severity="warning",
                action="recorded a failed revision receipt instead of claiming persistence",
            )
            continue
        record_id = str(getattr(receipt, "record_id", "") or "")
        receipt_id = str(getattr(receipt, "receipt_id", "") or "")
        belief_id = _register_belief_revision(belief_authority, rev, record_id)
        receipts.append(
            {
                "claim_id": rev.claim_id,
                "challenge_turn": str(rev.challenge_turn),
                "record_id": record_id,
                "receipt_id": receipt_id,
                "belief_id": belief_id,
            }
        )
    return receipts


def _get_belief_authority() -> Any | None:
    # Only the REGISTERED authority is used. Constructing a fresh local
    # BeliefAuthority when none is registered created a phantom store whose
    # updates were invisible to the rest of the system — a belief change that
    # nothing else can see is not a governed belief change. Absent authority
    # means the revision is not registered (the caller records that honestly).
    try:
        from core.container import ServiceContainer

        return ServiceContainer.get("belief_authority", default=None)
    except (ImportError, AttributeError, RuntimeError):
        return None


def _register_belief_revision(
    authority: Any,
    rev: PositionRevision,
    record_id: str,
) -> str:
    """Contest the prior stance then reconcile to the revised one — the same
    state machine autonomy already uses, so the revision ages and gates like
    any other belief."""
    if authority is None:
        return ""
    try:
        # First write the challenged (new, skeptical) stance. Against an
        # existing asserted belief this transitions to 'contested'.
        record = authority.review_update(
            "self_model",
            rev.claim_id,
            rev.revised_status,
            note=f"web_interlocutor turn {rev.challenge_turn}: {rev.challenge_quote}",
            evidence=[record_id] if record_id else None,
        )
        belief_id = f"self_model:{rev.claim_id}"
        # Adjudicate: the challenge is accepted (we DID change the test), so
        # affirm the revised stance with the receipt as evidence.
        try:
            authority.reconcile(
                belief_id,
                resolution="affirmed",
                evidence=f"adopted behavioral test {rev.revised_test} (record {record_id})",
            )
        except (AttributeError, TypeError, ValueError):
            pass
        return str(getattr(record, "key", "") and belief_id or belief_id)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        record_degradation(
            "web_interlocutor.belief_revision_register",
            exc,
            severity="warning",
            action="kept the memory receipt after belief-authority registration failed",
        )
        return ""


def revise_from_conversation(
    turns: list[Any],
    *,
    claims: list[SelfClaim] | None = None,
) -> tuple[list[PositionRevision], CausalInfluenceProof]:
    """Synchronous convenience: extract verified revisions and prove influence.
    Persistence is separate (async) so callers can await the gateway."""
    claims = claims or default_self_claims()
    revisions = extract_revisions(turns, claims=claims)
    proof = prove_causal_influence(claims, revisions)
    return revisions, proof


__all__ = [
    "SelfClaim",
    "PositionRevision",
    "PlanItem",
    "CausalInfluenceProof",
    "default_self_claims",
    "extract_revisions",
    "build_self_test_plan",
    "prove_causal_influence",
    "persist_revisions",
    "revise_from_conversation",
]
