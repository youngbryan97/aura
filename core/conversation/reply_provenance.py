"""Where a reply came from, and why that changes what counts as a defect.

``_BROKEN_LANE_BOILERPLATE_RE`` exists to catch a real failure: the model
narrating the runtime — "I dropped the heavy reasoning lane", "ask me again" —
*instead of* answering, while an answer was there to give. That is a defect
because the person asked a question and got a status report.

The same sentence from a different author is the opposite of a defect. When
``_build_degraded_live_reply`` says "I couldn't get a clear enough answer
together… ask me again", generation has already failed, every recovery has
already been tried, and the verified-floor lookup has already come back empty.
There is no answer being withheld. The sentence is true, and there is nothing
below it to fall back to.

The gate could not tell those apart, because it never asked who wrote the text.
So it read the runtime's own honest admission as lane chatter and logged
``assessment=runtime_boilerplate`` — twice, live, on 2026-08-04, on turns whose
real defect was an off-topic gate upstream discarding a correct answer.

The fix is not to forbid the words. Aura may say whatever is true. The fix is
that the gate now knows the difference between *talking about the runtime
instead of answering* and *reporting truthfully that no answer exists*, because
the caller that knows which one it is says so.

Provenance excuses exactly one family of reasons — "this is runtime talk rather
than an answer" — and nothing else. A leak is still a leak, a corrupted string
is still corrupt, and an admission that dresses itself up as an answer is
caught by ``admission_defects`` below.
"""

from __future__ import annotations

import hashlib
import re
import contextvars
import time
from dataclasses import dataclass
from enum import Enum

from core.conversation.session_scope import (
    current_conversation_session,
    current_conversation_turn,
)


class ReplyProvenance(str, Enum):
    """Who composed this text, and what was already known when they did."""

    #: The model produced it as its answer to the turn. Runtime talk here
    #: stands in for an answer, which is the failure the gate is for.
    MODEL_DRAFT = "model_draft"

    #: The runtime composed it AFTER generation and every recovery failed, and
    #: after the verified-floor lookup returned nothing. Saying so is the only
    #: honest thing left; there is no further fallback below it.
    HONEST_FAILURE = "honest_failure"

    #: The runtime composed it from evidence it READ rather than generated
    #: (a source excerpt, a file listing, a recorded measurement). Runtime
    #: vocabulary here describes the evidence, not a lane failure.
    READ_EVIDENCE = "read_evidence"


#: Reasons that mean "this is runtime talk rather than an answer". Every one of
#: them is an inference about the AUTHOR'S CHOICE — that they narrated the
#: machine when they could have answered. That inference is unavailable when no
#: answer existed to choose instead.
_RUNTIME_TALK_REASONS = frozenset(
    {
        "runtime_boilerplate",
        "non_answer_repair_floor",
        "low_signal_acknowledgement_placeholder",
        "friendly_failure_placeholder",
    }
)

#: Reasons no provenance can excuse: these are about what the text WOULD DO to
#: a person, not about what its author chose to talk about.
_NEVER_EXCUSED = frozenset(
    {
        "corrupted_language",
        "escaped_control_artifact",
        "internal_live_gate_leak",
        "internal_task_prompt_leak",
        "prompt_artifact",
        "raw_lane_telemetry",
        "raw_model_identity_leak",
        "raw_tool_result_fragment",
        "unsupported_embodiment_claim",
        "unsupported_external_provider_path_claim",
    }
)


def excused_reasons(provenance: ReplyProvenance | str | None) -> frozenset[str]:
    """Which gate reasons this author's position makes inapplicable."""

    try:
        resolved = ReplyProvenance(str(provenance or ReplyProvenance.MODEL_DRAFT))
    except ValueError:
        return frozenset()
    if resolved is ReplyProvenance.HONEST_FAILURE:
        return _RUNTIME_TALK_REASONS - _NEVER_EXCUSED
    if resolved is ReplyProvenance.READ_EVIDENCE:
        return frozenset({"runtime_boilerplate"}) - _NEVER_EXCUSED
    return frozenset()


@dataclass(frozen=True)
class AdmissionCheck:
    """What an honest failure has to contain to be worth sending."""

    ok: bool
    defects: tuple[str, ...]


#: Phrasings that claim an answer. An admission that no answer exists cannot
#: also assert one — that is the failure this check exists for, and it is the
#: only thing provenance must never be able to hide.
_ANSWER_CLAIMS = (
    "the answer is",
    "here is the answer",
    "here's the answer",
    "i found that",
    "the result is",
    "it turns out",
    "in summary,",
)


_WORD_RE = re.compile(r"[a-z0-9']+")

#: Three consecutive words carried over from the turn is not coincidence; one
#: or two ("the file", "you can") happen by chance in any English sentence.
_ECHO_RUN_TOKENS = 3


def _verbatim_echo_spans(user_message: str, reply_text: str) -> list[str]:
    """Runs of the person's own words, in order, that the reply gives back.

    Contiguity is what makes this evidence. A reply that reuses scattered
    vocabulary from the turn has demonstrated nothing; a reply carrying an
    unbroken run of what they wrote has demonstrably received it.
    """

    said = _WORD_RE.findall(str(user_message or "").lower())
    back = _WORD_RE.findall(str(reply_text or "").lower())
    if len(said) < _ECHO_RUN_TOKENS or len(back) < _ECHO_RUN_TOKENS:
        return []

    said_runs = {
        " ".join(said[i : i + _ECHO_RUN_TOKENS])
        for i in range(len(said) - _ECHO_RUN_TOKENS + 1)
    }
    spans: list[str] = []
    for i in range(len(back) - _ECHO_RUN_TOKENS + 1):
        run = " ".join(back[i : i + _ECHO_RUN_TOKENS])
        if run in said_runs:
            spans.append(run)
    return spans


def admission_defects(user_message: str, reply_text: str) -> AdmissionCheck:
    """Check an honest-failure reply against what makes it honest.

    Provenance removes one class of objection; it must not remove scrutiny. An
    admission earns its exemption by being a real admission: it says nothing
    happened, it shows what it understood so the person can see whether they
    were even parsed, and it does not assert a finding it does not have.
    """

    text = " ".join(str(reply_text or "").split())
    lowered = text.lower()
    defects: list[str] = []

    if not text:
        defects.append("empty_admission")
        return AdmissionCheck(False, tuple(defects))

    echoed = _verbatim_echo_spans(user_message, text)

    # Quoting the person cannot be asserting a finding. Without this, an
    # admission that echoes "tell me what the answer is" back to them is
    # accused of claiming an answer it explicitly said it did not have.
    claim_surface = lowered
    for span in echoed:
        claim_surface = claim_surface.replace(span, " ")
    if any(claim in claim_surface for claim in _ANSWER_CLAIMS):
        defects.append("admission_claims_an_answer")

    # Naming the subject back is what makes the admission useful — it lets the
    # person see whether the turn was even parsed correctly. A turn that named
    # no subject cannot be echoed, so it is not required to be.
    #
    # This was three literal phrases ("understood you to be asking", "i
    # understood", "you asked"), which made it a lexical gate deciding a
    # semantic property: on 2026-08-10 the degraded composer began showing the
    # person their own question back — the strongest possible evidence that the
    # turn was parsed — and was marked `admission_names_nothing_understood` for
    # not using one of the three approved openers. Giving their words back IS
    # naming the subject, and it is better evidence than the phrase.
    if str(user_message or "").strip() and not echoed:
        if not any(
            marker in lowered
            for marker in ("understood you to be asking", "i understood", "you asked")
        ):
            defects.append("admission_names_nothing_understood")

    return AdmissionCheck(not defects, tuple(defects))


#: Provenance has to survive the trip. A composer knows what it wrote and why,
#: but the gate that judges it runs a dozen frames away through call sites that
#: pass nothing but the string — ``_looks_semantically_glitched`` is the one
#: that produced the live ``assessment=runtime_boilerplate``, and it receives
#: exactly a user message and a reply. Threading a parameter through every one
#: of those would leave the next call site free to forget, which is the shape
#: of defect this file exists to close. So the declaration travels with the
#: TEXT: whoever composes it says once what it is, and every assessor inherits
#: that without being modified.
_DECLARED: contextvars.ContextVar[tuple[tuple[str, str, str, str, float], ...]] = (
    contextvars.ContextVar("conversation_reply_provenance", default=())
)
_DECLARED_LIMIT = 64
_DECLARED_TTL_SECONDS = 600.0


def _key(text: str) -> str:
    body = " ".join(str(text or "").split())
    return hashlib.sha256(body.encode("utf-8")).hexdigest() if body else ""


def declare_provenance(text: str, provenance: ReplyProvenance | str) -> None:
    """Record what a piece of composed text is, for whoever judges it later."""

    key = _key(text)
    if not key:
        return
    value = str(getattr(provenance, "value", provenance) or "")
    if not value:
        return
    session_id = current_conversation_session()
    turn_id = current_conversation_turn()
    now = time.time()
    rows = tuple(
        row
        for row in _DECLARED.get()
        if not (row[0] == key and row[2] == session_id and row[3] == turn_id)
        and now - row[4] <= _DECLARED_TTL_SECONDS
    )
    _DECLARED.set((*rows, (key, value, session_id, turn_id, now))[-_DECLARED_LIMIT:])


def declared_provenance(text: str) -> str:
    """What this exact text was declared to be, or "" if nobody said."""

    key = _key(text)
    if not key:
        return ""
    session_id = current_conversation_session()
    turn_id = current_conversation_turn()
    now = time.time()
    for row_key, value, row_session, row_turn, declared_at in reversed(_DECLARED.get()):
        if now - declared_at > _DECLARED_TTL_SECONDS:
            continue
        if (row_key, row_session, row_turn) == (key, session_id, turn_id):
            return value
    return ""


def forget_declared_provenance() -> None:
    """Drop every declaration. For tests and for process-level resets."""

    _DECLARED.set(())


__all__ = [
    "AdmissionCheck",
    "ReplyProvenance",
    "admission_defects",
    "declare_provenance",
    "declared_provenance",
    "excused_reasons",
    "forget_declared_provenance",
]


#: The reply the runtime sends when it has nothing it will stand behind.
#:
#: Written inline in two places and recognised in none, so evidence and this
#: could be joined: a correct finding followed by an admission of having no
#: finding. LIVE, 2026-08-27: "Wren is not top: Marek leads at 21 and Wren is
#: at 16." came back with this appended underneath it.
THE_HONEST_FAILURE = (
    "I couldn't get to an answer I'd stand behind on that one, and I "
    "won't send you a thinner one and pass it off as the real thing. "
    "Ask me again in a moment and I should have it."
)


def admits_no_answer(text: object) -> bool:
    """Whether this reply says it has nothing, and so may not follow evidence.

    Matched on the sentence rather than on a marker, because the marker is
    attached by the path that builds it and this is asked by paths that only
    receive it.
    """
    said = " ".join(str(text or "").split()).lower()
    if not said:
        return False
    if " ".join(THE_HONEST_FAILURE.split()).lower() in said:
        return True
    # The same admission, reworded by a repair pass.
    return (
        "couldn't get to an answer" in said
        or "could not get to an answer" in said
        or "won't send you a thinner one" in said
    )
