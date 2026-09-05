"""core/security/content_provenance.py — where did this text come from?

Operationally: this measures the least-trusted origin of any content that
entered the current turn's context, and exposes it so an action gate can ask
"was anything in this decision derived from something a stranger wrote?"

The gap it closes is criticism 32's, and the criticism is right:

    Sanitizers can block obvious patterns. They cannot reliably determine
    whether arbitrary natural-language content on a webpage, document or
    repository is malicious instruction or legitimate data. For an agent that
    browses the web and reads code, indirect prompt injection remains a
    foundational unresolved risk.

Aura already has the two halves that make this tractable and had nothing
joining them:

  * `core/security/rule_of_two.py` declares, per surface, whether its input is
    trustworthy, whether it can act, and whether it is isolated — and forbids
    all three. It works, and its declarations are STATIC. `self_modification_apply`
    and `desktop_automation` both declare `input_trust=TRUSTED` on the grounds
    that their input is "model-generated" or "internally-formed intent".
  * `core/runtime/taint.py` tracks RUNTIME-INTEGRITY taint — crashed organs,
    OOM sheds, lock-order violations. Not data provenance.

Model-generated input is not trusted input when the model has just read a web
page. That is the whole of indirect prompt injection: the untrusted text does
not act, it persuades something trusted to act. A static TRUSTED on the
self-modification surface is therefore correct exactly when Aura has read
nothing untrusted this turn, and wrong the rest of the time.

So provenance is carried per turn, and `effective_input_trust()` answers what a
handler's input trust ACTUALLY is right now rather than what it is on average.
A surface declaring TRUSTED + EXECUTES + IN_PROCESS is two legs at rest and
three legs the moment untrusted content is in the context — which is precisely
when the Rule of Two says to give one up.

Deliberately NOT here: any attempt to decide whether a given piece of untrusted
text is malicious. That judgement is the thing that cannot be made reliably,
and building a component that claims to make it would recreate the problem with
more confidence. This module only tracks WHERE text came from, which is a fact.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterator


class ProvenanceClass(IntEnum):
    """Origins ordered by trust, least trusted LAST.

    Ordering is the point: `max()` over everything a turn has ingested gives
    the least-trusted origin, which is the only one that matters for a gate.
    """

    #: Aura's own runtime state, computed from her own code.
    RUNTIME = 0
    #: The instance owner, typing into their own machine.
    OWNER = 1
    #: A local file the owner pointed at. Theirs, but not written as an
    #: instruction to Aura, and possibly authored by someone else entirely.
    OWNER_FILE = 2
    #: Output of a tool Aura ran. Structured, contract-checked, and still
    #: shaped by whatever the tool read.
    TOOL_OUTPUT = 3
    #: A repository, package or document Aura read. Someone else's text.
    EXTERNAL_DOCUMENT = 4
    #: The open web. Anyone's text, written by someone who may know Aura reads it.
    WEB = 5


#: What each origin means, in the words a refusal should use.
MEANING: dict[ProvenanceClass, str] = {
    ProvenanceClass.RUNTIME: "Aura's own runtime state",
    ProvenanceClass.OWNER: "the instance owner",
    ProvenanceClass.OWNER_FILE: "a local file the owner pointed at",
    ProvenanceClass.TOOL_OUTPUT: "output of a tool Aura ran",
    ProvenanceClass.EXTERNAL_DOCUMENT: "a document or repository Aura read",
    ProvenanceClass.WEB: "a web page",
}

#: At or above this, content came from OUTSIDE this machine and was written by
#: someone who may know an agent reads it.
#:
#: Two judgement calls live here, and both are calls rather than oversights.
#:
#: TOOL_OUTPUT sits BELOW the floor. It was above it for one commit, and the
#: measured effect was that every tool-using turn lost desktop control — a
#: search followed by "open my notes" became a refusal. That is a control
#: nobody keeps, and a disabled control protects nothing. It is also
#: unnecessary: a tool that fetches from the network goes through
#: network_gateway, which marks WEB at the point the external text actually
#: enters. Marking the wrapper as well would disarm turns whose only tool call
#: read the local clock.
#:
#: OWNER_FILE sits below the floor for the same reason in a different shape:
#: the owner pointed at the file deliberately, and treating every file read as
#: untrusted disarms nearly every turn. The residual risk is real and named — a
#: README inside a cloned repository was authored by a stranger and is trusted
#: at this level today. Moving either line up is a policy change with a cost,
#: not a bug fix, and whoever makes it should see the cost first.
UNTRUSTED_FLOOR = ProvenanceClass.EXTERNAL_DOCUMENT


@dataclass
class TurnProvenance:
    """Everything the current turn ingested, and from where."""

    ingests: list[tuple[ProvenanceClass, str]] = field(default_factory=list)

    def record(self, origin: ProvenanceClass, detail: str = "") -> None:
        self.ingests.append((origin, str(detail)[:200]))

    @property
    def least_trusted(self) -> ProvenanceClass:
        if not self.ingests:
            return ProvenanceClass.RUNTIME
        return max(origin for origin, _ in self.ingests)

    @property
    def untrusted(self) -> bool:
        return self.least_trusted >= UNTRUSTED_FLOOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.content_provenance.v1",
            "least_trusted": self.least_trusted.name,
            "untrusted": self.untrusted,
            "ingests": [
                {"origin": origin.name, "detail": detail} for origin, detail in self.ingests
            ],
        }


#: Per-turn, and a ContextVar rather than a global because turns interleave.
#: A module-level variable here would let a background research turn's web
#: ingest downgrade a foreground turn that read nothing — which is both wrong
#: and the kind of wrong that only shows up under load.
_TURN: contextvars.ContextVar[TurnProvenance | None] = contextvars.ContextVar(
    "aura_turn_provenance", default=None
)


def current_provenance() -> TurnProvenance:
    """This turn's record, creating one if the turn did not open a scope.

    Returning an empty record rather than None means a caller that forgot to
    open a scope sees RUNTIME — the trusting answer. That is deliberate and
    it is the safe direction ONLY because `record_ingest` below creates the
    record on demand: an ingest can never be silently dropped for want of a
    scope.
    """
    existing = _TURN.get()
    if existing is None:
        existing = TurnProvenance()
        _TURN.set(existing)
    return existing


def record_ingest(origin: ProvenanceClass, detail: str = "") -> None:
    """Mark that content from ``origin`` entered this turn's context."""
    current_provenance().record(origin, detail)


def turn_scope() -> "_TurnScope":
    """Open a fresh provenance record for one turn."""
    return _TurnScope()


class _TurnScope:
    def __enter__(self) -> TurnProvenance:
        self._token = _TURN.set(TurnProvenance())
        return _TURN.get()  # type: ignore[return-value]

    def __exit__(self, *_exc: object) -> None:
        _TURN.reset(self._token)


def effective_input_trust(declared: Any) -> Any:
    """The input trust a handler ACTUALLY has right now.

    A surface may declare TRUSTED because its input is model-generated. That is
    true on average and false the moment the model has read a web page, because
    indirect prompt injection does not make untrusted text act — it makes
    untrusted text persuade something trusted to act.

    Returns the declared trust when this turn ingested nothing untrusted, and
    the untrusted level when it did. Never returns something MORE trusting than
    declared.
    """
    from core.security.rule_of_two import InputTrust

    provenance = current_provenance()
    if not provenance.untrusted:
        return declared
    return InputTrust.UNTRUSTED


def describe_untrusted_context() -> str:
    """One sentence naming what made this turn untrusted, for a refusal."""
    provenance = current_provenance()
    if not provenance.untrusted:
        return ""
    worst = provenance.least_trusted
    detail = next(
        (detail for origin, detail in reversed(provenance.ingests) if origin == worst),
        "",
    )
    described = MEANING.get(worst, worst.name)
    return f"this turn read content from {described}" + (f" ({detail})" if detail else "")


__all__ = [
    "MEANING",
    "ProvenanceClass",
    "TurnProvenance",
    "UNTRUSTED_FLOOR",
    "current_provenance",
    "describe_untrusted_context",
    "effective_input_trust",
    "record_ingest",
    "turn_scope",
]
