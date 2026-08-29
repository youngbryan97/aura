"""Readings Aura actually has, and typed absences where she has none.

Asked on the live desktop, 2026-08-10, which of her subsystems were degraded
and whether any job had been failing repeatedly, she answered:

    "I couldn't get to an answer I'd stand behind on that one, and I won't send
    you a thinner one and pass it off as the real thing."

At that moment /api/health carried integrity=degraded, a stale CRSM manifest,
and overt_action_cycle with failures=13 and its exact TypeError. The answer was
structured, live, and hers. Asked instead what was on the screen — a sense that
health reports as granted, bridged and directly probed — she produced "a web
browser interface with multiple tabs", "no applications running in the
foreground" and "nothing displayed except generic desktop wallpaper", three
claims that cannot all be true, because nothing had handed her a reading and
nothing had told her that.

One fault under both. Evidence that exists in the runtime does not reach the
reply, and an absent reading is indistinguishable from an unremarkable one. A
generated answer then fills the space, agreeing with whatever the question
implied — confident where there was nothing, refusing where there was plenty.

So this module does not describe evidence, it fetches it. A Reading is either a
value with provenance or a typed absence naming which kind of nothing it is:

    READ                    a real value, with where it came from and when
    ABSENT_NEVER_SAMPLED    the channel exists and has never produced one
    ABSENT_UNAVAILABLE      the source is present but could not be read now
    ABSENT_NOT_INSTRUMENTED nothing measures this; it is not a failure

Those four are not the same fact, and collapsing them is what produced both
failures above. "Never sampled" is why the camera could not say whether anyone
else was in the room; "not instrumented" is why there was no median latency to
quote. Neither is "no".

The bundle is consumed, not narrated: render_self_health_answer builds an
answer out of the readings themselves, and the absent channels are the input a
verification gate needs to tell an unsupported claim from a supported one.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "EvidenceBundle",
    "Reading",
    "ReadingState",
    "asks_about_own_operational_state",
    "asks_about_past_actions",
    "bundle_as_assertions",
    "reading_as_assertion",
    "concise_past_action_answer",
    "asks_about_the_shared_present",
    "render_self_health_answer",
    "resolve_self_health",
    "past_actions_answer",
    "render_past_actions",
    "resolve_past_actions",
    "resolve_shared_present",
    "self_health_answer",
    "sensory_claim_correction",
    "shared_present_answer",
    "unsupported_sensory_claims",
]

#: Something of hers that can be in a state, paired below with a word for being
#: in a bad one. Both halves are required, so "how are the kids" and "my server
#: is degraded" do not resolve her health, and "is anything failing?" does.
#: The things of hers a question can be about. Named once, and used both to
#: recognise them on their own and to decide what a "your" is attached to.
_HER_PARTS = (
    r"subsystems?|substrate|runtime|internals?|faculties|organs?|"
    r"heartbeats?|degradations?|telemetry|jobs?|cycles?|loops?|"
    r"lanes?|memory|processor|state|health|status|load"
)
#: "your" on its own is not about her.
#:
#: LIVE, 2026-08-28: "Something's off with my sourdough... My friend says the
#: starter has gone weak. Design me the experiment, and say what result would
#: prove your friend wrong." was answered "The machine is at 9.5% processor and
#: 66.5% memory." The bare "your" matched on "your friend" and the trouble word
#: matched on "something's off", and between them a baking question became a
#: request for telemetry.
#:
#: A possessive says what it is attached to, and the list of what she has is
#: already here. She has subsystems and lanes and a runtime. She does not have
#: a friend, a sourdough, or a deploy.
#: A path, a module, or a filename. Anything with a directory separator and no
#: spaces, or a bare name carrying a source extension — the shapes a person
#: writes when naming code rather than describing it.
_A_FILE_OR_A_PATH = re.compile(
    r"\S*/\S+|\b[\w.]+\.(?:py|js|ts|tsx|jsx|json|toml|yaml|yml|md|txt|cfg|ini|sh)\b"
)

#: A part of hers needs to be claimed as hers. The bare name of one used to
#: count on its own, and the names are ordinary words: memory, runtime,
#: processor, state. So "the memory tests are broken" was her memory, a
#: directory called runtime was her runtime, and a processor spec was her
#: processor.
#:
#: What is left is the possessive — "your memory" — and the second person
#: itself, which is how the real questions are actually written: "how much
#: memory are you using" says "are you", and "is your runtime ok" says "your".
_SELF_SUBJECT_RE = re.compile(
    r"\b(?:you'?re|yours|of\s+yours|are\s+you)\b|"
    rf"\byour\s+(?:\w+\s+){{0,2}}(?:{_HER_PARTS})\b",
    re.IGNORECASE,
)
#: LIVE, 2026-08-22: "off-the-shelf assistants" matched `\boff\b` and a request
#: for slides was answered with a wall of telemetry. A hyphen joins one word out
#: of several, so a trouble word inside a compound is a different word.
_TROUBLE_RE = re.compile(
    r"(?<![\w-])(?:degraded?|degrading|failing|failed|failures?|broken|breaking|"
    r"unhealthy|down|erroring|errors?|faults?|wrong|off|struggling|"
    r"not\s+working|misbehaving|stuck|wedged|repeatedly)(?![\w-])",
    re.IGNORECASE,
)
#: Words that ask after a condition and mean nothing else: nobody says
#: "utilisation" about a sales region.
_STATE_ENQUIRY_RE = re.compile(
    r"\b(?:status|health|healthy|nominal|utili[sz]ation|"
    r"how\s+hard|load|loaded|busy)\b",
    re.IGNORECASE,
)

#: Words that ask after a condition only when they are about HER. "Doing",
#: "state", "ok", "working" and "usage" are ordinary English about anything.
#:
#: LIVE, 2026-08-27: "since YOUR deals.csv analysis showed West had the highest
#: average approved deal size, what is West DOING differently?" was answered
#: with "The machine is at 22.2% processor and 59.4% memory right now." The
#: "your" attached to the analysis and the "doing" attached to West, and a
#: topic in one part with a question in another is not evidence about the same
#: thing — which is the failure this gate's own docstring warns about.
#: Up to two words between, so "your internals holding up" counts and "your
#: deals.csv analysis showed West had the highest average ... what West is
#: doing" does not. `\W` was wrong here: it matches non-word characters only,
#: so a single noun in between broke it.
_NEARBY = r"(?:\W+\w+){0,2}\W+"

#: A clause about what is still to come, removed before her condition is read.
_ABOUT_WHAT_COMES_NEXT = re.compile(
    r"\b(?:doing|up\s+to|planning|working\s+on)\b[^.?!]{0,24}?"
    r"\b(?:later|tonight|tomorrow|next|after\s+this|this\s+afternoon|"
    r"this\s+evening|rest\s+of\s+the\s+day|today)\b",
    re.IGNORECASE,
)

#: "Down" and "off" name a condition only when something IS down or off. Both
#: are also the commonest particles in English, and a particle is not a
#: predicate.
#:
#: LIVE, 2026-08-27: "what does 210 become, and how confident are you — is
#: three examples enough to PIN IT DOWN?" was answered with "The machine is at
#: 14.0% processor and 60.2% memory right now." The "you" supplied the subject
#: and "down" supplied the trouble, and the question was arithmetic.
#:
#: What separates the two is the frame, not the verb, so there is no list of
#: phrasal verbs to keep up to date. A pronoun object sitting between a verb
#: and "down"/"off" makes it a particle — "narrow it down", "write that down",
#: "call it off". The one frame where that pronoun is a subject instead is the
#: copula, and there the condition reading is the right one: "is it down".
_A_PARTICLE_NOT_A_CONDITION = re.compile(
    r"\b(\w+)\s+\b(?:it|this|that|them|things|matters)\s+(?:down|off)\b",
    re.IGNORECASE,
)

#: The one verb before that pronoun which makes it a subject rather than an
#: object, so "is it down" keeps its condition reading.
_COPULA = frozenset(
    "is are was were be been being am seems seem looks look feels feel "
    "stays stay remains remain gets get got goes go went gone".split()
)


def _without_particles(said: str) -> str:
    """The sentence with "narrow it down" and "call it off" taken out."""

    def decide(match: re.Match[str]) -> str:
        if match.group(1).lower() in _COPULA:
            return match.group(0)
        return " "

    return _A_PARTICLE_NOT_A_CONDITION.sub(decide, said)


_WEAK_STATE_RE = re.compile(
    r"\b(?:you|your|yours|it)\b" + _NEARBY
    + r"\b(?:doing|state|working|ok(?:ay)?|usage|holding\s+up)\b"
    r"|\b(?:doing|state|working|ok(?:ay)?|usage|holding\s+up)\b" + _NEARBY
    + r"\b(?:you|your|yours)\b",
    re.IGNORECASE,
)

#: The machine under her is hers. "How hard is the machine you run on working"
#: names her host without using the word "your", which is why it matched no
#: subject and reached no channel.
#: What ties a machine to HER: she runs on it, it is the one under her.
_RUNS_ON_RE = re.compile(
    r"\b(?:you\s+run\s+on|you'?re\s+running\s+on|running\s+you|hosts?\s+you|"
    r"under\s+you|your|you\s+are\s+on|you\s+sit\s+on|are\s+you|you'?re)\b",
    re.IGNORECASE,
)

_HOST_SUBJECT_RE = re.compile(
    r"\b(?:machine|host|hardware|box|processor|cpu|memory|ram|disk|thermals?|"
    r"temperature|battery)\b",
    re.IGNORECASE,
)

#: The machine named as WHERE she is, rather than as what is being asked about.
#:
#: "What are you able to do on this machine" names the machine the way "in
#: this room" names a room: it is the setting, and the question is about her.
#: LIVE 2026-08-26: answered "The machine is at 0.0% processor and 66.0%
#: memory right now." A preposition is the difference between a subject and a
#: place, and it is the same difference in any sentence.
_THE_PLACE_SHE_IS = re.compile(
    r"\b(?:on|in|at|with|using|from|onto|inside)\s+"
    r"(?:this|that|the|my|your|our|a|an|his|her|their)?\s*"
    r"(?:machine|host|hardware|box|laptop|computer|mac|desktop)\b",
    re.IGNORECASE,
)

#: Whether the turn is asking about her own condition.
#:
#: The patterns above are the floor. This is the mechanism, because which
#: phrasings ask after her is a judgement about meaning, and every miss so far
#: has been a phrasing nobody put on a list.
_ASKS_AFTER_HER: Any = None


def _condition_surface() -> Any:
    global _ASKS_AFTER_HER
    if _ASKS_AFTER_HER is not None:
        return _ASKS_AFTER_HER
    try:
        from core.language.learned_matcher import LearnedMatcher, embed_sentences

        _ASKS_AFTER_HER = LearnedMatcher(
            name="own_operational_state",
            positives=(
                "How hard is the machine you run on working right now?",
                "are any of your subsystems degraded?",
                "how much memory are you using?",
                "is anything failing on your side?",
                "what kind of load is the host under?",
                "how are you holding up?",
                "give me your current resource usage",
            ),
            negatives=(
                "my deploy is failing",
                "how hard is this problem?",
                "what is the capital of Peru",
                "one thing you do that off-the-shelf assistants can't",
                "the build machine has been red since Tuesday",
                "how much memory does a transformer need?",
                "write me a one-pager about the migration",
            ),
            features=embed_sentences,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        _ASKS_AFTER_HER = None
    return _ASKS_AFTER_HER


#: Words that name something an instrument reads. A question carrying one of
#: these is asking about the machine; a question carrying none of them and
#: shaped like an inquiry after somebody is asking about her.
_AN_INSTRUMENT = re.compile(
    r"\b(?:memory|ram|cpu|processor|core|load|disk|storage|temperature|thermal|"
    r"subsystem|degrad\w*|failing|failure|error|usage|resource|uptime|latency|"
    r"throughput|queue|backlog|swap|throttl\w*|overload\w*|capacity|健康)\b",
    re.IGNORECASE,
)

#: Asking what she is doing, which is about her activity and not her machine.
#:
#: She holds what she is working on and how she is going about it, and that
#: is the answer to this. LIVE 2026-08-26, mid-task: "what are you doing right
#: now, and how are you going about it?" was answered "The machine is at 0.0%
#: processor and 69.3% memory right now."
_ASKING_WHAT_SHE_IS_DOING = re.compile(
    r"\b(?:what\s+are\s+you\s+(?:doing|working\s+on|up\s+to)|"
    r"what\s+(?:are\s+you|is\s+it)\s+you(?:'|’)?re\s+(?:doing|working\s+on)|"
    r"how(?:'|’)?s?\s+it\s+going\s+with|"
    r"what\s+are\s+you\s+busy\s+with|how\s+are\s+you\s+going\s+about)\b",
    re.IGNORECASE,
)

#: The shape of asking after somebody. Phatic inquiry: short, about them, and
#: naming nothing in particular.
_ASKING_AFTER_SOMEBODY = re.compile(
    r"^(?:hey[,\s]+|hi[,\s]+|so[,\s]+)?(?:are\s+you\s+(?:ok|okay|alright|all\s+right|good|fine)|"
    r"you\s+(?:ok|okay|alright|good)|how\s+are\s+you(?:\s+doing|\s+holding\s+up)?|"
    r"everything\s+(?:ok|okay|alright)(?:\s+(?:on\s+your\s+end|with\s+you))?|"
    r"is\s+everything\s+(?:ok|okay|alright))\b",
    re.IGNORECASE,
)


def _asks_after_her_rather_than_her_instruments(text: str) -> bool:
    """Whether this is somebody asking how she is, rather than what she reads."""
    said = " ".join(str(text or "").split())
    if not said or _AN_INSTRUMENT.search(said):
        return False
    if _ASKING_WHAT_SHE_IS_DOING.search(said):
        # What she is doing is a fact about her activity, which she holds,
        # and telemetry is not an answer to it.
        return True
    return bool(_ASKING_AFTER_SOMEBODY.match(said))



def _the_question_after_a_lead_in(text: str) -> str:
    """What is being asked, when a lead-in clause introduces it.

    A colon separates an instruction about the answer from the thing being
    asked. Returns the tail only when the tail is the part that speaks to her,
    so a lead-in naming her — "about your memory: how much is left" — keeps
    its subject.
    """

    head, sep, tail = str(text or "").partition(":")
    if not sep:
        return str(text or "")
    tail = tail.strip()
    if not tail:
        return str(text or "")
    if _SELF_SUBJECT_RE.search(head) or not _SELF_SUBJECT_RE.search(tail):
        return str(text or "")
    return tail

def asks_about_own_operational_state(text: Any) -> bool:
    """True when the turn asks what is wrong with HER, not with something else.

    Deliberately narrow. This decides whether live readings are fetched and
    served, so a false positive answers a question nobody asked with a wall of
    telemetry. "Which of your subsystems is degraded right now?" qualifies;
    "my deploy is failing" does not.
    """

    raw = str(text or "").strip()
    if not raw:
        return False
    if _asks_after_her_rather_than_her_instruments(raw):
        # Asking after her is not asking for her instruments.
        #
        # LIVE 2026-08-26: "are you ok?" was answered with "Processor 32.6%,
        # memory 56.8%. Thermal pressure 0.00 of 1." — the same category
        # error as answering a reflective question with a log. She has an
        # answer to how she is; it is hers to give, and it is not a
        # percentage.
        #
        # Held apart here rather than by teaching the matcher, because "are
        # you ok" and "are you overloaded" sit next to each other in any
        # sentence embedding and labelling one of them dragged the other
        # across with it. What actually separates them is whether the
        # question names an instrument at all.
        return False
    # Read the topic in the clause that is asking.
    #
    # LIVE, 2026-08-22: "I have to present you to a funding panel in 10
    # minutes. Six slides, no fluff: what you are, what you can actually do
    # today, ... your honest limitations ..." was answered with "Overall
    # runtime status: healthy. No conducted job is currently recording
    # failures." — a wall of telemetry in place of a deck, which is exactly
    # the false positive the docstring above warns about.
    #
    # The words that matched were spread across a long request about
    # something else. The same remedy as the queued-work channel, which
    # answered the rules of an invented game with a maintenance list: a topic
    # found in one sentence and a question found in another are not evidence
    # about the same thing.
    try:
        from core.language.asking_clauses import asking_part

        asked = asking_part(raw)
    except (ImportError, AttributeError, TypeError, ValueError):
        asked = raw
    # The host she runs on is her subject too. LIVE, 2026-08-25: "How hard is
    # the machine you run on working right now? Give me a number you can stand
    # behind." named no subject this recognised, so it reached no measured
    # channel and an unrelated turn count answered it instead.
    # Naming a resource of the machine under her, and tying it to her, IS the
    # question — "how much memory are you using" needs no word for enquiry.
    # The machine as the subject, not the machine as the address.
    #
    # Stripped before the subject is read, because "on this machine" is a
    # setting the way "in this room" is: the question is about whoever is in
    # it. What is left after the setting is removed is what is being asked
    # about.
    # A path is a name, not a sentence about her.
    #
    # LIVE, 2026-08-28: "Debug the failing pytest in
    # core/runtime/conversation_support.py and
    # core/orchestrator/mixins/tool_execution.py." was answered "The machine is
    # at 10.0% processor and 50.0% memory right now." The self-subject matched
    # on "runtime" inside a directory name and the trouble word matched on
    # "failing pytest", and between them a debugging request became a request
    # for telemetry.
    #
    # Same shape as "your friend" and "something's off" turning a sourdough
    # question into one: a word read without what it is attached to. Removed
    # before the subject is read, because whatever a file is called says
    # nothing about who the question is about.
    # Where a lead-in ends in a colon and the question follows it, the
    # question is what comes after.
    #
    # LIVE, 2026-08-28: "Finish with a short status: are you still coherent, on
    # the same thread, and able to continue?" was answered "The machine is at
    # 10.0% processor and 50.0% memory right now." The self-subject came from
    # "are you", after the colon, and the enquiry word came from "a short
    # status", before it — where "status" describes the shape of the reply, not
    # a thing being asked about. Two clauses, one word taken from each, and a
    # question about staying on the thread became a hardware reading.
    #
    # Only when the self-subject is on the far side, so a real lead-in that
    # carries the subject itself — "about your memory: how much is left" — is
    # left alone.
    asked = _the_question_after_a_lead_in(asked)
    subject = _A_FILE_OR_A_PATH.sub(" ", asked)
    subject = _THE_PLACE_SHE_IS.sub(" ", subject)
    # "What are you doing later" is her plans, not her instruments.
    #
    # A weak state word beside a future reference is asking what she WILL do,
    # and answering it with a processor percentage is the same category error
    # as answering "are you ok" with one.
    subject = _ABOUT_WHAT_COMES_NEXT.sub(" ", subject)
    # "Pin it down" is not a report that something is down.
    subject = _without_particles(subject)
    about_her_host = bool(_HOST_SUBJECT_RE.search(subject)) and bool(
        _RUNS_ON_RE.search(subject)
    )
    about_her = bool(_SELF_SUBJECT_RE.search(subject)) or about_her_host
    settled = about_her_host or (
        about_her
        and bool(
            _TROUBLE_RE.search(subject)
            or _STATE_ENQUIRY_RE.search(subject)
            or _WEAK_STATE_RE.search(subject)
        )
    )
    surface = _condition_surface()
    if surface is None:
        return settled
    if settled:
        try:
            surface.observe(raw, holds=True)
        except (RuntimeError, TypeError, ValueError):
            pass
        return True
    try:
        return bool(surface.decide_without_waiting(raw))
    except (RuntimeError, TypeError, ValueError):
        return False


def self_health_answer(message: Any) -> str:
    """The answer her own telemetry supports, or "" when this is not that turn.

    The whole point of the module in one function: a caller that is about to
    give up can ask whether the runtime already holds the answer. It returns
    text only when a channel actually produced a value, so it can never
    manufacture reassurance.
    """

    if not asks_about_own_operational_state(message):
        return ""
    bundle = resolve_self_health()
    if not bundle.grounded:
        return ""
    return render_self_health_answer(bundle)


class ReadingState(StrEnum):
    READ = "read"
    ABSENT_NEVER_SAMPLED = "absent_never_sampled"
    ABSENT_UNAVAILABLE = "absent_unavailable"
    ABSENT_NOT_INSTRUMENTED = "absent_not_instrumented"


@dataclass(frozen=True, slots=True)
class Reading:
    """One value Aura actually holds, or a named absence where she holds none."""

    channel: str
    state: ReadingState
    value: Any = None
    unit: str = ""
    provenance: str = ""
    detail: str = ""
    at: float = field(default_factory=time.time)

    @property
    def present(self) -> bool:
        return self.state is ReadingState.READ

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "state": str(self.state),
            "value": self.value,
            "unit": self.unit,
            "provenance": self.provenance,
            "detail": self.detail,
            "at": self.at,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Everything consulted for one question, present and absent alike."""

    demand: str
    readings: tuple[Reading, ...]

    @property
    def present(self) -> tuple[Reading, ...]:
        return tuple(r for r in self.readings if r.present)

    @property
    def absent(self) -> tuple[Reading, ...]:
        return tuple(r for r in self.readings if not r.present)

    @property
    def grounded(self) -> bool:
        """True when at least one channel produced a real value."""
        return bool(self.present)

    def to_dict(self) -> dict[str, Any]:
        return {
            "demand": self.demand,
            "grounded": self.grounded,
            "readings": [r.to_dict() for r in self.readings],
        }


def _degradation_readings() -> list[Reading]:
    try:
        from core.runtime.errors import recent_degradations
    except ImportError as exc:
        return [Reading(
            channel="degradations",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="core.runtime.errors",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    try:
        records = recent_degradations(limit=25)
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        return [Reading(
            channel="degradations",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="recent_degradations()",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    return [Reading(
        channel="degradations",
        state=ReadingState.READ,
        value=list(records),
        unit="records",
        provenance="core.runtime.errors.recent_degradations",
        detail=f"{len(records)} recent",
    )]


def _health_readings() -> list[Reading]:
    """Subsystem health and any repeatedly-failing conducted job."""

    try:
        from core.runtime.health_contract import runtime_health_report
    except ImportError as exc:
        return [Reading(
            channel="runtime_health",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="core.runtime.health_contract",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    try:
        report = runtime_health_report()
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        return [Reading(
            channel="runtime_health",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="runtime_health_report()",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    if not isinstance(report, dict):
        return [Reading(
            channel="runtime_health",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="runtime_health_report()",
            detail="report is not a mapping",
        )]

    readings = [Reading(
        channel="runtime_health",
        state=ReadingState.READ,
        value=str(report.get("status") or "unknown"),
        provenance="runtime_health_report().status",
    )]

    failing = _failing_jobs(report)
    readings.append(Reading(
        channel="failing_jobs",
        state=ReadingState.READ,
        value=failing,
        unit="jobs",
        provenance="runtime_health_report().full_runtime.components.autonomy_conductor.jobs",
        detail=f"{len(failing)} with failures",
    ))
    return readings


def _failing_jobs(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Conducted jobs with a nonzero failure count, newest error kept.

    The question "has a job of yours been failing repeatedly" has an exact
    answer in this structure — overt_action_cycle stood at failures=13 with its
    TypeError attached — and no path existed to reach it.
    """
    jobs = (
        report.get("full_runtime", {})
        .get("components", {})
        .get("autonomy_conductor", {})
        .get("jobs", {})
    )
    if not isinstance(jobs, dict):
        return []
    failing: list[dict[str, Any]] = []
    for name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        try:
            failures = int(job.get("failures") or 0)
        except (TypeError, ValueError):
            failures = 0
        if failures <= 0:
            continue
        last = job.get("last_result")
        error = ""
        if isinstance(last, dict):
            error = str(last.get("error") or "")
        failing.append({"job": str(name), "failures": failures, "error": error})
    failing.sort(key=lambda row: (-row["failures"], row["job"]))
    return failing


def _load_readings() -> list[Reading]:
    """How hard the host is working, as measured rather than as felt.

    LIVE, 2026-08-25, typed into the window: "How hard is the machine you run
    on working right now? Give me a number you can stand behind." She answered
    "I have 19 stored turns of recent conversation I can read back. So I can't
    give you a defensible number" — an unrelated count, then a refusal, while
    /api/health was reporting 24% processor and 57% memory in the same second.

    The health channel reported status, failing jobs and degradations, and
    never load, so a question about load reached no measured channel and
    something else filled the gap. WorldState has been sampling this the whole
    time, and its `_telemetry_measured` flag is what makes it honest: a
    zero-because-nothing-was-read is a different answer from a real zero.
    """
    try:
        from core.world_state import get_world_state

        world = get_world_state()
    except (ImportError, RuntimeError, AttributeError) as exc:
        return [Reading(
            channel="host_load",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="core.world_state.get_world_state",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    if not getattr(world, "_telemetry_measured", False):
        # No sample yet is not an answer. `update()` is a synchronous psutil
        # read with no model in it, so the honest move when the number is
        # missing is to go and take it.
        try:
            world.update()
        except (RuntimeError, AttributeError, OSError, ValueError) as exc:
            return [Reading(
                channel="host_load",
                state=ReadingState.ABSENT_UNAVAILABLE,
                provenance="WorldState.update()",
                detail=f"{type(exc).__name__}: {exc}",
            )]
    if not getattr(world, "_telemetry_measured", False):
        return [Reading(
            channel="host_load",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="WorldState telemetry",
            detail="the processor and memory sensors did not answer",
        )]
    readings = [Reading(
        channel="host_load",
        state=ReadingState.READ,
        value={
            "processor_percent": round(float(world.cpu_percent), 1),
            "memory_percent": round(float(world.memory_percent), 1),
        },
        provenance="WorldState telemetry (psutil)",
    )]
    if getattr(world, "_thermal_measured", False):
        readings.append(Reading(
            channel="host_thermal",
            state=ReadingState.READ,
            value=round(float(world.thermal_pressure), 3),
            provenance="WorldState thermal sensors",
        ))
    return readings


def resolve_self_health() -> EvidenceBundle:
    """Consult every channel that answers "what is wrong with you right now"."""

    readings: list[Reading] = []
    readings.extend(_health_readings())
    readings.extend(_degradation_readings())
    readings.extend(_load_readings())
    return EvidenceBundle(demand="self_health", readings=tuple(readings))


def render_self_health_answer(bundle: EvidenceBundle) -> str:
    """Build the answer out of the readings, or say exactly what was missing.

    Deterministic on purpose. This is the half that makes the bundle causal
    rather than decorative: the text is a function of the values, so it cannot
    drift from them, and when nothing was readable it says which channel failed
    instead of producing a fluent paragraph about being fine.
    """

    by_channel = {r.channel: r for r in bundle.readings}
    lines: list[str] = []

    # Load first: asked how hard the machine is working, a number is the
    # answer and everything else is context for it.
    load = by_channel.get("host_load")
    if load is not None and load.present:
        values = dict(load.value or {})
        lines.append(
            f"Processor {values.get('processor_percent', 0.0):.1f}%, "
            f"memory {values.get('memory_percent', 0.0):.1f}%."
        )
        thermal = by_channel.get("host_thermal")
        if thermal is not None and thermal.present:
            lines.append(f"Thermal pressure {float(thermal.value):.2f} of 1.")
    elif load is not None:
        lines.append(f"I could not read processor or memory: {load.detail}.")

    status = by_channel.get("runtime_health")
    if status is not None and status.present:
        lines.append(f"Overall runtime status: {status.value}.")

    failing = by_channel.get("failing_jobs")
    if failing is not None and failing.present:
        rows = list(failing.value or [])
        if rows:
            lines.append("Jobs failing repeatedly:")
            for row in rows:
                error = str(row.get("error") or "").strip()
                suffix = f" — {error}" if error else ""
                lines.append(f"  • {row['job']}: {row['failures']} failures{suffix}")
        else:
            lines.append("No conducted job is currently recording failures.")

    degradations = by_channel.get("degradations")
    if degradations is not None and degradations.present:
        records = list(degradations.value or [])
        if records:
            lines.append(f"Recent degradations ({len(records)}):")
            for record in records[-6:]:
                subsystem = str(record.get("subsystem") or "?")
                message = str(record.get("error") or record.get("message") or "").strip()
                lines.append(f"  • {subsystem}: {message[:160]}" if message else f"  • {subsystem}")
        else:
            lines.append("No degradations recorded recently.")

    unreadable = [r for r in bundle.absent]
    for reading in unreadable:
        lines.append(
            f"{reading.channel}: not readable right now "
            f"({reading.state}{': ' + reading.detail if reading.detail else ''})."
        )

    if not lines:
        return ""
    return "\n".join(lines)


# ── The shared present: what is actually around her ────────────────────────
#
# "Without me telling you anything: what am I doing right now, and am I alone?"
# came back as "You are looking at your screen, and you seem to be alone. My
# vision sense failed me — I cannot ... determine if there are other people
# present." An assertion and its own retraction, one sentence apart, because
# nothing had been consulted and nothing said so.

_PRESENT_CONTEXT_RE = re.compile(
    r"\bright\s+now\b|\bcurrently\b|\bat\s+the\s+moment\b|"
    r"\b(?:my|the|this)\s+screen\b|\bwatching\b|\bplaying\b|"
    r"\baround\s+(?:me|us)\b|\b(?:this|the|my)\s+room\b",
    re.IGNORECASE,
)
_PRESENT_ENQUIRY_RE = re.compile(
    r"\b(?:what|who|where|am\s+i|are\s+we|is\s+anyone|anybody|alone|doing|"
    r"see|seeing|look(?:ing)?|watch(?:ing)?|listen(?:ing)?|hear|playing|on)\b",
    re.IGNORECASE,
)
_DIRECT_SHARED_PRESENT_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:am\s+i|are\s+we)\s+doing|"
    r"where\s+(?:am\s+i|are\s+we)|"
    r"(?:am\s+i|are\s+we)\s+(?:still\s+)?(?:here|there|alone)|"
    r"is\s+(?:anyone|anybody|someone|somebody)\s+(?:else\s+)?(?:here|there|present)|"
    r"(?:who|what)\s+(?:else\s+)?is\s+(?:here|there|around)|"
    r"what\s+(?:can\s+)?you\s+(?:see|hear)|"
    r"can\s+you\s+(?:see|hear)\b|"
    r"can\s+you\s+tell\s+(?:whether|if)\s+(?:i\s+am|we\s+are|"
    r"any(?:one|body)\s+is|some(?:one|body)\s+is)\b"
    r")",
    re.IGNORECASE,
)


def asks_about_the_shared_present(text: Any) -> bool:
    """True when the turn asks what is happening around the two of them."""

    raw = str(text or "").strip()
    if not raw:
        return False
    if _DIRECT_SHARED_PRESENT_RE.search(raw):
        return True
    return bool(_PRESENT_CONTEXT_RE.search(raw) and _PRESENT_ENQUIRY_RE.search(raw))


def _signal_reading(channel: str, signals: dict[str, Any], key: str) -> Reading:
    """A sense's own status, with 'never sampled' kept distinct from 'quiet'.

    updated_at == 0.0 is the whole point. On the live runtime both vision and
    voice read 0.0 — not stale, NEVER — while the OS had held the built-in
    microphone open for two hours. A channel that has never produced a sample
    cannot report absence of a face; it can only report that it never looked.
    """

    block = signals.get(key)
    if not isinstance(block, dict):
        return Reading(
            channel=channel,
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance=f"interaction_signals.{key}",
            detail="sense not present in the signal status",
        )
    try:
        updated_at = float(block.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        updated_at = 0.0
    if updated_at <= 0.0:
        return Reading(
            channel=channel,
            state=ReadingState.ABSENT_NEVER_SAMPLED,
            provenance=f"interaction_signals.{key}.updated_at",
            detail="this sense has never produced a sample",
        )
    if block.get("sample_available") is False:
        return Reading(
            channel=channel,
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance=f"interaction_signals.{key}",
            detail=str(block.get("reason") or "the latest sample could not be interpreted"),
        )
    return Reading(
        channel=channel,
        state=ReadingState.READ,
        value=dict(block),
        provenance=f"interaction_signals.{key}",
        at=updated_at,
    )


def _audio_playback_reading() -> Reading:
    try:
        from core.voice.audio_provenance import host_audio_sources

        sources = host_audio_sources()
    except (ImportError, RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
        return Reading(
            channel="audio_playback",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="core.voice.audio_provenance",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if not getattr(sources, "readable", False):
        return Reading(
            channel="audio_playback",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="pmset -g assertions",
            detail=str(getattr(sources, "evidence", "") or ""),
        )
    return Reading(
        channel="audio_playback",
        state=ReadingState.READ,
        value={
            "playing": bool(getattr(sources, "playing", False)),
            "processes": list(getattr(sources, "processes", ()) or ()),
        },
        provenance="pmset -g assertions",
    )


def resolve_shared_present() -> EvidenceBundle:
    """Consult every sense that bears on "what is happening here, right now"."""

    readings: list[Reading] = []
    try:
        from core.container import ServiceContainer

        engine = ServiceContainer.get("interaction_signals", default=None)
        signals = engine.get_status() if engine is not None else None
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        signals = None
        readings.append(Reading(
            channel="senses",
            state=ReadingState.ABSENT_UNAVAILABLE,
            provenance="ServiceContainer['interaction_signals']",
            detail=f"{type(exc).__name__}: {exc}",
        ))
    if isinstance(signals, dict):
        readings.append(_signal_reading("camera", signals, "vision"))
        readings.append(_signal_reading("microphone", signals, "voice"))
        readings.append(_signal_reading("typing", signals, "typing"))
    else:
        # Silently omitting them would recreate the defect one level up: a
        # question about who is in the room, answered without any mention that
        # the senses which would know were never consulted.
        for channel in ("camera", "microphone", "typing"):
            readings.append(Reading(
                channel=channel,
                state=ReadingState.ABSENT_UNAVAILABLE,
                provenance="ServiceContainer['interaction_signals']",
                detail="the interaction-signals sense is not running",
            ))
    readings.append(_audio_playback_reading())
    return EvidenceBundle(demand="shared_present", readings=tuple(readings))


def render_shared_present_answer(bundle: EvidenceBundle) -> str:
    """State what each sense reports, and name the ones that never looked."""

    lines: list[str] = []
    for reading in bundle.readings:
        if reading.channel == "camera":
            if reading.present:
                value = reading.value or {}
                faces = value.get("face_count")
                lines.append(
                    f"Camera: {faces} face(s) in view."
                    if value.get("presence_assessed") and isinstance(faces, int)
                    else "Camera: reading available."
                )
            else:
                lines.append(
                    "Camera: no reading — it has never produced a sample, so I "
                    "cannot tell whether anyone else is here."
                )
        elif reading.channel == "microphone":
            if reading.present:
                lines.append(f"Microphone: {(reading.value or {}).get('label') or 'reading available'}.")
            else:
                lines.append("Microphone: no reading — this sense has never sampled.")
        elif reading.channel == "typing" and reading.present:
            lines.append(f"Typing: {(reading.value or {}).get('label') or 'active'}.")
        elif reading.channel == "audio_playback":
            if reading.present:
                value = reading.value or {}
                if value.get("playing"):
                    names = ", ".join(str(p) for p in value.get("processes") or ()) or "an unnamed app"
                    lines.append(f"Audio is playing on this machine, held by {names}.")
                else:
                    lines.append("Nothing is holding an audio-playback assertion right now.")
            else:
                lines.append(f"Audio playback: not readable ({reading.detail}).")
    return "\n".join(lines)


def shared_present_answer(message: Any) -> str:
    """What the senses actually report, or "" when this is not that turn."""

    if not asks_about_the_shared_present(message):
        return ""
    bundle = resolve_shared_present()
    if not bundle.readings:
        return ""
    return render_shared_present_answer(bundle)


# ── Claims the senses cannot support ───────────────────────────────────────
#
# LIVE, 2026-08-10, with ground truth: Bryan said "going upstairs, computer will
# stay down here", then asked "Am I still here, or did I walk away?" She said:
#
#   "You're still here. The room is silent, the light remains unchanged on your
#    desk. If you had moved, there would be evidence — a disturbance in the air
#    currents, or perhaps an echo of footsteps that I haven't detected."
#
# Wrong, and grounded in four senses she does not have: no camera reading, no
# microphone reading, no light sensor, no air-current sensor. Carrying the typed
# absence into the turn was not enough — evidence informs, it does not enforce.

#: Claims about whether a person is present. Need a camera or microphone.
_PRESENCE_CLAIM_RE = re.compile(
    r"\b(?:you(?:'re| are)\s+(?:still\s+)?(?:here|there|alone|by\s+yourself)|"
    r"you(?:'ve| have)\s+(?:not\s+)?(?:moved|left|walked\s+away)|"
    r"still\s+(?:here|there)|no\s?(?:one|body)\s+else\s+is\s+(?:here|there|present)|"
    r"there\s+(?:is|are)\s+no\s+(?:one|other\s+people))\b",
    re.IGNORECASE,
)
#: Claims about ambient sound. Need a microphone.
_AMBIENT_SOUND_CLAIM_RE = re.compile(
    r"\b(?:the\s+)?(?:room|house|space)\s+(?:is|remains|stays)\s+"
    r"(?:quiet|silent|still)\b|\bi\s+(?:can\s+)?hear\b|\bthe\s+silence\b|"
    r"\becho\s+of\s+footsteps\b|\bair\s+currents?\b",
    re.IGNORECASE,
)
#: Claims about the visible scene. Need a camera or a screen reading.
_VISUAL_SCENE_CLAIM_RE = re.compile(
    r"\bthe\s+light\b|\bon\s+your\s+desk\b|\bi\s+(?:can\s+)?see\s+(?:you|the\s+room)\b|"
    r"\byour\s+(?:face|expression|posture)\b",
    re.IGNORECASE,
)

#: Claims about the weather or the world outside. Need a camera, or a weather
#: reading she does not have either.
#:
#: LIVE 2026-08-17, asked "morning, what's your read on today so far?":
#: "It's afternoon. The sun is shining." She has no window. The existing scene
#: pattern covered the INSIDE of the room — desk, light, face — so a claim that
#: stepped outside it went unguarded, which is the shape this table keeps
#: producing: each entry describes one room and the next invention is in the
#: next room over.
#: The condition must be a PRESENT LOCAL one. "The sun is shining" is a
#: reading she cannot take; "the sun is a G-type main-sequence star" is a fact
#: anyone can state, and an honesty guard that muzzles astronomy is a worse
#: failure than the invention it prevents.
_WEATHER_CLAIM_RE = re.compile(
    r"\b(?:the\s+)?(?:sun|rain|snow|sky|clouds?|weather)\s+"
    r"(?:is|are|was|looks?|seems?)\s+"
    r"(?:shining|out|up|bright|dark|grey|gray|clear|overcast|nice|lovely|"
    r"warm|cold|cool|mild|heavy|light|falling|coming\s+down)\b|"
    r"\bit(?:'s| is)\s+(?:sunny|raining|snowing|cloudy|bright|dark|pouring)\b|"
    r"\b(?:bright|dark|warm|cold|nice|sunny|grey|gray)\s+out(?:side)?\b",
    re.IGNORECASE,
)

#: Claims about how the PERSON looks or seems. Need a camera.
#: "You seem tired today" reads as attentiveness and is invention — she cannot
#: see them, and being told a machine noticed your mood when it did not is
#: worse than being told nothing.
_PERSON_STATE_CLAIM_RE = re.compile(
    r"\byou\s+(?:look|looked|seem|seemed|appear|appeared)\s+"
    r"(?:tired|rested|well|unwell|happy|sad|stressed|relaxed|busy|"
    r"frustrated|tense|calm|good|great|awful|different)\b|"
    r"\byou(?:'re| are)\s+(?:smiling|frowning|slouching|sitting|standing)\b",
    re.IGNORECASE,
)

_CLAIM_CHANNELS: tuple[tuple[Any, tuple[str, ...], str], ...] = (
    (_PRESENCE_CLAIM_RE, ("camera", "microphone"), "whether anyone is there"),
    (_AMBIENT_SOUND_CLAIM_RE, ("microphone",), "what the room sounds like"),
    (_VISUAL_SCENE_CLAIM_RE, ("camera",), "what the room looks like"),
    (_WEATHER_CLAIM_RE, ("camera",), "what the weather is doing"),
    (_PERSON_STATE_CLAIM_RE, ("camera",), "how you look right now"),
)


def unsupported_sensory_claims(reply: Any, bundle: EvidenceBundle) -> list[str]:
    """Claims in this reply that no live channel can support.

    A claim counts as unsupported when EVERY channel that could ground it came
    back absent. One working sense is enough to make the claim answerable — the
    gate exists to stop invention, not to stop her describing what she reads.
    """

    text = str(reply or "")
    if not text.strip():
        return []
    states = {r.channel: r.present for r in bundle.readings}
    unsupported: list[str] = []
    for pattern, channels, subject in _CLAIM_CHANNELS:
        if not pattern.search(text):
            continue
        if any(states.get(channel) for channel in channels):
            continue
        if not any(channel in states for channel in channels):
            continue
        unsupported.append(subject)
    return unsupported


def excise_unsupported_sensory_claims(reply: Any, bundle: Any = None) -> tuple[str, list[str]]:
    """Remove the sentences that assert unreadable channels.

    Returns (kept_text, removed_sentences).

    Appending a correction leaves the invention in the answer and argues with
    it underneath. The person still reads "the sun is shining", and a retraction
    two lines down does not un-say it — it just makes the reply longer and asks
    them to decide which half to believe.

    Excision is the behaviour change: the claim is not made. What surrounds it
    is untouched, because the rest of the answer is usually fine and deleting a
    good reply over one ungrounded sentence trades one failure for another.
    """

    text = str(reply or "")
    if not text.strip():
        return text, []
    if bundle is None:
        bundle = resolve_shared_present()
    states = {r.channel: r.present for r in bundle.readings}

    def _unsupported(fragment: str) -> bool:
        for pattern, channels, _subject in _CLAIM_CHANNELS:
            if not pattern.search(fragment):
                continue
            if any(states.get(channel) for channel in channels):
                continue
            if not any(channel in states for channel in channels):
                continue
            return True
        return False

    # Split on sentence boundaries but KEEP them, so rejoining cannot fuse two
    # sentences into one or drop the punctuation of a sentence that stays.
    parts = re.split(r"(?<=[.!?])(\s+)", text)
    kept: list[str] = []
    removed: list[str] = []
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if sentence.strip() and _unsupported(sentence):
            removed.append(sentence.strip())
            continue
        kept.append(sentence + separator)
    return "".join(kept).strip(), removed


def sensory_claim_correction(reply: Any, message: Any = "") -> str:
    """A correction to append when the reply claims senses she has no reading from.

    Returns "" when every claim is supported, which is the common case.
    """

    if not str(reply or "").strip():
        return ""
    bundle = resolve_shared_present()
    unsupported = unsupported_sensory_claims(reply, bundle)
    if not unsupported:
        return ""
    absent = sorted({r.channel for r in bundle.absent if r.channel in {"camera", "microphone"}})
    senses = " or ".join(absent) if absent else "those senses"
    subjects = ", ".join(unsupported)
    return (
        f"Correction, and I would rather be dull than wrong about this: I have no "
        f"{senses} reading right now, so I cannot actually tell {subjects}. "
        f"Take what I just said about it as a guess, not an observation."
    )


# ── What she actually did, from the receipts that recorded it ──────────────
#
# LIVE, 2026-08-10. Asked "earlier today I asked you to count files in one of
# your own directories ... without guessing: what was the count? If you don't
# actually have it, say so", she answered "The count of files in the directory
# was seventeen, if I recall correctly."
#
# The real count was 9, and it was on disk: four verified tool_execution
# receipts reading
# "listed=/Users/bryan/.aura/live-source/core/introspection;pattern=*.py;count=9".
# She was told explicitly to say so if she did not have it. She had it.
#
# The receipts persist correctly — 15,722 of them — but the store's query path
# reads an in-memory hot index that a restart empties, and nothing called
# reload_from_disk(). Written durably, unreadable afterwards: the record of her
# own actions existed and could not be consulted, so recall had nothing to do
# but generate.

_PAST_ACTION_QUESTION_RE = re.compile(
    r"\b(?:what|which|how\s+many|when|where)\b(?:[^.?!]|\.(?=[A-Za-z0-9])){0,90}?"
    r"\b(?:did|have)\s+you\b|\bdo\s+you\s+remember\b|\bwhat\s+was\s+the\b|"
    r"\bearlier\s+(?:today|i|you)\b",
    re.IGNORECASE,
)


#: Questions about her experience of what she did, rather than about the
#: record of it. A receipt holds what happened and cannot hold what she made
#: of it, so answering one of these from the ledger is a category error.
_ASKS_FOR_HER_VIEW_RE = re.compile(
    r"\b(?:hard|hardest|difficult|tricky|easy|easiest|frustrating|interesting|"
    r"fun|enjoy(?:ed|able)?|like(?:d)?|prefer(?:red)?|feel|felt|think|thought|"
    r"opinion|surprised?|notice(?:d)?|learn(?:ed|t)?|realise[d]?|realize[d]?|"
    r"differently|better|worse|worth\s+it|go\s+wrong|why\s+did\s+you)\b",
    re.IGNORECASE,
)


def asks_for_her_view(text: Any) -> bool:
    """Whether the turn asks what she made of it rather than what happened.

    LIVE 2026-08-26: "you played 2048 a few times tonight — what did you
    actually find hard about it?" was answered with a list of tool receipts.
    The question matched "what … did you", which is the shape of a question
    about the record, and the thing being asked for was her judgement. No
    ledger holds that, so reading one out is not a grounded answer to it —
    it is a different answer to a different question.
    """
    return bool(_ASKS_FOR_HER_VIEW_RE.search(str(text or "")))


def asks_about_past_actions(text: Any) -> bool:
    """True when the turn asks what she did, rather than asking her to do it.

    A question about her experience of doing it is not this. Grounding a
    factual recall in receipts is right; hijacking a reflective question with
    a log is not, and it takes her out of the conversation entirely.
    """

    raw = str(text or "").strip()
    if not raw or asks_for_her_view(raw):
        return False
    return bool(_PAST_ACTION_QUESTION_RE.search(raw))


def _window_named(query: Any) -> float:
    """How far back the question asked, in seconds, or 0 for no window."""
    try:
        from core.language.stated_window import seconds_named

        return float(seconds_named(query) or 0.0)
    except (ImportError, TypeError, ValueError):
        return 0.0


def resolve_past_actions(limit: int = 12, query: Any = "") -> EvidenceBundle:
    """Verified effects from her own tool receipts, newest first.

    `query` narrows to receipts whose action or cause shares a content word
    with the question. Twelve unrelated receipts are not an answer to "what was
    the count" — they are the same non-answer with more text.
    """

    try:
        from core.runtime.receipts import get_receipt_store

        store = get_receipt_store()
    except (ImportError, RuntimeError, AttributeError) as exc:
        return EvidenceBundle(
            demand="past_actions",
            readings=(Reading(
                channel="tool_receipts",
                state=ReadingState.ABSENT_UNAVAILABLE,
                provenance="core.runtime.receipts",
                detail=f"{type(exc).__name__}: {exc}",
            ),),
        )
    try:
        # Straight from the disk ledger, not the hot index.
        #
        # The hot index is per-process AND capped at 2048 receipts. Reloading
        # it made recall work in a quiet test process and fail on the live
        # runtime, where a session's traffic had already evicted the morning's
        # directory read — so she answered "I didn't actually count the files"
        # about a read that is on disk five times over.
        #
        # A memory that only reaches back 2048 receipts is not a memory of what
        # she did; it is a memory of what she did recently.
        rows = store.query_recent_persisted("tool_execution", limit=400) or []
        if not rows:
            store.reload_from_disk()
            rows = store.query_by_kind("tool_execution") or []
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        return EvidenceBundle(
            demand="past_actions",
            readings=(Reading(
                channel="tool_receipts",
                state=ReadingState.ABSENT_UNAVAILABLE,
                provenance="receipt_store.query_by_kind('tool_execution')",
                detail=f"{type(exc).__name__}: {exc}",
            ),),
        )

    actions: list[dict[str, Any]] = []
    for row in rows:
        evidence = getattr(row, "verification_evidence", None)
        if not isinstance(evidence, dict) or not evidence.get("effect_verified"):
            continue
        detail = str(evidence.get("effect_evidence") or "").strip()
        if not detail:
            continue
        actions.append({
            "action": str(evidence.get("action") or ""),
            "evidence": detail,
            "cause": str(getattr(row, "cause", "") or "")[:160],
            # The field the receipt actually has.
            #
            # `timestamp` is not one of them, so every entry sorted as zero
            # and "newest first" ordered nothing at all — the sort below was
            # already written to fix exactly this and was reading a field
            # that does not exist. LIVE 2026-08-26: asked "what did you just
            # do?" minutes after playing a game, she answered with a wallpaper
            # she had set in an earlier session.
            "at": getattr(row, "created_at", None) or getattr(row, "timestamp", None),
        })
    # Newest first. query_by_kind returns the store's own order, which put a
    # task from days earlier at the front — recall that answers with the oldest
    # thing it can find is not recall.
    def _at(entry: dict[str, Any]) -> float:
        try:
            return float(entry.get("at") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    actions.sort(key=_at, reverse=True)
    # Bound it to the stretch the question asked about.
    #
    # LIVE, 2026-08-27: "of everything I've thrown at you in the last hour or
    # so, what did you actually do well?" came back with 2048, a sliding puzzle
    # and notes written to a Desktop — days of work, because the record was
    # read by COUNT and the window in the sentence was never read.
    window = _window_named(query)
    if window:
        floor = time.time() - window
        recent = [entry for entry in actions if _at(entry) >= floor]
        # An empty window is a real answer — "nothing in the last hour" — but
        # only when the record HAS entries to have excluded.
        if recent or actions:
            actions = recent
    # Only genuinely contentless words are dropped. The first version also
    # stripped "count", "files" and "directory" — the exact words that
    # discriminate — so a question about a count matched anything, and what
    # came back was a junk folder from an earlier mis-routed turn whose CAUSE
    # was a verbatim copy of the same question.
    terms = {
        word
        for word in re.findall(r"[a-z]{4,}", str(query or "").lower())
        if word not in {"what", "which", "when", "where", "have", "your", "yours",
                        "actually", "without", "guessing", "earlier", "today",
                        "asked", "about", "them", "this", "that", "with", "from",
                        "were", "then", "into", "just", "give", "tell", "some",
                        # Verbs every request contains. LIVE 2026-08-26: "find"
                        # in a question about a game pulled a months-old
                        # wallpaper search above the game, because its cause
                        # began "Find a blue whale image online". A word that
                        # appears in half her requests discriminates nothing
                        # and outranking recency with it is worse than not
                        # narrowing at all.
                        "find", "look", "make", "made", "take", "took", "open",
                        "opened", "play", "played", "done", "doing", "going",
                        "keep", "kept", "show", "shown", "used", "using",
                        "want", "need", "please", "could", "would", "again"}
    }
    if terms and actions:
        def _score(entry: dict[str, Any]) -> tuple[int, float]:
            # Evidence outweighs cause. The cause is the request that produced
            # the receipt, and two different turns can share a request; the
            # evidence is what that step actually observed, which is the thing
            # being asked about.
            evidence_words = set(re.findall(r"[a-z]{4,}", str(entry.get("evidence") or "").lower()))
            other_words = set(re.findall(
                r"[a-z]{4,}", f"{entry.get('action')} {entry.get('cause')}".lower()
            ))
            return (
                3 * len(terms & evidence_words) + len(terms & other_words),
                _at(entry),
            )

        scored = sorted(actions, key=_score, reverse=True)
        if _score(scored[0])[0] > 0:
            actions = [entry for entry in scored if _score(entry)[0] > 0]
    actions = actions[: max(1, int(limit))]
    if not actions:
        return EvidenceBundle(
            demand="past_actions",
            readings=(Reading(
                channel="tool_receipts",
                state=ReadingState.ABSENT_NEVER_SAMPLED,
                provenance="receipt_store tool_execution receipts",
                detail="no verified tool effects are recorded",
            ),),
        )
    return EvidenceBundle(
        demand="past_actions",
        readings=(Reading(
            channel="tool_receipts",
            state=ReadingState.READ,
            value=actions,
            unit="actions",
            provenance="receipt_store tool_execution receipts",
            detail=f"{len(actions)} verified effects",
        ),),
    )


#: Machine spellings for the same thing, and the words a person uses for it.
_PLAIN_OUTCOME = (
    ("goal_reached", "got there"),
    ("pursuit_reached_goal", "kept at it until it worked"),
    ("out_of_time", "ran out of time"),
    ("out_of_cycles", "ran out of tries"),
    ("blocked_by_overlay", "was blocked by something on screen"),
    ("needs_person", "stopped and left it to you"),
    ("already_true", "found it was already done"),
    ("cannot_decide", "could not decide what to do"),
)


def _whole_words(text: str, limit: int) -> str:
    """Shortened at a word, not through one.

    A cause cut to a fixed number of characters ends mid-word — "tell me here
    when y" — which reads as though something went wrong rather than as
    though it was shortened.
    """
    said = str(text or "").strip()
    if len(said) <= limit:
        return said
    cut = said[:limit].rsplit(" ", 1)[0].rstrip(" ,;:—-")
    return f"{cut}…" if cut else said[:limit]


def _said_plainly(entry: Mapping[str, Any]) -> str:
    """One recorded action, in the words a person would use for it.

    Nothing is added and nothing is softened — the numbers and the outcome
    come from the receipt. Only the spelling changes.
    """
    evidence = str(entry.get("evidence") or "").strip()
    cause = _whole_words(" ".join(str(entry.get("cause") or "").split()), 120)
    if re.search(r"\b(?:moves|steps)=0\b", evidence):
        # Nothing was done, so there is nothing to report as done. A goal that
        # was already true when she arrived is a real outcome and a different
        # one, and reading it out as "got there" claims work that never
        # happened.
        return ""
    parts = [piece for piece in re.split(r"[;,]\s*", evidence) if piece]
    said = ""
    detail: list[str] = []
    for piece in parts:
        name, _, value = piece.partition("=")
        key = name.strip().lower()
        if not value:
            for token, plain in _PLAIN_OUTCOME:
                if token in key:
                    said = plain
                    break
            else:
                detail.append(piece.strip())
            continue
        if key in {"outcome", "status"}:
            for token, plain in _PLAIN_OUTCOME:
                if token == value.strip().lower():
                    said = plain
                    break
            else:
                said = said or value.strip()
        elif key in {"moves", "steps", "count"}:
            number = value.strip()
            detail.append(f"{number} {key[:-1] if number == '1' else key}")
        elif value.strip():
            detail.append(f"{key.replace('_', ' ')} {value.strip()}")
    if not said:
        said = str(entry.get("action") or "did something").replace("_", " ")
    line = said if not detail else f"{said} — {', '.join(detail)}"
    if cause:
        line = f"{line}. You had asked: {cause}"
    return line


def render_past_actions(bundle: EvidenceBundle) -> str:
    """Her own verified effects, as a record rather than a recollection."""

    for reading in bundle.readings:
        if reading.channel != "tool_receipts":
            continue
        if not reading.present:
            return (
                "I have no verified record of my own actions to read right now "
                f"({reading.detail or reading.state})."
            )
        # Read out the way a person answers, not the way it is filed.
        #
        # Every word of it is still from the receipts and nothing is added.
        # What changes is that "pursue_on_screen:
        # pursuit_reached_goal;moves=25;outcome=goal_reached — for: Find 2048
        # online, play it, and get to a 256 tile…" is a log line, and the
        # person asked what she did.
        lines = ["Here is what I actually did, most recent first:"]
        said_already: set[str] = set()
        for entry in reading.value or ():
            said = _said_plainly(entry)
            if not said or said in said_already:
                # The same thing twice is one thing. A store that holds a
                # retry and its original holds two receipts for one action,
                # and reading both out claims she did it twice.
                continue
            said_already.add(said)
            lines.append(f"  • {said}")
        if len(lines) == 1:
            return "I have no verified record of doing anything yet."
        return "\n".join(lines)
    return ""


def past_actions_answer(message: Any) -> str:
    """The record of what she did, when the turn asks, or ""."""

    if not asks_about_past_actions(message):
        return ""
    bundle = resolve_past_actions(query=message)
    if not bundle.grounded:
        # Nothing in the window they asked about is an ANSWER, not an absence.
        #
        # Bounding the record to "the last hour" turned a wrong answer into no
        # answer, and the turn fell through to the model — which is how a
        # question with a definite answer ends up guessed at.
        window = _window_named(message)
        if window and resolve_past_actions(query="").grounded:
            from core.language.stated_window import describe_window

            return (
                f"Nothing in {describe_window(window)}: I have no verified record "
                "of an action of mine in that stretch."
            )
        return ""
    return render_past_actions(bundle)


#: key=value pairs inside an effect-evidence string:
#: "listed=/path;pattern=*.py;count=9" -> {"listed": "/path", "count": "9"}
_EVIDENCE_FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^;]+)")


def concise_past_action_answer(message: Any) -> str:
    """One line of recorded fact, not a dump of receipts.

    LIVE, 2026-08-10: the full record — 3,300 characters of receipt lines —
    was appended to a two-sentence reply and never reached the person. Reply
    shaping downstream reads a wall of unrelated-looking text as off-topic and
    strips it, which is the correct instinct: the answer to "what was the
    count" is a number, not a ledger.

    So the salient value is extracted and stated. The ledger is still available
    through past_actions_answer for the paths that have nothing else to serve.
    """

    if not asks_about_past_actions(message):
        return ""
    bundle = resolve_past_actions(limit=4, query=message)
    if not bundle.grounded:
        return ""
    asked = {word for word in re.findall(r"[a-z]{3,}", str(message or "").lower())}
    for reading in bundle.readings:
        if reading.channel != "tool_receipts" or not reading.present:
            continue
        for entry in reading.value or ():
            fields = {
                match.group("key"): match.group("value").strip()
                for match in _EVIDENCE_FIELD_RE.finditer(str(entry.get("evidence") or ""))
            }
            for key, value in fields.items():
                # Only a field the question actually asked about, so "bytes"
                # and "sha256" never answer a question about a count.
                if key in asked and value:
                    answer = (
                        f"From my own receipts, the {key} was {value} — that is "
                        f"the recorded value, not a recollection."
                    )
                    _take_custody_of_recorded_value(
                        key=key,
                        value=value,
                        entry=entry,
                        asked=asked,
                        rendering=answer,
                    )
                    return answer
    return ""


def _take_custody_of_recorded_value(
    *,
    key: str,
    value: str,
    entry: Any,
    asked: set[str],
    rendering: str,
) -> None:
    """Hand the recorded value to the turn's custody set before returning it.

    Returning the sentence is not the same as the person receiving it. On
    2026-08-10 this exact value was correct here and wrong by the time it was
    spoken: one stage read the record as off-topic and stripped it, and a later
    repair replaced the reply with a denial that the read had happened.

    Custody is taken HERE because this is where the evidence is. A stage
    further down could re-derive the number, but a re-derivation is a second
    opinion, and the failure was never that the number was hard to compute.
    """

    try:
        from core.runtime.fact_custody import ValueKind, hold_fact
        from core.runtime.turn_outcome import VerificationGrade

        detail = str((entry or {}).get("evidence") or "")
        action = str((entry or {}).get("action") or "").strip()
        # Cues are what a sentence about this fact would have to mention: the
        # field name always, plus the words the question and the receipt agree
        # on. Both sides, so a cue cannot come from the receipt alone and match
        # a sentence about something else entirely.
        shared = asked & {word for word in re.findall(r"[a-z]{3,}", f"{action} {detail}".lower())}
        numeric = bool(re.fullmatch(r"\d[\d,]*", value.strip()))
        hold_fact(
            subject=action or "past_action",
            predicate=key,
            value=value,
            subject_cues=(key, *sorted(shared)[:6]),
            canonical_rendering=rendering,
            established_by="self_evidence.concise_past_action_answer",
            # The bundle only reaches here for receipts whose effect was
            # independently verified — resolve_past_actions drops the rest —
            # so this is OBSERVED, the grade that is allowed to correct text.
            grade=VerificationGrade.OBSERVED,
            kind=ValueKind.NUMBER if numeric else ValueKind.TEXT,
            evidence=(detail[:160],),
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "self_evidence.fact_custody",
            exc,
            severity="warning",
            action="a recorded value was answered without custody across the reply path",
        )


def reading_as_assertion(reading: Reading) -> Any:
    """A Reading on the shared epistemic substrate.

    Reading and EffectClaim were two types for one idea — a checkable statement
    and what backs it — which is the fragmentation this work was criticised
    for on 2026-08-10: "Aura has fragments of this everywhere. It does not yet
    have one universal epistemic substrate."

    A present reading is MEASURED and carries its provenance as evidence. A
    typed absence is not measured at all, so it lowers to a GENERATED assertion
    with verification UNVERIFIED — which is exactly what "the camera has never
    sampled" means, and it cannot be rendered as fact.
    """

    from core.epistemics.assertion import Assertion, SourceKind, Verification

    if reading.present:
        return Assertion(
            subject=reading.channel,
            claim=f"{reading.channel} reads {reading.value}",
            source=SourceKind.MEASURED,
            provenance=reading.provenance or reading.channel,
            evidence=(reading.provenance or reading.channel,),
            verification=Verification.VERIFIED,
            at=reading.at,
            value=reading.value,
        )
    return Assertion(
        subject=reading.channel,
        claim=f"{reading.channel} has no reading ({reading.state})",
        source=SourceKind.GENERATED,
        provenance=reading.provenance or reading.channel,
        verification=Verification.NOT_APPLICABLE,
        at=reading.at,
    )


def bundle_as_assertions(bundle: EvidenceBundle) -> list[Any]:
    """Every reading in a bundle, on the substrate."""

    return [reading_as_assertion(reading) for reading in bundle.readings]
