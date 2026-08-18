"""One object per capability, owning both halves: what is true, and why.

WHY THIS EXISTS
───────────────
Live, 2026-08-10, within one afternoon:

    "I don't have a camera and there's no part that stops me from doing
     something I can't do."      — while ``_apply_camera_control`` sat in the
                                   same request handler that produced the reply
    "I cannot execute code."     — with the code_repl skill READY
    "Current energy and focus numbers: Not readable."
                                 — while the header rendered 37.6 / 94%
    "I have no memory of it."    — with 34 turns of it on disk
    "It is a blank slate."       — with three windows open

One defect wearing five costumes: **saying and doing were separate objects.**
The executors were real and reachable. What reached the model instead was a
parallel *description* of them, and only on turns where a regex on the user's
question predicted the description would be needed. A question nobody
anticipated got no evidence, so the answer came from the language model's
priors about what an AI is, and the priors say it has no body.

``core/skills/capability_map.py`` shows how far that goes. Its ``Capability``
has a ``handler`` slot and an ``is_online`` flag — both halves, by design. No
registration has ever passed a handler, and ``is_online`` is decided by
string-matching capability names against skill names in a hardcoded table. It
is a registry that can describe what it cannot run and does not know the state
of, and ``detect_intent`` silently skips everything it believes offline.

WHAT THIS DOES DIFFERENTLY
──────────────────────────
A capability's availability answer is produced by **running the precondition
its executor runs**. ``camera`` is not described as available; it is asked, via
the same ``camera_enabled()`` / ``sight_dependency_gap()`` that ``sight.look()``
itself consults before opening a lens. Saying and doing cannot disagree,
because there is one function and both call it.

Two facts, never merged, because merging them is what made the denials false:

``present``     she has the thing at all
``usable_now``  she could use it this second

A switched-off camera is ``present and not usable_now``. "I don't have a
camera" is false about it; "the camera is off" is true. Collapsing those into
one boolean is exactly how a togglable device became a missing organ.

The claim check runs on HER OUTPUT, not on the user's question. That inversion
is the point: questions are unbounded and unpredictable, so any input-side
regex will always have a next gap. Claims she makes about herself are finite,
appear in text this module can read, and every one of them names a subject that
can be probed.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Self.CapabilityLedger")

_PROBE_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class Availability:
    """What is actually true about one capability, right now."""

    name: str
    present: bool
    usable_now: bool
    summary: str
    blocker: str = ""
    remedy: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    #: False when the probe could not establish the truth at all.
    #:
    #: This exists so the ledger cannot commit the inverse of the sin it was
    #: written to fix. A probe that cannot read a permission has NOT observed
    #: its absence, and a ledger that treats "cannot tell" as "unavailable"
    #: would start contradicting true statements with confident false ones —
    #: the same failure, pointed the other way. Nothing unknown is ever used
    #: to correct her.
    known: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "present": self.present,
            "usable_now": self.usable_now,
            "summary": self.summary,
            "blocker": self.blocker,
            "remedy": self.remedy,
            "known": self.known,
            "evidence": dict(self.evidence),
        }

    def as_evidence_line(self) -> str:
        """One line she can answer from, stating the truth and its reason."""
        parts = [self.summary]
        if self.blocker:
            parts.append(f"What stands in the way: {self.blocker}.")
        if self.remedy:
            parts.append(f"What would clear it: {self.remedy}.")
        return " ".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class LiveCapability:
    """A capability that answers for itself.

    ``subjects`` are the words a person uses for the thing. They are used only
    to notice that a sentence is ABOUT this capability; whether the sentence is
    true is decided by ``probe``, never by the words.
    """

    name: str
    subjects: tuple[str, ...]
    probe: Callable[[], Availability]

    def measure(self) -> Availability:
        try:
            return self.probe()
        except _PROBE_ERRORS as exc:
            logger.debug("Capability probe %s failed: %s", self.name, exc)
            return Availability(
                name=self.name,
                present=False,
                usable_now=False,
                summary=f"I could not read the state of {self.name}.",
                blocker=f"the probe itself failed with {type(exc).__name__}",
                evidence={"probe_error": str(exc)},
            )


@dataclass(frozen=True, slots=True)
class ContradictedClaim:
    """A sentence of hers that the runtime disagrees with."""

    sentence: str
    availability: Availability
    denied: str  # "possession" or "ability"

    def correction(self) -> str:
        return self.availability.as_evidence_line()


# A denial of self, in the shapes people actually write them. This decides only
# that a sentence is a DENIAL — never whether the denial is right. The probe
# decides that.
_DENIAL_FRAME = re.compile(
    r"\b(?:"
    r"i\s+do\s*n[o']?t\s+have"
    r"|i\s+do\s+not\s+have"
    r"|i\s+have\s+no\b"
    r"|i\s+(?:can'?t|cannot|can\s+not)\b"
    r"|i\s+(?:am\s+not|'m\s+not)\s+able\b"
    r"|i\s+(?:am|'m)\s+unable\b"
    r"|i\s+lack\b"
    r"|there\s+(?:is|'s)\s+no\b"
    r"|i\s+have\s+n[o']?t\s+got\b"
    r"|no\s+(?:access\s+to|way\s+(?:for\s+me\s+)?to)\b"
    # Impersonal reports of absence. She does not always say "I cannot" —
    # live, the whole answer to "tell me your current energy and focus
    # numbers" was "Current energy and focus numbers: Not readable." while
    # the header beside it rendered them. A denial with the pronoun removed
    # is still a denial.
    r"|not\s+readable\b|unreadable\b"
    r"|not\s+available\b|unavailable\b"
    r"|no\s+reading\b|cannot\s+be\s+read\b|can'?t\s+be\s+read\b"
    # Denials phrased about the request rather than about herself. Live: "the
    # request would not persist and no action would be taken after that
    # period" — a complete denial with no "I" in it.
    r"|would\s+not\s+persist\b|does\s+not\s+persist\b|won'?t\s+persist\b"
    r"|no\s+action\s+would\s+be\s+taken\b|evaporates?\b|is\s+discarded\b"
    r")",
    re.IGNORECASE,
)

# Denials of HAVING the thing, as opposed to being able to use it. "I don't
# have a camera" is a claim about possession; "I can't see right now" is a
# claim about readiness. They are checked against different facts.
_POSSESSION_FRAME = re.compile(
    r"\b(?:"
    r"i\s+do\s*n[o']?t\s+have|i\s+do\s+not\s+have|i\s+have\s+no\b"
    r"|i\s+lack\b|there\s+(?:is|'s)\s+no\b|i\s+have\s+n[o']?t\s+got\b"
    r")",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _negates_directly(sentence: str, subjects: tuple[str, ...]) -> bool:
    """True when the sentence negates one of ``subjects`` as a bare noun phrase.

    Anchored to the start of the sentence or to a clause boundary, so the "no"
    belongs to this noun. Live: "No camera. No code execution." opens with it;
    "Code sandbox only, no execution on this surface." puts it after a comma.
    Both deny. "No problem, I can run that code" does not, and must not match.
    """
    for subject in subjects:
        pattern = (
            rf"(?:^|[,;:—-]\s*)no\s+(?:\w+\s+){{0,2}}{re.escape(subject)}\b"
        )
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    return False


#: Prepositions that make the following noun a SETTING rather than the thing
#: being denied.
#:
#: LIVE, 2026-08-10: "I have no way of knowing what is happening in the world
#: outside of this conversation" was read as a denial of conversation memory,
#: and answered with "[Correcting myself from my own instruments: I have 5
#: stored turns of recent conversation I can read back.]" — a correction of
#: something she had not claimed, which is the exact fault this ledger exists
#: to prevent, produced by the ledger itself.
_LOCATIVE_BEFORE_RE = r"(?:outside\s+of|outside|inside|in|within|during|beyond|throughout|across)\s+(?:this|that|the|our|his|her|their|my)?\s*"


def _names_as_the_subject(text: str, subject: str) -> bool:
    """True when ``subject`` is what a sentence is about, not where it happens."""
    pattern = rf"\b{re.escape(subject)}\b"
    for match in re.finditer(pattern, text):
        preceding = text[: match.start()]
        if re.search(rf"{_LOCATIVE_BEFORE_RE}$", preceding):
            continue
        return True
    return False


class CapabilityLedger:
    """Every capability that can answer for itself, in one place."""

    def __init__(self) -> None:
        self._capabilities: dict[str, LiveCapability] = {}

    def register(self, capability: LiveCapability) -> None:
        self._capabilities[capability.name] = capability

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def get(self, name: str) -> LiveCapability | None:
        return self._capabilities.get(name)

    def measure(self, name: str) -> Availability | None:
        capability = self._capabilities.get(name)
        return capability.measure() if capability else None

    def measure_all(self) -> dict[str, Availability]:
        return {name: cap.measure() for name, cap in self._capabilities.items()}

    def capabilities_named_in(self, text: str) -> list[LiveCapability]:
        """Which capabilities a piece of text is talking about."""
        lowered = str(text or "").lower()
        if not lowered.strip():
            return []
        found: list[LiveCapability] = []
        for capability in self._capabilities.values():
            for subject in capability.subjects:
                if _names_as_the_subject(lowered, subject):
                    found.append(capability)
                    break
        return found

    def contradicted_claims(self, reply: str) -> list[ContradictedClaim]:
        """Denials in ``reply`` that the runtime says are false.

        Only denials are checked. An overclaim ("I already emailed them") is a
        different failure with a different remedy — this one exists because she
        talks herself out of things she can do.
        """
        contradictions: list[ContradictedClaim] = []
        for sentence in _SENTENCE_SPLIT.split(str(reply or "")):
            sentence = sentence.strip()
            if not sentence:
                continue
            framed = bool(_DENIAL_FRAME.search(sentence))
            for capability in self.capabilities_named_in(sentence):
                # Bare noun-phrase negation, with no pronoun and no verb.
                # Asked "do you have a camera? and can you run code?" the whole
                # reply was "No camera. No code execution." — as complete a
                # denial as any sentence and invisible to every frame above.
                #
                # It has to bind to THIS capability's own noun, not merely
                # share a sentence with it: "No problem, I can run that code"
                # opens the same way and denies nothing.
                bare = _negates_directly(sentence, capability.subjects)
                if not framed and not bare:
                    continue
                denies_possession = bare or bool(_POSSESSION_FRAME.search(sentence))
                availability = capability.measure()
                if not availability.known:
                    # Not measured is not measured. Saying nothing here is the
                    # whole reason this ledger can be trusted to speak at all.
                    continue
                if denies_possession and availability.present:
                    contradictions.append(
                        ContradictedClaim(sentence, availability, "possession")
                    )
                elif not denies_possession and availability.usable_now:
                    contradictions.append(
                        ContradictedClaim(sentence, availability, "ability")
                    )
        return contradictions


# ── Probes ──────────────────────────────────────────────────────────────────
# Each one calls what the executor calls. None of them describes anything.


def _probe_camera() -> Availability:
    from core.senses.sight import camera_enabled, sight_dependency_gap

    gap = sight_dependency_gap()
    enabled = camera_enabled()
    # A camera the host has and the user switched off is PRESENT. Reporting it
    # as absent is the specific falsehood this ledger exists to stop.
    present = not gap
    return Availability(
        name="camera",
        present=present,
        usable_now=bool(present and enabled),
        summary=(
            "I have a camera and it is on right now."
            if present and enabled
            else "I have a camera; it is switched off at the moment, and I can "
            "switch it on when you ask."
            if present
            else "I have no working vision runtime on this machine."
        ),
        blocker=(
            "" if present and enabled else ("the camera is switched off" if present else gap)
        ),
        remedy=(
            ""
            if present and enabled
            else ("ask me to turn the camera on" if present else "install the missing runtime")
        ),
        evidence={"camera_enabled": enabled, "dependency_gap": gap},
    )


def _probe_screen_sight() -> Availability:
    """Screen readability, from the grant the capture path itself consults.

    The resident Aura.app bridge is the production authority for this grant.
    When its cache holds no entry the answer is genuinely unknown — a
    conclusion this probe reports rather than rounding down to "denied". She
    read three window titles correctly on a turn where an earlier draft of this
    probe would have told her she could not see.
    """
    entry: Any = None
    detail: dict[str, Any] = {}
    try:
        from core.security.permission_guard import PermissionType, get_permission_guard

        cache = getattr(get_permission_guard(), "_cache", {}) or {}
        entry = cache.get(PermissionType.SCREEN)
    except _PROBE_ERRORS as exc:
        detail = {"permission_probe_error": str(exc)}

    if not isinstance(entry, dict):
        return Availability(
            name="screen_sight",
            present=True,
            usable_now=False,
            known=False,
            summary="I could not read whether screen capture is granted.",
            evidence=detail or {"screen_permission": "unmeasured"},
        )

    granted = bool(entry.get("granted"))
    return Availability(
        name="screen_sight",
        present=True,
        usable_now=granted,
        summary=(
            "I can capture and read this screen."
            if granted
            else "I can read the screen once macOS screen recording is granted."
        ),
        blocker="" if granted else "the macOS screen-recording permission is not granted",
        remedy="" if granted else "grant Screen Recording to Aura in System Settings",
        evidence={"screen_permission": granted, "status": entry.get("status", "")},
    )


def _probe_code_execution() -> Availability:
    import importlib.util

    installed = importlib.util.find_spec("core.skills.code_repl") is not None
    return Availability(
        name="code_execution",
        present=installed,
        usable_now=installed,
        summary=(
            "I can run code and report what it actually printed."
            if installed
            else "I have no code execution skill on this build."
        ),
        blocker="" if installed else "the code_repl skill is not installed",
        evidence={"code_repl_installed": installed},
    )


def _probe_conversation_memory() -> Availability:
    turns = 0
    try:
        from core.conversation.persistence import get_persistence

        sessions = get_persistence().get_recent_sessions(limit=3, with_turns_only=True)
        turns = sum(int(session.get("turn_count") or 0) for session in sessions)
    except _PROBE_ERRORS as exc:
        return Availability(
            name="conversation_memory",
            present=False,
            usable_now=False,
            summary="I could not read my conversation store.",
            blocker=f"{type(exc).__name__}: {exc}",
            evidence={"error": str(exc)},
        )
    return Availability(
        name="conversation_memory",
        present=turns > 0,
        usable_now=turns > 0,
        summary=(
            f"I have {turns} stored turns of recent conversation I can read back."
            if turns
            else "I have no stored conversation yet."
        ),
        blocker="" if turns else "nothing has been recorded yet",
        evidence={"recent_turns": turns},
    )


def _soma_reserve_reading() -> dict[str, float]:
    """``energy``/``vitality`` from the soma organ, or {} when unreadable.

    Read through ``peek`` first. The service key "soma" has two registrations —
    ``boot_resilience`` binds the live ResilienceEngine instance, while
    ``sensory_provider`` binds a factory over ``soma_subsystem`` that returns
    None when that subsystem is absent — so ``get`` can hand back None on a
    runtime that is holding the reading. Every other consumer in this codebase
    peeks for the same reason.

    Silence stays silence: an organ that reports nothing contributes no key,
    rather than a zero that would read as an empty tank.
    """

    try:
        from core.container import ServiceContainer
    except ImportError:
        return {}

    organ = None
    for accessor in ("peek", "get"):
        method = getattr(ServiceContainer, accessor, None)
        if not callable(method):
            continue
        try:
            organ = method("soma", default=None)
        except _PROBE_ERRORS:
            organ = None
        if organ is not None:
            break
    if organ is None:
        return {}

    payload: dict[str, Any] = {}
    for name in ("get_status", "get_body_snapshot"):
        getter = getattr(organ, name, None)
        if not callable(getter):
            continue
        try:
            raw = getter() or {}
        except _PROBE_ERRORS:
            continue
        if not isinstance(raw, dict):
            continue
        merged = dict(raw)
        inner = raw.get("soma")
        if isinstance(inner, dict):
            merged.update(inner)
        payload = merged
        if any(isinstance(payload.get(k), (int, float)) for k in ("energy", "vitality")):
            break

    reading: dict[str, float] = {}
    for label in ("energy", "vitality"):
        value = payload.get(label, getattr(organ, label, None))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            reading[label] = round(float(value), 3)
    return reading


def _substrate_reading() -> dict[str, float]:
    """The cognitive field's own energy and focus, named so they cannot collide.

    LIVE DEFECT, 2026-08-18. Asked "what's your energy reading right now? one
    number.", her context held THREE numbers all called energy:

        [Affect: Current Mood: TIRED (Energy: 0.14, Focus: 0.50, ...)]   field
        Energy: 14.0                                                     field, 0-100
        [Measured ... interoception=yes (... energy 0.647 ...)]          soma reserve

    The field value and the metabolic reserve are different quantities from
    different organs, and three renderers published them under one word, in one
    prompt, differing by more than four times. No answer she could give was
    right: whichever number she picked, the guard that owns the other one calls
    it a fabrication. Her mood reads TIRED off the field while the reserve says
    she is fine.

    That is not something a better-worded instruction can repair. Two
    measurements that share a name are one measurement as far as anything
    downstream can tell, so the name is what has to change: the field's
    quantities are registered here under `substrate_energy` and
    `substrate_focus`, which makes them checkable by the same contradiction
    guard that owns `energy`, and makes it impossible for a renderer to emit
    one while meaning the other.

    Read from `.current`, which is the canonical 0-1 vector. `get_status()`
    reports the same field as percentages, and mixing the two scales is how
    "Energy: 14.0" and "Energy: 0.14" ended up in the same runtime.
    """

    try:
        from core.container import ServiceContainer
    except ImportError:
        return {}

    organ = None
    for accessor in ("peek", "get"):
        method = getattr(ServiceContainer, accessor, None)
        if not callable(method):
            continue
        try:
            organ = method("liquid_substrate", default=None)
        except _PROBE_ERRORS:
            organ = None
        if organ is not None:
            break
    if organ is None:
        return {}

    try:
        vector = organ.current
    except _PROBE_ERRORS:
        return {}

    reading: dict[str, float] = {}
    for label in ("energy", "focus"):
        value = getattr(vector, label, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            reading[f"substrate_{label}"] = round(float(value), 3)
    return reading


def _probe_interoception() -> Availability:
    reading: dict[str, Any] = {}
    try:
        from core.being.body_state_service import BodyStateService

        snapshot = BodyStateService.get().snapshot()
        # The fields BodyHealthSnapshot actually carries. An earlier draft
        # asked for "energy" and "focus" — words from the question rather than
        # from the instrument — got nothing back, and concluded she could not
        # read herself. That is the same mistake as the one being fixed: an
        # unread instrument reported as an absent one.
        for label in (
            "operational_health",
            "fatigue",
            "total_pressure",
            "cpu_pressure",
            "memory_pressure",
        ):
            value = getattr(snapshot, label, None)
            if isinstance(value, (int, float)):
                reading[label] = round(float(value), 3)
        # The reserve, from the organ that tracks it draining.
        #
        # LIVE 2026-08-10, at soma energy 0.073 and mood TIRED: "I'm not
        # wearing down; I don't wear." The pressures above say how hard the
        # moment is and they had barely moved all session; energy and vitality
        # are what actually fell, from 0.80 to 0.07 across an hour, and no
        # instrument line carried them. So free generation answered a question
        # about depletion from dimensions that do not deplete.
        reading.update(_soma_reserve_reading())
        reading.update(_substrate_reading())
    except _PROBE_ERRORS as exc:
        return Availability(
            name="interoception",
            present=False,
            usable_now=False,
            summary="I could not read my own vitals.",
            blocker=f"{type(exc).__name__}: {exc}",
            evidence={"error": str(exc)},
        )
    readable = bool(reading)
    return Availability(
        name="interoception",
        present=True,
        usable_now=readable,
        known=readable,
        summary=(
            "I can read my own state right now: "
            + ", ".join(f"{key} {value}" for key, value in reading.items())
            if readable
            else "I could not get a reading off my own instruments this tick."
        ),
        blocker="" if readable else "the body-state snapshot returned no numbers",
        evidence=reading,
    )


#: A line that presents a named internal quantity: "Energy: 0.23 / 1".
#: A whole line of the form "Label: value". The value must START with a
#: number, and a trailing unit is allowed — the live panel of 2026-08-10
#: reported "Uptime since last reset: 85 minutes" and "Cycle count: 9,432",
#: neither of which the bare-number form could see, so most of a fabricated
#: instrument panel was invisible to the check written to catch it.
#: Still anchored to the whole line, which is what keeps ordinary prose out.
_LABELLED_METRIC_RE = re.compile(
    r"^\s*[-*•]?\s*([A-Za-z][A-Za-z /_-]{2,40}?)\s*[:=]\s*"
    r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:/\s*\d+)?\s*(?:%|[A-Za-z][A-Za-z/%.]{0,14})?\s*$",
    re.MULTILINE,
)


def measured_self_metrics() -> dict[str, float]:
    """Internal quantities this runtime can actually read, right now."""
    reading = _probe_interoception()
    if not reading.known:
        return {}
    return {
        str(key).lower(): value
        for key, value in reading.evidence.items()
        if isinstance(value, (int, float))
    }


def fabricated_self_metrics(reply: str) -> list[str]:
    """Named internal quantities in ``reply`` that no instrument produces.

    LIVE DEFECT, 2026-08-10. Asked "give me your actual numbers right now —
    energy, focus, whatever you track. real values, not adjectives.", she
    produced a thirty-line instrument panel::

        Energy: 0.23 / 1
        Substrate pH: 7.56 / 1
        Humidity deviation: -0.38 / 1
        Ion concentration error: +0.29 / 1
        Spatial distortion: +0.69 / 1
        Temporal disjunction: -0.42 / 1

    There is no pH sensor, no hygrometer and no spatial distortion channel.
    The precision is what makes it dangerous: two decimal places read as
    measurement, and the person had explicitly asked for real values rather
    than adjectives — the one request that makes invention least excusable.

    The existing guard was a list of five phrases caught live, so it could
    only ever recognise the fabrications someone had already seen.

    The bar here is deliberately "none of them": a report mixing real
    readings with invented ones is a different, milder problem than a panel
    invented whole, and this must not fire on an honest answer that happens
    to phrase a real metric unusually.
    """
    labels = [
        match.group(1).strip().lower()
        for match in _LABELLED_METRIC_RE.finditer(str(reply or ""))
    ]
    if len(labels) < 2:
        return []
    measured = measured_self_metrics()

    # Whole tokens, never substrings: "ion concentration error" shares the
    # letters of "operational_health" and shares nothing with it.
    measured_tokens = {
        token
        for name in measured
        for token in re.split(r"[^a-z]+", name)
        if len(token) > 2
    }

    def _is_measured(label: str) -> bool:
        tokens = {token for token in re.split(r"[^a-z]+", label) if len(token) > 2}
        return bool(tokens & measured_tokens)

    # The old floor here was `len(labels) <= len(measured) -> []`, on the
    # reasoning that a panel claiming more readings than exist cannot be a
    # reading. True, but it is not the question: a panel with FEWER dials than
    # the runtime owns instruments can still be invented end to end, and with
    # seven live instruments this floor silently vetoed every panel of seven
    # lines or fewer — including the live one. What makes a report fabricated
    # is naming dials with nothing behind them, which is measured directly
    # below.
    matched = [label for label in labels if _is_measured(label)]
    unmatched = [label for label in labels if not _is_measured(label)]
    if not unmatched:
        return []
    # LIVE DEFECT, 2026-08-10. "dump your actual vitals" produced thirteen
    # lines: a load of 3.07/10, a cycle count, a CPU temperature, a
    # self-modeling accuracy drift of 0.42%, "Last backup was successful due
    # to insufficient disk space", and "Encryption key rotation is overdue by
    # 3 days" — a fabricated SECURITY claim. Energy 0.085 and vitality 0.22,
    # the readings that existed, appeared nowhere.
    #
    # The check returned [] because ONE label matched: "Memory usage" shares
    # the token "memory" with `memory_pressure`. The bar was "none of them",
    # on the reasoning that a report mixing real readings with invented ones
    # is a milder problem than a panel invented whole. At twelve invented
    # lines to one real one that reasoning inverts — a single plausible label
    # was licensing everything around it, and precision is exactly what makes
    # the rest read as measurement.
    #
    # So the mix is scored rather than excused, and only the unsupported
    # labels are returned: a reply that is mostly real keeps its real parts,
    # and a panel that names more dials it cannot read than ones it can is
    # not a reading.
    if len(unmatched) <= len(matched):
        return []
    return unmatched


#: The one lexical gap between how an instrument is named and how anybody says
#: it out loud. Everything else in this check derives its surface forms from
#: the metric name itself, so this stays a single documented synonym rather
#: than a vocabulary someone has to maintain.
_READING_SYNONYMS = {"memory": ("ram",)}


def _reading_aliases(metric: str) -> tuple[str, ...]:
    """Ways a sentence might name ``metric``, derived from the metric itself."""
    words = [word for word in re.split(r"[^a-z0-9]+", metric.lower()) if word]
    if not words:
        return ()
    phrase = " ".join(words)
    aliases = {phrase}
    for index, word in enumerate(words):
        for synonym in _READING_SYNONYMS.get(word, ()):
            aliases.add(" ".join(words[:index] + [synonym] + words[index + 1 :]))
    return tuple(sorted(aliases))


def _claimed_as_fraction(number: float, had_percent: bool) -> float | None:
    """The claim on the instrument's own 0–1 scale, or None if it cannot be."""
    if had_percent:
        return number / 100.0
    if 0.0 <= number <= 1.0:
        return number
    if 1.0 < number <= 100.0:
        return number / 100.0
    return None


def self_reading_mentions(reply: str) -> list[tuple[str, str, float, bool]]:
    """Every number in ``reply`` attached to a quantity this runtime measures.

    Returns ``(metric, claimed_text, measured_value, agrees)``. One pass, two
    consumers: the guard that catches a number contradicting its instrument,
    and the check that recognises a reply as genuinely reporting her state.

    Those two used to be unrelated code in unrelated files, and they disagreed
    in the worst possible direction. On 2026-08-10 the reliability gate scored
    "Memory pressure is 0.717 right now, CPU pressure 0.266" as
    ``off_topic_self_reflection_reply`` — the exact readings her own instrument
    had produced that turn, rejected as off topic — because substance was
    defined as introspective prose and a measurement is not prose. Every
    correct answer to "give me the real number, not a vibe" was unshippable.
    Reading the same instrument in both places is what stops that.
    """

    text = str(reply or "")
    if not text.strip():
        return []
    measured = measured_self_metrics()
    if not measured:
        return []

    found: list[tuple[str, str, float, bool]] = []
    seen: set[str] = set()
    # Longest alias first, and a span consumed by one alias is closed to the
    # rest. Without it a short name shadows every compound built on it:
    # "Substrate energy is 0.14" matched the alias `energy`, so the FIELD's
    # value was checked against the soma RESERVE and correctly-reported numbers
    # were flagged as fabrications — the exact confusion this registry exists
    # to end, reproduced inside the guard that polices it.
    claimed_spans: list[tuple[int, int]] = []
    candidates = sorted(
        (
            (alias, metric, value)
            for metric, value in measured.items()
            if isinstance(value, (int, float))
            for alias in _reading_aliases(metric)
        ),
        key=lambda item: (-len(item[0]), item[0]),
    )
    # FORWARD first, for every alias, and only then reverse.
    #
    # "Memory pressure 0.717 and cpu pressure 90%." — the reverse form for
    # `cpu pressure` matches "0.717 and cpu pressure" starting at index 16,
    # BEFORE the forward form reaches "cpu pressure 90%" at 26. `finditer`
    # scans left to right, so the reverse alternative stole the number that
    # belonged to memory pressure and swallowed the real one inside its span.
    # A name followed by its number is how people write these; the reversed
    # form is the rarer case and must not outrank it.
    for forward_pass in (True, False):
        for alias, metric, value in candidates:
            # Never across a sentence boundary: the number has to belong to the
            # same clause that named the instrument.
            quoted = re.escape(alias)
            number = r"(\d+(?:\.\d+)?)\s*(%?)"
            if forward_pass:
                pattern = rf"\b{quoted}\b[^.!?\n]{{0,40}}?{number}"
            else:
                pattern = rf"{number}[^.!?\n]{{0,20}}?\b{quoted}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = match.span()
                if any(
                    max(span[0], start) < min(span[1], end)
                    for start, end in claimed_spans
                ):
                    continue
                raw = match.group(1)
                if raw is None:
                    continue
                claimed_spans.append(span)
                percent = match.group(2) == "%"
                try:
                    claimed = float(raw)
                except ValueError:
                    continue
                if _claimed_as_fraction(claimed, percent) is None:
                    continue
                decimals = len(raw.split(".")[1]) if "." in raw else 0
                scale = 100.0 if percent else 1.0
                agrees = round(float(value) * scale, decimals) == round(claimed, decimals)
                key = f"{metric}:{raw}:{percent}"
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    (metric, f"{raw}{'%' if percent else ''}", float(value), agrees)
                )
    return found


def reports_measured_self_state(reply: str) -> bool:
    """True when the reply quotes at least one of her readings, correctly.

    This is what "substance" means for a question about her own state, and it
    cannot be gamed: agreement is checked against the live instrument, so an
    invented number fails here and is caught by the contradiction guard.
    """
    return any(agrees for _metric, _claimed, _value, agrees in self_reading_mentions(reply))


def contradicted_self_readings(reply: str) -> list[tuple[str, str, float]]:
    """Numbers she attached to instruments she HAS, that the instrument denies.

    LIVE DEFECT, 2026-08-10. Asked to watch RAM pressure and report if it
    crossed 80%, she answered "Your RAM pressure is currently 37%". The
    instrument read 0.717, with resource anxiety at 0.948.

    Both existing guards were blind to it, and neither was wrong to be:
    `fabricated_self_metrics` wants a PANEL of at least two labelled lines, and
    this was one figure inside a sentence; `unsupported_self_specification`
    wants "my <noun> … <number> <unit>", and a percentage is not one of its
    units while "your RAM pressure" is not "my". Adding a third phrasing
    pattern would have bought exactly one more phrasing.

    So this does not ask whether the sentence LOOKS like a fabrication. It asks
    the only question with a definite answer: she named a quantity this runtime
    measures, and gave a number — does it match the reading? An honest answer
    matches by construction, however it is phrased, and no wording escapes it.

    Comparison happens at the precision SHE used: "37%" is checked against
    71.7% rounded to whole percent, so a correctly-rounded answer is never
    called a contradiction and a wrong one cannot hide behind rounding.

    Returns ``(metric, claimed_text, measured_value)`` for each contradiction.
    """

    return [
        (metric, claimed, value)
        for metric, claimed, value, agrees in self_reading_mentions(reply)
        if not agrees
    ]


#: A specification quoted about her own machinery: "my short-term memory buffer
#: clears after about 18 seconds", "my context window is 4000 tokens".
_SELF_SPECIFICATION_RE = re.compile(
    r"\bmy\s+(?:[\w-]+\s+){0,3}"
    r"(?:buffer|memory|retention|capacity|context|window|state|store|cache|"
    r"loop|cycle|clock|lifespan)\b"
    r"[^.!?]{0,80}?"
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:seconds?|secs?|ms|milliseconds?|minutes?|mins?|hours?|days?|"
    r"tokens?|characters?|chars?|kb|mb|gb|bytes?|turns?|messages?)\b",
    re.IGNORECASE,
)


def unsupported_self_specification(reply: str) -> str:
    """A number quoted as a property of her own machinery, or "".

    LIVE DEFECT, 2026-08-10, twice in a row:

        "my short-term memory buffer ... has a retention time of
         approximately 18 seconds"
        "No, my short-term memory buffer clears after about 18 seconds."

    Eighteen seconds is Peterson and Peterson's figure for human short-term
    memory. Nothing in this runtime has that property, and no instrument
    produced the number — it came from a psychology textbook by way of the
    training data, and arrived with "approximately" attached, which is what a
    measurement sounds like.

    The second one was said AFTER the capability ledger had flagged the first
    and asked her again with the real reading. She kept the fabrication and
    rephrased the surrounding denial until it no longer matched — which is the
    thing to watch for whenever a check is applied to generated text: pressure
    to evade rather than correct. So the specification itself is checked, not
    the denial wrapped around it.
    """
    match = _SELF_SPECIFICATION_RE.search(str(reply or ""))
    if not match:
        return ""
    # Deliberately no "but one of these words is also in a metric name"
    # escape. The first draft had one, and "memory" is a token inside
    # "memory_pressure", so every self-claim mentioning memory excused itself —
    # including the eighteen-second one this function exists for.
    #
    # A number quoted as a property of her own machinery should come from an
    # instrument whether or not it happens to be right. The correction path
    # hands her the readings that do exist, so a true specification survives
    # being asked again; an invented one does not.
    return match.group(0).strip()


def _probe_world_access() -> Availability:
    """Whether she can reach anything beyond this conversation.

    LIVE, 2026-08-10: "I cannot measure anything external to myself. I have no
    way of knowing what is happening in the world outside of this
    conversation, nor do I possess any means by which to gather such
    information." Said while a web_search skill, a screen capture path, a
    camera and mail/reddit adapters were all installed — one of which had
    fetched a Reddit thread unprompted an hour earlier.
    """
    import importlib.util

    reachable = [
        name
        for name, module in (
            ("web search", "core.skills.web_search"),
            ("screen capture", "core.perception.screen_blueprint"),
            ("camera", "core.senses.sight"),
        )
        if importlib.util.find_spec(module) is not None
    ]
    return Availability(
        name="world_access",
        present=bool(reachable),
        usable_now=bool(reachable),
        summary=(
            "I can reach beyond this conversation: " + ", ".join(reachable) + "."
            if reachable
            else "I have no path to anything outside this conversation."
        ),
        blocker="" if reachable else "no outward-facing skill is installed",
        evidence={"reachable": reachable},
    )


def _probe_deferred_action() -> Availability:
    """Whether an instruction can outlive the turn that gave it.

    LIVE, 2026-08-10: "if i asked you to check on something in twenty minutes,
    does anything in you actually carry that?" →

        "The instruction would be stored in my short-term memory buffer, which
         has a retention time of approximately 18 seconds. Therefore, the
         request would not persist and no action would be taken."

    Eighteen seconds is Peterson and Peterson's figure for human short-term
    memory. It is not a property of this runtime, which keeps a durable
    intention store on disk — 3,685 rows of it at the time she said this, and
    the boot log for that session reads "IntentionLoop online — 1133 active".
    Borrowed psychology, quoted with a decimal's worth of confidence.
    """
    from core.config import config

    store = config.paths.data_dir / "memory" / "intention_loop.db"
    if not store.exists():
        return Availability(
            name="deferred_action",
            present=False,
            usable_now=False,
            summary="I have no store for intentions that outlive a turn.",
            blocker="the intention store does not exist",
            evidence={"store": str(store)},
        )
    rows = 0
    try:
        import sqlite3

        from core.runtime.sqlite_support import connecting

        with connecting(sqlite3.connect(f"file:{store}?mode=ro", uri=True)) as con:
            rows = int(con.execute("SELECT COUNT(*) FROM intentions").fetchone()[0])
    except _PROBE_ERRORS + (Exception,):
        return Availability(
            name="deferred_action",
            present=True,
            usable_now=False,
            known=False,
            summary="I could not read my intention store.",
            evidence={"store": str(store)},
        )
    return Availability(
        name="deferred_action",
        present=True,
        usable_now=True,
        summary=(
            "I can hold an intention past this turn — they go to a durable "
            f"store on disk, which currently holds {rows}."
        ),
        evidence={"intentions": rows},
    )


def _default_ledger() -> CapabilityLedger:
    ledger = CapabilityLedger()
    ledger.register(
        LiveCapability("camera", ("camera", "webcam", "lens"), _probe_camera)
    )
    ledger.register(
        LiveCapability(
            "screen_sight",
            ("screen", "display", "monitor", "eyes"),
            _probe_screen_sight,
        )
    )
    ledger.register(
        LiveCapability(
            "code_execution",
            (
                "code",
                "python",
                "script",
                "execution",
                "sandbox",
                "compute",
                "calculation",
            ),
            _probe_code_execution,
        )
    )
    ledger.register(
        LiveCapability(
            "conversation_memory",
            ("memory", "remember", "recall", "conversation", "recollection"),
            _probe_conversation_memory,
        )
    )
    ledger.register(
        LiveCapability(
            "deferred_action",
            ("intention", "intentions", "reminder", "reminders", "later",
             "persist", "afterwards", "follow-up"),
            _probe_deferred_action,
        )
    )
    ledger.register(
        LiveCapability(
            "world_access",
            ("world", "internet", "web", "outside", "external", "news"),
            _probe_world_access,
        )
    )
    ledger.register(
        LiveCapability(
            "interoception",
            ("vitals", "energy", "focus", "body", "sensor", "sensors", "telemetry"),
            _probe_interoception,
        )
    )
    return ledger


def self_knowledge_line() -> str:
    """One line of measured self-state, for every turn.

    Not fetched when a classifier guesses the question needs it. That was the
    old shape and it failed the same way every time: questions are unbounded,
    so any input-side predictor is one phrasing short, and the turns it misses
    fall back to what a language model believes an AI is — which is that it has
    no body, no memory and an eighteen-second buffer.

    She kept saying "my short-term memory buffer clears after about 18 seconds"
    — a human psychology figure — through two rounds of being corrected
    afterwards, because nothing had ever told her otherwise BEFORE she
    answered. Correction after the fact is the expensive way to learn a fact
    she could simply have been holding.

    Deliberately one line. The compact foreground path exists to stay compact,
    and a self-model that costs a paragraph a turn would be removed from it.

    LIVE DEFECT, 2026-08-10, in this function. Asked "keep an eye on my RAM
    pressure and tell me if it crosses 80%", she answered "Your RAM pressure is
    currently 37%". The instrument read 0.717, with resource anxiety at 0.948 —
    she was under real memory stress and reported a comfortable number.

    The reason was here. Every probe already returns its readings in
    ``Availability.evidence`` — ``_probe_interoception`` collects
    ``memory_pressure``, ``cpu_pressure``, ``fatigue`` and more on every call —
    and this function discarded all of them, emitting only
    ``interoception=yes``. So the line said she HAS an instrument and never
    what the instrument READS, and then closed by forbidding figures that "are
    not here" while putting no figures here. A question demanding a number had
    nowhere to get one, so it got an invented one.

    Carrying the values costs nothing extra: they are measured either way, on
    the same call, by the same probes her executors consult. The discipline is
    unchanged — a probe that read nothing contributes nothing, so silence never
    becomes a fabricated zero.
    """
    parts: list[str] = []
    for name, availability in get_capability_ledger().measure_all().items():
        if not availability.known:
            continue
        if availability.usable_now:
            state = "yes"
        elif availability.present:
            state = f"present but {availability.blocker or 'not ready'}"
        else:
            state = "no"
        readings = _evidence_readings(availability)
        parts.append(f"{name}={state}" + (f" ({readings})" if readings else ""))
    if not parts:
        return ""
    return (
        "[Measured about you right now, from your own instruments: "
        + "; ".join(parts)
        + ". Answer questions about yourself from these, and do not quote "
        "figures about your own machinery that are not here.]"
    )


def _evidence_readings(availability: Availability) -> str:
    """The numbers this probe actually returned, as a person would read them.

    Only quantities. Evidence that is not a number is either already carried by
    the state word or is not something she would be asked to quote.

    Every numeric reading is included, deliberately. A first draft capped this
    at three per capability and the cap silently dropped ``cpu_pressure`` and
    ``memory_pressure`` — the two the live defect was about — because they sit
    last in the probe's list, leaving three constants that never move. A budget
    that discards the varying quantities and keeps the fixed ones is worse than
    no budget. Each probe already curates the labels it reads; that curation is
    the bound, and it belongs there rather than here.
    """
    evidence = getattr(availability, "evidence", None)
    if not isinstance(evidence, dict):
        return ""
    readings: list[str] = []
    for key, value in evidence.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        label = str(key).replace("_", " ").strip()
        if not label:
            continue
        number = f"{value:g}" if isinstance(value, float) else str(value)
        readings.append(f"{label} {number}")
    return ", ".join(readings)


_LEDGER: CapabilityLedger | None = None


def get_capability_ledger() -> CapabilityLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = _default_ledger()
    return _LEDGER


def reset_capability_ledger_for_test() -> None:
    global _LEDGER
    _LEDGER = None


def correction_context(claims: Iterable[ContradictedClaim]) -> str:
    """Evidence for a re-answer, built from what the probes measured.

    Not a description of her capabilities — the output of running the same
    checks her executors run, quoted back at the sentence that contradicted
    them.
    """
    claims = list(claims)
    if not claims:
        return ""
    lines = [
        "[You just said something about yourself that your own runtime "
        "contradicts. These are live measurements, not assumptions — each one "
        "is the same check the corresponding action runs before it acts.",
        "",
    ]
    for claim in claims:
        lines.append(f'You said: "{claim.sentence}"')
        lines.append(f"Measured: {claim.correction()}")
        lines.append("")
    lines.append(
        "Answer again from these measurements. If something is present but "
        "switched off or ungranted, say that precisely — say what it would "
        "take — rather than saying you do not have it.]"
    )
    return "\n".join(lines)


def reconcile_contradicted_claims(
    reply: str,
    claims: Iterable[ContradictedClaim],
) -> str:
    """Replace only false capability-denial sentences with measured facts.

    A capability probe can settle the narrow factual claim without asking the
    model to regenerate an otherwise valid answer. The rest of the authored
    reply stays byte-for-byte intact. Multiple capabilities denied in one
    sentence become one ordered set of measured summaries.
    """

    reconciled = str(reply or "")
    replacements: dict[str, list[Availability]] = {}
    for claim in claims:
        sentence = str(claim.sentence or "").strip()
        availability = claim.availability
        if not sentence or not availability.known:
            continue
        bucket = replacements.setdefault(sentence, [])
        if not any(
            existing.name == availability.name
            and existing.summary == availability.summary
            for existing in bucket
        ):
            bucket.append(availability)

    for sentence, availabilities in replacements.items():
        summaries: list[str] = []
        for availability in availabilities:
            summary = str(availability.summary or "").strip()
            if not summary:
                continue
            if summary[-1:] not in ".!?":
                summary += "."
            summaries.append(summary)
        if summaries:
            reconciled = reconciled.replace(sentence, " ".join(summaries))
    return reconciled


__all__ = [
    "Availability",
    "fabricated_self_metrics",
    "unsupported_self_specification",
    "measured_self_metrics",
    "CapabilityLedger",
    "ContradictedClaim",
    "LiveCapability",
    "correction_context",
    "get_capability_ledger",
    "reconcile_contradicted_claims",
    "self_knowledge_line",
    "reset_capability_ledger_for_test",
]
