"""The intermediate cognitive code: what Aura's state says, before any words.

    SELF        continuity high / agency mid
    GOAL        active priority-high
    MEMORY      recall-strong contradiction
    UNCERTAINTY high
    TEMPORAL    future
    INTEND      (abstained: no trained head)

Ugly on purpose. This is not a draft of a reply and it is not shown to anyone:
it is a reading of the endogenous state in symbols, taken *before* the
transformer is asked for anything. Two things follow from that ordering.

It is a measurement instrument. When the transformer explains Aura's state
there is always the question of whether the machinery contained that
information or whether a pretrained model wrote a plausible account after
being handed context. A reading taken from z_Aura before generation is a
second, independent channel: intervene on one named dimension, watch the code
move, then watch the language move.

And it is an arbitration surface. A proposal that arrives from the transformer
can be checked against the code — against a goal that is actually held, an
uncertainty that is actually high — instead of being taken as her thought
because it came back from the model.

**Provenance is per line.** Most lines are derived from z_Aura by a stated
rule. Some — the active concepts, the referents — cannot be: 74 floats cannot
encode which entities are live, and pretending otherwise is exactly the
overclaim this pathway exists to avoid. Those lines are read from the organs
and marked ``organ`` so nothing downstream mistakes them for a readout.

**Nothing here is trained except what says it is.** ``INTEND`` is a learned
head over the state, and until one is fitted the line abstains. An abstention
prints as an abstention. It never prints as a guess.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.brain.llm.endogenous_state import (
    FEATURE_INDEX,
    FEATURES,
    EndogenousState,
)

logger = logging.getLogger("Aura.CognitiveCode")

#: Speech acts the INTEND head may choose between once fitted. The list is the
#: contract: a head trained against one list cannot be read against another.
SPEECH_ACTS: tuple[str, ...] = (
    "assert",
    "ask",
    "propose",
    "report",
    "acknowledge",
    "decline",
    "reflect",
)

#: How many concepts and referents a code may carry. A code is a reading, not
#: a dump of the working set.
MAX_CONCEPTS = 6
MAX_REFERENTS = 4


def _band(state: EndogenousState, feature: str) -> str | None:
    """Low / mid / high, cut at the thirds of the dimension's declared range.

    The cuts come from the range the dimension was declared with rather than
    from a number chosen here, so a dimension that runs -1..1 and one that
    runs 0..1 band the same way without anyone maintaining a table.
    """
    index = FEATURE_INDEX.get(feature)
    if index is None or not state.present[index]:
        return None
    declared = FEATURES[index]
    span = declared.high - declared.low
    if span <= 0:
        return None
    position = (float(state.values[index]) - declared.low) / span
    if position < 1.0 / 3.0:
        return "low"
    if position < 2.0 / 3.0:
        return "mid"
    return "high"


def _argmax_present(state: EndogenousState, features: Sequence[str]) -> str | None:
    """The strongest of a competing set, or nothing if none was answered for."""
    best: tuple[float, str] | None = None
    for name in features:
        index = FEATURE_INDEX.get(name)
        if index is None or not state.present[index]:
            continue
        value = float(state.values[index])
        if best is None or value > best[0]:
            best = (value, name.split(".", 1)[-1])
    return best[1] if best else None


@dataclass(frozen=True)
class CodeLine:
    """One line of the code, and where its content came from."""

    field: str
    value: str
    provenance: str  # "state" | "organ" | "head" | "abstained"

    def render(self) -> str:
        return f"{self.field:<12}{self.value}"


@dataclass(frozen=True)
class CognitiveCode:
    """A reading of z_Aura in symbols. Never language, and never presentable."""

    lines: tuple[CodeLine, ...]
    state_digest: str = ""
    coverage: float = 0.0
    interventions: tuple[str, ...] = ()

    #: Kept as a field rather than a constant so it travels with any record
    #: this code is serialised into. The proto-token generator learned this
    #: the hard way: "the path ran" and "its text may be shown to a person"
    #: are two different questions and need two different answers.
    is_user_presentable: bool = field(default=False, init=False)

    def render(self) -> str:
        return "\n".join(line.render() for line in self.lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lines": [
                {"field": line.field, "value": line.value, "provenance": line.provenance}
                for line in self.lines
            ],
            "state_digest": self.state_digest,
            "coverage": round(self.coverage, 4),
            "interventions": list(self.interventions),
            "is_user_presentable": False,
        }

    def get(self, field_name: str) -> str | None:
        for line in self.lines:
            if line.field == field_name:
                return line.value
        return None

    def abstained(self) -> tuple[str, ...]:
        return tuple(line.field for line in self.lines if line.provenance == "abstained")

    def diff(self, other: CognitiveCode) -> dict[str, tuple[str | None, str | None]]:
        """Which lines moved between two readings.

        This is what a state trajectory looks like from outside: z1 → z2 → z3
        printed as the fields that changed, with no generated text involved.
        """
        fields = {line.field for line in self.lines} | {line.field for line in other.lines}
        moved: dict[str, tuple[str | None, str | None]] = {}
        for name in sorted(fields):
            before, after = self.get(name), other.get(name)
            if before != after:
                moved[name] = (before, after)
        return moved


def _self_line(state: EndogenousState) -> CodeLine:
    parts = []
    for feature, label in (
        ("self.continuity", "continuity"),
        ("self.agency", "agency"),
        ("self.integrity", "integrity"),
    ):
        band = _band(state, feature)
        if band:
            parts.append(f"{label}-{band}")
    if not parts:
        return CodeLine("SELF", "absent", "abstained")
    return CodeLine("SELF", " ".join(parts), "state")


def _affect_line(state: EndogenousState) -> CodeLine:
    valence = _band(state, "affect.valence")
    arousal = _band(state, "affect.arousal")
    if valence is None and arousal is None:
        return CodeLine("APPRAISAL", "absent", "abstained")
    parts = []
    if valence:
        parts.append(f"valence-{valence}")
    if arousal:
        parts.append(f"arousal-{arousal}")
    curiosity = _band(state, "affect.curiosity")
    if curiosity == "high":
        parts.append("curious")
    return CodeLine("APPRAISAL", " ".join(parts), "state")


def _goal_line(state: EndogenousState) -> CodeLine:
    index = FEATURE_INDEX["goal.active"]
    if not state.present[index]:
        return CodeLine("GOAL", "absent", "abstained")
    if float(state.values[index]) < 0.5:
        return CodeLine("GOAL", "none-held", "state")
    parts = ["active"]
    priority = _band(state, "goal.priority")
    if priority:
        parts.append(f"priority-{priority}")
    if state.is_present("goal.blocked") and state.get("goal.blocked") >= 0.5:
        parts.append("blocked")
    if state.is_present("goal.conflict") and state.get("goal.conflict") >= 0.5:
        parts.append("conflicted")
    return CodeLine("GOAL", " ".join(parts), "state")


def _memory_line(state: EndogenousState) -> CodeLine:
    recall = _band(state, "memory.recall_hits")
    if recall is None:
        return CodeLine("MEMORY", "absent", "abstained")
    parts = [f"recall-{recall}"]
    confidence = _band(state, "memory.recall_confidence")
    if confidence:
        parts.append(f"grounding-{confidence}")
    if state.is_present("memory.contradiction") and state.get("memory.contradiction") >= 0.5:
        parts.append("contradiction")
    return CodeLine("MEMORY", " ".join(parts), "state")


def _uncertainty_line(state: EndogenousState) -> CodeLine:
    confidence = _band(state, "uncertainty.confidence")
    if confidence is None:
        return CodeLine("UNCERTAINTY", "absent", "abstained")
    inverted = {"low": "high", "mid": "mid", "high": "low"}[confidence]
    parts = [inverted]
    support = _band(state, "uncertainty.evidence_support")
    if support:
        parts.append(f"evidence-{support}")
    return CodeLine("UNCERTAINTY", " ".join(parts), "state")


def _attention_line(state: EndogenousState) -> CodeLine:
    focus = _band(state, "attention.focus")
    novelty = _band(state, "attention.novelty")
    if focus is None and novelty is None:
        return CodeLine("ATTENTION", "absent", "abstained")
    parts = []
    if focus:
        parts.append(f"focus-{focus}")
    if novelty:
        parts.append(f"novelty-{novelty}")
    return CodeLine("ATTENTION", " ".join(parts), "state")


def _recurrence_line(state: EndogenousState) -> CodeLine:
    depth = _band(state, "recurrence.depth")
    if depth is None:
        return CodeLine("RECURRENCE", "absent", "abstained")
    parts = [f"depth-{depth}"]
    convergence = _band(state, "recurrence.convergence")
    if convergence:
        parts.append(f"convergence-{convergence}")
    return CodeLine("RECURRENCE", " ".join(parts), "state")


def _temporal_line(state: EndogenousState) -> CodeLine:
    strongest = _argmax_present(
        state, ("temporal.past", "temporal.present", "temporal.future")
    )
    if strongest is None:
        return CodeLine("TEMPORAL", "absent", "abstained")
    horizon = _band(state, "temporal.horizon")
    value = strongest if horizon is None else f"{strongest} horizon-{horizon}"
    return CodeLine("TEMPORAL", value, "state")


def _service(name: str) -> Any:
    """The same organ lookup the state assembler uses, and for the same reason."""
    from core.brain.llm.endogenous_state import _service as resolve

    return resolve(name)


#: What reading a label off an organ may raise. An organ mid-update or with a
#: property that computes is expected; anything else belongs in a log.
_ACCESSOR_ERRORS = (
    AttributeError,
    KeyError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _labels_from(organ: Any, accessors: Sequence[str], limit: int) -> tuple[str, ...]:
    for accessor in accessors:
        candidate = getattr(organ, accessor, None)
        try:
            value = candidate() if callable(candidate) else candidate
        except _ACCESSOR_ERRORS as exc:
            logger.debug("organ accessor %s declined: %s", accessor, exc)
            continue
        if isinstance(value, Mapping):
            value = list(value.keys())
        if isinstance(value, (list, tuple)) and value:
            out = []
            for item in value[:limit]:
                text = str(item).strip().replace(" ", "-")[:32]
                if text:
                    out.append(text)
            if out:
                return tuple(out)
    return ()


def _concepts_line() -> CodeLine:
    """Which concepts are live. Read from the organs, never from the state.

    Seventy-four floats cannot encode which entities are active, so this line
    does not pretend to be a readout. It is marked ``organ`` and an experiment
    that ablates a state channel must not expect it to move.
    """
    for key in ("atomspace", "attention_manager", "workspace", "semantic_memory"):
        organ = _service(key)
        if organ is None:
            continue
        labels = _labels_from(
            organ,
            ("active_concepts", "focused_atoms", "top_attention", "active_entities"),
            MAX_CONCEPTS,
        )
        if labels:
            return CodeLine("CONCEPTS", " ".join(labels), "organ")
    return CodeLine("CONCEPTS", "absent", "abstained")


def _referents_line() -> CodeLine:
    for key in ("dialogue_state", "conversation_state", "referent_tracker"):
        organ = _service(key)
        if organ is None:
            continue
        labels = _labels_from(
            organ, ("active_referents", "referents", "salient_entities"), MAX_REFERENTS
        )
        if labels:
            return CodeLine("REFERENTS", " ".join(labels), "organ")
    return CodeLine("REFERENTS", "absent", "abstained")


@dataclass(frozen=True)
class IntentHead:
    """A fitted head over the state, choosing among ``SPEECH_ACTS``."""

    weights: np.ndarray  # (len(SPEECH_ACTS), STATE_DIM)
    bias: np.ndarray
    acts: tuple[str, ...] = SPEECH_ACTS
    #: Below this the head declines to choose. A head that always answers is
    #: a head that guesses in the middle, which is worse than the rule it
    #: replaces.
    min_probability: float = 0.5

    def predict(self, state: EndogenousState) -> tuple[str | None, float]:
        z = np.where(state.present, state.values, 0.0).astype(np.float64)
        logits = self.weights @ z + self.bias
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        if confidence < self.min_probability:
            return None, confidence
        return self.acts[best], confidence


def _intend_line(state: EndogenousState, head: IntentHead | None) -> CodeLine:
    if head is None:
        return CodeLine("INTEND", "abstained: no trained head", "abstained")
    act, confidence = head.predict(state)
    if act is None:
        return CodeLine(
            "INTEND", f"abstained: best {confidence:.2f} under floor", "abstained"
        )
    return CodeLine("INTEND", f"{act} ({confidence:.2f})", "head")


def read_code(
    state: EndogenousState,
    *,
    intent_head: IntentHead | None = None,
    include_organ_lines: bool = True,
) -> CognitiveCode:
    """Read z_Aura as a cognitive code, before anything is generated."""
    lines = [
        _self_line(state),
        _affect_line(state),
        _goal_line(state),
        _memory_line(state),
        _uncertainty_line(state),
        _attention_line(state),
        _recurrence_line(state),
        _temporal_line(state),
    ]
    if include_organ_lines:
        lines.extend([_concepts_line(), _referents_line()])
    lines.append(_intend_line(state, intent_head))
    return CognitiveCode(
        lines=tuple(lines),
        state_digest=state.digest,
        coverage=state.coverage,
        interventions=tuple(i.feature for i in state.interventions),
    )


__all__ = [
    "MAX_CONCEPTS",
    "MAX_REFERENTS",
    "SPEECH_ACTS",
    "CodeLine",
    "CognitiveCode",
    "IntentHead",
    "read_code",
]
