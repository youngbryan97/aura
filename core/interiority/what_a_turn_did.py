"""core/interiority/what_a_turn_did.py — what a person's turn did, written where the appraisal reads it.

Measured on 17 October: six messages ("Thank you, that was genuinely brilliant
work.", "You got it wrong again. This is useless.", "My dog died this
morning.", "I just got the job!!", "What time is it?", "No, it's 1049.") went
through the interior layer and came out with the same valence, +0.112, and
relevance 0.0. Over 9,334 recorded turns her valence stayed between 0.00 and
0.10. The appraisal is computed from the relational ledger (what the event
advances or blocks, who caused it, whether a standard was broken), and for a
person's turn nothing wrote those notes: the event's object was the message's
own first 64 characters, a new string every turn, and ``note_attribution`` and
``note_norm_judgement`` had no writer outside the tests.

This module is that writer. It reads a turn three ways, none of them a word
list and none of them a prompt:

* what the person expressed, from a classifier over her own embedding model
  trained on GoEmotions (Demszky et al. 2020: 58k Reddit comments, 27 emotions
  and neutral, labelled by people), with a held-out score in its report;
* whether it was addressed to her, from the second-person rate the text
  features already measure;
* whether it corrected her, from the learning-in-use path that already notices
  "No, it's 1049."

and writes, for the event:

* a correction: the object is her last answer, it is blocked (goal delta -1),
  and she caused it;
* praise or blame addressed to her: the object is her last answer, its goal
  delta is the valence of what was expressed, and the cause is split between
  her (judgement of her work: disapproval, disappointment) and them (hostility:
  anger, annoyance, disgust), with hostility also a breach of how she is to be
  treated;
* anything else: the object is the person's wellbeing, its goal delta is the
  valence of what they expressed, the cause is circumstance, and their distress
  goes on the ``context`` channel the other-minds reader takes it from. What is
  good or bad for someone she is attached to is congruent or not with her care
  for them (Ortony, Clore and Collins 1988: happy-for, sorry-for); with no bond
  the relevance check finds no stake and nothing moves, which is correct.

The valence of an emotion follows the paper's own grouping into positive,
negative and ambiguous.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from core.interiority.event import InteriorEvent
from core.interiority.evidence import measured
from core.interiority.text_features import statistics

logger = logging.getLogger("Aura.Interiority.TurnReading")

#: GoEmotions' labels in the dataset's own order (simplified configuration).
LABELS: tuple[str, ...] = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
)

#: Demszky et al. 2020, section 4 (sentiment grouping of the taxonomy).
POSITIVE = frozenset({"admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude",
                      "joy", "love", "optimism", "pride", "relief"})
NEGATIVE = frozenset({"anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment",
                      "fear", "grief", "nervousness", "remorse", "sadness"})

#: Of the negative feelings, those that are hostility toward the one addressed, and those that are a
#: judgement of what that one did. The split decides whether blame lands on her or on them.
HOSTILE = frozenset({"anger", "annoyance", "disgust"})
JUDGING = frozenset({"disapproval", "disappointment"})
#: What is distress in the one speaking: what happened to them, not what they think of her.
DISTRESS = frozenset({"sadness", "grief", "fear", "nervousness", "remorse", "embarrassment", "disappointment"})

#: Where the trained reader lives, under her data directory.
HEAD_RELATIVE = "interiority/turn_reader_head.npz"
#: The embedding instruction the reader was trained under (core/memory/embedding_model.py).
TASK = "feeling_in_a_message"


@dataclass(frozen=True)
class TurnReading:
    """What a turn expressed and whom it was for, with how sure the reading is."""

    feelings: Mapping[str, float] = field(default_factory=dict)
    corrected_her: bool = False
    addressed_to_her: bool = False
    confidence: float = 0.0

    @property
    def read(self) -> bool:
        return bool(self.feelings)

    def _mass(self, labels: frozenset[str]) -> float:
        return sum(p for name, p in self.feelings.items() if name in labels)

    @property
    def valence(self) -> float | None:
        """Positive mass less negative mass, over what carries either; None when nothing was read."""
        if not self.feelings:
            return None
        positive, negative = self._mass(POSITIVE), self._mass(NEGATIVE)
        total = positive + negative
        return 0.0 if total <= 0.0 else (positive - negative) / total

    @property
    def hostile_share(self) -> float:
        hostile, judging = self._mass(HOSTILE), self._mass(JUDGING)
        return 0.0 if hostile + judging <= 0.0 else hostile / (hostile + judging)

    @property
    def distress(self) -> float:
        return min(1.0, self._mass(DISTRESS))


class TurnReader:
    """A linear multi-label head over her embedding of a message, as trained by tools/train_turn_reader.py."""

    def __init__(self, weights: Any, bias: Any, labels: tuple[str, ...], *, embedding_model: str,
                 report: Mapping[str, Any] | None = None) -> None:
        import numpy as np

        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.labels = tuple(labels)
        self.embedding_model = embedding_model
        self.report = dict(report or {})
        if self.weights.ndim != 2 or self.weights.shape[0] != len(self.labels) \
                or self.bias.shape != (len(self.labels),):
            raise ValueError("turn reader head does not match its labels")

    def feelings(self, vector: Any) -> dict[str, float]:
        import numpy as np

        v = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if norm == 0.0 or v.shape[0] != self.weights.shape[1]:
            return {}
        logits = self.weights @ (v / norm) + self.bias
        return {name: 1.0 / (1.0 + math.exp(-float(z))) for name, z in zip(self.labels, logits, strict=True)}

    def save(self, path: Path) -> Path:
        import json

        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, bias=self.bias, labels=np.array(self.labels),
                 embedding_model=np.array(self.embedding_model), report=np.array(json.dumps(self.report)))
        return path

    @classmethod
    def load(cls, path: Path) -> TurnReader | None:
        import json

        import numpy as np

        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return cls(data["weights"], data["bias"], tuple(str(x) for x in data["labels"]),
                       embedding_model=str(data["embedding_model"]),
                       report=json.loads(str(data["report"])))


def head_path() -> Path:
    """Where her reader lives: beside the interior ledger, under her state root."""
    from core.runtime.state_ownership import state_root

    return state_root() / "data" / HEAD_RELATIVE


def her_turn_reader() -> TurnReader | None:
    """Her trained reader, or None when none has been trained on this machine."""
    try:
        return TurnReader.load(head_path())
    except (OSError, ValueError, KeyError) as exc:
        logger.debug("no turn reader: %s", exc)
        return None


def read_turn(message: str, *, reader: TurnReader | None, embed: Callable[[str], Any] | None,
              corrected_her: bool = False) -> TurnReading:
    """What ``message`` expressed and whether it was addressed to her or corrected her."""
    # Addressed to her when it speaks to someone: in a conversation with her, that someone is her.
    addressed = statistics(message).second_person_rate > 0.0
    feelings: dict[str, float] = {}
    if reader is not None and embed is not None and message.strip():
        vector = embed(message)
        if vector is not None:
            feelings = reader.feelings(vector)
    # How sure the reading is: the classifier's own strongest probability; a correction is a fact.
    confidence = max(feelings.values()) if feelings else (1.0 if corrected_her else 0.0)
    return TurnReading(feelings=feelings, corrected_her=corrected_her, addressed_to_her=addressed,
                       confidence=confidence)


def note_what_it_did(ledger: Any, event: InteriorEvent, reading: TurnReading, *, person: str) -> InteriorEvent:
    """Write what the turn did into the ledger's notes and return the event pointed at it.

    ``ledger`` is the relational ledger (``get_interiority().ledger``); the notes go
    on ``ledger.notes``. Nothing is written when nothing was read.
    """
    notes = ledger.notes
    observations = dict(event.observations)
    if reading.corrected_her:
        pointed = replace(event, object="her_last_answer")
        notes.note_goal_delta("her_last_answer", -1.0)
        notes.note_attribution(pointed.event_id, own=1.0, other=0.0, circumstance=0.0)
        return pointed
    valence = reading.valence
    if valence is None:
        return event
    if reading.addressed_to_her and abs(valence) > 0.0:
        pointed = replace(event, object="her_last_answer")
        notes.note_goal_delta("her_last_answer", valence)
        if valence < 0.0:
            other = reading.hostile_share
            notes.note_attribution(pointed.event_id, own=1.0 - other, other=other, circumstance=0.0)
            if other > 0.0:
                notes.note_norm_judgement(pointed.event_id, fit=-other, endorsement=reading.confidence)
        else:
            notes.note_attribution(pointed.event_id, own=1.0, other=0.0, circumstance=0.0)
        return pointed
    wellbeing = f"{person or 'them'}:wellbeing"
    notes.note_goal_delta(wellbeing, valence)
    if reading.distress > 0.0:
        observations["context"] = measured(reading.distress, source="turn_reader:distress",
                                           confidence=reading.confidence)
    pointed = replace(event, object=wellbeing, observations=observations)
    notes.note_attribution(pointed.event_id, own=0.0, other=0.0, circumstance=1.0)
    return pointed


__all__ = ["DISTRESS", "HEAD_RELATIVE", "head_path", "HOSTILE", "JUDGING", "LABELS", "NEGATIVE", "POSITIVE", "TASK",
           "TurnReader", "TurnReading", "her_turn_reader", "note_what_it_did", "read_turn"]
