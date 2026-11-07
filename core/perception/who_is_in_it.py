"""Who is in what she watches: the people named in what she hears and in what her eyes say of it.

Caring about the people in a story is much of why it holds an audience: viewers enjoy most when liked characters fare
well, fear for a liked character is what suspense is made of, and a character is liked mostly for how they act
(Zillmann's affective disposition theory; Raney). To care about someone in it she has to know who is in it, and her
senses give her words, not people. This finds the people named in those words, with a small model trained for it
(dslim/distilbert-NER: DistilBERT fine-tuned on CoNLL-2003, Apache-2.0, 66M parameters), on the CPU, off the event
loop, from the weights kept on this machine. Nothing is asked of her model; the names are read, as her ear writes words
down and her eyes read the words on a picture.

A name and a longer name it is part of are one person ("Allen" is Allen Pan once "Allen Pan" has been heard).
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("Aura.Perception.WhoIsInIt")

__all__ = ["NER_MODEL", "people_in", "same_person"]

NER_MODEL = "dslim/distilbert-NER"
#: How sure the model must be that a name is a person's.
SURE_OF_A_PERSON = 0.6
#: The least memory the host must have free for the model to be loaded, in bytes: it and the library it runs on take a
#: few hundred megabytes, and her own model is refused generation when the host runs short. Below it, nothing is read
#: and it is tried again later.
ROOM_NEEDED = 3 * 1024**3

_LOCK = threading.RLock()
_READER: dict[str, Any] = {}


def _reader() -> Any | None:
    """The model, loaded once from this machine's copy of its weights; None where it is not here."""
    with _LOCK:
        if "pipeline" in _READER or "failed" in _READER:
            return _READER.get("pipeline")
        if not _room_for_it():
            return None
        try:
            from core.runtime import model_lane_control as lanes
        except ImportError:
            lanes = None
        if lanes is not None and "lease" not in _READER:
            # Held through the lane like every model this host loads, so the lane controller counts its memory and can
            # take it back; it is loaded again the next time it is needed. A refusal is tried again later.
            try:
                _READER["lease"] = lanes.acquire_synchronous_in_process_model_lane(
                    owner_id="who-is-in-it",
                    model_path=NER_MODEL,
                    purpose="serve",
                    request_gb=LANE_GB,
                    priority=20,
                    preemptible=True,
                    evict=_evict,
                    metadata={"engine": "who_is_in_it", "model_role": "named_people", "device": "cpu"},
                )
            except lanes.ModelLaneControlError as exc:
                logger.info("who is in what she watches waits for room in the model lane: %s", exc)
                return None
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(NER_MODEL, local_files_only=True)
            _READER["pipeline"] = pipeline("token-classification", model=model, tokenizer=tokenizer,
                                           aggregation_strategy="first", device="cpu")
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            from core.runtime.errors import record_degradation

            _READER["failed"] = str(exc)
            _give_back(_READER.pop("lease", None), "load_failed")
            record_degradation("who_is_in_it", exc, severity="info", action="who is in what she watches is not read")
            return None
        return _READER["pipeline"]


#: What the model and its library are counted at in the model lane, in GB.
LANE_GB = 0.6


def _evict(_owner: Any, reason: str) -> bool:
    """Give the model back to the lane; it is loaded again the next time a name is read."""
    with _LOCK:
        _READER.pop("pipeline", None)
        lease = _READER.pop("lease", None)
    _give_back(lease, f"eviction:{reason}")
    logger.info("The model that reads who is named was given back to the lane (%s).", reason)
    return True


def _give_back(lease: Any, reason: str) -> None:
    release = getattr(lease, "release", None)
    if callable(release):
        release(reason=f"who_is_in_it_{reason}")


def _room_for_it() -> bool:
    try:
        from core.runtime import resource_psutil as psutil

        return float(psutil.virtual_memory().available) >= ROOM_NEEDED
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return True


def people_in(text: str) -> list[str]:
    """The people named in ``text``, as they are named, each once, in order; [] where none is, or the model is not
    here."""
    text = " ".join(str(text or "").split())
    if not text:
        return []
    reader = _reader()
    if reader is None:
        return []
    try:
        found = reader(text[:1000])
    except (RuntimeError, ValueError, IndexError) as exc:
        logger.debug("who is named was not read: %s", exc)
        return []
    named: list[str] = []
    for entity in found:
        name = " ".join(str(entity.get("word") or "").replace("##", "").split())
        if entity.get("entity_group") == "PER" and float(entity.get("score") or 0.0) >= SURE_OF_A_PERSON and len(name) > 1:
            if name.lower() not in {other.lower() for other in named}:
                named.append(name)
    return named


def same_person(one: str, other: str) -> bool:
    """Whether two names are one person: equal, or every word of the shorter is a word of the longer."""
    a, b = set(one.lower().split()), set(other.lower().split())
    return bool(a and b) and (a <= b or b <= a)
