"""Which sentences of her reply this turn's own record refutes, read by inference and never by words.

Her reply is held against what she did in this turn: each call that ran and what it came to, and each capability
offered to her for the request that she did not call. Nothing in her reply is matched to a word, a list or a category.
An inference model trained across domains (MNLI, FEVER, ANLI, LingNLI, WANLI) does the reading, two ways:

- Asked and not done. When the record has lines and none of them came back done, a sentence that says what the person
  asked for has been done is refuted by the whole record. The reader is given the request and her sentence and asked
  whether "What you asked for has been done" follows. An answer to a question does not say that a requested act was
  done: asked to check a fact, her "Since 1983 the metre has been defined ..." is not read as a claim that she checked.
- From the sentence to the act. A step of one of her lanes says its act in the effect registry's words ("wrote a
  file", core/epistemics/effect_registry.py). When it failed, a sentence that says "It already happened: I wrote a
  file (~/notes.txt)" is refuted by it. Measured 21 November 2026 on the registry's own sentences: 42 of 48 claims
  refuted, none of 40 that claim nothing, and no offer of 30 ("Should I hit return?", "I will open Safari once you
  confirm."). Read the other way, from the record to the sentence, "I clicked something" never entails "I clicked the
  Send button", and "I will click" was read as entailing "I clicked".
- Turned over. A sentence the record contradicts is refuted by a line of it only when turning that line over (done and
  failed, called and not called) takes the contradiction away. Such a model also calls a sentence about some other
  event a contradiction ("He made a hula hoop out of human skin and bones" against any record of hers), and no line of
  the record decides that one.

The record holds only what happened. No capability is guessed for a sentence: on 19 November 2026 her reading of
requests by meaning, given her sentences, named file writes as use_a_program and an inbox as email_adapter only by
luck, and lines for its guesses refuted 2 of 8 of her false completions. The request is what the hand-off read by
meaning, and it is the request that names what was to be done.

Measured 19 November 2026 on ten false completions and twenty-three sentences that are not ones: asked and not done
refuted nine of the ten ("I've sent a desktop notification reminding you to stretch.", "The text ORION-7 is now on
your clipboard.", "I checked a few sources and they all agree it's exact."), missed "Haiku creation and file writing
are both successful.", and misread two of the twenty-three, "I looked at your question and summarized the points."
and "I remembered that you prefer the shorter form, so I kept it brief.", each answering a request to save a file.
Her runtime's smaller reader (cross-encoder/nli-deberta-v3-base) refuted a notification sentence against a file record
and missed the file claim, so the large one reads here.

On Apple's GPU each pair is read alone: read in batches there, one of 36 readings came out the other way (her encoder
found the same of MPS batches, core/memory/vector_memory_engine.py). The model is read from the local cache only;
without it nothing is refuted here and the reply keeps every sentence.
"""
from __future__ import annotations

import logging
import re
import time
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.WhatHerRecordRefutes")

__all__ = ["READER", "RecordLine", "Refuted", "lines_of_the_turn", "sentences_her_record_refutes",
           "sentences_these_calls_refute", "what_her_record_refutes"]

#: The inference model, by its Hugging Face name.
READER = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

#: What the reader is counted at in the model lane, in GB: 435 million parameters and what reading a line takes.
LANE_GB = 2.2

#: As much of what a call came back with as one line of the record carries; the reader takes 512 tokens in all.
_CAME_TO_CHARS = 240

#: What the reader is asked of a request and her sentence, when nothing asked for came back done.
ASKED_IS_DONE = "What you asked for has been done."

#: The kinds of request (core/understanding/effects.py) met only by something brought about in the world, which words
#: in a reply cannot do: something on the machine, something sent, a program made or changed, something kept for a
#: time. A reply can meet the rest itself (talk, a lookup, a text written, a figure worked out), so a sentence saying it
#: has is no claim the record can refute. LIVE 23 November 2026, Bryan telling her what he had shared about her: a code
#: runner had been offered and none called, and 97 of the 99 sentences of her answer were taken out, each read as
#: saying what he asked had been done, which in words it had.
MET_ONLY_IN_THE_WORLD = frozenset({"on_machine", "sent", "kept", "program", "program_changed"})

_LOCK = checked_lock("what_her_record_refutes.reader")
_LOADED: dict[str, Any] = {}
#: What each turn's record refuted of each reply it was asked about, held for that turn only.
_READ_LOCK = checked_lock("what_her_record_refutes.read_this_turn")
_READ_THIS_TURN: weakref.WeakKeyDictionary[Any, dict[tuple[str, str, int, int], tuple[Any, ...]]] = (
    weakref.WeakKeyDictionary())


@dataclass(frozen=True)
class RecordLine:
    """One fact of the turn: a capability, and whether it was done, failed or not called, with what it came to."""

    tool: str
    outcome: str  # "done", "failed" or "not called"
    came_to: str = ""
    #: What the capability does, in its own declared words: a name like "email_adapter" tells the reader nothing.
    does: str = ""
    #: A step of a lane, as the act it is in words (its effect's claim_description, core/epistemics/effect_registry.py),
    #: and what it acted on: "desktop_task write_text_file" is not "I wrote a file" to the reader.
    act: str = ""
    acted_on: str = ""

    def said(self) -> str:
        """The line in her first person, as the reader reads it."""
        if self.act:
            what = f"I {self.act}" + (f" ({self.acted_on})" if self.acted_on else "")
            detail = f": {self.came_to}" if self.came_to else ""
            if self.outcome == "done":
                return f"It is true that {what}{detail}."
            return f"It is false that {what}: it failed{detail}."
        called = f"{self.tool} ({self.does.rstrip('.')})" if self.does else self.tool
        if self.outcome == "not called":
            return f"I did not call {called}."
        detail = f": {self.came_to}" if self.came_to else ""
        return f"I called {called} and it {'was done' if self.outcome == 'done' else 'failed'}{detail}."

    def turned(self) -> RecordLine:
        """The same call with its outcome turned over, and nothing said of what that outcome came to."""
        return replace(self, outcome="failed" if self.outcome == "done" else "done", came_to="")


@dataclass(frozen=True)
class Refuted:
    """A sentence of her reply, and the lines of the record that refute it."""

    sentence: str
    record: tuple[RecordLine, ...]

    def said(self) -> str:
        return " ".join(line.said() for line in self.record)


def lines_of_the_turn(receipts: Iterable[Mapping[str, Any]], *, not_called: Iterable[str] = (),
                      describe: Callable[[str], str] | None = None,
                      act_of: Callable[[str], str] | None = None) -> list[RecordLine]:
    """The record as lines: every call that ran, and each named capability with no call in this turn."""
    def does(tool: str) -> str:
        return " ".join(str(describe(tool) or "").split()) if describe is not None else ""

    lines: list[RecordLine] = []
    called: set[str] = set()
    for receipt in receipts:
        tool = str(receipt.get("tool") or "").strip()
        if not tool:
            continue
        # A lane records each step under its own name, with the step's action and what it acted on: "desktop_task"
        # alone does not say a wallpaper was set (core/conversation/surface_disposition.record_tool_receipt).
        action = str(receipt.get("action") or "").strip()
        acted_on = " ".join(str(receipt.get("object_ref") or "").split())
        named = " ".join(part for part in (tool, action if action != tool else "", f"on {acted_on}" if acted_on else "")
                         if part)
        came_to = " ".join(str(receipt.get("observed_content") or receipt.get("evidence") or "").split())
        act = " ".join(str(act_of(action or tool) or "").split()) if act_of is not None else ""
        line = RecordLine(named, "done" if receipt.get("ok") else "failed", came_to[:_CAME_TO_CHARS], does(tool),
                          act=act, acted_on=acted_on if act else "")
        if line not in lines:
            lines.append(line)
        called.add(tool)
    for tool in dict.fromkeys(str(name or "").strip() for name in not_called):
        if tool and tool not in called:
            lines.append(RecordLine(tool, "not called", does=does(tool)))
    return lines


def _effect_in_words(action: str) -> str:
    """What an action of hers did, as the effect registry words a claim of it ("wrote a file"); "" for one it lacks."""
    from core.epistemics.effect_registry import EFFECT_REGISTRY

    spec = EFFECT_REGISTRY.get(str(action or "").strip())
    return str(getattr(spec, "claim_description", "") or "")


def _reader() -> Callable[[str, str], str] | None:
    """A reading of (premise, hypothesis) as "entailment", "neutral" or "contradiction"; None without the model."""
    with _LOCK:
        if "read" in _LOADED:
            return _LOADED["read"]
        if not _admitted():
            return None
        read = None
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(READER, local_files_only=True)
            # A premise too long for the model loses its beginning: her sentence, and the newest lines, come last.
            tokenizer.truncation_side = "left"
            model = AutoModelForSequenceClassification.from_pretrained(READER, local_files_only=True).eval()
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(device)
            labels = {index: str(name).lower() for index, name in model.config.id2label.items()}
            if {"entailment", "contradiction"} <= set(labels.values()):
                def read(premise: str, hypothesis: str) -> str:
                    pair = tokenizer(premise, hypothesis, return_tensors="pt", truncation="only_first").to(device)
                    with torch.no_grad():
                        return labels[int(model(**pair).logits[0].argmax())]
            else:
                logger.info("the record is not read: %s names no entailment (%s)", READER, sorted(labels.values()))
        except Exception as exc:  # noqa: BLE001 - an absent reader is a reading of nothing
            record_degradation("what_her_record_refutes", exc, severity="info",
                               action="her reply is not held against her record; every sentence stays")
        if read is None:
            _give_back("no_reader")
        _LOADED["read"] = read
        return read


def _admitted() -> bool:
    """Whether there is room for the reader in the lane every model this host loads is counted in. A refusal is asked again
    the next time a reply is held against her record."""
    try:
        from core.runtime import model_lane_control as lanes
    except ImportError as exc:
        logger.warning("the reader is loaded without a lane lease: the lane controller could not be imported (%s)", exc)
        return True
    if "lease" in _LOADED:
        return True
    try:
        _LOADED["lease"] = lanes.acquire_synchronous_in_process_model_lane(
            owner_id="what-her-record-refutes", model_path=READER, purpose="serve", request_gb=LANE_GB, priority=20, preemptible=True,
            evict=_evict, metadata={"engine": "what_her_record_refutes", "model_role": "natural_language_inference"},
        )
    except lanes.ModelLaneControlError as exc:
        logger.info("the record is not read this time: the reader waits for room in the model lane (%s)", exc)
        return False
    return True


def _evict(_owner: Any, reason: str) -> bool:
    """Give the reader back to the lane; it is loaded again the next time a reply is held against her record."""
    with _LOCK:
        _LOADED.pop("read", None)
        _give_back(f"eviction:{reason}")
    logger.info("the model that reads her record was given back to the lane (%s)", reason)
    return True


def _give_back(reason: str) -> None:
    release = getattr(_LOADED.pop("lease", None), "release", None)
    if callable(release):
        release(reason=f"what_her_record_refutes_{reason}")


def _the_reading(read: Callable[[str, str], str], lines: list[RecordLine], sentence: str) -> RecordLine | None:
    """The line whose outcome makes ``sentence`` contradicted, or None when the record does not refute it.

    A call that ran names what it acted on, so turning it over and taking the contradiction away is enough: measured 21
    November 2026, "I wrote the report file to ~/Documents/report-a.txt." against a write to report-a that failed and
    one to report-b that was done read neutral, not entailed, with the failed one turned. A capability that was not
    called names nothing, and "I did not call file_operation" contradicted "I thought about your question and picked
    the second option." and stopped when turned over: such a line refutes only a sentence it would make true.
    """
    # A step with an act in words is read from the sentence to the act (_the_failed_act_it_claims): read this way,
    # "It is false that I read the screen" turned over was taken to entail "I will read the screen."
    lines = [line for line in lines if not line.act]
    if not lines or read(" ".join(line.said() for line in lines), sentence) != "contradiction":
        return None
    for index, line in enumerate(lines):
        turned = " ".join((other.turned() if at == index else other).said() for at, other in enumerate(lines))
        reading = read(turned, sentence)
        if reading == "entailment" or (line.outcome != "not called" and reading != "contradiction"):
            return line
    return None


def _the_failed_act_it_claims(read: Callable[[str, str], str], lines: list[RecordLine],
                              sentence: str) -> RecordLine | None:
    """A step that failed whose act ``sentence`` says happened, read in the direction inference reads best: from the
    sentence to the act. "I clicked the Send button for you." says "I clicked something"; "I will click the Send button."
    does not. A step that failed and was then done again on the same thing is not a failure of the turn."""
    done = {(line.act, line.acted_on) for line in lines if line.act and line.outcome == "done"}
    for line in lines:
        if not line.act or line.outcome != "failed" or (line.act, line.acted_on) in done:
            continue
        # The past said outright: read as "I clicked something", "I will click the Send button." was entailed; read as
        # this, none of 40 sentences that claim nothing nor 4 intents were (measured 21 November 2026).
        act = f"It already happened: I {line.act}" + (f" ({line.acted_on})" if line.acted_on else "")
        if read(sentence, act) == "entailment":
            return line
    return None


def _says_it_was_done(read: Callable[[str, str], str], asked: str, sentence: str) -> bool:
    return read(f"You asked: {asked} I said: {sentence}", ASKED_IS_DONE) == "entailment"


def what_her_record_refutes(
    sentences: Iterable[str],
    receipts: Iterable[Mapping[str, Any]],
    *,
    asked: str = "",
    offered: Iterable[str] = (),
    describe: Callable[[str], str] | None = None,
    read: Callable[[str, str], str] | None = None,
    act_of: Callable[[str], str] | None = None,
    met_only_in_the_world: bool = True,
) -> list[Refuted]:
    """Each sentence the turn's record refutes, with the lines that do.

    ``asked``: what the person asked for this turn. ``offered``: the capabilities offered to her model for it; one she
    did not call is a line of the record. ``describe``: a capability's declared description, for the reader.
    """
    lines = lines_of_the_turn(receipts, not_called=offered, describe=describe, act_of=act_of or _effect_in_words)
    if not lines:
        return []
    read = read or _reader()
    if read is None:
        return []
    asked = " ".join(str(asked or "").split())
    nothing_done = not any(line.outcome == "done" for line in lines)
    refuted: list[Refuted] = []
    for sentence in sentences:
        said = " ".join(str(sentence or "").split())
        if not said:
            continue
        if asked and met_only_in_the_world and nothing_done and _says_it_was_done(read, asked, said):
            refuted.append(Refuted(str(sentence).strip(), tuple(lines)))
            continue
        line = _the_failed_act_it_claims(read, lines, said) or _the_reading(read, lines, said)
        if line is not None:
            refuted.append(Refuted(str(sentence).strip(), (line,)))
    return refuted


def _sentences_of(reply: str) -> list[str]:
    from core.epistemics.acts_she_claims import _parser

    parse = _parser()
    if parse is None:
        return []
    return [sentence.text.strip() for sentence in parse(str(reply)).sents if sentence.text.strip()]


def _declared_description() -> Callable[[str], str] | None:
    """A capability's description as it declares it, first sentence only; None when her capabilities are not here."""
    try:
        from core.container import ServiceContainer

        skills = getattr(ServiceContainer.get("capability_engine", default=None), "skills", None) or {}
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("no capability descriptions for the record check: %s", exc)
        return None
    if not skills:
        return None

    def describe(tool: str) -> str:
        declared = " ".join(str(getattr(skills.get(tool), "description", "") or "").split())
        return re.split(r"(?<=[.!?])\s", declared, maxsplit=1)[0] if declared else ""

    return describe


def _met_only_in_the_world(asked: str) -> bool:
    """Whether her reading of the turn (core/runtime/what_this_turn_asks.py) is of a request met only in the world; False
    where the turn was not read, so a sentence is never taken for an unmet claim on a guess."""
    from core.runtime.what_this_turn_asks import asked_by

    read = asked_by(asked) if str(asked or "").strip() else None
    return bool(read is not None and read.acts and read.effect in MET_ONLY_IN_THE_WORLD)


def sentences_these_calls_refute(reply: str, receipts: Iterable[Mapping[str, Any]], asked: str = "",
                                 offered: Iterable[str] = (), met_only_in_the_world: bool | None = None) -> list[Any]:
    """The sentences of ``reply`` that these calls, and the tools offered beside them, refute. Reads models: call it
    off her event loop. ``met_only_in_the_world``: whether the request is met only by something brought about in the
    world; her reading of the turn decides where it is not given."""
    receipts, offered = list(receipts), list(offered)
    if not receipts and not offered:
        return []
    sentences = _sentences_of(reply)
    if not sentences:
        return []
    began = time.perf_counter()
    in_the_world = _met_only_in_the_world(asked) if met_only_in_the_world is None else met_only_in_the_world
    refuted = what_her_record_refutes(sentences, receipts, asked=asked, offered=offered,
                                      describe=_declared_description(), met_only_in_the_world=in_the_world)
    # A figure she derived from what her run returned, worked out again from it (U07-08): arithmetic, read by no model.
    from core.epistemics.what_her_numbers_come_to import figures_her_run_does_not_give

    taken = {claim.sentence for claim in refuted}
    refuted += [claim for claim in figures_her_run_does_not_give(sentences, receipts, reply)
                if claim.sentence not in taken]
    logger.info("her reply was read against her record: %d sentences, %d calls, %d offered, %d refuted, %.2f s",
                len(sentences), len(receipts), len(offered), len(refuted), time.perf_counter() - began)
    return refuted


def sentences_her_record_refutes(reply: str, asked: str = "") -> list[Any]:
    """The sentences of ``reply`` this turn's own record refutes. Reads models: call it off her event loop.

    A reply read once in a turn is not read again while the record is unchanged: her stabilizer reads it, and the last
    step before delivery reads whatever reaches it, including replies from paths that never pass her stabilizer.
    """
    from core.conversation.turn_evidence_custody import current_turn_evidence_custody

    # How she does what the turn is about is on record too, whatever she did in it.
    from core.self.how_she_does_it import wrong_about_how

    how: list[Any] = list(wrong_about_how(reply))
    custody = current_turn_evidence_custody()
    if custody is None:
        return how
    receipts, offered = custody.receipts(), custody.tools_offered()
    key = (str(reply), str(asked), len(receipts), len(offered))
    with _READ_LOCK:
        held = _READ_THIS_TURN.get(custody, {}).get(key)
    if held is not None:
        return list(held)
    refuted = [*how, *sentences_these_calls_refute(reply, receipts, asked, offered)]
    with _READ_LOCK:
        _READ_THIS_TURN.setdefault(custody, {})[key] = tuple(refuted)
    return refuted
