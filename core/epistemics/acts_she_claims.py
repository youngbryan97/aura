"""What she says she did, read by its grammar and its meaning, and checked against what this turn's record shows.

core/epistemics/unevidenced_action.py decides whether a sentence claims an act in the world from lists: verbs of
thinking and speaking, irregular past forms, nouns that name things in the world, patterns for "if" and "would". Each
live miss added to a list (16 November 2026, "sent", "notification", "reminder" and "alert" after "I've sent a desktop
notification" passed with nothing sent). A list of words does not know what a sentence says.

Here a claim is found by the sentence's structure and named by what it means:

- Grammar (a dependency parse): she is the subject ("I", "we"), the verb is completed (past, or perfect with "have"),
  and nothing makes it less than an assertion of her own: no negation, no modal, no condition ("if", "unless",
  "whether"), no question, no quotation marks around it, and no other speaker whose words it is reported in.
- Meaning: the act is the verb with what it takes, as she wrote it ("sent a desktop notification reminding you to
  stretch"), and which of her capabilities it is, if any, is read by the same reading that takes a person's request
  to a capability (core/intent/capability_by_meaning.py), with its abstention. An act that is no capability of hers
  (thinking, telling, noticing) names none and asks no record.
- Record: an act that names a capability is backed when this turn holds a receipt of that capability that came back
  done.

The parse needs spaCy and its small English model; without them nothing is read here and the caller keeps what it had.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.ActsSheClaims")

__all__ = ["ClaimedAct", "acts_she_claims", "claims_her_record_does_not_back"]

#: The English model the parse reads with.
PARSER = "en_core_web_sm"

_PARSER_LOCK = checked_lock("acts_she_claims.parser")
_PARSER_HELD: list[Any] = []


@dataclass(frozen=True)
class ClaimedAct:
    """One act she says she did: the sentence, the act as she wrote it, and the capability it means ("" if none)."""

    sentence: str
    act: str
    capability: str = ""


def _parser() -> Any | None:
    with _PARSER_LOCK:
        if not _PARSER_HELD:
            try:
                import spacy

                _PARSER_HELD.append(spacy.load(PARSER, disable=["ner"]))
            except (ImportError, OSError) as why:
                logger.info("acts she claims are not read: no %s parser (%s)", PARSER, why)
                _PARSER_HELD.append(None)
        return _PARSER_HELD[0]


def _hers(token: Any) -> bool:
    """A subject that is the speaker: a first-person personal pronoun in the nominative."""
    morph = token.morph
    return (token.dep_ in ("nsubj", "nsubjpass") and morph.get("Person") == ["1"]
            and morph.get("PronType") == ["Prs"])


def _clause_of(verb: Any) -> list[Any]:
    """The verb and the verbs it is conjoined to, nearest first: "I opened Safari and searched" shares one subject,
    one negation and one modal across both verbs."""
    chain = [verb]
    while chain[-1].dep_ == "conj" and chain[-1].head is not chain[-1]:
        chain.append(chain[-1].head)
    return chain


def _whose(verb: Any) -> str:
    """Whose act the verb asserts: "her" (a first-person subject, or a passive with no agent but her), "another", or
    "" when the clause has no subject to read."""
    for clause in _clause_of(verb):
        children = list(clause.children)
        subjects = [child for child in children if child.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            continue
        if any(_hers(subject) for subject in subjects):
            return "her"
        passive = any(child.dep_ == "auxpass" for child in children)
        agents = [token for child in children if child.dep_ == "agent" for token in child.subtree
                  if token.dep_ == "pobj"]
        if passive and (not agents or any(agent.lower_ == "me" for agent in agents)):
            return "her"
        return "another"
    return ""


def _said_by_another(verb: Any) -> bool:
    """Whether the clause sits inside what someone else said or thought ("you said I sent it")."""
    for above in verb.ancestors:
        subjects = [child for child in above.children if child.dep_ in ("nsubj", "nsubjpass")]
        if subjects and not any(_hers(subject) for subject in subjects):
            return True
    return False


def _completed(verb: Any) -> bool:
    """A finished act asserted as hers: past, or perfect, with no negation, modal or condition on it."""
    children = [child for clause in _clause_of(verb) for child in clause.children]
    if any(child.dep_ == "neg" for child in children):
        return False
    if any(child.dep_ in ("aux", "auxpass") and child.tag_ == "MD" for child in children):
        return False
    # Finished: "sent", "have sent", "has been sent", "was sent". Not "is sent", "is being sent" or "is sending".
    perfect = verb.tag_ == "VBN" and any(
        (child.dep_ == "aux" and child.lemma_ == "have") or (child.dep_ == "auxpass" and child.tag_ == "VBD")
        for child in children)
    if not (perfect or verb.tag_ == "VBD"):
        return False
    return not any(child.dep_ == "mark" and child.lower_ in ("if", "unless", "whether")
                   for clause in (verb, *verb.ancestors) for child in clause.children)


def _quoted(token: Any, sentence: Any) -> bool:
    """Whether ``token`` stands between quotation marks in its sentence."""
    before = sentence.doc[sentence.start:token.i].text
    return sum(before.count(mark) for mark in ('"', "“", "”")) % 2 == 1


def _the_act(verb: Any) -> str:
    """The verb with what it takes, in her words: its objects, particles, prepositions and complements."""
    # A passive's subject is what was acted on ("the file has been saved"), so it is part of the act.
    takes = {"dobj", "obj", "dative", "prt", "prep", "attr", "oprd", "xcomp", "ccomp", "acomp", "advmod", "npadvmod",
             "nsubjpass"}
    words = {verb}
    for child in verb.children:
        if child.dep_ in takes:
            words.update(child.subtree)
    return " ".join(token.text for token in sorted(words, key=lambda token: token.i))


def acts_she_claims(text: str) -> list[tuple[str, str]]:
    """Each act ``text`` asserts she did, as (sentence, act); [] when there is no parser."""
    parse = _parser()
    if parse is None or not str(text or "").strip():
        return []
    found: list[tuple[str, str]] = []
    for sentence in parse(str(text)).sents:
        said = sentence.text.strip()
        if said.endswith("?"):
            continue
        for verb in sentence:
            if verb.pos_ not in ("VERB", "AUX") or _whose(verb) != "her":
                continue
            if _completed(verb) and not _said_by_another(verb) and not _quoted(verb, sentence):
                found.append((said, _the_act(verb)))
    return found


def _receipts_done(receipts: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(receipt.get("tool") or "") for receipt in receipts if receipt.get("ok")}


def claims_her_record_does_not_back(
    text: str,
    receipts: Iterable[Mapping[str, Any]],
    meant: Callable[[str], str | None],
) -> list[ClaimedAct]:
    """The acts ``text`` asserts she did that name a capability with no receipt of it done in ``receipts``.

    ``meant`` reads an act to the capability it means, or None when none clearly is (her reading of requests by
    meaning). An act that names no capability is not a claim the record can back or refute, and is left alone.
    """
    done = _receipts_done(receipts)
    unbacked: list[ClaimedAct] = []
    for sentence, act in acts_she_claims(text):
        capability = str(meant(act) or "")
        if capability and capability not in done:
            unbacked.append(ClaimedAct(sentence, act, capability))
    return unbacked
