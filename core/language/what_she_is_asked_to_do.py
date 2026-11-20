"""The parts of what a person said that ask her to act, read from the grammar of who is to act.

Naming a game, an app or a page in a sentence does not ask her to open it. "I think you got 2048 in 968 moves. What
are the chances I can beat you?" names a game she plays, and on 19 November 2026 it opened 2048 and she started
playing: the recogniser that reads a goal to keep at (watched_goal) found the game, and nothing asked whether the
person had asked her to do anything. Whether a sentence is an instruction is not in its words but in its grammar:
who is to act, and how the sentence puts it to her.

A clause asks her to act when the one who is to act is her:

- an imperative, whose missing subject is the person addressed ("beat Stockfish for me", "open Notes"),
- a request put to her as a question with a modal ("can you open Safari", "would you play it"), not a question
  about what she thinks ("do you think you could beat it" asks for an opinion, through "do"),
- what the speaker wants her to do ("I want you to play", "you should look them up").

A clause whose subject is the speaker or someone else, a report of what was done, or a question about chances,
reasons or opinions asks her nothing; the game or app named in it is what is being talked about. The dependency
parse that reads this is the one her claims checking already uses (core/epistemics/acts_she_claims.py).
"""
from __future__ import annotations

import logging
from typing import Any

from core.language.english_grammar import parse as _parse

logger = logging.getLogger("Aura.WhatSheIsAsked")

__all__ = ["the_parts_that_ask_her", "the_sentences_that_ask_her"]


def _person(token: Any) -> str:
    person = token.morph.get("Person")
    return person[0] if person else ""


def _subject(verb: Any) -> Any | None:
    return next((c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")), None)


def _the_addressee(token: Any) -> bool:
    return token is not None and _person(token) == "2"


def _asks_her(verb: Any) -> bool:
    """Whether this verb's clause puts an act to the person addressed."""
    subject = _subject(verb)
    auxiliaries = [c for c in verb.children if c.dep_ in ("aux", "auxpass")]
    if any(c.dep_ == "neg" for c in verb.children):
        return False  # "don't open it" asks her not to act
    if subject is None and verb.tag_ == "VB" and all(a.tag_ == "VB" for a in auxiliaries):
        # An imperative: the one to act is the one addressed. A helper in its bare form belongs to it ("go play
        # chess"); a tensed or modal one makes it a statement or a question.
        return True
    if _the_addressee(subject):
        # A modal put to her: "can you", "would you", "you should". "Do you think ..." is a question about her mind,
        # and has "do" for its auxiliary, not a modal.
        return any(a.tag_ == "MD" for a in auxiliaries)
    if subject is not None and _person(subject) == "1":
        # What the speaker wants her to do: "I want you to play it", "I need you to look". The act wanted is in its
        # bare form; "I think you got 2048" reports one, finished.
        for child in verb.children:
            if child.dep_ in ("xcomp", "ccomp") and child.pos_ in ("VERB", "AUX") and child.tag_ == "VB":
                agent = _subject(child) or next((c for c in verb.children if c.dep_ in ("dobj", "obj")), None)
                if _the_addressee(agent):
                    return True
    return False


#: Clauses that hang off the act asked: what it is said of, when, or what condition ends it. They are not the act;
#: "Remember when you played 2048" asks her to remember, and the game is in what is remembered.
_HANGING = ("advcl", "ccomp", "relcl", "acl", "csubj", "parataxis")


def _clause_text(verb: Any) -> str:
    """The act asked, in its own words: the verb's subtree without clauses conjoined to it (read on their own) and
    without the clauses that hang off it."""
    skip = {t.i for c in verb.children
            if (c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")) or c.dep_ in _HANGING for t in c.subtree}
    return " ".join(t.text for t in verb.subtree if t.i not in skip).strip()


def the_parts_that_ask_her(text: str) -> list[str] | None:
    """The clauses of ``text`` that ask her to act, in order; [] when none does; None when it cannot be read."""
    words = str(text or "").strip()
    if not words:
        return []
    doc = _parse(words)
    if doc is None:
        return None
    asked: list[str] = []
    for sentence in doc.sents:
        for verb in sentence:
            if verb.pos_ not in ("VERB", "AUX") or verb.dep_ not in ("ROOT", "conj"):
                continue
            head = verb
            while head.dep_ == "conj" and head.head is not head:
                head = head.head
            # A verb conjoined to one that asks her ("open Notes and write") is asked too, unless it has a subject of
            # its own.
            if _asks_her(verb) or (verb is not head and _subject(verb) is None and _asks_her(head)):
                asked.append(_clause_text(verb))
    return asked


def _points_back(sentence: Any) -> bool:
    """Whether the sentence names what it is about with a word that points back: "play it", "open that"."""
    return any(t.pos_ == "PRON" and t.dep_ in ("dobj", "obj", "pobj", "nsubjpass")
               and (_person(t) == "3" or t.morph.get("PronType") == ["Dem"]) for t in sentence)


#: The tenses a finite statement's verb carries: present (third person and not), and past.
_TENSED = ("VBZ", "VBP", "VBD")


def _a_question(sentence: Any) -> bool:
    """Put to her as a question: it ends in a question mark, opens with a wh-word, or puts its helper before its
    subject ("can you", "is it")."""
    if sentence.text.rstrip().endswith("?"):
        return True
    words = [t for t in sentence if not t.is_punct]
    if words and words[0].tag_ in ("WDT", "WP", "WP$", "WRB"):
        return True
    root = sentence.root
    subject = _subject(root)
    helpers = [c for c in root.children if c.dep_ in ("aux", "auxpass")]
    return subject is not None and any(h.i < subject.i for h in helpers)


def _only_talk(sentence: Any) -> bool:
    """A statement about someone's doing that does not put an act to her: it has a subject of its own, is not a
    question, and is not a request made in the speaker's or her name ("I want you to", "you should")."""
    if _a_question(sentence):
        return False
    root = sentence.root
    if _subject(root) is None:
        return False  # an imperative, a fragment, or a command the parse could not place: not ruled out
    # A statement is finite: its verb is tensed, or carries a tensed or modal helper. "Finder ... create a folder"
    # is not; it is "first open Finder, then create a folder" read wrongly, and a command.
    helpers = [c for c in root.children if c.dep_ in ("aux", "auxpass")]
    if root.tag_ not in _TENSED and not any(h.tag_ in (*_TENSED, "MD") for h in helpers):
        return False
    for verb in sentence:
        # A command inside the sentence ("keep playing 2048, I want to see a 1024 tile"), unless it is a clause
        # that hangs off another as its time or condition ("until you get", "when you are done").
        if (verb is not root and verb.pos_ in ("VERB", "AUX") and verb.tag_ == "VB" and _subject(verb) is None
                and verb.dep_ in ("advcl", "ccomp", "parataxis", "dep", "conj")
                and not any(c.dep_ == "mark" or (c.dep_ == "aux" and c.tag_ != "VB") for c in verb.children)):
            return False
    verbs = [root, *[c for c in root.children if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")]]
    return not any(_asks_her(verb) for verb in verbs)


def the_sentences_that_ask_her(text: str) -> list[str] | None:
    """The sentences of ``text`` that put something to her, each with the sentence before it when it points back to
    what that one named; [] when every sentence is only talk; None when it cannot be read.

    Only talk is a statement about someone's doing that asks her nothing: "I played chess last night", "I bet I could
    beat you", "you were good at 2048", "I think you got 2048 in 968 moves". A game or an app named there is what is
    being talked about. Questions, requests, commands and fragments are kept, and what they are for is read from them
    as it was before; a command the parse cannot place has no subject, so it is kept too."""
    words = str(text or "").strip()
    if not words:
        return []
    doc = _parse(words)
    if doc is None:
        return None
    sentences = list(doc.sents)
    kept: list[int] = []
    for index, sentence in enumerate(sentences):
        if _only_talk(sentence):
            continue
        if index > 0 and _points_back(sentence) and index - 1 not in kept:
            kept.append(index - 1)
        kept.append(index)
    return [sentences[i].text.strip() for i in sorted(set(kept))]
