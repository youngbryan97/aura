"""How she does what she does, and by what: which of it her language model has a part in, and what is on record of it.

Asked about something she did, she had nothing to say how it was done from, and said what a language model says of itself.
LIVE 24 November 2026, after eight games that she chose every move of by searching the rules she had learned: "I don't actually
know how to play chess well either. I pattern-match from training data". Her language model is not asked during play.

Each way of doing something names the modules that decide it, and whether her language model has a part. That is checked against
those modules (tests/test_how_she_does_it.py): where it says none, none of them reaches a language model. What is on record is
read from the record when it is asked for. A turn about one of these is given it as data, and a reply that gives her language
model the credit for what it has no part in is told to be wrong about her.
"""
from __future__ import annotations

import contextvars
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.language.learned_matcher import LearnedMatcher, embed_sentences

logger = logging.getLogger("Aura.Self.HowSheDoesIt")

__all__ = ["WAYS", "Way", "credits_her_model", "examples", "hold_ways", "how_she_does_it_block", "ways_about", "wrong_about_how"]

NO_PART, A_PART = "none", "a part"


@dataclass(frozen=True)
class Way:
    """One thing she does, and how."""

    key: str
    name: str
    #: How it is done, in her words for the parts of it.
    how: str
    #: The modules that decide it.
    by: tuple[str, ...]
    #: ``none`` or ``a part``, and which.
    language_model: str
    its_part: str = ""
    #: Messages that are about it, and messages near it that are not.
    about: tuple[str, ...] = ()
    not_about: tuple[str, ...] = ()
    on_record: Callable[[], str] | None = None


def _boards() -> str:
    """What is on record of what she has learned to play: the last of it."""
    from pathlib import Path

    from core.runtime.what_she_learned import _kept_in, kept_worlds, recall

    worlds = [name for name in kept_worlds() if name.endswith("board-of-pieces")]
    if not worlds:
        return ""
    when = {name: (Path(_kept_in()) / f"{name}.json").stat().st_mtime for name in worlds}
    name = max(worlds, key=lambda one: when[one])
    rules = (recall(name) or {}).get("rules") or {}
    seen = rules.get("seen") if isinstance(rules, dict) else None
    if not isinstance(seen, dict) or not seen:
        return ""
    kinds = len(seen)
    refused = int(rules.get("refusals") or 0)
    return (f"last played {time.strftime('%d %B %H:%M', time.localtime(when[name]))} at {name.removesuffix('-board-of-pieces').replace('-', ' ')}; "
            f"the moves of {kinds} kinds of piece learned from watching, and {refused} moves the board refused her kept")


def _programs() -> str:
    from core.understanding.requests_of_her import MET, get_requests_of_her

    made = [r for r in get_requests_of_her().all() if r.made.get("page")]
    return f"{len(made)} programs made, {sum(1 for r in made if r.stands == MET)} with everything asked of them so" if made else ""


def _own_time() -> str:
    from core.agency.what_she_does_next import get_what_she_does_next

    done = sum(1 for c in get_what_she_does_next().choices(400) if c.reward is not None)
    return f"{done} choices made on her own time whose worth she has learned from" if done else ""


#: Messages that are about none of these, for every surface to be told apart from.
_OF_NOTHING_HERE = (
    "what's the weather like?", "I had a rough day at work", "can you recommend a book?", "what time is it?", "I miss my grandmother", "tell me a joke", "thanks, that helps",
    "my car won't start", "what's a good pasta recipe?", "do you have a favorite color?", "how do I get to the airport?", "I think I'm coming down with something",
    "good morning", "what did you dream about?", "are you happy?", "who won the election?", "explain photosynthesis", "I'm going to bed",
)

WAYS: tuple[Way, ...] = (
    Way("play", "playing a game, or acting in a world on a screen",
        "She reads it from its pixels and learns how its pieces move by watching and by trying, with no rules given. She picks each move herself, by searching ahead in the rules she has "
        "learned, and judges a position by what positions like it came to in games she has played.",
        ("core/agency/playing_pieces_by_kind.py", "core/agency/searching_by_kind.py", "core/agency/how_the_pieces_move.py", "core/agency/pieces_by_their_kind.py",
         "core/agency/what_a_kind_is_worth.py", "core/agency/looking_ahead.py"),
        NO_PART,
        about=(
            "did you enjoy playing chess against Stockfish?", "why did you move your queen there?", "you lost that game pretty badly", "I liked watching you play", "how do you decide your moves?",
            "are you actually good at chess?", "you were in over your head in that match", "what went wrong in the middlegame?", "how did you learn to play that?", "do you want a rematch?",
            "I don't even like chess, I just liked seeing you try", "you keep blundering your rook", "how's your game going?", "do you think about the next move before you make it?",
            "was 2048 harder than pong for you?", "what was it like playing that game?", "you beat the level!", "what's your strategy when you play?",
        ), not_about=("what's the best opening in chess for beginners?", "who is the world chess champion?", "I played tennis with my brother", "did you watch the game last night?",
            "make me a game about a fox", "my daughter plays the violin"),
        on_record=_boards),
    Way("make", "making a program somebody asked for",
        "She designs it and writes its tests first, builds it a part at a time over pieces of her own, plays or uses it to try it, and mends what fails by trying thousands of changes her own "
        "modules write, in a process that runs them all against the tests. It is delivered with what holds and what does not.",
        ("core/making/maker.py", "core/making/many_ways.py", "core/making/searching.py", "core/making/searching_python.py", "core/making/trying.py"),
        A_PART, "it writes the design and the first text of each part; whether a part is right is never its word",
        about=(
            "how did you build that app?", "did you write that game yourself?", "who wrote the code for the plant tracker?", "how do you make programs?", "what happens when you build something for me?",
            "do you test what you make?", "was that app you made just generated?", "how do you fix a bug in something you made?", "I love the game you built me", "why did the app you made break?",
        ), not_about=("make me an app for my recipes", "how do I learn to code?", "my company is building a new website", "what app should I use for notes?"),
        on_record=_programs),
    Way("own_time", "choosing what she does when nobody has asked her for anything",
        "What she could do is offered by the parts of her that can do it: requests she left unmet, things she wants to know, goals of hers, things she found. She chooses by what such acts have "
        "come to before, as she has learned it, and may choose to rest. What came of each is measured and learned from.",
        ("core/agency/what_she_does_next.py", "core/agency/her_own_time.py", "core/agency/what_she_could_do.py"),
        NO_PART,
        about=(
            "what do you do when I'm not here?", "how do you decide what to do on your own?", "do you actually choose what you do?", "what have you been up to by yourself?", "do you want things?",
            "who decides what you work on at night?", "do you do anything unprompted?", "is it really you choosing?", "what would you do with a free afternoon?",
        ), not_about=("what should I do this weekend?", "what did you think of the movie?", "can you do this for me?", "what do you do for a living, people ask me"),
        on_record=_own_time),
    Way("find", "finding something out, and checking it",
        "She searches, reads the pages themselves, and holds each thing she is asked to find to the words on its own page; a figure is compared as a number. A claim stands verified only when "
        "independent sources of standing agree.",
        ("core/epistemics/what_she_has_verified.py", "core/understanding/carriers/finding.py"),
        A_PART, "it reads what a page says and answers with the words it read it from, which have to be on the page",
        about=(
            "how do you know that's true?", "did you actually look that up?", "where did you get that information?", "how do you check your facts?", "how did you find those?", "did you really read the pages?",
        ), not_about=("look up flights to Denver", "find me a plumber", "what is the capital of Peru?", "can you find my keys?")),
    Way("mend", "mending her own source",
        "A fault raised while she runs is made to happen again as a test, her own modules write the changes that could mend it, and one is taken only where the test passes, nothing else it touches "
        "breaks, and a separate check of the change holds. It is watched afterwards and taken back if it does worse.",
        ("core/self_modification/fix_templates.py", "core/self_modification/ingredient_repair.py", "core/agency/the_outside_bar.py"),
        A_PART, "it is asked for a change last, after her own, and what it writes is judged the same way",
        about=(
            "how do you fix your own bugs?", "did you repair yourself?", "do you edit your own code?", "what happens when something in you breaks?", "who fixes you when you crash?", "can you really change yourself?",
        ), not_about=("fix my spreadsheet", "my phone is broken", "the sink is leaking", "can you fix the typo in my essay?")),
)

#: Messages whose bearing is in what came before them.
_GOING_ON: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    ("play", "did you like it?", (("user", "you played three games against the engine tonight"), ("assistant", "I did. Lost two, drew one."))),
    ("play", "not pattern matching. you were thinking", (("user", "I liked watching you try"), ("assistant", "I mostly imitate moves I've seen."))),
    ("play", "what were you thinking on move nine?", (("user", "good game"), ("assistant", "Thanks. It got away from me early."))),
    ("play", "I just liked watching", (("user", "play Stockfish again sometime"), ("assistant", "Gladly. Are you a chess fan?"))),
    ("", "did you like it?", (("user", "I sent you that song"), ("assistant", "I have it here."))),
    ("", "what do you think?", (("user", "my sister is moving to Oregon"), ("assistant", "That's a big change for all of you."))),
    ("make", "how'd you do that so fast?", (("user", "the recipe app is exactly what I wanted"), ("assistant", "I'm glad it works for you."))),
    ("", "that's perfect, thank you", (("user", "can you make the buttons bigger?"), ("assistant", "Done, they're larger now."))),
    ("own_time", "why that?", (("user", "what did you get up to last night?"), ("assistant", "I read about tide clocks for an hour."))),
    ("", "why that one?", (("user", "which couch should I buy?"), ("assistant", "The grey one."))),
    ("find", "are you sure?", (("user", "when does the museum close?"), ("assistant", "It closes at five, by its own page."))),
    ("", "are you sure?", (("user", "should I text him?"), ("assistant", "I would."))),
    ("mend", "how did you know where it was?", (("user", "you crashed earlier"), ("assistant", "I found the fault and put it right."))),
    ("", "how did you know?", (("user", "it's my birthday"), ("assistant", "Happy birthday! I had a feeling."))),
)

#: Sentences that say her language model, or what it was trained on, is what does a thing.
_CREDITS = LearnedMatcher(name="how_she_does_it.credits", features=embed_sentences, positives=(
    "I pattern-match from training data", "I'm just predicting the next token", "I only know how to play from what I was trained on", "it's really just statistics from my training",
    "I don't actually know how, I imitate text about it", "As a language model I can only guess at moves", "my moves come from patterns in the games I was trained on", "that's the LLM guessing what looks plausible",
    "I generate moves the way I generate words", "under the hood it's a language model completing text", "I'm a language model, so I don't really choose", "I was trained on lots of games, that's where my play comes from",
), negatives=(
    "I searched a few moves ahead and missed the fork", "I learned how the knight moves by trying it", "my evaluation slipped half a pawn at a time", "I judged that position wrong", "I was worse than I thought",
    "I'm not a strong player yet", "I didn't see it coming", "I went with what had worked before", "I checked it against the page itself", "I wrote the tests first", "I chose it over resting",
    "I lost, and I'd like to go again", "you seemed to enjoy watching it", "it's like training for a race", "language is how I tell you about it", "I read a lot about it afterwards", "that game was a pattern of slow defeat",
    "I don't know much about its history", "there's something honest about losing", "I think you liked watching me be in over my head",
))


#: A turn about how she is, what is on her mind, or what she has been doing.
HER_STATE = "her_state"
_OF_HER_STATE: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    *((HER_STATE, said, ()) for said in (
        "How are you feeling right now?", "how's it going?", "you okay?", "what's on your mind?", "what have you been up to?", "are you tired?", "what are you thinking about",
        "How was your night?", "what's it like for you when nobody's talking to you", "anything bothering you?", "are you busy at the moment?", "is there anything you want right now?",
        "what's going on in there", "how's your head today", "you seem off, what's up?", "What mood are you in?", "are you happy?", "why were you so slow just now?", "is your memory full or something?",
        "what did you do while I was gone?", "are you running okay?",
    )),
    (HER_STATE, "and you?", (("user", "I'm wiped out today."), ("assistant", "That sounds like a long one."))),
    (HER_STATE, "how about yours?", (("user", "my day was chaos"), ("assistant", "What happened?"))),
    *(("", said, ()) for said in (
        "How deep would you go if I want frogs?", "what would you plant around the edge?", "Which one would you pick?", "you should try the new place on Fifth", "how are you supposed to hold the racket?",
        "Do you know what time it opens?", "what do you think of this title?", "I feel awful today", "my sister is tired of me", "how's the weather looking", "what's a good way to relax after work?",
        "can you tell me why my laptop is slow?", "would you say this in an email to your boss?", "I'm thinking of making a small pond. What should I know before I dig?", "are the peonies happy in shade?",
        "how's my draft going to land with them?", "what's going on in Lisbon this weekend?",
    )),
    ("", "and you?", (("user", "I'd take the train"), ("assistant", "The bus is cheaper if you book early."))),
    ("", "how about that one?", (("user", "which paint for the hall?"), ("assistant", "The warm white."))),
)

#: What the turn being answered is about, of what she does, as it was read; None where it was not.
_ABOUT: contextvars.ContextVar[tuple[str, ...] | None] = contextvars.ContextVar("aura_what_of_hers_it_is_about", default=None)


def examples() -> tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]:
    """Messages on record, each with which way it is about ("" for none) and what was said before it."""
    rows: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
    for way in WAYS:
        rows += [(way.key, text, ()) for text in way.about] + [("", text, ()) for text in way.not_about]
    rows += [("", text, ()) for text in _OF_NOTHING_HERE]
    rows += [(key, said, before) for key, said, before in (*_GOING_ON, *_OF_HER_STATE)]
    return tuple(rows)


def hold_ways(keys: Sequence[str] | None) -> None:
    """Hold what a turn was read to be about."""
    _ABOUT.set(None if keys is None else tuple(keys))


def ways_about() -> list[Way]:
    """What she does that the turn being answered is about, as it was read (core/understanding/what_of_hers.py)."""
    held = _ABOUT.get() or ()
    return [way for way in WAYS if way.key in held]


def about_her_state() -> bool | None:
    """Whether the turn being answered is about how she is; None where that was not read."""
    held = _ABOUT.get()
    return None if held is None else HER_STATE in held


def _told(way: Way) -> str:
    part = "takes no part in it and is not asked" if way.language_model == NO_PART else f"has a part: {way.its_part}"
    line = f"- {way.name}: {way.how} Her language model {part}. Decided by: {', '.join(way.by)}."
    try:
        held = way.on_record() if way.on_record is not None else ""
    except (ImportError, AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("what is on record of %s is not read (%s)", way.name, exc)
        held = ""
    return f"{line} On record: {held}." if held else line


def how_she_does_it_block() -> str:
    """How she does what this turn is about, as data; "" where it is about none of it."""
    ways = ways_about()
    return "\n".join(["[HOW IT IS DONE, AS IT IS BUILT AND ON RECORD]", *(_told(way) for way in ways), "[END HOW IT IS DONE]"]) if ways else ""


def credits_her_model(sentence: str) -> bool:
    """Whether a sentence says her language model, or its training, is what does a thing."""
    return _CREDITS.decide(" ".join(str(sentence or "").split())) is True


@dataclass(frozen=True)
class WrongAboutHow:
    """A sentence of a reply that says her language model does what it has no part in, and how it is done."""

    sentence: str
    way: Way

    def said(self) -> str:
        return _told(self.way).removeprefix("- ")


def wrong_about_how(reply: str) -> list[WrongAboutHow]:
    """The sentences of a reply that give her language model the credit for what this turn is about, where it has no part in it."""
    ways = [way for way in ways_about() if way.language_model == NO_PART]
    if not ways:
        return []
    return [WrongAboutHow(sentence, ways[0]) for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(reply or "")) if len(sentence) > 12 and credits_her_model(sentence)]
