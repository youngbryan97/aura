"""The situations she knows: what a place puts her in, how she can tell, and what that sets her play to.

Games put players in the same few situations again and again, and so do programs
and sites. Things fall and are to be caught, or dodged. Something is to be kept
away from whoever is after it until help comes. A thing is tossed to whoever is
there to take it. Everything of a kind is to be collected before the clock runs
out. A press is made when two things line up. What was shown is done again.
Something is made, and nothing is won. A person knows each at once and plays it
the way it asks before any of its things has touched them: they get under what
is to be caught, keep wide of what is coming at them, and know that the clock
running out while they hold on is the win, not the loss.

Each situation here is said for anywhere (none names a game, a character or a
site) as what play needs of it, not as advice to read. It is known by its
words, on a floor of phrasings and a learned surface over them
(core/language/learned_matcher.py) that decides the phrasings the floor misses
and abstains otherwise. What it sets is used directly by her play
(core/agency/playing_as_it_happens.py, core/agency/the_way_it_is_played.py):

* the ways it is played (core/agency/ways_of_playing.py), offered after any the
  place's own words name;
* what she supposes of a thing of a kind not yet met, by how it moves: what
  falls is to be caught or kept clear of, what heads for her kept clear of or
  shot at, until meeting it says otherwise;
* how wide she keeps of what costs her;
* whether lasting is the task, and what the clock running out means.

Each situation's structure follows the description of a game the general
video game playing work settled on (an avatar, the roles of the things, what
meeting each does, what ends it: VGDL, Schaul 2013; Perez-Liebana et al.,
GVGAI). ``elsewhere`` says where the same situation stands outside games.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["PLAYED_IN_TIME", "SITUATIONS", "Situation", "caution_of", "lasting_is_the_task", "played_in_time", "prior_stance",
           "situations_in", "ways_of"]

#: Stances, as core/agency/what_meeting_things_does.py names them.
MEET, AVOID, SHOOT = "meet", "avoid", "shoot"

#: What the clock running out means in a situation.
WIN, LOSS = "win", "loss"


@dataclass(frozen=True)
class Situation:
    """One situation, said for anywhere: how it is told, and what it sets play to."""

    name: str
    #: The floor: phrasings that say it, matched where a word begins.
    said: re.Pattern[str]
    #: Sentences that say it and sentences that do not, for the learned surface.
    examples: tuple[str, ...]
    counter_examples: tuple[str, ...]
    #: Her ways of playing it (core/agency/ways_of_playing.py).
    ways: tuple[str, ...] = ()
    #: What she supposes of a thing not yet met that falls from above her, and of one heading for her.
    falling: str = ""
    heading_for_her: str = ""
    #: How much wider than usual she keeps of what costs her.
    caution: float = 1.0
    #: Whether holding on is the task, and what the clock running out means.
    lasting: bool = False
    clock_ends_in: str = ""
    #: Where the same situation stands outside games.
    elsewhere: str = ""


def _said(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


SITUATIONS: tuple[Situation, ...] = (
    Situation(
        "catching what falls",
        _said(r"\b(?:catch|collect|grab|get) (?:\w+ ){0,3}(?:as (?:it|they) falls?|that falls?|falling|from the sky|"
              r"before (?:it|they) (?:hits?|lands?|reach(?:es)?) the (?:ground|floor|bottom))|\bcatch (?:the |all )?\w+"),
        ("Catch the falling eggs in your basket.", "Collect the fruit before it hits the ground.",
         "Catch as many cookies as you can."),
        ("Dodge the falling rocks.", "Throw the ball to your friend.", "Click the start button."),
        ways=("steer",), falling=MEET,
        elsewhere="taking in a stream of items as they arrive, before each expires",
    ),
    Situation(
        "keeping clear of what comes",
        _said(r"\b(?:dodge|avoid|evade|duck|watch out for|look out for|don'?t (?:get|be) (?:hit|caught|touched)|"
              r"keep clear of|stay away from|steer clear of)\b"),
        ("Dodge the falling debris.", "Avoid the bombs and don't get hit.", "Watch out for the guards."),
        ("Catch the falling eggs.", "Click on the ghosts to take their picture.", "Type your question above."),
        ways=("steer", "jump"), falling=AVOID, heading_for_her=AVOID, caution=2.0,
        elsewhere="keeping a system within its limits while load arrives",
    ),
    Situation(
        "keeping something away from whoever is after it",
        _said(r"\b(?:keep (?:\w+ ){0,3}(?:away from|safe|out of (?:reach|the hands))|don'?t let (?:him|her|them|it) "
              r"(?:get|grab|catch|take|reach)|before (?:she|he|they) (?:grabs?|catches|gets?|takes?))"),
        ("Keep Bloo safely in the arms of his friends until Mac arrives.", "Don't let her grab the puppy.",
         "Keep the ball away from the other team."),
        ("Grab the ball.", "Catch the falling eggs.", "Use the arrow keys to move."),
        ways=("send", "click things", "steer"), heading_for_her=AVOID, caution=1.5, lasting=True, clock_ends_in=WIN,
        elsewhere="keeping work out of a failing path until help is available",
    ),
    Situation(
        "holding on until the end",
        _said(r"\b(?:survive|stay alive|keep alive|hold (?:out|on) (?:until|till|for)|last (?:as long as|until|till|for)|"
              r"until (?:\w+ )?(?:arrives?|comes|gets here|time runs out)|as long as (?:you can|possible))"),
        ("Survive for two minutes.", "Hold out until help arrives.", "Last as long as you can."),
        ("Collect all the coins before time runs out.", "Win the race.", "Make your own song."),
        ways=("steer",), heading_for_her=AVOID, caution=1.5, lasting=True, clock_ends_in=WIN,
        elsewhere="keeping a service up through an outage window",
    ),
    Situation(
        "getting them all against a clock",
        _said(r"\b(?:collect|find|gather|grab|pick up|get) (?:all|every|each)\b.{0,40}\b(?:before|in time|time runs out|"
              r"timer|seconds|minutes)|\bbefore (?:the )?time runs out\b"),
        ("Collect all the gems before time runs out.", "Find every hidden item in sixty seconds."),
        ("Survive until the end.", "Avoid the bombs.", "Make your own character."),
        ways=("steer", "click things"), clock_ends_in=LOSS,
        elsewhere="finishing a checklist before a deadline",
    ),
    Situation(
        "shooting what comes",
        # A word for shooting at something is said of whatever it is shot at; "hit" only of those that come at her.
        _said(r"\b(?:shoot|blast|zap|fire at|splat|knock out|defeat|destroy|smash) (?:the |all |every |each )?\w+|"
              r"\bhit (?:\w+ ){0,3}(?:enemies|monsters|robots|invaders|aliens|bad guys|targets|ghosts|knights|zombies|"
              r"attackers)\b"),
        ("Shoot the robots before they reach you.", "Click to hit enemies with your paddle.", "Blast the invaders."),
        ("Catch the falling eggs.", "Avoid the bombs.", "Click Next to continue."),
        ways=("shoot", "steer", "click things"), heading_for_her=SHOOT,
        elsewhere="clearing incoming faults as they appear",
    ),
    Situation(
        "tossing to whoever takes it",
        _said(r"\b(?:throw|toss|pass|hand|send) (?:\w+ ){0,4}(?:to|towards?) (?:one of )?(?:the |his |her |your |their )?"
              r"(?:friends?|teammates?|partners?|catchers?|receivers?|players?|team)\b"),
        ("Use your mouse to aim and throw Bloo to one of his friends.", "Pass the ball to your teammate."),
        ("Throw the snowball at the other team.", "Catch the falling eggs.", "Press space to jump."),
        ways=("click things", "send"), heading_for_her=AVOID, caution=1.5,
        elsewhere="handing a task to whoever is free to take it, away from whoever would drop it",
    ),
    Situation(
        "getting to the way out",
        _said(r"\b(?:reach|get to|make (?:it|your way) to|find) (?:the )?(?:exit|end|door|finish|goal|top|other side|"
              r"way out)\b|\bescape\b"),
        ("Reach the exit of each level.", "Make your way to the top.", "Escape from the lab."),
        ("Collect the coins.", "Throw the ball to your friend.", "Make your own song."),
        ways=("steer", "jump", "a world bigger than the view"),
        elsewhere="getting to a page or setting through a series of screens",
    ),
    Situation(
        "steering something somewhere",
        # Brought somewhere by her steering, in time: a lander onto its pad, a car into its space, a boat through its
        # rings. Offline 2026-10-11 "Use the keys to guide Tommy's Lunar Lander safely onto the landing platform" put
        # her in no situation, and the lander crashed on each try while she pressed one key every few seconds.
        _said(r"\b(?:guide|steer|drive|fly|pilot|land|park|navigate|manoeuvre|maneuver)\b (?:[\w'’]+ ){0,5}?(?:to|into|onto|"
              r"through|safely|softly|gently|around|past|across|between|in its|in the)\b"),
        ("Use the arrow keys to guide Tommy's lunar lander safely onto the landing platform!",
         "Park each car in its space without hitting the others.", "Drive the boat through all the rings."),
        ("Use your arrow keys to move the tiles.", "Click on the doors.", "Catch the falling eggs."),
        ways=("steer",), heading_for_her=AVOID, caution=1.5,
        elsewhere="bringing a process to a target state without overshooting",
    ),
    Situation(
        "pressing at the right moment",
        _said(r"\b(?:when|as soon as|once) (?:the |it |they )?\w+ (?:lines? up|is over|reaches|hits|passes|is in)|"
              r"\b(?:at the right (?:moment|time)|in time with|on the beat|right timing)\b"),
        ("When the hamster lines up with the pillow, click again.", "Press space when the needle is over the green.",
         "Hit the key on the beat."),
        ("Use the arrow keys to move.", "Catch the falling eggs.", "Collect all the coins."),
        ways=("time a press",),
        elsewhere="acting when a watched condition becomes true",
    ),
    Situation(
        "doing again what was shown",
        _said(r"\b(?:repeat|copy|follow|memori[sz]e|do the same) (?:the )?(?:\w+ )?(?:pattern|sequence|moves|steps|"
              r"order|dance)\b|\bsimon says\b"),
        ("Repeat the sequence of arrows.", "Copy the dance moves in order.", "Memorize the pattern and repeat it."),
        ("Catch the falling eggs.", "Shoot the robots.", "Make your own song."),
        ways=("copy a sequence",),
        elsewhere="replaying a procedure that was demonstrated",
    ),
    Situation(
        "matching what goes together",
        _said(r"\b(?:match|find) (?:the |all the )?(?:pairs?|matching|same|ones? that match)\b|\bmemory game\b|"
              r"\bmatch (?:the |all the )?\w+ (?:that|which) (?:are|look) (?:the )?(?:same|alike)\b"),
        ("Find the matching pairs.", "Match the cards that are the same."),
        ("Catch the eggs.", "Shoot the robots.", "Reach the exit."),
        ways=("remember what was shown",),
        elsewhere="reconciling records that belong together",
    ),
    Situation(
        "setting off a chain",
        _said(r"\b(?:chain reaction|connect (?:the )?\w+ (?:to|from)|from the \w+ to the \w+|build a (?:path|way|track)|"
              r"place (?:the )?parts?)\b"),
        ("Drag each tube so the gumball rolls from the chute to the bucket.", "Build a chain reaction to trap him."),
        ("Catch the eggs.", "Shoot the robots.", "Type your question."),
        ways=("carry", "a chain", "turn what was placed"),
        elsewhere="composing steps whose outputs feed the next until the result is reached",
    ),
    Situation(
        "making something",
        _said(r"\b(?:(?:create|make|design|build|draw|paint|compose|mix) your own|use your imagination|dress (?:up|him|her))\b"),
        ("Create your own character.", "Use your imagination to make the craziest scenes.", "Mix your own song."),
        ("Make it to the end of the level.", "Collect all the coins.", "Shoot the robots."),
        ways=("make",),
        elsewhere="producing a document, a design or a piece of music to a brief",
    ),
    Situation(
        "launching for distance",
        _said(r"\b(?:as far as (?:you can|possible)|for distance|farthest|furthest|how far)\b"),
        ("Launch the hamster as far as you can.", "See how far you can fly."),
        ("Reach the exit.", "Catch the eggs.", "Survive as long as you can."),
        ways=("send", "time a press"),
        elsewhere="tuning one setting for the largest result",
    ),
    Situation(
        "building up",
        _said(r"\b(?:stack|pile|build (?:a |the )?(?:tower|stack|pile)|as high as)\b"),
        ("Stack the blocks as high as you can.", "Build a tower of the same colour."),
        ("Catch the eggs.", "Reach the exit.", "Shoot the robots."),
        ways=("stack",),
        elsewhere="building on what is done without toppling it",
    ),
    Situation(
        "a fight at close quarters",
        _said(r"\b(?:punch|kick|melee|fight|brawl|beat up|strike)\b"),
        ("Press S to punch and D to kick.", "Fight your way out.", "Use Z to melee attack close enemies."),
        ("Throw the ball to your friend.", "Catch the eggs.", "Make your own song."),
        ways=("strike", "steer", "jump"), heading_for_her=SHOOT, caution=1.5,
        elsewhere="handling what is close before reaching for what is far",
    ),
)

_SURFACES: dict[str, Any] = {}


def _surface(situation: Situation) -> Any:
    """The learned surface for one situation, made the first time it is asked for."""
    found = _SURFACES.get(situation.name)
    if found is None:
        from core.language.learned_matcher import LearnedMatcher
        from core.language.model_features import model_hidden_features

        found = _SURFACES[situation.name] = LearnedMatcher(
            name=f"situation_{situation.name.replace(' ', '_')}", positives=situation.examples,
            negatives=situation.counter_examples, features=model_hidden_features)
    return found


def _sentences(said: str) -> list[str]:
    lines = "\n".join(" ".join(line.split()) for line in str(said or "").splitlines())
    return [s for s in re.split(r"(?<=[.!?;])\s+|\n", lines) if s]


def situations_in(said: str) -> tuple[Situation, ...]:
    """The situations a place's words put her in: each the floor is sure of, which teaches the surface the sentence it
    was said in, and each the surface has decided a sentence says; in the order they are listed."""
    found: list[Situation] = []
    for situation in SITUATIONS:
        for sentence in _sentences(said):
            if situation.said.search(sentence):
                _surface(situation).observe(sentence, holds=True)
                found.append(situation)
                break
            if len(sentence.split()) >= 4 and _decided(situation, sentence):
                found.append(situation)
                break
    return tuple(found)


def _decided(situation: Situation, sentence: str) -> bool:
    try:
        return bool(_surface(situation).decide_without_waiting(sentence))
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        logger.debug("The %s surface could not decide a sentence: %s", situation, exc)
        return False


def ways_of(situations: Iterable[Situation]) -> list[str]:
    """The ways the situations are played, in their order, each once."""
    return list(dict.fromkeys(way for situation in situations for way in situation.ways))


def caution_of(situations: Sequence[Situation]) -> float:
    """How much wider than usual she keeps of what costs her: as the most careful of her situations asks."""
    return max((situation.caution for situation in situations), default=1.0)


def lasting_is_the_task(situations: Iterable[Situation]) -> bool:
    return any(situation.lasting for situation in situations)


#: How fast a thing must go for which way it goes to say anything, in working pixels a second.
GOING = 12.0


def prior_stance(situations: Sequence[Situation], thing: Any, mine: Any) -> str:
    """What she supposes of a thing not yet met, by how it moves, as her situations say: "" where they say nothing.

    A thing falling from above her (going mostly down, from higher up) is what the situation says falling things are;
    one heading for her (its going takes it nearer) is what it says of those.
    """
    if not situations or thing is None:
        return ""
    vx, vy = float(getattr(thing, "vx", 0.0)), float(getattr(thing, "vy", 0.0))
    if math.hypot(vx, vy) < GOING:
        return ""
    falling = vy > 0 and abs(vy) > 1.5 * abs(vx) and (mine is None or thing.y < mine.y)
    if falling:
        stance = next((s.falling for s in situations if s.falling), "")
        if stance:
            return stance
    if mine is not None:
        toward = (mine.x - thing.x) * vx + (mine.y - thing.y) * vy
        if toward > 0:
            return next((s.heading_for_her for s in situations if s.heading_for_her), "")
    return ""


#: The situations played in time: what comes, falls or runs out comes whether she is ready or not.
PLAYED_IN_TIME = frozenset({"catching what falls", "keeping clear of what comes", "keeping something away from whoever is after it",
                            "holding on until the end", "getting them all against a clock", "shooting what comes",
                            "pressing at the right moment", "a fight at close quarters", "steering something somewhere"})


def played_in_time(situations: Sequence[Situation]) -> bool:
    """Whether any of the situations a place's words put her in is played in time."""
    return any(situation.name in PLAYED_IN_TIME for situation in situations)

