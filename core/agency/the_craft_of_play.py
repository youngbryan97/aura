"""The craft of play: how people go about the situations games put them in, as watching them play shows it.

Knowing what a mechanic is (core/agency/mechanics_she_knows.py) is not knowing how to play it well. That is a craft,
and people show it every time they play: before pressing anything they know which one they are; they try the controls
where nothing can hurt them; they get in line with a target before they throw; they only step aside when something
falling will land where they stand; they get under what is worth catching and let the bombs drop; they cash in what
they hold before it topples; they open every door on the way; they read the silhouettes before hunting the room;
they ask everyone what they want and keep a list.

Each craft here was drawn from hours of people playing the games of the design map and their kind (the lunchroom
brawler, the falling-debris dodger, the egg catcher, the stacker who banks, the blueprint hunt, the resort errand-runner,
the one-on-one fighter, the move-planner, the chain-reaction puzzle, the cavern platformer, the kart derby with its
shop, the machine builder, the lesson-by-lesson trainer...), and said for any place: none names a game, a character
or a site. Each says how to tell a place asks for it (its words, or the mechanics found there), what a person does
first, what they do from moment to moment, how they think about it over a round, what they take care not to do, how
it goes on as the place goes on, and what of it carries over to places that are not games.

What she finds here goes into the guide to the place she is in (core/cognition/a_guide_to_a_place.py) and from there
into what she reasons with at every move; and what it says to do is what her play does where her play has a way to.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["CRAFTS", "PRINCIPLES", "Craft", "crafts_for"]


@dataclass(frozen=True)
class Craft:
    """One situation a place puts a player in, and how people play it."""

    name: str
    #: Words that say a place asks for it, and the mechanics (core/agency/mechanics_she_knows.py) that do.
    words: str
    mechanics: tuple[str, ...]
    #: What a person does first; then from moment to moment; how they think about it over a round; what they take care
    #: not to do; how it goes on; and what carries over to work that is not a game.
    first: str
    moment: str
    strategy: str
    mistakes: str
    goes_on: str = ""
    carries_over: str = ""
    #: Why people play it so: the reason under the craft, by which it is carried to places that never named it.
    why: str = ""

    @property
    def foundational(self) -> bool:
        """Whether it is part of playing anything (knowing which one is you, learning the controls, reading, reading the
        screen, trying again), not of what this place in particular asks."""
        return self.name in FOUNDATIONAL

    def said_in(self, text: str) -> int:
        """How many different signs of it ``text`` shows."""
        return len({m.group(0).lower() for m in re.finditer(self.words, text or "", re.I)}) if self.words else 0


#: The crafts that are part of playing anything.
FOUNDATIONAL = frozenset({"knowing which one is me", "learning the controls safely", "reading before acting",
                          "reading the screen's furniture", "trying again, differently"})

CRAFTS: tuple[Craft, ...] = (
    Craft("knowing which one is me",
          r"\b(?:you are|playing as|play as|your (?:character|hero|player|ship|car)|guide|control|move)\b",
          ("steering", "pointer steering"),
          "find the one you are before you press anything: the one the words name, the one the health bar is named "
          "for, the one the view follows, the one in the middle of things; then press each control once and watch it",
          "act through it: every plan is about where it is, where it can go, and what can reach it",
          "keep track of it when the screen is busy; if you lose it, find it again before doing anything else",
          "taking the mouse cursor, a score, or something that only moves by itself for yourself",
          "a new stage, a respawn or a new form can move or change you: find yourself again at each",
          "in any program, know which window, field or document your actions go into before you type",
          why="every act goes through one body; until you know which, no act can be judged by what it did"),
    Craft("learning the controls safely",
          r"\b(?:controls?|use the|press|arrow keys|mouse|click|space ?bar)\b",
          ("keys shown", "steering", "jumping", "shooting"),
          "read the controls screen; then, where nothing threatens yet, try each control once and see what it does",
          "use the control that does what the moment needs; a control's effect is learned by what follows it",
          "learn the few controls that matter first, the rest as play calls for them",
          "pressing everything at once, or learning controls in front of danger",
          "games add controls as they go ('two new moves!'): try each new one when it is given",
          "in a new program, try a command where it can be undone before relying on it",
          why="a control's effect is learned cheaply where a wrong press costs nothing, and dearly where it does"),
    Craft("reading before acting",
          r"\b(?:how to play|instructions|tutorial|lesson|tip|hint|press .{1,20} to (?:continue|start)|next)\b",
          ("information shown", "story", "menus"),
          "read the instructions, the lesson, the line a character says: they say what wins, what hurts and how",
          "when words appear mid-play, they are usually a new rule or a hint: read them before going on",
          "what the place says outranks what you guessed; a lesson ('press 2 to block') is the move to use later",
          "clicking through instructions unread, then playing blind",
          "lessons build on each other: what one taught, the next assumes",
          "read a page's own help and error text before trying another way",
          why="the place knows its own rules; a sentence read costs a second, a rule guessed wrong costs a life"),
    Craft("lining up to throw or shoot",
          r"\b(?:throw|toss|shoot|fire|blast|hit (?:your|the) opponents?|attack)\b",
          ("shooting", "aiming", "enemies"),
          "find what you are throwing at, and what you throw with; check how far a throw goes",
          "get in line with a target (its row, its column, its height) before throwing; throw when lined up and "
          "nothing is between; then step out of its line before it throws back",
          "pick the target that is alone, close, and not looking at you; keep moving between throws",
          "throwing from where nothing is in line; standing still in someone's line of fire",
          "more opponents, faster throws: keep to the edges and pick them off one at a time",
          "aim before you act: line up the thing you act on with what it is meant to reach",
          why="a throw goes straight; only one in line can land, and being in line works both ways"),
    Craft("sidestepping what falls",
          r"\b(?:falling|debris|dodge|avoid|watch out|rocks?|bombs?)\b",
          ("hazards", "a world going by"),
          "stand where you can move either way, in the middle; watch what falls and where it will land",
          "move only when something will land where you are, and only as far as clears it; then come back to the middle",
          "the danger is the one that lands soonest near you; small steps beat big ones",
          "running from everything, which takes you into the next thing falling",
          "more falls faster: keep the steps small and early",
          "deal with the problem that will land first, not the biggest one",
          why="every step away from one danger is a step toward another; the middle keeps both ways open"),
    Craft("getting under what to catch",
          r"\b(?:catch|collect .{0,20}(?:falling|before they)|before (?:they|it) (?:crash|hit|fall)|eggs?)\b",
          ("collecting", "hazards"),
          "tell what is worth catching from what is not (bombs, rocks): the bad ones are let fall",
          "go to where the next good one will land, the one that lands soonest first; stay out from under the bad ones",
          "when two will land at once, take the one you can reach, and the one worth more",
          "chasing one that cannot be reached and missing two that could",
          "faster and more at once: stay near the middle so every landing is close",
          "triage: take what you can reach in time, let go of what you cannot",
          why="what lands first decides; reach is limited, so value is what can be reached in time"),
    Craft("cashing in before you lose it",
          r"\b(?:stack|bank|cash in|collect your|take .{0,20}to safety|bonus points|combo|multiplier|before they fall)\b",
          ("stacking", "score"),
          "find what you are holding that can still be lost (a stack, a combo, what you carry) and how to make it safe",
          "keep building while it is steady; cash in when it starts to wobble, gets hard to hold, or danger comes near",
          "more held is worth more, but all of it can go at once: cash in a little early rather than too late",
          "holding out for one more and losing the lot",
          "the risk climbs faster than the reward: cash in sooner as it gets harder",
          "save your work before the risky step",
          why="what is held can be lost all at once, and the chance of losing it grows as it grows"),
    Craft("opening every door",
          r"\b(?:open(?:ing)? doors?|doors?|secret|passages?|chests?|what'?s behind)\b",
          ("reaching a place", "power-ups", "enemies"),
          "go along and open each door you pass: what is behind can be a way on, a power-up or a monster",
          "stand in front of a door, open it, and be ready to back off from what comes out",
          "remember what each door held; the way on is often behind one of them",
          "running past doors and missing the one that leads on",
          "later doors hold worse things: open them ready to act",
          "look in every menu and tab of a program once; remember where things are",
          why="what you have not looked at cannot help you; looking is cheap next to missing the way on"),
    Craft("going screen to screen",
          r"\b(?:next room|rooms?|exit|arrow|go (?:left|right|up|down)|areas?|floor|level \d)\b",
          ("reaching a place", "levels"),
          "find the ways out of the screen: an arrow, a doorway, an open edge, a sign",
          "cross the screen toward the way out, clearing what is in the way; walk off the edge to the next",
          "keep a map in your head: which way you came, which ways you have not tried",
          "walking back and forth over the same screens",
          "each room adds something: a new danger, a new thing to use",
          "navigate a site page by page with the way back in mind",
          why="a place bigger than one screen is a map; without one you walk the same ground twice"),
    Craft("finding what is shown",
          r"\b(?:find|items? to (?:collect|find)|blueprint|hidden|look for|spot|search)\b",
          ("clicking things", "collecting"),
          "look at what you are to find first (a list, silhouettes, a blueprint): its shape, its size, its colour",
          "scan the scene in order (left to right, top to bottom); click a thing when its shape matches; "
          "move on when one is found",
          "the small and the half-hidden are the last found: look at edges, under things and behind them",
          "clicking at random, which wastes time and may cost points",
          "later scenes hide things better: look for the outline, not the colour",
          "find a setting or a file by what it looks like and where such things are kept",
          why="a search in order finds everything once; a search at random finds the same things again"),
    Craft("fetching and trading",
          r"\b(?:wants? (?:a|an|the|some|to|my)|give (?:\w+ )?(?:the|a|an|my)|bring (?:me|him|her|them|the)|trade|"
          r"exchange|in return|deliver|talk to)\b",
          ("using things", "story"),
          "talk to everyone you meet and note what each wants and what each gives",
          "keep a list: who wants what, who has what; pick up what is lying about; give each what they asked for",
          "chains: what one gives is what another wants; work back from the thing you need",
          "forgetting who wanted what; walking back and forth without a plan",
          "each trade opens the next part of the story",
          "track the requests of each person you are working for, and close them one by one",
          why="people's wants link up; the thing you need is held by someone who wants something else"),
    Craft("planning moves before they play out",
          r"\b(?:choose your moves|select (?:\d|your)|pick (?:your|\d)|moves|combo|turn)\b",
          ("choosing from options", "following a sequence"),
          "read what each move does; see what the other side does in its turn",
          "choose moves that answer what the other side did last; keep a strong move for when it counts",
          "a mix beats a repeat; three of a kind may be a special; heal when low",
          "choosing the same thing every round",
          "new moves are given as rounds are won: try them",
          "plan a sequence of steps, then watch it run and adjust",
          why="the other side answers what you did; the move that beats the last one is the one to make"),
    Craft("fighting one on one",
          r"\b(?:fight|versus|vs\.?|opponent|punch|kick|block|attack|knock out|wins)\b",
          ("striking", "health"),
          "learn which control attacks, which blocks, which is special",
          "close the distance, attack when in reach and the other is open; block when it winds up; back off when "
          "it is your turn to be hit",
          "health is the score: trade hits only when it is ahead; use the special when it is charged",
          "standing in reach without attacking or blocking",
          "harder opponents read you: change your timing",
          "pick your moment: act when the other side is not ready",
          why="an attack opens you as it lands; the one who hits when the other is open wins the trade"),
    Craft("setting off a chain",
          r"\b(?:chain|reaction|burst|pop|splash|drops?|combo)\b",
          ("chains", "matching"),
          "count what you have to spend (drops, moves) and what is to be cleared",
          "spend where the most will set each other off: at the edge of a full cluster, where its burst reaches others",
          "the first move matters most; keep a few in hand for the last stragglers",
          "spending on lone ones while clusters stand",
          "fewer in hand, more to clear: look further ahead",
          "do the step that unblocks the most others first",
          why="one act that sets off others does the work of many; spend where it spreads"),
    Craft("building a way through",
          r"\b(?:pipes?|parts?|build|connect|route|from .{1,20} to|machine|add a part|drop)\b",
          ("chains", "putting things in place"),
          "find where the thing starts and where it must end up",
          "lay parts from the start toward the end, each joining the last; test it; fix the gap where it stops",
          "the simplest path that joins them is the best; spare parts are for detours",
          "laying parts that join nothing",
          "longer gaps and fewer parts: plan the whole way before laying",
          "build a pipeline step by step and test each join",
          why="a path is only as good as its weakest join; test it before trusting it"),
    Craft("spending to get stronger",
          r"\b(?:shop|buy|cost|cash|coins?|upgrade|garage|store|\$\d)\b",
          ("score",),
          "see what is for sale, what it costs, and what you have",
          "buy what fixes your weakest point first; cash unspent does nothing",
          "compare what each thing adds per what it costs",
          "spending on looks, or saving everything",
          "better parts cost more as you go on",
          "spend effort where it removes the biggest bottleneck",
          why="the weakest part limits the whole; spending there raises the most"),
    Craft("reaching a bar to go on",
          r"\b(?:you need \d|to advance|points to|reach \d|at least \d|target|goal of \d|par)\b",
          ("score", "levels"),
          "read the number you need, and how many tries you have",
          "keep track of how close you are; when behind, take the riskier ways that pay more",
          "the bar is the goal, not the most points: once past it, play safe",
          "playing for show when behind, or for risk once ahead",
          "the bar rises each stage",
          "know the success criteria before you start; check against them as you go",
          why="the goal is the bar, not the most; the risk worth taking depends on how far below it you are"),
    Craft("collecting them all against a clock",
          r"\b(?:\d+\s*/\s*\d+|all the|every (?:one|coin|candy)|time left|before time runs out)\b",
          ("collecting", "a clock"),
          "read how many there are and how long you have",
          "sweep the area in order, nearest first, without going back over ground you cleared",
          "if time is short, go where many are close together",
          "zigzagging after far ones and leaving near ones",
          "bigger areas, more hidden ones: sweep edges and corners",
          "batch the work by place, not by the order it was asked for",
          why="time spent walking is time not collecting; order by nearness, never cross ground twice"),
    Craft("climbing and jumping a way up",
          r"\b(?:ladders?|platforms?|climb|jump|ledge|cage|key|rescue)\b",
          ("jumping", "reaching a place", "rescue and allies"),
          "see the route: where the ladders go, which gaps can be jumped, where the goal is",
          "climb to change level; jump from the edge, not the middle; deal with enemies on your level before going on",
          "plan the route from the goal back to you; keys open cages",
          "jumping without a landing in sight",
          "longer gaps, moving platforms: time the jump to the platform",
          "work out the route to the goal before taking the first step",
          why="a route is found backward from the goal; a jump without a landing is a fall"),
    Craft("learning lesson by lesson",
          r"\b(?:lesson|training|mastered|practice|learn|today you will)\b",
          ("levels", "information shown"),
          "each lesson teaches one thing: read it, do exactly that",
          "repeat the taught move until it is passed; the test is what was taught",
          "carry each lesson into the next; the last puts them all together",
          "doing anything but what the lesson asks",
          "lessons get harder and combine",
          "learn one tool at a time, then combine them",
          why="a lesson is a test of one thing; the later ones assume it"),
    Craft("trying again, differently",
          r"\b(?:try again|game over|you lose|lost|out of (?:time|lives)|retry)\b",
          ("winning and losing", "lives"),
          "say what went wrong: what cost you, when, and what you were doing",
          "change one thing and try again; keep what worked",
          "two tries the same way that fail are enough: try another way; persistence, not repetition",
          "doing the same thing again and hoping",
          "each try teaches: a little further each time is progress",
          "after a failure, change the approach before retrying",
          why="the same act in the same place gives the same result; only a change can give another"),
    Craft("reading the screen's furniture",
          r"\b(?:score|health|life|lives|time|level|round|power)\b",
          ("information shown", "health", "lives", "a clock", "score"),
          "find the score, the health bar (and whose it is), the clock, the lives, the level",
          "glance at them between acts: they say how you are doing and what changed",
          "health low, play safe; time low, take risks; score behind, go for more",
          "taking a readout for something to play with",
          "the furniture changes as new things are added: read it again",
          "watch the indicators a program shows (progress, errors, counts) as you work",
          why="the furniture is the place telling you how you stand; it is read, not played"),
    Craft("telling a story by acting",
          r"\b(?:story|captured|rescue|help .{1,20} escape|stole|mission|attention|agent|operation)\b",
          ("story", "rescue and allies"),
          "read the story: who is in trouble, who did it, what the mission is",
          "each scene is a step of the mission; the story says which way is forward",
          "the villain is to be beaten or escaped; the friends are to be helped",
          "skipping the story and not knowing what the goal is",
          "the story turns as levels are cleared: read each new part",
          "know the purpose of the task, not only its steps",
          why="the story is the goal said as people say it; it says who is friend and who is not"),
    Craft("launching for distance",
          r"\b(?:launch|fling|distance|how far|angle|boost)\b",
          ("sending by strength", "pull"),
          "find what sets the angle and what sets the power",
          "launch at a middling angle with as much power as it allows; use boosts while flying",
          "note how far each launch went and change angle or power a little each time",
          "launching the same way after a poor one",
          "upgrades add power: launch harder as you buy them",
          "tune one setting at a time and measure the result",
          why="an outcome that depends on two settings is learned by changing one at a time"),
)


#: The reasons under every craft: what each of them is a case of, for a place that asks for none of them by name.
PRINCIPLES: tuple[str, ...] = (
    "know which one you are, and what your controls do, before judging anything by what followed it",
    "try a thing first where a mistake is cheap; commit where you know what it does",
    "read what the place says of itself before guessing; what it says outranks what you assumed",
    "keep your options open: stay where you can go either way, move as little as clears the danger",
    "deal with what comes soonest first, then with what is worth most among what you can still reach",
    "what you hold can be lost; the more you hold, the sooner you should make it safe",
    "look at everything once, in order, and remember it: a map beats a search",
    "work backward from the goal to find the way; the bar to reach decides how much risk is worth it",
    "change one thing at a time and see what it did; after two failures the same way, change the way",
)


def crafts_for(text: str, mechanics: Iterable[str] = (), *, most: int = 4) -> list[Craft]:
    """The crafts a place asks for, by what it says and the mechanics found in it; those both speak of first."""
    found = set(mechanics)
    scored = []
    for order, craft in enumerate(CRAFTS):
        by_words = craft.said_in(text)
        by_mechanics = len(found & set(craft.mechanics))
        if by_words or by_mechanics:
            scored.append((-(2 * by_words + by_mechanics) + (1 if craft.foundational else 0), order, craft))
    return [craft for *_rank, craft in sorted(scored)[:most]]
