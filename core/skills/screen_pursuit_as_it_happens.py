"""The screen pursuit's reflexes: what moves on its own is played as it happens.

The screen pursuit reads a screen, decides, acts once and reads again. On a
menu, an instruction screen or a board that is the right pace. Once a game is
running it is not: LIVE 2026-10-03 she played Flash games on her own page at
one move every ten to fifteen seconds while the game went on without her.

So for the length of a hand-over a second, faster way of playing sits under
the first. Before each of the pursuit's looks it checks whether the picture is
moving on its own. If it is, it plays that stretch as it happens
(core/agency/playing_as_it_happens.py), with the controls the game's own words
named, and gives the pursuit back the screen it ends on: a game over, a level
done, a menu. The pursuit reads that the way it reads any screen.

A run is over, for whoever handed the game over, when she has played some of
it and a way to start again has appeared that was not there at the start. What
to do then (play again, go back to the list) belongs to the request, which the
caller holds.
"""
from __future__ import annotations

import contextvars
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.what_stops_it import current
from core.verify import invariant
from core.skills.games_on_her_record import (
    _keep_what_she_learned,
    _what_she_kept_of,
)
from core.skills.games_on_her_record import (
    _this_game as _this_game,
)

logger = logging.getLogger("Aura.ScreenPursuit.AsItHappens")

__all__ = ["AS_IT_HAPPENS", "PlayingAsItHappens", "a_handed_over_run_is_over", "looked_at_as_it_happens"]


def _a_different_screen(said: str, before: str, *, shortest: int = 3) -> bool:
    """Whether two readings of a screen are of different screens: little of the words of either is in the other (a
    reading a little differently made is the same screen). Words shorter than ``shortest`` letters are not counted."""
    word = re.compile(rf"[a-z]{{{shortest},}}")
    now, then = set(word.findall(said.lower())), set(word.findall(before.lower()))
    return len(now & then) < 0.5 * max(1, min(len(now), len(then))) if now or then else False


#: The situations a place's words may put her in that are played in time, as things happen, not a move at a time
#: (core/agency/situations_she_knows.py).
def _played_in_time(said: str) -> bool:
    """Whether what a place has said puts her in a situation played in time."""
    from core.agency.situations_she_knows import played_in_time, situations_in

    return played_in_time(situations_in(said))


#: The situations a place's words may put her in that are played neither by turns on a board nor by pieces coming down
#: a well.
def _played_neither_by_turns_nor_by_pieces(reflexes: Any, said: str = '') -> bool:
    """Whether the place's words say it is played as one of the situations that is neither a board nor a well: then
    neither is looked for, each a watch of its own. Offline 2026-10-11, timed, a food fight's play began four and a
    half seconds late, spent looking for a board and for a piece coming down a well."""
    said = said or " ".join([*reflexes.words[-6:], str(reflexes.keep.get("counsel") or "")])
    # Nor in a thing for making, as she understands the place: offline 2026-10-12 an alien maker's save screen was taken
    # for three in a row, and three minutes went on searching moves on it, none made.
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    from core.agency.how_a_game_works import how_it_works, understood

    how = understood(guide, also=said) if guide is not None and hasattr(guide, 'controls') else how_it_works(said)
    return all(how.picture_check_fits(check) is False for check in ('board', 'well'))


def worked_by_typing(keep: dict[str, Any], guide: Any, named: Any) -> str:
    """How a place of questions and answers that she has answered by typing is worked, whatever moves on its screen
    after: by typing what it asks; '' for anywhere else, a game that only asked her name before play among them, and
    for a screen that names keys of its own. Offline 2026-10-12 a riddler's answer screen played its animation, was
    taken for play, and arrows were pressed at it for thirty-six seconds, twice, while the answer it gave went unread."""
    if not keep.get("typed") or named:
        return ""
    # What play showed outweighs what its words were taken to say: she typed an answer to a question, or the place
    # answered what she typed (live 2026-10-12 a riddler was read as a game of following a sequence, and was played with
    # keys for three minutes after it had answered).
    if keep.get("answered") or keep.get("asked_a_question"):
        return "typing what it asks"
    if guide is None:
        return ""
    return "typing what it asks" if "questions" in (getattr(guide, "mechanics", None) or {}) else ""


def _keys_of_play(guide: Any, read: str) -> list[str]:
    """The keys the guide knows play is worked by; where she understands surely that nothing of hers is moved by keys
    (core/agency/how_a_game_works.py), not those its program alone listens for: offline 2026-10-11 a game whose words
    said the mouse moves its hero was played with up and down, which its program listened for."""
    from core.agency.how_a_game_works import understood
    from core.cognition.a_guide_to_a_place import PROGRAM

    keys = list(guide.keys_for_play())
    how = understood(guide, also=read)
    if how.me.known() and not how.arrows_are_a_guess_worth_trying():
        keys = [key for key in keys if getattr(guide.controls.get(key), "source", PROGRAM) != PROGRAM]
    return keys


def _what_she_understands_said(keep: dict[str, Any], guide: Any, read: str) -> None:
    """How she understands the place works (core/agency/how_a_game_works.py), said once a game as play begins, for
    whoever watches to hear what she is going by."""
    if guide is None or getattr(guide, "how_said", None) is None:
        return
    from core.agency.how_a_game_works import understood

    said = understood(guide, also=read).said()
    if said and said != guide.how_said:
        guide.how_said = said
        _said_while_playing(said)


def _what_play_showed_of_how_it_works(stretch: dict[str, Any]) -> None:
    """What a stretch of play showed of how the place works, kept on its guide (core/agency/how_a_game_works.py): a body
    her keys moved, or one that followed her pointer; and said where it changes what she understands."""
    from core.agency.how_a_game_works import seen_to_answer, understood
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    if guide is None:
        return
    before = understood(guide).said()
    if stretch.get("follows_the_pointer"):
        seen_to_answer(guide, "the pointer")
    elif stretch.get("keys_that_move_her"):
        seen_to_answer(guide, "keys")
    now = understood(guide).said()
    if now and now != before and now != getattr(guide, "how_said", now):
        guide.how_said = now
        _said_while_playing(now.replace("As I understand it:", "Now I understand it better:", 1))


#: The fewest words of the language, new on the screen that asked, for what she typed to have been answered.
NEW_WORDS_AN_ANSWER = 1


def _answered_since_typed(keep: dict[str, Any], observation: dict[str, Any]) -> str:
    """What a place answered to what she typed into it, where it answered on the screen that asked: the words that
    asked still on it, and words of the language on it that were not, nor what she typed. A place that took her name
    and went on to another screen (its game) has answered nothing: the words that asked are gone. Offline 2026-10-11 a
    secret-identity machine answered her name, offered to try again, and she pressed TRY AGAIN and wiped the answer,
    having taken it for no more than a way to start again. The screen's words where it answered, else ""."""
    import re

    from core.language.words_of_the_language import the_word

    typed, before, asked_by = keep.get("typed"), keep.get("typed_on"), str(keep.get("typed_asked_by") or "")
    if not typed or before is None:
        return ""
    def words(text: str) -> set[str]:
        # Words of the language only: what a screen's lettering is misread as ("ETRYGOAIN") is neither asked nor new.
        return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", text) if the_word(w) is not None}

    now = " ".join(str(observation.get("text") or "").split())
    words_now, asking = words(now), words(asked_by)
    if not asking or len(asking & words_now) < 0.5 * len(asking):
        return ""
    from core.language.a_way_on import how_much_it_leads_on

    # A way to go again appearing is how such a place offers another go, not what it answered.
    new = {word for word in words_now - words(f"{before} {typed}") if how_much_it_leads_on(word) <= 1.0}
    if len(new) < NEW_WORDS_AN_ANSWER:
        return ""
    part = _the_new_part(now, f"{before} {typed}")
    return part if part and new & {w.lower() for w in re.findall(r"[A-Za-z]{4,}", part)} else now


#: Little words that join the words of a name or a phrase.
_JOINING_WORDS = frozenset({"the", "a", "an", "of", "and", "de", "von", "van", "le", "la", "el", "da", "du"})


def _the_new_part(now: str, before: str) -> str:
    """The longest run of a screen's words that were not on it before, nor misreadings of those (a letter or two off), nor
    a way to go again: what it answered, not the whole screen ("HENRY THE HATER", not "YOUR NAME HERE ... YOUR SECRET
    IDENTITY" around it)."""
    import re

    from core.language.a_way_on import how_much_it_leads_on

    def letters(token: str) -> str:
        return re.sub(r"[^a-z]", "", token.lower())

    was = {letters(token) for token in before.split() if letters(token)}

    def near(word: str) -> bool:
        # A misreading is a letter off in a short word, two in a long one: "HENRY" is no misreading of "HERE".
        off = 1 if len(word) <= 6 else 2
        return any(abs(len(word) - len(old)) <= off and _apart(word, old) <= off for old in was)

    # The little words that join a name together ("THE HYPNOTIST", "HENRY THE HATER") were on the screen before, in its
    # other words; between or before new words they are part of what is new.
    runs, run, glue = [], [], []
    for token in now.split():
        word = letters(token)
        fresh = bool(word) and word not in was and not near(word) and how_much_it_leads_on(token) <= 1.0 \
            and how_much_it_leads_on(word) <= 1.0 and "again" not in word
        if fresh:
            run.extend([*glue, token.strip("*•…")])
            glue = []
        elif word in _JOINING_WORDS:
            glue.append(token.strip("*•…"))
        else:
            if run:
                runs.append(run)
            run, glue = [], []
    if run:
        runs.append(run)
    best = max(runs, key=len, default=[])
    return " ".join(token for token in best if token)


def _apart(a: str, b: str) -> int:
    """How many letters apart two words are (edits)."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def a_hand_among_things() -> bool:
    """Whether she understands herself to be a hand among things in the place before her (core/agency/how_a_game_works.py):
    what answers the pointer there is what she plays with, and is looked for however the place goes on."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    if guide is None or not hasattr(guide, "controls"):
        return False
    from core.agency.how_a_game_works import POINTS, understood

    return understood(guide).me.value == POINTS


async def _searched_the_places(reflexes: Any, now: float) -> bool:
    """Where she is a hand among things against a clock, and many places on the screen answer the pointer, each is
    opened in turn, fast, and what each held kept (core/agency/searching_among_places.py); whether it was. Offline
    2026-10-11 a corridor of eighteen doors was opened one every five to fourteen seconds and the clock ran out."""
    from core.agency.searching_among_places import AT_LEAST, search_the_places

    places = list(getattr(reflexes, "answer_the_pointer", ()) or ())
    if len(places) < AT_LEAST or reflexes.way_on_shown or not a_hand_among_things() or not _it_goes_on_without_her(reflexes):
        return False
    search = reflexes.keep.setdefault("searching", {"held": {}, "paired": set(), "said": False})
    if not search["said"]:
        search["said"] = True
        _said_while_playing("There's a clock and many places alike: opening each in turn, quickly, and keeping in mind what "
                            "is behind each.")
    done = await search_the_places(reflexes.look, reflexes, places, seconds=min(STRETCH_S, reflexes.ends_at - now),
                                   held=search["held"], paired=search["paired"])
    if done.get("ended") == "the screen turned over":
        search["held"], search["paired"] = {}, set()
    if done.get("ended") == "the view moved":
        # The places are found again where the view now is; what was behind each goes with it.
        reflexes.answer_the_pointer = []
    reflexes.played_at = time.monotonic()
    return bool(done.get("opened") or done.get("pairs"))


def _made_by_placing(can_do: Any) -> bool:
    """Whether what is made here is made by carrying things into place, so that only placed work, measured, shows it
    made (core/skills/screen_step_evidence.py): things have been carried, or are said to be, or she understands herself
    to be a hand that puts things in place (core/agency/how_a_game_works.py). Elsewhere a thing for making is made by
    choosing and its own finishing control: offline 2026-10-11 an agent generator showed the agent card it made, nothing
    was placed in it, and she went on clicking for four minutes."""
    # A carry that went where it was taken, not one tried to see what it does: offline 2026-10-11 pictures dragged onto
    # an alien maker's title to see what they did made it a place of placing, and its alien could never be done.
    if any(bool(went) for went in (getattr(can_do, "carried_to", None) or {}).values()) or getattr(can_do, "carrying_said", False):
        return True
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    if guide is None or not hasattr(guide, "controls"):
        return False
    from core.agency.how_a_game_works import PLACES, understood

    return understood(guide).me.value == PLACES or guide.in_play("dragging")


def _it_goes_on_without_her(reflexes: Any) -> bool:
    """Whether, as she understands the place (core/agency/how_a_game_works.py), it goes on without her: from all its
    guide has read and what is on the screens before her, and what her play there has shown."""
    # Signs to be followed may be coming along a line whether she is ready or not (core/agency/signs_that_come_along.py):
    # until that is seen or ruled out, the place is taken to go on. Offline 2026-10-12 a book of arrows was waited on to
    # come to rest after its NEXT, and its first round was lost before she looked at it.
    if _signs_may_come(reflexes):
        return True
    said = " ".join([*reflexes.words[-6:], str(reflexes.keep.get("counsel") or "")])
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    if guide is None:
        return _played_in_time(said)
    from core.agency.how_a_game_works import understood

    return understood(guide, also=said).goes_on_without_her()


def _signs_may_come(reflexes: Any) -> bool:
    """Whether signs to be followed may be coming along a line in this place: its words ask for signs to be followed,
    and a line has been seen here or has not yet been looked for and ruled out."""
    keep = reflexes.keep
    if not keep.get("signs_told"):
        # Things coming along a line in a place that asks nothing of the kind are things of the place. Offline
        # 2026-10-12 eggs falling in a game of catching them were seen going together down a line, the line was kept,
        # and in the next two runs each egg was waited on as a sign with no key to it.
        return False
    if keep.get("signs_showing_over"):
        # A showing has just ended: whatever screen is up is read whole before it is taken for one of play again.
        # Offline 2026-10-12 a room's last sign went by, the screen that followed offered NEXT, and it was looked
        # at as play is, quickly and without its words, for as long as the game had left.
        return False
    return bool(keep.get("signs_going") or keep.get("signs_seen")
                or not keep.get("signs_shown_in_turn") and int(keep.get("signs_none") or 0) < A_LINE_LOOKED_FOR)


def a_screen_of_play_in_time(reflexes: Any, labels: Any = None) -> bool:
    """Whether the screen before her is one of play in a place whose words put her in a situation played in time
    (dodging, shooting what comes, catching): no way on is shown on it (``labels``, the screen's writing, where it has
    not been read into ``reflexes`` yet). Such a screen is not waited on to come to rest, nor swept with the pointer:
    offline 2026-10-11 a food fight's round of ten seconds was over before play began, sixteen seconds after it showed,
    spent sweeping the pointer over it and waiting for it to stand still after a click."""
    from core.language.a_way_on import how_much_it_leads_on

    if reflexes is None or not _it_goes_on_without_her(reflexes):
        return False
    if labels is None:
        return not reflexes.way_on_shown
    return not any(0 < len(label) <= 40 and how_much_it_leads_on(label) > 1.0 for label in labels)


async def _it_moves_on_its_own(reflexes: PlayingAsItHappens, watched: Any) -> bool:
    """Whether the screen moves on its own, watched afresh only where the screen is another, or a while has passed."""
    known = reflexes.still_screen
    if (known is not None and _the_same_screen(known[0], reflexes.said_now)
            and time.monotonic() - known[1] < STILL_SCREEN_KEPT_S):
        return False
    moves = await watched(reflexes.look)
    reflexes.still_screen = None if moves else (reflexes.said_now, time.monotonic())
    if moves and not reflexes.way_on_shown:
        # A screen with no way on that goes on by itself is play going on without her: kept for every part of her.
        from core.agency.how_a_game_works import seen_going_on_by_itself
        from core.cognition.a_guide_to_a_place import THE_GUIDE

        seen_going_on_by_itself(THE_GUIDE.get(), True)
    return moves


#: How long one screen taken for play may stand, the same words on it, before she looks it over with the pointer.
STUCK_ON_PLAY_S = 15.0


def stuck_on_one_screen_of_play(reflexes: Any, said: str) -> bool:
    """Whether the screen taken for play has stood, the same in its words, for STUCK_ON_PLAY_S while she played it: a
    person whose keys have done nothing for a while moves the mouse over the screen to see what it answers. Offline
    2026-10-12 an instructions screen whose backdrop moved, read once without its "CONTINUE", was taken for play, and
    she pressed right on it for three and a half minutes, right having done nothing every time."""
    if reflexes is None:
        return False
    now = time.monotonic()
    kept = reflexes.play_screen
    if kept is None or not _the_same_screen(kept[0], said):
        reflexes.play_screen = (said, now)
        return False
    return now - kept[1] >= STUCK_ON_PLAY_S


def _the_same_screen(before: Any, said: Any) -> bool:
    """Whether a screen read now is one read before, a reading a little differently made being the same screen."""
    return (isinstance(before, str) and isinstance(said, str)
            and (before == said or bool(before and said) and not _a_different_screen(said, before)))


def begin_run(keep: dict[str, Any], game: str = "") -> None:
    """A new attempt keeps world knowledge and measures its own state. ``game`` is the name the game is kept under
    (`_this_game`): where it is the game the last attempt was at, unchanged, what she is in it and what her keys do
    are kept too, and only which thing on the screen she is now is found again.

    A game mended between attempts is another game, and its keys are measured afresh: a paddle whose up key was put
    right does not go the way it went. But the same game begun again has the keys it had. LIVE 2026-10-11 a lander
    that is down in five seconds was begun ten times, each time from knowing nothing of herself, and the five seconds
    went on finding out; and a two-minute game had its keys tried again at the start of each of three runs."""
    from core.agency.how_the_contest_stands import ContestStands
    from core.agency.which_one_answers_to_her import WhichIsHers

    contest = keep.get("contest")
    keep["contest"] = ContestStands(wins=contest.wins) if contest is not None else ContestStands()
    hers = keep.get("hers")
    if game and keep.get("hers_in", game) == game and hers is not None and hers.kind is not None:
        hers.numbered_afresh()
    else:
        keep["hers"] = WhichIsHers()
    keep["hers_in"] = game
    meeting = keep.get("meeting")
    if meeting is not None:
        meeting.begin_run()
    # The clock and the speeds of a new attempt are its own; what the place pulls, and how firmly she keeps to a thing
    # gone for, are the place's.
    pressures = keep.get("pressures")
    if pressures is not None:
        pressures.begin_run()
    gone_for = keep.get("gone_for")
    if gone_for is not None:
        gone_for.number, gone_for.passed_over = None, {}
    # What she plays for begins its counts afresh and keeps what the runs before showed; who is who keeps what each
    # kind has shown.
    for kept in (keep.get("plays_for"), keep.get("who")):
        if kept is not None:
            kept.begin_run()

#: How many times one screen that offers a way on is left to be gone on from before what moves on it is played: a
#: way on may take a second look to be known for a control, and a second click to take.
WAYS_ON_TRIED = 4

#: How long a screen found not to move on its own is taken to stay so while its words do not change, in seconds.
STILL_SCREEN_KEPT_S = 12.0

#: How long the reflexes leave a moving screen alone after finding nothing on
#: it that answers to her: a title screen that animates is not a game yet.
LEAVE_A_MOVING_MENU_S = 20.0

#: The longest one stretch of play runs before the pursuit gets a look.
STRETCH_S = 300.0

#: The longest a screen whose movement keeps answering to nothing is left alone before it is tried again, in seconds.
LEFT_ALONE_AT_MOST_S = 180.0

#: How often what the page draws is looked at for a board of two sides in turns, in seconds.
BOARD_LOOKED_FOR_EVERY_S = 3.0

#: How often the one thing she has been sending in a place is looked for on its screen.
HERS_LOOKED_FOR_EVERY_S = 2.0

#: How many of the things a screen offers are tried before the choice is left to the pursuit, and how long the screen
#: is given to take one in.
CHOICES_TRIED = 3
AFTER_A_CHOICE_S = 0.8
#: How many of a place's screens of words are kept for what its board's rules are.
WORDS_KEPT = 60

#: How long a showing is watched for places lit in turn, and how often one screen's showing is followed.
WATCH_A_SHOWING_S = 6.0

#: Words saying a place is worked by the pointer pressed on things, so a place it shows is done again by a click on it.
_CLICKED = re.compile(r"\b(?:click|tap|touch|press (?:on|the) (?:button|light|square|tile))", re.I)
FOLLOW_AT_MOST = 6

#: The longest she waits for her model's answer to a question a screen asks, in seconds.
ANSWER_WITHIN_S = 20.0

#: How long she tries out a thing whose own words set her to make something, not to win: it has no end of its own.
TRYING_OUT_S = 90.0
#: The longest an attempt at making continues without measured current work.
TRYING_OUT_MOST_S = 240.0
#: Where what is made is not carried into place but chosen and finished by its own controls (a generator, a randomizer,
#: "I'm done"), how many of her acts must have changed it for something to have been made.
CHANGES_MADE_AT_LEAST = 5

#: Why a run of a thing for making is over.
MADE_NOT_WON = "it is for making things, and I have tried it out"
#: How many stretches in a row may end on a still screen, with nothing to show, before a place played in time is left to
#: the pursuit a while all the same.
STILL_STRETCHES_AT_MOST = 3
#: How long a question put to a thing is given to be answered before the answer is read off the screen.
ANSWER_SHOWS_S = 2.5
MAKING_STALLED = "no created work was verified before the making attempt's time ran out"

#: Words that say a screen teaches how to play: it is read and gone on from, not played on. LIVE 2026-10-09 a game's
#: putter lesson ("HOW TO USE THE PUTTER: 1. Click on your ball...") drew its "next" in letters she could not read, and
#: she sent shots at the lesson for three minutes.
_RULES = re.compile(r"\b(?:how to (?:play|use|win)|instructions|controls|how do (?:i|you)|tutorial)\b|(?:^|:)\s*1\.\s+[A-Za-z]", re.I)


def reads_as_rules(said: str) -> bool:
    """Whether a screen's words teach how to play, rather than show the play."""
    return bool(_RULES.search(said or ""))


def asks_to_choose(said: str) -> bool:
    """Whether a screen's words ask her to choose something: a player, a ball, a level, a character."""
    from core.language.a_way_on import asks_to_choose as asks

    return asks(said)


def _lays_out_a_choice(said: str) -> bool:
    """Whether a screen that also offers a way on is still one to choose on first: its words are mostly the asking."""
    return len(said.split()) <= 12


#: The fewest words a screen that is not plainly rules must have to be read for a legend.
LEGEND_WORDS = 6

#: The ways the reflexes play, as core/agency/the_way_it_is_played.py names them.
SEND_WAY, AS_IT_HAPPENS_WAY = "send", "as it happens"


def _newly_says_the_game_is_over(observation: dict[str, Any], during: set[str]) -> str:
    """The words by which the screen says the game is over, where nothing read while she played had them in it."""
    from core.language.how_a_game_ended import what_says_the_game_is_over

    ending = what_says_the_game_is_over(str(observation.get("text") or ""))
    if not ending or any(ending in " ".join(str(said).lower().split()) for said in during):
        return ""
    return ending


@dataclass
class PlayingAsItHappens:
    """Eyes and hands on the part of her own page that draws, at the speed it goes."""

    page: Any
    band: tuple[float, float, float, float]
    goal: str
    ends_at: float
    keep: dict[str, Any] = field(default_factory=dict)
    stretches: list[dict[str, Any]] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    first_ways_back: frozenset[str] | None = None
    quiet_until: float = 0.0
    #: The words of the screen she was left quiet on, and of the screen as last looked at: quiet is for that screen.
    quiet_on: str = ""
    said_now: str = ""
    #: The words of the screen last watched and found not to move on its own, and when.
    still_screen: tuple[str, float] | None = None
    over_because: str = ""
    recalled: bool = False
    #: The words of the screen that ended the run, for whoever asks how it ended.
    ending_words: str = ""
    #: The same screen's pieces of writing, each as it was read.
    ending_parts: list[str] = field(default_factory=list)
    #: Something went on going while she held a key: a world that waits for her and moves under her hand.
    under_her_hand: bool = False
    #: The places on the screen last read where the pointer turns into a hand.
    answer_the_pointer: list[tuple[float, float]] = field(default_factory=list)
    #: Whether the screen last read is one of play in a place played in time (`a_screen_of_play_in_time`).
    play_in_sight: bool = False
    #: The words of the screen of play last read, and since when they have stood (`stuck_on_one_screen_of_play`).
    play_screen: tuple[str, float] | None = None
    #: Screens this run reached that the game had not shown her before, told apart by their words.
    new_screens: int = 0
    #: Her bearings on the last screen she took in (core/cognition/her_bearings.py).
    bearings: Any = None
    #: How often play as it happens was held back for the way on a screen offered, by the screen's words.
    held_for_a_way_on: dict[str, int] = field(default_factory=dict)
    #: Since when its words have been known to set her to make something, not to win.
    for_making_since: float | None = None
    #: How many of her acts had changed the screen when making began, and how many have now.
    made_before: int = 0
    acts_that_changed: int = 0
    #: Whether the last screen read showed a label that goes on (Play, Start, Next): a menu, not a world to send into.
    way_on_shown: bool = False
    menu_pixels: Any = None
    #: The short labels her screens have shown (a button's "LAUNCH", "Next"): words on their own, not what it says to do.
    labels_seen: set = field(default_factory=set)
    #: Play that knows when its game has stopped (a well nothing more comes down) said it has: with a way to start again
    #: on screen, the run is over, though that way was on screen before.
    play_stopped: bool = False
    #: A well was tried this run and no key of hers moved its piece; and how long she has waited at a well for a piece.
    no_well_here: bool = False
    waited_at_a_well: float = 0.0
    #: When she last played anything, and the screens not seen before counted when the last stretch was judged.
    played_at: float = 0.0
    _screens_judged: int = 0
    _clip: dict[str, float] | None = None
    _focused: bool = False
    _frames: Any = None
    _canvas: Any = None
    #: Her taps, each let go of a moment after it (core/skills/keys_tapped_as_a_hand_taps.py).
    _taps: Any = None
    _board_probe: Any = None

    # -- eyes ---------------------------------------------------------------

    async def _where(self) -> dict[str, float]:
        if self._clip is None:
            from core.perception.what_her_page_shows import the_page_size

            wide, tall = await the_page_size(self.page)
            left, top, right, bottom = self.band
            self._clip = {
                "x": left * wide, "y": top * tall,
                "width": max(1.0, (right - left) * wide), "height": max(1.0, (bottom - top) * tall),
            }
        return self._clip

    async def look(self) -> tuple[Any, float] | None:
        from core.perception.frames_as_they_are_drawn import CanvasFrames, PageFrames

        clip = await self._where()
        # The drawing read from its own canvas where it can be; else frames as
        # the browser draws them; a screenshot a picture only where neither.
        if self._canvas is None:
            self._canvas = CanvasFrames(self.page)
        if not self._canvas.unavailable:
            read = await self._canvas.look(clip)
            if read is not None:
                return read
        if self._frames is None:
            self._frames = PageFrames(self.page)
        if not self._frames.unavailable:
            streamed = await self._frames.look(clip)
            if streamed is not None:
                return streamed
        try:
            # At the page's own pixels: on a high-density screen a picture
            # at device pixels is four times the size for the same view, and
            # live play ran at thirteen pictures a second (2026-10-04).
            from core.perception.a_picture_of_her_page import picture_of

            at = time.monotonic()
            picture = await picture_of(self.page, clip, css=True, kind="jpeg")
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return None
        if picture is None:
            return None
        return picture, at

    async def close(self) -> None:
        """Stop streaming frames, when the run is over, and let go of every key tapped."""
        if self._taps is not None:
            await self._taps.let_go_of_all()
        if self._board_probe is not None:
            await self._board_probe.close()
            self._board_probe = None
        if self._canvas is not None:
            await self._canvas.close()
            self._canvas = None
        if self._frames is not None:
            await self._frames.close()
            self._frames = None

    # -- hands --------------------------------------------------------------

    async def _focus(self) -> None:
        from core.skills.screen_pursuit_on_a_page import focus_drawing_for_input

        self._focused = await focus_drawing_for_input(self.page)

    @staticmethod
    def _key(name: str) -> str:
        """A key by the browser's own name for it: "backspace" is "Backspace", a letter is itself."""
        from core.skills.screen_pursuit_on_a_page import _KEY_NAMES

        lowered = str(name or "").lower()
        if lowered in _KEY_NAMES:
            return _KEY_NAMES[lowered]
        if re.fullmatch(r"f\d{1,2}", lowered):
            return lowered.upper()
        return name if len(name) == 1 or not name.isalpha() else name[:1].upper() + name[1:]

    async def _keyed(self, act: Any, key: str) -> None:
        """A key pressed, let go or tapped; one the browser does not know is said and passed over, not the end of the
        game: LIVE 2026-10-09 a name typed into a game's scorecard was cleared with "backspace", which the browser
        calls "Backspace", and the error ended the whole game."""
        try:
            await act(self._key(key))
        except Exception as why:  # noqa: BLE001 - the browser's refusal of one key name
            if "Unknown key" not in str(why):
                raise
            logger.info("the browser knows no key %r; passed over", key)

    async def down(self, key: str) -> None:
        context = current(whose="realtime_input.down")
        context.check()
        if key == MOUSE_BUTTON:
            await self.page.mouse.move(*await self._at(*_WHERE_A_BUTTON_PRESS_GOES))
            context.check()
            await self.page.mouse.down()
            self._focused = True
            return
        await self._focus()
        context.check()
        if self._taps is not None:
            self._taps.held(key)
        await self._keyed(self.page.keyboard.down, key)

    async def up(self, key: str) -> None:
        if key == MOUSE_BUTTON:
            await self.page.mouse.up()
            return
        if self._taps is not None:
            self._taps.held(key)
        await self._keyed(self.page.keyboard.up, key)

    async def tap(self, key: str) -> None:
        context = current(whose="realtime_input.tap")
        context.check()
        if key == MOUSE_BUTTON:
            await self.click(*_WHERE_A_BUTTON_PRESS_GOES)
            return
        await self._focus()
        context.check()
        if self._taps is None:
            from core.skills.keys_tapped_as_a_hand_taps import KeysTapped

            self._taps = KeysTapped(lambda name: self._keyed(self.page.keyboard.down, name),
                                    lambda name: self._keyed(self.page.keyboard.up, name))
        await self._taps.tap(key)

    async def pressed(self, key: str) -> None:
        """A key pressed for the pursuit the way a person presses one to see what it does: held a moment, watched."""
        from core.agency.playing_as_it_happens import it_goes_while_held

        if await it_goes_while_held(self.look, self, key) and not self.under_her_hand:
            self.under_her_hand = True
            logger.info("something went on going while %s was held: the world waits for her and moves under her hand", key)

    async def _at(self, x: float, y: float) -> tuple[float, float]:
        clip = await self._where()
        return clip["x"] + min(1.0, max(0.0, x)) * clip["width"], clip["y"] + min(1.0, max(0.0, y)) * clip["height"]

    async def click(self, x: float, y: float) -> None:
        context = current(whose="realtime_input.click")
        context.check()
        at = await self._at(x, y)
        context.check()
        # As a hand clicks, quickly: in play a click is a throw or a jump, and a moment's rest is a moment lost.
        from core.skills.screen_pursuit_on_a_page import as_a_hand_clicks

        await as_a_hand_clicks(self.page, *at, arrives_s=0.0, presses_s=0.04)
        self._focused = True

    async def point(self, x: float, y: float) -> None:
        context = current(whose="realtime_input.point")
        context.check()
        at = await self._at(x, y)
        context.check()
        await self.page.mouse.move(*at)

    async def press(self, x: float, y: float) -> None:
        """The button pressed at a place and kept down, for a pull or a hold (core/agency/playing_by_shots.py)."""
        context = current(whose="realtime_input.press")
        context.check()
        at = await self._at(x, y)
        context.check()
        await self.page.mouse.move(*at)
        context.check()
        await self.page.mouse.down()
        self._focused = True

    async def release(self) -> None:
        await self.page.mouse.up()

    async def type_text(self, text: str) -> None:
        """Words typed as a person types them, a key at a time, into whatever has the focus."""
        await self.page.keyboard.type(text, delay=40)

    # -- the pursuit's hooks ---------------------------------------------------

    def read(self, observation: dict[str, Any]) -> None:
        """What a look of the pursuit showed: the screen's words, and its ways back at the start."""
        from core.language.a_way_on import how_much_it_leads_on
        from core.skills.screen_pursuit_bearings import restart_controls
        from core.skills.visible_words_from_their_pixels import current_visible_words

        observation = current_visible_words(self, observation)
        from core.skills.a_new_attempt_must_be_seen import note_attempt_activity

        note_attempt_activity(self, observation)
        self.keep.pop("signs_showing_over", None)                # the screen after a showing of signs has been read
        _ways_on_read_right_before(self, observation)
        said = " ".join(str(observation.get("text") or "").split())
        labels = [" ".join(str(r.get("text") or "").split()) for r in observation.get("layout") or () if isinstance(r, dict)]
        self.said_now = said
        # Where the pointer turns into a hand on this screen: what answers a click, for what sends what she types.
        self.answer_the_pointer = [(float(r["center_x"]), float(r["center_y"]))
                                   for r in [*(observation.get("shapes") or ()), *(observation.get("layout") or ())]
                                   if isinstance(r, dict) and r.get("answers_the_pointer") and "center_x" in r]
        if said and (not self.words or self.words[-1] != said):
            # A screen not taken in before, glanced at as a whole, beside her play (core/perception/what_a_scene_is.py).
            from core.cognition.a_guide_to_a_place import THE_GUIDE
            from core.perception.what_a_scene_is import glance_beside

            guide = THE_GUIDE.get()
            if guide is not None:
                from core.cognition.what_this_place_is import ask_for_a_reading

                glance_beside(self.look, guide, screen=said, then=lambda: ask_for_a_reading(
                    guide, getattr(guide, "ask_her_model", None), task=self.goal, tell=_said_while_playing))
        self.way_on_shown = any(0 < len(label) <= 40 and how_much_it_leads_on(label) > 1.0 for label in labels)
        from core.skills.current_words_before_play import remember_menu_pixels

        remember_menu_pixels(self, observation, menu=self.way_on_shown or reads_as_rules(said))
        self.labels_seen.update(label for label in labels if 0 < len(label.split()) <= 3)
        if said and (not self.words or self.words[-1] != said):
            self.words.append(said)
            del self.words[:-12]
            self._a_screen_not_seen_before(said)
            # Asked once, held to for the game: the next look, after the title's PLAY, is a watch for what is shown, and
            # the screens it is done on (WATCH, a score) never say it again.
            from core.agency.doing_again_what_was_shown import asks_to_follow

            if asks_to_follow(said):
                self.keep["asked_to_follow"] = True
            # And asked by the place's own screens, not by what she looked up or was counselled: offline 2026-10-12
            # advice found for a game of falling eggs read as asking for signs to be followed, and a line of signs
            # was looked for in it for eleven seconds of its third run.
            from core.agency.signs_that_come_along import told_to_follow_signs

            if told_to_follow_signs(said):
                self.keep["signs_told"] = True
        if self.first_ways_back is None and observation.get("ok", True):
            self.first_ways_back = restart_controls(observation)

    def _a_menu_first(self) -> bool:
        """Whether to read the screen, and go on from it, before playing what moves on it.

        A screen not yet read is read first; a screen whose words offer a way on (Start, Play, Next) is a menu,
        whatever moves on it, and is clicked through first: LIVE 2026-10-08 a title's fireflies were played for
        fifty-three seconds with START under them. A few times at most for one screen, in case its way on does nothing.
        """
        from core.cognition.what_this_place_is import a_way_on_glimpsed
        from core.language.a_way_on import offers_a_way_on

        if not self.words:
            return True
        said = self.words[-1]
        if not self.way_on_shown and not offers_a_way_on(said) and not reads_as_rules(said) and not a_way_on_glimpsed():
            return False
        key = said[:80]
        self.held_for_a_way_on[key] = self.held_for_a_way_on.get(key, 0) + 1
        return self.held_for_a_way_on[key] <= WAYS_ON_TRIED

    def _a_screen_not_seen_before(self, said: str) -> None:
        """Count a screen the game has not shown her before: getting to one is getting somewhere, scored or not.

        LIVE 2026-10-08 a game's menus, story and player choice took most of a round, nothing was scored on
        them, and the game was left as its play began. Kept across the rounds of one game; two screens are one
        where they share most of their words (core/agency/where_things_lead.py).
        """
        from core.agency.where_things_lead import SAME_SCREEN, screen_words

        words = screen_words([], said)
        if len(words) < 2:
            return
        seen: list[frozenset[str]] = self.keep.setdefault("screens_seen", [])
        if any(len(words & before) >= SAME_SCREEN * len(words | before) for before in seen):
            return
        seen.append(words)
        del seen[:-60]
        self.new_screens += 1

    def run_is_over(self, observation: dict[str, Any], *, can_do: Any = None) -> bool:
        """Whether she has played, and the screen now offers a way to start again it did not offer at first."""
        if self.over_because:
            return True
        self.keep["answered"] = self.keep.get("answered") or _answered_since_typed(self.keep, observation)
        if self.keep.get("answered"):
            # A question put to a thing that answers questions, and answered, is the thing done once.
            self.over_because, self.ending_words = MADE_NOT_WON, str(self.keep["answered"])
            return True
        from core.language.how_a_game_ended import what_it_asks_of_a_player

        # Menus and backgrounds can change without producing work. Use the
        # same current occupancy evidence that permits construction controls.
        if self.for_making_since is None and what_it_asks_of_a_player(" ".join(self.words)) == "make":
            self.for_making_since, self.made_before = time.monotonic(), self.acts_that_changed
        if self.for_making_since is not None:
            making = time.monotonic() - self.for_making_since
            if making >= TRYING_OUT_S and can_do is not None:
                from core.skills.screen_step_evidence import current_work_is_measured

                made = current_work_is_measured(can_do, observation) or (
                    not _made_by_placing(can_do) and self.acts_that_changed - self.made_before >= CHANGES_MADE_AT_LEAST)
            else:
                made = False
            if made:
                self.over_because = MADE_NOT_WON
                return True
            if making >= TRYING_OUT_MOST_S:
                self.over_because = MAKING_STALLED
                return True
        if not any(stretch.get("pictures") for stretch in self.stretches):
            return False
        from core.skills.screen_pursuit_bearings import restart_controls

        # Against what was on screen while she played, where she read it during
        # play: the screen that ended the last run already offered a way back,
        # so the first reading of this one is no measure of what is new.
        during = {said for stretch in self.stretches for said in stretch.get("words_seen") or ()}
        before = during if during else (self.first_ways_back or frozenset())
        offered = restart_controls(observation)
        appeared = {control for control in offered if control not in before}
        if offered and not appeared and self.play_stopped:
            # The game itself stopped: LIVE-like 2026-10-11 a second game of falling pieces ended on the screen the
            # first had ended on, its PLAY AGAIN no longer new, and she played the end screen for a minute and a half.
            appeared = set(offered)
        said_over = _newly_says_the_game_is_over(observation, during)
        if said_over and not appeared:
            # The screen says the game is over in words it did not show while she played, whatever way on it offers:
            # LIVE 2026-10-11 a two-minute game ended on "Your final score" over the Play its title had shown, she
            # played on into the next game, and each of her three runs was ended by her own clock part-way through one.
            self.over_because = f"the screen says the game is over ({said_over})"
            self.ending_words = " ".join(str(observation.get("text") or "").split())
            self.ending_parts = _lines_of(observation.get("layout") or [])
            logger.info("the run is over: %s", self.over_because)
            return True
        if offered and not appeared:
            logger.info("a way to start again is on screen (%s), but it was there while she played", ", ".join(sorted(offered)))
        if appeared:
            self.over_because = f"a way to start again appeared ({', '.join(sorted(appeared))})"
            self.ending_words = " ".join(str(observation.get("text") or "").split())
            self.ending_parts = _lines_of(observation.get("layout") or [])
            logger.info("the run is over: %s", self.over_because)
            return True
        return False

    def _no_place_for_shots(self) -> bool:
        """Whether the screen is one to be gone on from, not sent into: a way on shown, words that teach the rules, or
        words that ask her to choose. LIVE 2026-10-09 a shot on a putter lesson pressed its "back", and she went round
        title, welcome and lesson for a round without reaching the course.
        """
        said = self.words[-1] if self.words else ""
        return self.way_on_shown or reads_as_rules(said) or asks_to_choose(said)

    def held_to(self) -> Any:
        """The way she has taken up in this game, kept across its rounds (core/agency/the_way_it_is_played.py)."""
        from core.agency.the_way_it_is_played import HeldTo

        held = self.keep.get("held_to")
        if held is None:
            held = self.keep["held_to"] = HeldTo()
        return held

    def _judge(self, way: str, stretch: dict[str, Any]) -> None:
        """A stretch played one way, with the screens not seen before that came up since the last was judged."""
        self.played_at = time.monotonic()
        stretch.setdefault("new_screens", self.new_screens - self._screens_judged)
        self._screens_judged = self.new_screens
        said = " ".join([*self.words[-6:], str(self.keep.get("counsel") or "")])
        held = self.held_to()
        holding = held.way == way
        held.took(way, stretch, said)
        if holding and held.way != way:
            logger.info("left off playing it by %s: it had its turn and got nowhere", way)

    def goes_on_playing(self, observation: dict[str, Any], played_before: float) -> bool:
        """Whether, after a look, play goes straight on the way she holds to, without the pursuit looking round.

        Not when the run is over, a menu is up, nothing was played since the look before, or the way has stopped paying:
        then the screen is the pursuit's, to go on from as it goes on from any screen.
        """
        if self.played_at <= played_before or time.monotonic() >= self.ends_at:
            return False
        if not observation.get("ok", True) or self.way_on_shown or self.run_is_over(observation):
            return False
        return bool(self.held_to().holding())

    async def read_a_legend(self) -> None:
        """A screen that draws things beside words saying what to do about them is read as a legend, once a screen,
        while it is still up: what each drawn thing is, kept for play to know them by (core/perception/what_a_legend_shows.py).
        """
        import asyncio

        import numpy as np

        from core.perception.what_a_legend_shows import stance_of, what_a_legend_shows
        from core.perception.what_the_pixels_show import recognize_text

        said = self.words[-1] if self.words else ""
        read_before: set[str] = self.keep.setdefault("legends_read", set())
        # A screen of rules, or one with words enough to tell things apart: a score's "+ 12 ft." is neither.
        enough = reads_as_rules(said) or len(re.findall(r"[A-Za-z]{3,}", said)) >= LEGEND_WORDS
        if not said or said[:120] in read_before or not stance_of(said) or not enough:
            return
        read_before.add(said[:120])
        seen = await self.look()
        if seen is None:
            return
        picture = np.asarray(seen[0])
        regions = await asyncio.to_thread(recognize_text, np.ascontiguousarray(picture[..., ::-1]))
        shown = what_a_legend_shows(picture, regions)
        # A legend sets several things each beside what to do about it. One line with such a word in it, on a screen
        # that is not one of rules, is a title or a banner: LIVE 2026-10-11 "GET READY" over a game and a game's own
        # name with SMASH in it were each read as a thing shown to be got or shot.
        if not shown or (len({drawn.words for drawn in shown}) < 2 and not reads_as_rules(said)):
            return
        kept = list(self.keep.get("legend") or [])
        kept += [drawn for drawn in shown if all(drawn.words != k.words or drawn.where != k.where for k in kept)]
        self.keep["legend"] = kept[-24:]
        captions = list(dict.fromkeys(drawn.words for drawn in shown))
        logger.info("a legend: %s", [(d.words[:40], d.stance, d.colour) for d in shown])
        _said_while_playing("Its rules screen shows what things are: " + "; ".join(f"\u201c{c}\u201d" for c in captions[:3]) + ".")

    async def while_it_moves(self) -> None:
        """Play whatever is moving on its own, until it stops moving."""
        from core.agency.playing_as_it_happens import (
            controls_named_in,
            play_as_it_happens,
            the_world_moves_on_its_own,
        )
        from core.agency.the_way_it_is_played import SEND, ways_asked
        from core.perception.what_the_pixels_show import recognize_text

        now = time.monotonic()
        held = self.held_to()
        # Left alone on a moving screen that gave her nothing, she is not left alone on the next one: LIVE 2026-10-10 PLAY
        # led from an animated menu into a boat game, which played itself to GAME OVER in the quiet left on the menu.
        if now < self.quiet_until and self.said_now != self.quiet_on and _a_different_screen(self.said_now, self.quiet_on):
            self.quiet_until = 0.0
        from core.agency.the_controls_a_game_names import the_pointer_only_clicks, tried_in_turn
        from core.cognition.a_guide_to_a_place import THE_GUIDE as _THE_GUIDE

        # Everything the place has said of how it is played, not only its last screens: its instructions as read, kept
        # by the guide. Offline 2026-10-11 a golf game's "click on your ball and hold down the mouse button" was six
        # screens behind by the time the course came up, and on the course she clicked the words painted on it.
        heard = list(getattr(getattr(_THE_GUIDE.get(), "rules", None), "heard", None) or [])
        said = " ".join([*heard[-12:], *self.words[-6:], str(self.keep.get("counsel") or "")])
        # A place whose words name the pointer only to click on things, drag them or start is clicked through by the
        # pursuit, however its scenery moves (core/agency/the_controls_a_game_names.py the_pointer_only_clicks).
        only_clicks = the_pointer_only_clicks(said)
        in_time = (_played_in_time(said) or _it_goes_on_without_her(self)) and not only_clicks
        # What her own acts have shown outweighs what the place's last screens say: where she has been sending one
        # thing of hers in this place, struck where it lies, and it is in sight again, it is sent again. Offline
        # 2026-10-12 she holed a golf ball on the first hole, and on the second, its tee in plain sight, clicked shapes
        # "to see what they do" for a minute: the putter's instructions were six screens behind her.
        if now < self.ends_at and not self.way_on_shown and await _hers_to_send_is_in_sight(self, now):
            await self._by_shots(now, said, shown_by_her_acts=True)
            return
        # A place played in time is not left quiet but on its menus: a stretch with no body of hers found in it is how
        # a game played without one (a food fight seen from behind the player's own eyes) goes.
        if (now < self.quiet_until and not held.holding() and not (in_time and not self.way_on_shown and not _stuck_still(self))
                or now >= self.ends_at):
            _logged_leaving_alone(self, now)
            return
        # A screen that asks for words typed is answered before it is played or gone on from: offline 2026-10-11 a name
        # field's caret blinked, the screen was taken for one that moves on its own, and its GO was clicked for five
        # minutes with no name in the field.
        if await self._typed_what_it_asks():
            return
        # And a screen that asks her to take one of the things it lays out has one taken, before anything is pressed to
        # see what it does: offline 2026-10-12 "Choose a ball. Pick your favorite color!" stood over nine balls for a
        # minute and a half while she pressed return, space and the sign.
        if await _chose_what_it_asks(self):
            return
        # Once a place has asked for what it shows to be done again, a showing is watched for before anything else: the
        # screen after PLAY was still read as the title it replaced, taken for a menu, and the pursuit pressed a key to
        # see what it did while the first arrow was being shown (offline 2026-10-11). Where nothing is shown, as on a
        # menu, the watching gives up in a moment and the menu is gone on from as before.
        # What it shows may be coming along a line, each sign a key to be held as it gets there: that is played as it
        # happens (core/agency/signs_that_come_along.py), and is not waited on to light up and stop. Offline 2026-10-12
        # a book of arrows sent them down a strip; she watched for a showing to end, pressed a key to have it shown
        # again, and the three arrows that went by in the meantime were her three lives.
        # Seen once in a game, it is known for the game: the next screen of play is played from its first moment.
        # And where its words ask for signs to be followed, the loop that only answers signs has the screen once it is no
        # menu, and finds the line itself: offline 2026-10-12 the watch for a showing went first, saw nothing in order
        # for ten seconds, and the first round was lost before a key was held.
        signs_here = bool(self.keep.get("signs_told"))
        if not self.under_her_hand and self._a_menu_first():
            # Past a menu is another screen: a line not found before it may be looked for again.
            self.keep["signs_none"] = 0
            return
        if signs_here and await _answered_the_signs(self, now):
            return
        # A screen whose words ask for what it shows to be done again is watched, and followed, before it is played.
        if not self.keep.get("signs_going") and await self._did_again_what_it_showed():
            return
        # Asked afresh: the watch just made may have found signs coming.
        coming = bool(self.keep.get("signs_going"))
        in_time = in_time or coming
        # A board of two sides in turns is searched, not steered (core/agency/playing_a_board.py); a well with a piece
        # coming down it is played by where each piece lands best (core/agency/playing_a_well.py). Neither is looked for
        # where the place's words say it is played some other way.
        # Many places alike to open against a clock are opened in turn, fast (core/agency/searching_among_places.py).
        if not coming and await _searched_the_places(self, now):
            return
        otherwise = coming or _played_neither_by_turns_nor_by_pieces(self, said)
        if not otherwise and await _played_a_board(self, now):
            return
        if not otherwise and await _played_a_well(self, now):
            return
        # The way the place says it is played, before what the world does by itself: a thrower bobbing where he stands
        # is not a world to steer (core/agency/the_way_it_is_played.py).
        asked = _ways_asked_in_this_game(self, ways_asked(said, self.words[-1] if self.words else ""))
        way = held.choose(asked, said)
        if way == SEND and not self._no_place_for_shots():
            await self._by_shots(now, said)
            return
        # Played as it happens where it moves on its own, and where it moves for as long as she holds a key. Whether a
        # screen moves on its own is watched once a screen, not before every move on it: LIVE-like 2026-10-11 a menu
        # whose figures bobbed in place was watched for three and a half seconds before each of her clicks.
        # And where the place's words put her in a situation played in time (dodging, catching, shooting what comes),
        # the screen after its menus is played as it happens without first waiting to see it move: offline 2026-10-11
        # a food fight whose opponent rose from behind the tables now and then was watched still, played with arrows by
        # the slow pursuit, and lost in thirty seconds; LIVE 2026-10-10 a boat game the same.
        if not self.under_her_hand and not in_time and not await _it_moves_on_its_own(self, the_world_moves_on_its_own):
            # A world that waits for her: words it asks for are typed; where it says to send something, by shots. Only
            # there: LIVE 2026-10-11 a still level select, in a game played as it happens, had three minutes of pulls
            # and holds let go at it, and nothing sent.
            if (not await self._typed_what_it_asks() and SEND not in held.let_go and SEND in asked
                    and not self._no_place_for_shots()):
                await self._by_shots(now, said)
            return
        if not self.keep.get("hers") and not self.recalled:
            self.recalled = True
            self.keep.update(_what_she_kept_of(self.page))
        # What the place's screens said, and what taking stock found out (core/cognition/taking_stock.py), read alike.
        # And what the place said of how it is played, kept by the guide: offline 2026-10-12 a food fight's second round
        # began on its "BETTER LUCK NEXT TIME", nothing of "click to throw" among the words of it, and in every round
        # after the first she threw at no one.
        read = " ".join([self.goal, *heard[-12:], *self.words[-6:], str(self.keep.get("counsel") or "")])
        named, pointer_first = controls_named_in(read,
                                                keys_without_words=(), during_play=True)
        # And what the guide to the place holds of its controls: its page's words, its program, what she looked up
        # (core/cognition/a_guide_to_a_place.py), words before code.
        from core.cognition.a_guide_to_a_place import SCREEN as SCREEN_SAID
        from core.cognition.a_guide_to_a_place import THE_GUIDE, confirmed_by

        guide = THE_GUIDE.get()
        from core.cognition.a_guide_to_a_place import pointer_acts
        from core.cognition.what_this_place_is import (
            done_by_its_lesson,
            not_steered,
            said_not_steered,
        )

        done_by = not_steered(guide) if guide is not None and not named and not pointer_first else ""
        done_by = done_by or ("clicking and dragging things" if only_clicks and not self.under_her_hand else "")
        if guide is not None and guide.rules.a_lesson_plays(self.labels_seen):
            # A lesson new to her is playing: it is read, not played (core/cognition/reading_the_rules.py).
            logger.info("a lesson is playing: reading it, not playing it")
            self.quiet_until = time.monotonic() + guide.rules.beat()
            return
        # What its lesson asks is said by the place itself, and outweighs a key a screen happened to name.
        done_by = done_by or said_not_steered(guide) or done_by_its_lesson(guide)
        done_by = done_by or worked_by_typing(self.keep, guide, named)
        if done_by and not coming:
            # Nothing it says names a control for moving, and what it is says it is done by another way: what moves on
            # it is left moving, and the screen is acted on as a screen (core/cognition/what_this_place_is.py).
            logger.info("it moves on its own, but it is done by %s, not steered: leaving it to be acted on", done_by)
            self.quiet_until, self.quiet_on = time.monotonic() + LEAVE_A_MOVING_MENU_S, self.said_now
            return
        acts = pointer_acts(guide)
        if guide is not None:
            named = list(dict.fromkeys([*named, *_keys_of_play(guide, read)]))
            # Where keys move her and the pointer only aims, her body is looked for by the keys: what follows the
            # pointer then is where she aims.
            pointer_first = (pointer_first or guide.pointer_named()) and not guide.pointer_aims()
        if acts:
            # The pointer pressed is an act of hers ("click to jump"): its button is a key, tried and timed as any key
            # is, and what follows the mouse is the cursor, not her.
            logger.info("the pointer is pressed to %s: its button is one of her keys", acts)
            named = [*named, MOUSE_BUTTON]
            pointer_first = False
        # The game's controls are every key any of its screens has named, not
        # only this screen's: LIVE 2026-10-04 a run begun from the end screen
        # ("Press SPACE to play again") was played with space alone, and the
        # arrows the title screen had named were never pressed.
        from core.agency.body_trials_keep_the_world import keys_with_current_sources
        keys = keys_with_current_sources(guide, self.keep.get("named_keys") or [], named,
                                         _keys_of_play(guide, read) if guide is not None else [])
        # A thing whose words name the mouse and no key is played with the mouse: LIVE 2026-10-09 a putt game, "click the
        # ball, hold down the mouse while you aim", was played with the arrows a game is usually played with.
        if not keys and not (pointer_first or self.keep.get("pointer_named")):
            keys = controls_named_in("")[0]
        # Keys named only for doing something ("space to jump") say nothing of moving, and a person takes the arrows to
        # move: LIVE 2026-10-07 a platformer whose screen named space and up was played with those alone, its hero never
        # walked. Where nothing named moves her (no arrows, no WASD, no pointer), the arrows are tried too.
        if not pointer_first and not {"up", "down", "left", "right", "w", "a", "s", "d"} & set(keys):
            keys = [*keys, "left", "right", "up", "down"]
        pointer_first = (pointer_first or bool(self.keep.get("pointer_named"))) and not acts
        self.keep["named_keys"], self.keep["pointer_named"] = keys, pointer_first
        keys = tried_in_turn(keys, " ".join(self.words[-2:]))
        logger.info("it moves on its own: playing it as it happens with %s%s", keys, " and the pointer" if pointer_first else "")
        _what_she_understands_said(self.keep, guide, read)
        # The person's ask, apart from the screen's words: what she plays for is theirs first.
        self.keep["asked"] = self.goal
        stretch = await play_as_it_happens(
            self.look, self, keys=keys, seconds=min(STRETCH_S, self.ends_at - now),
            say=_said_while_playing, read_words=recognize_text, keep=self.keep,
            pointer_first=pointer_first, getting_somewhere=_getting_somewhere,
            told=read,
            waits_for_her=self.under_her_hand,
        )
        self.stretches.append(stretch)
        _what_play_showed_of_how_it_works(stretch)
        # A view of a bigger world she had not been in is somewhere new, as a screen she had not seen is.
        self.new_screens += int(stretch.get("new_views") or 0)
        self._judge(AS_IT_HAPPENS_WAY, stretch)
        if guide is not None:
            # What play read and confirmed, into the guide: what changed said, and what it learned of every place kept.
            for line in guide.take_in(SCREEN_SAID, stretch.get("words_seen") or [])[:1]:
                _said_while_playing(line)
            confirmed_by(guide, stretch)
        if (stretch.get("runtime_checks") or {}).get("violations"):
            self.over_because = "runtime contract violated"
        _keep_what_she_learned(self.page, self.keep)
        logger.info("a stretch played as it happened: %s", {k: stretch.get(k) for k in ("seconds", "ended", "hers", "keys_that_move_her", "pictures_a_second", "learned", "gains", "losses", "new_views", "views_along", "input_key_downs", "clicks", "taps")})
        _left_alone_after(self, stretch)

    async def _did_again_what_it_showed(self) -> bool:
        """Where the place's words ask for what it shows to be done again: each showing watched and done again, round
        after round, by the controls her presses have been seen to light (core/agency/doing_again_what_was_shown.py,
        core/agency/what_a_press_lights.py). Whether anything was done."""
        from core.agency.doing_again_what_was_shown import asks_to_follow, follow
        from core.agency.playing_as_it_happens import controls_named_in
        from core.agency.the_way_it_is_played import FOLLOW
        from core.agency.what_a_press_lights import WhatAPressLights
        from core.perception.what_the_pixels_show import recognize_text

        # Asked on any screen of this place, and held to for the game: the ask is on the title screen, and the screen it
        # is done on says WATCH and a score (offline 2026-10-11, a book of arrows: after PLAY nothing on the screen said
        # to follow, and she pressed keys to see what they did until her lives were gone).
        if not self.keep.get("asked_to_follow"):
            if not asks_to_follow(" ".join(self.words)):
                return False
            self.keep["asked_to_follow"] = True
        if FOLLOW in self.held_to().let_go:
            # It has had its turn here and got nowhere: the place is left to the pursuit until it says something new.
            return False
        tries = self.keep.setdefault("followed", {})
        key = self.words[-1][:80] if self.words else ""
        if tries.get(key, 0) >= FOLLOW_AT_MOST:
            return False
        tries[key] = tries.get(key, 0) + 1
        said = " ".join(self.words)
        named, _pointer_first = controls_named_in(said, keys_without_words=(), during_play=True)
        keys = list(dict.fromkeys([*(self.keep.get("named_keys") or []), *named]))
        lights = self.keep.setdefault("what_a_press_lights", WhatAPressLights())

        def a_round(record: dict[str, Any]) -> None:
            # Taken up as the way this place is played, so play goes on with it from look to look, and the pursuit does
            # not press keys to see what they do between one showing and the next.
            self.stretches.append(record)
            self.keep["signs_shown_in_turn"] = True         # its signs are lit and done again: no line of them to look for
            self.held_to().choose([FOLLOW], " ".join(self.words[-6:]))
            self._judge(FOLLOW, record)

        from core.agency.signs_that_come_along import SignsAreComing, watching_for_signs

        try:
            rounds = await follow(watching_for_signs(self.look, told=False), self, lights, keys=keys,
                                  clicks=bool(_CLICKED.search(said)), rounds_at_most=FOLLOW_AT_MOST,
                                  watch_s=WATCH_A_SHOWING_S, read_words=recognize_text, say=_said_while_playing,
                                  each_round=a_round)
        except SignsAreComing as seen:
            # What it shows comes along a line, each sign a key's: played as it happens, from this moment.
            logger.info("signs are coming along a line, each a key's: playing it as it happens")
            self.keep["signs_going"] = seen.args[0]
            for key in keys:
                await self.up(key)
            return False
        if not rounds:
            logger.info("asked to follow what is shown, and nothing was shown in order")
        return bool(rounds)

    async def _typed_what_it_asks(self) -> bool:
        """Where the screen's words ask for words typed, typed (core/agency/typing_what_is_asked.py), twice at most for
        one ask. Whether anything was typed."""
        from core.agency.typing_what_is_asked import type_what_is_asked
        from core.language.words_asked_for import ANSWER, QUESTION, words_asked_for
        from core.perception.what_the_pixels_show import recognize_text

        asked = words_asked_for(self.words[-1]) if self.words else None
        if asked is None:
            return False
        # What she typed before and the place answered is not typed again: the answer is what she came for.
        if self.keep.get("typed") and self.words:
            answered = _answered_since_typed(self.keep, {"text": self.words[-1]})
            if answered:
                self.keep["answered"] = self.keep.get("answered") or answered
                return False
        # Counted by what is asked for, not by how the screen happened to be read: a name asked for is typed twice at
        # most, however many misreadings of its heading there are.
        tries = self.keep.setdefault("typed_for", {})
        # One ask however its words are misread, but the screen's own numbering makes another ("Player 2" after "Player
        # 1"): offline 2026-10-11 a golf game asked the second player's name, and OK was pressed thirty times with none.
        # The numbering is the ask's own, in the words that asked, not every number on the screen: offline 2026-10-12
        # the stray numbers read off a name machine's screen made each reading another ask, and her name was typed into
        # it eight times while what it answered stood on the screen.
        numbered = " ".join(re.findall(r"\b(?:player|p|no\.?|number|#)\s*(\d+)\b", asked.asked_by, re.I))
        which = f"{asked.kind}:{numbered}"
        if tries.get(which, 0) >= 2:
            return False
        tries[which] = tries.get(which, 0) + 1
        done = await type_what_is_asked(asked, self.look, recognize_text, self, goal=self.goal,
                                        answer=_an_answer_she_has, say=_said_while_playing)
        logger.info("typed what the screen asks for: %s", done)
        if done.get("typed"):
            # What she put to it, for saying what it answered: a thing asked once has its question and its answer. Typed
            # once is answered: offline 2026-10-11 a name typed and left in its box was typed into again, "AuAurara".
            tries[which] = 2
            self.keep["typed"] = str(done["typed"])
            # What the place had shown before she typed, all its screens' readings: a label read cleanly only after is
            # no answer for having been misread before (offline 2026-10-11 "YOUR SECRET IDENTITY", read "GOUR SECRET
            # DENTLTY" until then, was reported as the machine's answer).
            self.keep["typed_kind"], self.keep["typed_on"] = str(done.get("kind") or ""), " ".join(self.words[-12:])
            self.keep["typed_asked_by"] = asked.asked_by
            # A question she put, or one put to her that she answered: a place of questions and answers.
            self.keep["asked_a_question"] = self.keep.get("asked_a_question") or done.get("kind") in (QUESTION, ANSWER)
            if done.get("kind") in (QUESTION, ANSWER):
                from core.agency.how_a_game_works import seen_to_type
                from core.cognition.a_guide_to_a_place import THE_GUIDE

                seen_to_type(THE_GUIDE.get())
            if done.get("kind") == QUESTION:
                self.keep["answered"] = await _what_it_answered(self.look, recognize_text)
        return bool(done.get("typed"))

    async def _by_shots(self, now: float, said: str = "", shown_by_her_acts: bool = False) -> None:
        """Shots, where the place's words speak of sending a thing by a press pulled or held and let go, or her own
        shots here have shown it.

        Not on a screen that offers a way on: a menu is gone through by its labels, whatever the game is played by
        once it begins. LIVE 2026-10-09 two games' menus ("Enter Code or Play", a title's START) were shot at.
        """
        from core.agency.how_a_game_works import seen_to_send
        from core.agency.playing_by_shots import play_by_shots, sends_by_letting_go
        from core.cognition.a_guide_to_a_place import THE_GUIDE as _THE_GUIDE
        from core.perception.what_the_pixels_show import recognize_text

        # Everything the place has said of how it is played, as whoever chose this way read it, and its last screens.
        told = "\n".join([said, *(_without_labels(line, self.labels_seen) for line in self.words[-6:]),
                          str(self.keep.get("counsel") or "")])
        if not shown_by_her_acts and (not sends_by_letting_go(told) or now < self.quiet_until):
            return
        if self._no_place_for_shots():
            return
        logger.info("it waits for her to send something: playing it by shots")
        shots = self.keep.setdefault("by_shots", {})
        stretch = await play_by_shots(self.look, self, seconds=min(STRETCH_S, self.ends_at - now), keep=shots,
                                      say=_said_while_playing, read_words=recognize_text, aim_at=_where_it_goes(told),
                                      her_name=_her_name_here(self),
                                      chosen=(self.keep.get("chose") or {}).get("colour"))
        logger.info("shots played: %s", stretch)
        record = {"pictures": stretch.get("shots", 0), "gains": stretch.get("gains", 0), "losses": 0,
                  "seconds": time.monotonic() - now, "ended": stretch.get("ended"), "by_shots": stretch}
        self.stretches.append(record)
        self._judge(SEND_WAY, record)
        _keep_what_she_learned(self.page, self.keep)
        if stretch.get("sent"):
            seen_to_send(_THE_GUIDE.get())
        elif not stretch.get("sends_from") and not shots.get("lies_where_struck"):
            self.quiet_until, self.quiet_on = time.monotonic() + LEAVE_A_MOVING_MENU_S, self.said_now
            self.held_to().never_did_it(SEND_WAY, " ".join([*self.words[-6:], str(self.keep.get("counsel") or "")]))

    def what_it_came_to(self) -> str:
        """One line on the play, for whoever handed the game over."""
        played = [s for s in self.stretches if s.get("pictures")]
        if not played:
            return ""
        seconds = sum(float(s.get("seconds") or 0.0) for s in played)
        last = played[-1]
        parts = [f"played it as it happened for {seconds:.0f} s"]
        if last.get("hers"):
            parts.append(f"I was the {last['hers']}")
        learned = last.get("learned") or {}
        if learned:
            parts.append("; ".join(f"{colour}: {stance}" for colour, stance in learned.items()))
        counters = last.get("counters") or {}
        if last.get("standing"):
            parts.append(str(last["standing"]).rstrip("."))
        elif counters:
            parts.append(", ".join(f"{name} {value}" for name, value in counters.items() if not name.startswith("number")))
        if self.over_because:
            parts.append(f"the run is over: {self.over_because}")
        return "; ".join(part for part in parts if part)


#: The mouse button pressed as a key, and where over the game it is pressed: the middle, a little low, clear of the
#: corners where a game keeps its pause and sound.
MOUSE_BUTTON = "mouse"
_WHERE_A_BUTTON_PRESS_GOES = (0.5, 0.6)


def _without_labels(said: str, labels: set) -> str:
    """A screen's words with its short labels taken out, each leaving a break: a button's "LAUNCH" is not part of the
    sentence written beside it (LIVE 2026-10-10 it read as one with the rules, and a steered game was played by shots)."""
    for label in sorted(labels, key=len, reverse=True):
        said = re.sub(rf"(?<!\w){re.escape(label)}(?!\w)", "\n", said)
    return said


async def _what_it_answered(look: Any, read_words: Any) -> str:
    """The screen's words once a question she put has gone in: its answer, kept as it showed it. Offline 2026-10-11 a
    fortune teller answered, she clicked on through its screens, and what it had said was gone when she came to say
    it."""
    import asyncio

    import numpy as np

    await asyncio.sleep(ANSWER_SHOWS_S)
    seen = await look()
    if seen is None:
        return ""
    regions = await asyncio.to_thread(read_words, np.asarray(seen[0])[:, :, 2::-1].copy())
    return " ".join(" ".join(str(region.get("text") or "").split()) for region in regions).strip()


def _where_it_goes(told: str) -> Any:
    """Where what she sends is to end up, as the pastime the place names says (a golf ball, into the hole), where that
    can be found on the picture (core/cognition/what_a_pastime_is_played_with.py): what anyone knows of a game named for
    golf before they play it. None where the place names no pastime."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.cognition.what_a_pastime_is_played_with import pastime_named_in, where_it_goes

    guide = THE_GUIDE.get()
    pastime = pastime_named_in(" ".join([str(getattr(guide, "place", "") or ""), told]))
    if pastime is None or not pastime.toward_looks:
        return None
    logger.info("a place of %s: what she sends goes to the %s", pastime.name, pastime.toward)
    return lambda picture, hers=None: where_it_goes(pastime, picture, hers)


async def _chose_what_it_asks(reflexes: PlayingAsItHappens) -> bool:
    """Where the screen's words ask her to choose and it lays things out to choose among, one is taken
    (core/agency/choosing_among_what_is_shown.py) and what she took is kept: it is hers from then on. Each thing
    once at most, a few times for one screen. Whether she took one."""
    import asyncio

    import numpy as np

    from core.agency.choosing_among_what_is_shown import (
        the_one_to_take,
        what_is_offered,
        what_she_took,
    )
    from core.perception.what_the_pixels_show import recognize_text

    said = reflexes.words[-1] if reflexes.words else ""
    if not asks_to_choose(said) or reflexes.way_on_shown and not _lays_out_a_choice(said):
        return False
    tried: list[Any] = reflexes.keep.setdefault("choices_tried", {}).setdefault(said[:80], [])
    if len(tried) >= CHOICES_TRIED:
        return False
    seen = await reflexes.look()
    if seen is None:
        return False
    picture = np.asarray(seen[0])
    try:
        words = await asyncio.to_thread(recognize_text, np.ascontiguousarray(picture[:, :, 2::-1]))
        # The screen in front of her now, not the one its words were last read from: offline 2026-10-12 a ball was
        # chosen, and on the hole's card that followed she "chose" a word of its name.
        now_says = " ".join(str(region.get("text") or "") for region in words)
        # Not by its little words: "ROUND ONE" after "Choose a token, pick the one you like" shares only "one".
        if not asks_to_choose(now_says) and _a_different_screen(now_says, said, shortest=4):
            return False
        taken = the_one_to_take(what_is_offered(picture, words, said), tried)
    except _A_FAULT_OF_HERS as why:
        logger.info("what the screen offers could not be made out: %s", why)
        return False
    if taken is None:
        return False
    tried.append(taken)
    _said_while_playing(f"It asks me to choose: I'm taking {what_she_took(taken)}.")
    logger.info("a choice asked (%r): taking %s", said[:60], taken)
    await reflexes.click(*taken.place)
    reflexes.keep["chose"] = {"colour": taken.colour, "label": taken.label, "asked": said[:120]}
    reflexes.played_at = time.monotonic()
    await asyncio.sleep(AFTER_A_CHOICE_S)
    return True


#: How many times a line of signs is looked for in a place whose words ask for signs to be followed and none has yet
#: been seen coming along one: after that its signs are taken to be shown some other way (lit in turn, say).
A_LINE_LOOKED_FOR = 2


async def _answered_the_signs(reflexes: PlayingAsItHappens, now: float) -> bool:
    """A stretch of answering the signs that come along a line, by the loop that does only that
    (core/agency/signs_that_come_along.py). Whether the screen was hers to answer signs on.

    Where none came, a line known from before is waited on once more; after that the screen is left to whatever else
    may play it. What is known of a line is kept while signs on it are judged: offline 2026-10-12 a line's ink was
    taken from another screen, nothing was seen on it afterwards, and it was read that way for the rest of the game."""
    from core.agency.signs_that_come_along import answer_signs_as_they_come

    keep = reflexes.keep
    none = int(keep.get("signs_none") or 0)
    if not _signs_may_come(reflexes):
        return False
    before = keep.get("signs_going")
    # Signs being answered are not left to go by while the round is looked at: offline 2026-10-12 a round's four
    # minutes were up twelve signs from the end of a game's last room, and the game was lost in the ten seconds the
    # looking took. The loop gives the screen back by itself when none are coming.
    ends = max(reflexes.ends_at, float(keep.get("all_of_it_ends") or 0.0))
    stretch = await answer_signs_as_they_come(reflexes.look, reflexes, seconds=min(STRETCH_S, ends - now),
                                              known=before, say=_said_while_playing)
    learned = stretch.pop("known")
    logger.info("signs answered as they came: %s", stretch)
    if not stretch.get("judged"):
        keep["signs_none"] = none + 1
        if before and not none:
            return True
        keep["signs_going"] = None
        return False
    keep["signs_going"], keep["signs_none"], keep["signs_seen"] = learned or before, 0, True
    keep["signs_showing_over"] = stretch.get("ended") == "no signs are coming"
    # A sign answered and judged is getting somewhere, whether or not the place counts it in sight.
    stretch["gains"], stretch["losses"] = int(stretch.get("judged") or 0), 0
    reflexes.stretches.append(stretch)
    reflexes.played_at = time.monotonic()
    reflexes._judge(AS_IT_HAPPENS_WAY, stretch)
    _keep_what_she_learned(reflexes.page, keep)
    return True


def _her_name_here(reflexes: PlayingAsItHappens) -> str:
    """The name she gave the place, where she gave one: what it draws beside that name marks which thing is hers."""
    from core.language.words_asked_for import NAME

    return str(reflexes.keep.get("typed") or "") if reflexes.keep.get("typed_kind") == NAME else ""


async def _hers_to_send_is_in_sight(reflexes: PlayingAsItHappens, now: float) -> bool:
    """Whether the one thing she has been sending in this place is on the screen now; looked for every few seconds,
    and only where she has sent one."""
    from core.agency.playing_by_shots import hers_is_in_sight
    from core.perception.what_the_pixels_show import recognize_text

    shots = reflexes.keep.get("by_shots") or {}
    if shots.get("sends_what") is None or not shots.get("lies_where_struck"):
        return False
    if now - float(reflexes.keep.get("hers_looked_for_at", -1e9)) < HERS_LOOKED_FOR_EVERY_S:
        return False
    reflexes.keep["hers_looked_for_at"] = now
    seen = await reflexes.look()
    if seen is None:
        return False
    try:
        return hers_is_in_sight(seen[0], shots, _her_name_here(reflexes), recognize_text)
    except _A_FAULT_OF_HERS as why:
        logger.info("her own thing could not be looked for: %s", why)
        return False


#: Words that ask whether she is sure of something that loses what she has; and the controls that agree and decline.
_A_WARNING = re.compile(r"\bare you sure\b|\b(?:lose|erase|delete|clear|reset)\s+(?:all\s+)?(?:your\s+|the\s+|current\s+)*"
                        r"(?:progress|work|changes|data|game|alien|save[ds]?)\b|\breally\s+want\s+to\s+quit\b", re.I)
_AGREES = re.compile(r"^\W*(?:ok(?:ay)?|yes|sure|confirm|quit|delete|erase|reset)\W*$", re.I)
_DECLINES = re.compile(r"^\W*(?:cancel|no|back|keep\s+playing|go\s+back|never\s*mind)\W*$", re.I)


def _starting_over_read_whole(observation: dict[str, Any]) -> None:
    """A "START" read alone with an "OVER" (or "AGAIN") just under or beside it is the one control "START OVER", a way
    back that throws away what was made, not a way on: offline 2026-10-12 an alien builder's START OVER, its two words
    set one above the other, was read as START, pressed twenty times, and each time asked whether to lose the alien."""
    import math

    layout = [region for region in observation.get("layout") or () if isinstance(region, dict) and "center_x" in region]
    # A screen that asks whether she is sure, warning what will be lost, is not agreed to: a player out to finish keeps
    # what is made and the game going. Its agreeing controls are left out of what she can press, and the way back
    # stays. Offline 2026-10-12 "ARE YOU SURE? YOU WILL LOSE ALL CURRENT PROGRESS ON YOUR ALIEN. CANCEL OK".
    said = " ".join(str(observation.get("text") or "").split())
    if _A_WARNING.search(said) and any(_DECLINES.match(" ".join(str(r.get("text") or "").split())) for r in layout):
        observation["layout"] = [region for region in observation.get("layout") or ()
                                 if not (isinstance(region, dict) and _AGREES.match(" ".join(str(region.get("text") or "").split())))]
    for region in layout:
        if " ".join(str(region.get("text") or "").split()).lower() not in ("start", "begin"):
            continue
        place = (float(region["center_x"]), float(region["center_y"]))
        for other in layout:
            word = " ".join(str(other.get("text") or "").split()).lower()
            if word in ("over", "again", "over!") and math.dist(place, (float(other["center_x"]), float(other["center_y"]))) <= 0.06:
                region["text"] = f"{region['text']} {other.get('text')}"
                break


def _ways_on_read_right_before(reflexes: PlayingAsItHappens, observation: dict[str, Any]) -> None:
    """Retire coordinate-only label guesses after current pixel recovery.

    The pixel-backed reader runs first. Position alone cannot replace a
    current counter, icon or unknown word with an earlier control label.
    """
    reflexes.keep.pop('ways_on_read', None)
    _starting_over_read_whole(observation)


@invariant('skills.coordinates_cannot_rename_current_words', scope='skills',
           owner='core/skills/screen_pursuit_as_it_happens.py', observational=False)
def _coordinate_label_invariant() -> tuple:
    from types import SimpleNamespace

    reflex = SimpleNamespace(keep={'ways_on_read': [('Start', .5, .65, time.monotonic())]})
    observation = dict(text='0%', layout=[dict(text='0%', center_x=.5, center_y=.65)])
    _ways_on_read_right_before(reflex, observation)
    assert observation['layout'][0]['text'] == observation['text'] == '0%'
    assert 'ways_on_read' not in reflex.keep
    return ()


def _ways_asked_in_this_game(reflexes: PlayingAsItHappens, asked: list[str]) -> list[str]:
    """The ways the place asks to be played, as its screens in front of her say, else as any of its screens said
    before: offline 2026-10-12 "Click on your ball and hold down the mouse button" was read six screens before the
    course (players, a name, a colour), was no longer among the words she went by there, and the course was
    played as a world going by itself, with not one putt."""
    if asked:
        reflexes.keep["ways_read"] = list(dict.fromkeys([*asked, *(reflexes.keep.get("ways_read") or [])]))
        return asked
    return list(reflexes.keep.get("ways_read") or [])

def _left_alone_after(reflexes: PlayingAsItHappens, stretch: dict[str, Any]) -> None:
    """After a stretch that found nothing of hers and gained and lost nothing, the screen is left to the pursuit a
    while; and stretches in a row that ended on a still screen are counted (see `_stuck_still`)."""
    nothing = not stretch.get("hers") and not stretch.get("gains") and not stretch.get("losses")
    still = nothing and str(stretch.get("ended") or "").startswith("nothing on the screen")
    reflexes.keep["still_stretches"] = int(reflexes.keep.get("still_stretches") or 0) + 1 if still else 0
    if nothing:
        reflexes.quiet_until, reflexes.quiet_on = time.monotonic() + _left_alone_for(reflexes), reflexes.said_now
        reflexes.under_her_hand = False

def _stuck_still(reflexes: PlayingAsItHappens) -> bool:
    """Whether play as it happens has ended on a still screen, with nothing to show for it, too many times in a row
    to be begun again at once, though the place goes on in time: offline 2026-10-12 a golf course and a menu of
    difficulties were each played for three seconds at a time, a hundred times over, and nothing else done."""
    return int(reflexes.keep.get("still_stretches") or 0) >= STILL_STRETCHES_AT_MOST


def _logged_leaving_alone(reflexes: PlayingAsItHappens, now: float) -> None:
    """That a screen is being left alone, logged once each time it is begun."""
    if now < reflexes.quiet_until and reflexes.keep.get("quiet_said") != reflexes.quiet_until:
        reflexes.keep["quiet_said"] = reflexes.quiet_until
        logger.info("leaving this screen alone for %.0fs more", reflexes.quiet_until - now)


def _left_alone_for(reflexes: PlayingAsItHappens) -> float:
    """How long a moving screen that answered to nothing is left alone: twice as long each time the same screen does it
    again. LIVE 2026-10-11 a building game's characters idled on its board, and every twenty seconds she spent nineteen
    finding again that nothing there answered to her."""
    kept = reflexes.keep.get("left_alone_on")
    times = kept[1] + 1 if kept is not None and not _a_different_screen(reflexes.said_now, kept[0]) else 1
    reflexes.keep["left_alone_on"] = (reflexes.said_now, times)
    return min(LEFT_ALONE_AT_MOST_S, LEAVE_A_MOVING_MENU_S * 2 ** (times - 1))


async def _played_a_board(reflexes: PlayingAsItHappens, now: float) -> bool:
    """Where what the page draws is a board of two sides in turns, it is played by search, its rules read from what the
    place has said (core/agency/playing_a_board.py); whether it was. Looked for every few seconds, not every look."""
    from core.agency.playing_a_board import a_board_game, play_a_board
    from core.perception.a_board_seen import a_board_in
    from core.perception.a_picture_examined_without_holding_the_loop import UNMEASURED, PictureProbe
    from core.perception.visual_probe_cache import VisualProbeMiss, VisualProbeWitness

    if now - float(reflexes.keep.get("board_looked_for_at", -1e9)) < BOARD_LOOKED_FOR_EVERY_S:
        return False
    reflexes.keep["board_looked_for_at"] = now
    seen = await reflexes.look()
    # What the place said of its rules is kept from one stretch of play to the next: LIVE 2026-10-11 a second stretch
    # of a draughts game, its rules screen long gone, was played without the piece its rules had changed.
    heard: list[str] = reflexes.keep.setdefault("words_of_the_place", [])
    heard += [w for w in reflexes.words if w not in heard]
    del heard[:-WORDS_KEPT]
    words = " ".join([*heard, str(reflexes.keep.get("counsel") or "")])
    context = (id(getattr(reflexes, "page", None)), tuple(getattr(reflexes, "band", ())),
               reflexes.keep.get("hers_in", ""))
    witness = VisualProbeWitness.measure(seen[0], context=context, inputs=words) if seen is not None else None
    rejected = reflexes.keep.get("no_board_on")
    if isinstance(rejected, VisualProbeMiss) and rejected.applies(witness, now=now):
        return False
    if seen is not None and getattr(reflexes, '_board_probe', None) is None:
        reflexes._board_probe = PictureProbe()
    board = await reflexes._board_probe.examine(a_board_in, seen[0]) if seen is not None else None
    if board is UNMEASURED:
        reflexes.keep['board_looked_for_at'] = -1e9
        return False
    if board is None or not a_board_game(board, words):
        reflexes.keep["no_board_on"] = VisualProbeMiss(witness, now) if witness is not None else None
        return False
    reflexes.keep.pop("no_board_on", None)
    logger.info("a board of two sides in turns (%s): playing it by search", a_board_game(board, words))
    stretch = await play_a_board(reflexes.look, reflexes, words, seconds=max(1.0, min(STRETCH_S, reflexes.ends_at - now)),
                                 say=_said_while_playing)
    logger.info("a board played: %s", stretch)
    # What the other side lost is what she gained, and turns the game took are the game going on.
    reflexes.stretches.append({"pictures": stretch.get("moves", 0), "turns": stretch.get("moves", 0),
                               "gains": int(stretch.get("ended") == "won") + int(stretch.get("taken") or 0),
                               "losses": int(stretch.get("ended") == "lost") + int(stretch.get("lost") or 0),
                               "seconds": time.monotonic() - now, "ended": stretch.get("ended"),
                               "settled": stretch.get("ended")})
    if stretch.get("ended") in ("won", "lost"):
        reflexes.over_because = f"the board game ended: {stretch['ended']}"
    else:
        # A board played until its time ran out is looked for again at once, not tried with keys meanwhile.
        reflexes.keep["board_looked_for_at"] = -1e9
    return True


#: What a fault in her own seeing or playing of a well can raise: any of these leaves the screen to her other ways.
_A_FAULT_OF_HERS = (ImportError, IndexError, KeyError, ValueError, TypeError, AttributeError, ArithmeticError, RuntimeError,
                    OSError)
#: The keys a piece coming down a well is tried with, beside any the place has named.
_KEYS_FOR_A_PIECE = ("left", "right", "up", "down", "space")


#: How many times in one game a ground may be taken for a well, with no key of hers moving its piece, before it is
#: not looked for there again.
NO_WELL_AFTER = 2
#: How long, in one run, a well she has played before is waited at for a piece to come down it before the screen is
#: left to her other ways.
WAITS_AT_A_WELL_S = 45.0


async def _played_a_well(reflexes: PlayingAsItHappens, now: float) -> bool:
    """Where what the page draws is a well with a piece coming down it, each piece is put where it lands best, her keys
    and what clears found by trying and watching (core/agency/playing_a_well.py); whether it was. Looked for every few
    seconds, and not again this run where nothing she pressed moved the piece. A well she has played before, seen with
    nothing yet coming down it, is waited at: LIVE-like 2026-10-11 a second game of falling tiles opened on a moment
    with no piece in it, was steered about in for fifty seconds as a place to move through, and was lost by then."""
    from core.agency.playing_a_well import (
        NOTHING_MOVES_IT,
        a_well_with_a_piece_coming_down,
        play_a_well,
    )
    from core.perception.a_picture_examined_without_holding_the_loop import UNMEASURED
    from core.perception.visual_probe_cache import VisualProbeMiss, VisualProbeWitness

    if reflexes.no_well_here or int(reflexes.keep.get("no_well_found", 0)) >= NO_WELL_AFTER \
            or now - float(reflexes.keep.get("well_looked_for_at", -1e9)) < BOARD_LOOKED_FOR_EVERY_S:
        return False
    reflexes.keep["well_looked_for_at"] = now
    first = await reflexes.look()
    context = (id(getattr(reflexes, "page", None)), tuple(getattr(reflexes, "band", ())),
               reflexes.keep.get("hers_in", ""))
    inputs = " ".join([*getattr(reflexes, "words", []), str(reflexes.keep.get("counsel") or "")])
    witness = VisualProbeWitness.measure(first[0], context=context, inputs=inputs) if first is not None else None
    rejected = reflexes.keep.get("no_well_on")
    if isinstance(rejected, VisualProbeMiss) and rejected.applies(witness, now=now) and not reflexes.keep.get("well_keys"):
        return False
    last = first

    async def watching():
        nonlocal first, last
        if first is not None:
            last, first = first, None
        else:
            last = await reflexes.look()
        return last

    empty: list[Any] = []
    try:
        well = await a_well_with_a_piece_coming_down(watching, reflexes.keep.get("well_seen"), empty)
    except _A_FAULT_OF_HERS as why:
        # Looking for a well is never what ends a game: LIVE 2026-10-11 a fault in it ended every game she opened.
        from core.runtime.errors import record_degradation

        logger.exception("looking for a well failed")
        record_degradation("playing_a_well", why, action="no well is looked for again this run")
        reflexes.no_well_here = True
        return False
    if well is UNMEASURED:
        reflexes.keep['well_looked_for_at'] = -1e9
        return False
    if well is None:
        waited = time.monotonic() - now
        if empty and reflexes.keep.get("well_keys") and reflexes.waited_at_a_well + waited < WAITS_AT_A_WELL_S:
            reflexes.waited_at_a_well += waited
            reflexes.keep["well_looked_for_at"] = -1e9
            return True
        witness = VisualProbeWitness.measure(last[0], context=context, inputs=inputs) if last is not None else None
        reflexes.keep["no_well_on"] = VisualProbeMiss(witness, now) if witness is not None else None
        return False
    reflexes.keep.pop("no_well_on", None)
    reflexes.keep["well_seen"] = well
    logger.info("candidate cells %d across and %d down with apparent falling motion: checking current control responses",
                well.columns, well.rows)
    heard: list[str] = reflexes.keep.setdefault("words_of_the_place", [])
    heard += [w for w in reflexes.words if w not in heard]
    del heard[:-WORDS_KEPT]
    words = " ".join([*heard, str(reflexes.keep.get("counsel") or "")])
    keys = list(dict.fromkeys([*_KEYS_FOR_A_PIECE, *(k for k in reflexes.keep.get("named_keys") or [] if k != MOUSE_BUTTON)]))
    try:
        stretch = await play_a_well(reflexes.look, reflexes, well, words,
                                    seconds=max(1.0, min(STRETCH_S, reflexes.ends_at - now)),
                                    say=_said_while_playing, offered=keys, known=reflexes.keep)
    except _A_FAULT_OF_HERS as why:
        # A fault in playing the well is hers to hear of and not the game's to lose by: the screen goes to her other
        # ways for the rest of the run. Offline 2026-10-11 one such fault ended a whole run two minutes in.
        from core.runtime.errors import record_degradation

        logger.exception("playing the well failed")
        record_degradation("playing_a_well", why, action="left the well to her other ways for this run")
        reflexes.no_well_here = True
        return False
    logger.info("a well played: %s", stretch)
    if stretch.get("ended") == NOTHING_MOVES_IT:
        # Not a well this run; and taken for one twice in a game with no key moving its piece, not a well in this game.
        reflexes.no_well_here = True
        reflexes.keep["no_well_found"] = int(reflexes.keep.get("no_well_found", 0)) + 1
        reflexes.keep.pop("well_seen", None)
        return False
    if not stretch.get('controls_confirmed'):
        return False
    reflexes.keep["no_well_found"] = 0
    reflexes.waited_at_a_well = 0.0
    reflexes.stretches.append({"pictures": stretch.get("pieces", 0), "turns": stretch.get("pieces", 0),
                               "gains": int(stretch.get("cleared") or 0), "losses": 0,
                               "seconds": time.monotonic() - now, "ended": stretch.get("ended"),
                               "settled": stretch.get("ended")})
    reflexes.played_at = time.monotonic()
    if stretch.get("ended") == "out of time":
        # A well played until its time ran out is looked for again at once, not tried with keys meanwhile.
        reflexes.keep["well_looked_for_at"] = -1e9
    elif stretch.get("pieces"):
        # Nothing more comes down, or the well is covered: the game has stopped. The screen it stopped on is read, not
        # steered about in.
        reflexes.play_stopped = True
        reflexes.quiet_until, reflexes.quiet_on = time.monotonic() + LEAVE_A_MOVING_MENU_S, reflexes.said_now
    return True


def _lines_of(regions: list[dict[str, Any]]) -> list[str]:
    """A screen's writing as lines: pieces at one height, left to right.

    Text recognition hands back pieces, and a title in large letters comes
    back as one piece a word ("You", "win!") read in among the smaller line
    below it. Pieces whose middles are within half a piece's height of each
    other are one line.
    """
    pieces = []
    for region in regions:
        text = str(region.get("text") or "").strip()
        if not text:
            continue
        height = float(region.get("height") or 0.0)
        middle = float(region.get("center_y") if region.get("center_y") is not None else float(region.get("y") or 0.0) + height / 2)
        left = float(region.get("x") or region.get("center_x") or 0.0)
        pieces.append((middle, height, left, text))
    lines: list[list[tuple[float, float, float, str]]] = []
    for piece in sorted(pieces):
        for line in lines:
            if abs(line[0][0] - piece[0]) <= 0.5 * max(line[0][1], piece[1], 1e-6):
                line.append(piece)
                break
        else:
            lines.append([piece])
    return [" ".join(text for _m, _h, _l, text in sorted(line, key=lambda p: p[2])) for line in lines]



#: How long her model is given to answer a question a place asks where the place waits for her answer.
ANSWER_WHILE_IT_WAITS_S = 75.0


def _time_to_answer() -> float:
    """How long her model may take over a question a place asks: a place that waits for her answer is given the time a
    thought takes; one that goes on without her, only a moment. Offline 2026-10-11 a riddle on a screen that waited was
    not put to her model at all, since it needed some thirty-five seconds and was allowed twenty."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    if guide is not None and hasattr(guide, "controls"):
        from core.agency.how_a_game_works import understood

        if understood(guide).goes_on_without_her():
            return ANSWER_WITHIN_S
    return ANSWER_WHILE_IT_WAITS_S


async def _an_answer_she_has(question: str) -> str | None:
    """Her answer to a question a screen asks, from her model as one witness, within a few seconds; else none."""
    from core.cognition.taking_stock import from_her_model
    from core.rebuilding.her_model import ask_her_model

    try:
        heard = await from_her_model(ask_her_model)([question], _time_to_answer())
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError, TimeoutError) as why:
        logger.info("no answer to %r: %s", question[:80], why)
        return None
    for one in heard:
        said = re.sub(r"^(?:the answer is|it is|it's)\s+", "", str(one.text or "").strip(), flags=re.I).rstrip(".")
        if said:
            return said
    return None


def _said_while_playing(line: str) -> None:
    from core.skills.screen_pursuit import _tell

    _tell(line)


def _getting_somewhere(said: str) -> None:
    from core.runtime.still_getting_somewhere import it_got_somewhere
    from core.skills.sovereign_browser_one_question import SAYING_IT_MOVES

    (SAYING_IT_MOVES.get() or it_got_somewhere)(said)


#: The reflexes for the hand-over in progress, when there is one.
AS_IT_HAPPENS: contextvars.ContextVar[PlayingAsItHappens | None] = contextvars.ContextVar(
    "aura_playing_as_it_happens", default=None
)


async def looked_at_as_it_happens(look: Any) -> dict[str, Any]:
    """``await look()``, with the hand-over's reflexes around it when there are any.

    What moves on its own is played first, and the look is of the screen that
    play ends on; the reflexes then read what the look saw.
    """
    reflexes = AS_IT_HAPPENS.get()
    played_before = reflexes.played_at if reflexes is not None else 0.0
    if reflexes is not None:
        from core.skills.current_words_before_play import read_current_words

        fresh = await read_current_words(reflexes, goes_on=_it_goes_on_without_her(reflexes))
        if fresh is None or not reflexes.run_is_over(fresh):
            await reflexes.while_it_moves()
    seen = await look()
    if reflexes is None:
        return seen
    reflexes.read(seen)
    seen = await _waited_out_while_made_ready(reflexes, look, seen)
    await reflexes.read_a_legend()
    # A screen of play in a place played in time is played the moment it is seen, not after a move has been decided on
    # it and made: offline 2026-10-11 the pursuit clicked a food fight's corner to see what it did, two seconds of a
    # ten-second round, before play began.
    # Where signs may be coming along a line, so is the screen that a choice just made led to: offline 2026-10-12 a
    # game went from its choice of who to be straight into play, the choice counted as having played, a key was
    # pressed to see what it did, and the round was lost six seconds on with no sign answered.
    if ((reflexes.played_at <= played_before or _signs_may_come(reflexes)) and a_screen_of_play_in_time(reflexes)
            and not reflexes.run_is_over(seen)):
        await reflexes.while_it_moves()
        if reflexes.played_at > played_before:
            seen = await look()
            reflexes.read(seen)
    # And a screen that asks her to choose among things it shows is answered the moment it is read, as a person
    # answers it, not after a move has been made on it to see what it does.
    words = getattr(reflexes, "words", None)
    if words and asks_to_choose(words[-1]) and await _chose_what_it_asks(reflexes):
        seen = await look()
        reflexes.read(seen)
    # Held to a way that may still pay, she plays on with it: the pursuit is given the screen when the run is over, a
    # menu is up, or the way has had its turn and got nowhere (core/agency/the_way_it_is_played.py).
    while reflexes.goes_on_playing(seen, played_before):
        played_before = reflexes.played_at
        await reflexes.while_it_moves()
        seen = await look()
        reflexes.read(seen)
    return seen


#: The longest a screen that says it is still being made ready is waited out, and how often it is looked at meanwhile.
READY_WAIT_S = 45.0
READY_LOOK_EVERY_S = 1.5


async def _waited_out_while_made_ready(reflexes: PlayingAsItHappens, look: Any, seen: dict[str, Any]) -> dict[str, Any]:
    """A screen whose words say it is still being made ready (loading, downloading, launching), with no way on shown, is
    looked at again until it says otherwise, for a while: what is on it is not yet the thing. LIVE 2026-10-09 an
    archive's "Downloading game metadata..." screen had its logo clicked to see what it did."""
    import asyncio

    from core.language.a_way_on import says_it_is_being_made_ready

    waited = 0.0
    said_it = False
    while (waited < READY_WAIT_S and not reflexes.way_on_shown
           and says_it_is_being_made_ready(" ".join(str(seen.get("text") or "").split()))):
        if not said_it:
            said_it = True
            _said_while_playing("It's still loading; waiting for it.")
        await asyncio.sleep(READY_LOOK_EVERY_S)
        waited += READY_LOOK_EVERY_S
        seen = await look()
        reflexes.read(seen)
    if waited:
        logger.info("waited %.1fs for a screen that said it was being made ready", waited)
    return seen


def a_handed_over_run_is_over(observation: dict[str, Any], made: int = 0, *, can_do: Any = None) -> bool:
    """A run played and over ends a hand-over; what follows belongs to whoever handed it over. ``made`` is how many of
    her acts so far changed what was on the screen."""
    reflexes = AS_IT_HAPPENS.get()
    if reflexes is not None:
        reflexes.acts_that_changed = made
    return reflexes is not None and reflexes.run_is_over(observation, can_do=can_do)
