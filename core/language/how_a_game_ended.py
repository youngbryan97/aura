"""How a game ended, read from the words on its last screen, and whether a request asks for a win.

An end screen says who won in a sentence: "You win!", "The computer wins.",
"Game over", "Congratulations, you beat level 3", "Try again". Who won is the
subject of the winning verb. When it is the player ("you", "player") she won;
when it is an explicit opponent role she lost. A named person has no side until that side is known.
A screen that names no winner but says the
game is over, or asks to try again, is a loss, and so is a player with nothing left to lose ("Lives: 0"). Two
sides' scores written out on a terminal screen ("Game over. You 5 Computer 3") settle it by the numbers.

It abstains when the words do not say, so an unread ending is never reported
as a win.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Final

from core.verify import invariant

__all__ = ["asks_to_win", "declares_a_win", "how_it_ended", "how_it_ended_in", "names_a_measure", "offers_a_win",
           "says_a_round_is_over", "says_an_unresolved_result", "says_play_is_ongoing", "the_rating_in", "the_score_in",
           "what_it_asks_of_a_player", "what_says_the_game_is_over"]

_WINNING: Final = re.compile(r"\b(win|wins|won|winner|victory|victorious|beat|beats|defeated|champion)\b")
_RESULT_VERB: Final = re.compile(r"\b(win|wins|won|winner|victory|victorious|beat|beats|defeat|defeats|defeated|champion|"
                                 r"lose|loses|lost|died|crashed|failed|caught|captured)\b")
_THE_PLAYER: Final = frozenset({"you", "player", "player 1", "player one", "p1", "your", "yours"})
_THE_OPPONENT: Final = frozenset({"computer", "cpu", "opponent", "enemy", "rival", "bot", "ai", "com", "away",
                                "player 2", "player two", "p2", "other player"})
_THE_TASK: Final = frozenset({"game", "round", "stage", "level", "wave", "mission", "world"})
#: An end lost, as end screens say it. LIVE 2026-10-09 "Out of Time. You have failed." was read as no end at all.
#: And a player consoled: offline 2026-10-12 "Oh, no! Dad's home! ... Better luck next time!" was read as no end at
#: all, and "Welcome home! Great job, kids!", the same game won, as the same.
_OVER: Final = re.compile(r"\b(game over|you lose|you lost|try again|out of lives|no lives|time'?s up|you died|defeat|"
                          r"(?:you )?(?:have )?failed|mission failed|out of time|time over|you crashed|crash(?:ed)?|"
                          r"(?:you were|you got|you've been) (?:caught|captured)|busted|better luck next time)\b")
_PRAISED: Final = re.compile(
    r"\b(congratulations|congrats|well done|you did it|you made it|victory|"
    r"(?:great|good|nice|excellent|awesome|fantastic|super) (?:job|work)|"
    r"(?:level|stage|round|wave|mission|world)\s*\d*\s*(?:complete|completed|clear|cleared|passed|beaten))\b"
)
_SIDES: Final = re.compile(r"\b(you|player)\s*:?\s*(\d+)\D{1,24}?\b([a-z]+)\s*:?\s*(\d+)\b")
#: What a player has of their own to lose, at nothing: "Lives: 0", "health 0", "0 hearts".
_NOTHING_LEFT: Final = re.compile(r"\b(lives|life|health|hp|hearts|chances|balls left|tries left)\s*:?\s*0\b|\b0\s+(lives|hearts|chances)\b")

#: Current play contradicts a predicted end, even where the board or its counters were misread.
_ONGOING: Final = re.compile(
    r"\b(?:your|my|his|her|their|our|the computer'?s|computer'?s|opponent'?s|player(?:\s*\d)?'?s)\s+turn\b|"
    r"\b(?:it(?:'s| is) )?(?:your|my|his|her|their) move\b|"
    r"\b(?:game|round|level|mission|match|play)\s+(?:is\s+)?(?:in progress|still running|continues|not over)\b|"
    r"\bwaiting for (?:the )?(?:computer|opponent|other player)(?:'s)? (?:to )?(?:move|turn)\b|"
    r"\b(?:resume|continue) (?:the |your )?(?:game|round|match)\b"
)
#: A result named as a rule, possibility, denied event or past run does not report this run's end.
_UNASSERTED: Final = re.compile(
    r"\b(?:if|when|whenever|unless|until|before|after|once|could|would|should|will|might|may|can|cannot|can't|"
    r"must|not|never|don't|doesn't|didn't|did not|do not|does not|avoid|prevent)\b|"
    r"\b(?:to|how to|ways? to|in order to)\s+(?:win|lose|beat|defeat|crash|die)\b|"
    r"\bfirst\s+(?:(?:player|one|side)\s+)?to\b|"
    r"\b(?:last|previous|earlier)\s+(?:round|game|run|attempt|match)\b|"
    r"\b(?:or|otherwise)\s+(?:you|the player|player|the computer|computer)\b"
)
#: A subject-free base verb with an object is an action to perform, not a result already achieved.
_DIRECTIVE_RESULT: Final = re.compile(
    r"(?:^|:\s*)(?:please\s+)?(?:win|lose|beat|defeat|crash|die)\s+[a-z0-9]+\b"
)


def says_play_is_ongoing(words: str) -> bool:
    """Whether the current screen asks for a turn or explicitly says play is still going."""
    # OCR may split one prompt into separate pieces ("YOUR", "TURN TO MAKE A MOVE").
    for clause in _clauses(str(words or "").replace("\n", " ")):
        if not _ONGOING.search(clause):
            continue
        if re.search(r"\b(?:if|when|whenever|unless|until|before|after|once|will|would|could|might)\b|"
                     r"\b(?:last|previous|earlier)\s+(?:round|game|run|attempt|match)\b", clause):
            continue
        if re.search(r"\bturn\s+(?:is\s+)?(?:over|ended|finished|complete)\b", clause):
            continue
        return True
    return False


def _clauses(words: str) -> list[str]:
    """Separate writing before normalization removes its line breaks."""
    return [_plain(clause) for clause in re.split(r"[.!?;\n]+", str(words or "")) if clause.strip()]


def _asserted(clause: str) -> bool:
    """Whether an ending named in this clause reports a present fact."""
    return not (_UNASSERTED.search(clause) or _DIRECTIVE_RESULT.search(clause))


def _plain(text: str) -> str:
    # A zero drawn in a game's font is read as a letter as often as a digit
    # ("Computer o", "You Ø"); standing alone where a score stands, it is one.
    value = re.sub(r"(?<=\s)[oOØø](?=\s|$)", "0", str(text or ""))
    # "Won't" is "will not", not "won": offline 2026-10-13 a game's instructions, "you won't score as well", were read
    # as the game saying she had won it, on its first screen. However its apostrophe is read: as another mark, as a
    # space, or not at all ("won´t", "won t", "wont"), as it was again on the same screen in a later pass.
    value = re.sub(r"\bwon(?:\s*[^\w\s]\s*|\s+)?t\b", "will not", value, flags=re.IGNORECASE)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(value.casefold().split())


def _who_won(clause: str, found: re.Match[str] | None = None) -> str:
    """The subject of a result verb in a clause, or "" where there is none."""
    found = found or _RESULT_VERB.search(clause)
    if not found:
        return ""
    before = clause[: found.start()].strip(" ,!.:")
    before = re.sub(r"\b([a-z]+)'(?:ve|s|re|d)\b", r"\1", before)
    words = re.findall(r"[a-z0-9]+", before)
    # Auxiliaries and a nearby score are not the subject ("you have won", "You You 5 win!").
    while len(words) > 1 and (words[-1] in ("have", "has", "had", "been", "is", "are", "was", "were", "got", "the") or words[-1].isdigit()):
        if len(words) >= 2 and words[-2] == "player":
            break
        words.pop()
    subject = " ".join(words[-2:]) if len(words) >= 2 and (words[-2] == "player" or words[-2:] == ["other", "player"]) else (words[-1] if words else "")
    if found.group(1) in ("winner", "champion", "victory", "victorious"):
        after = re.findall(r"[a-z0-9]+", clause[found.end():])
        if not subject or subject == "the":
            while after and after[0] in ("is", "are", "for", "to", "the"):
                after.pop(0)
            subject = " ".join(after[:2]) if len(after) >= 2 and after[0] == "player" else (after[0] if after else "")
    return subject


def how_it_ended_in(parts: list[str]) -> str:
    """How it ended, read from a screen's separate pieces of writing first.

    Text read off a screen in one run interleaves its pieces: LIVE 2026-10-04
    "You win!" in large letters over "You 5 Computer 0" came out as "You You 5
    win! Computer o Press SPACE to play again", in which nobody wins, and
    "play again" said only that it was over, so a win was taken for a loss.
    A piece that says who won, or what each side scored, is taken first; a
    screen that only says it is over is read whole.
    """
    words = "\n".join(parts)
    if says_play_is_ongoing(words):
        return ""
    readings = [_who_it_says_won(part) for part in parts]
    if None in readings:
        return ""
    stated = set(readings) - {""}
    if len(stated) > 1:
        return ""
    # Where no piece says who won, the pieces are read on from one to the next as well, as a person reads them: a reader
    # that split "GAME" from "OVER" left every run of a game unread (offline 2026-10-12, five runs, each over).
    return next(iter(stated)) if stated else how_it_ended(words) or how_it_ended(" ".join(parts))


def _who_it_says_won(text: str) -> str | None:
    """The result from explicit roles; None for an unbound participant or conflicting results, "" for no result."""
    if not text:
        return ""
    if says_play_is_ongoing(text):
        return ""
    outcomes = set()
    for clause in _clauses(text):
        if not _asserted(clause):
            continue
        for result in _RESULT_VERB.finditer(clause):
            who = _who_won(clause, result)
            if not who or who in _THE_TASK:
                continue
            if who not in _THE_PLAYER and who not in _THE_OPPONENT:
                return None
            player = who in _THE_PLAYER
            verb = result.group(1)
            if verb in ("lose", "loses", "lost", "died", "crashed", "failed", "caught", "captured"):
                player = not player
            if verb == "defeated" and re.search(r"\b(?:was|were|is|are|been|got)\s*$", clause[:result.start()]):
                player = not player
            outcomes.add("won" if player else "lost")
    return None if len(outcomes) > 1 else next(iter(outcomes)) if outcomes else ""


def says_an_unresolved_result(words: str) -> bool:
    """Whether current result assertions conflict or name a participant whose side is not known.

    An unread screen leaves room for another source of terminal evidence. An explicit unresolved result cannot be
    replaced by a prediction that silently assigns its named participant to one side.
    """
    clauses = [clause for clause in _clauses(words) if _asserted(clause)]
    readings = [_who_it_says_won(clause) for clause in clauses]
    if None in readings or len(set(readings) - {""}) > 1:
        return True
    sides = _SIDES.search(_plain(words))
    return bool(sides and sides.group(3) not in _THE_OPPONENT | {"to", "and", "points"} and
                any(_ROUND_OVER.search(clause) or _OVER.search(clause) or _PRAISED.search(clause) for clause in clauses))


#: What an end screen says over a round, won or lost, as against the rules' words for winning and losing ("defeat the
#: robots", "don't crash"), which are read in play too.
_ROUND_OVER: Final = re.compile(
    r"\b(game over|try again|play again|retry|replay|you (?:lose|lost|win|won|died|crashed)|out of (?:lives|time)|"
    r"time'?s up|(?:level|stage|round|mission|wave) (?:complete|completed|cleared|failed)|mission failed|well done|"
    r"congratulations)\b")


def says_a_round_is_over(words: str) -> bool:
    """Whether writing says a round is over, the way an end screen says it: "GAME OVER", "TRY AGAIN", "Level Complete"."""
    if says_play_is_ongoing(words):
        return False
    return bool(how_it_ended(words) or any(_asserted(clause) and (_ROUND_OVER.search(clause) or _repeat_at_the_end(clause))
                                         for clause in _clauses(words)))


def _repeat_at_the_end(clause: str) -> bool:
    """An OCR-damaged repeat label at the end of an asserted screen reading.

    Use the same bounded spelling comparison as navigation. A result counter
    before the label must not prevent the actor from handing the screen back.
    This supplies no winner or score.
    """
    from core.language.a_way_on import nearly_says

    words = clause.split()
    return any(nearly_says(' '.join(words[-size:]), ('try again', 'play again'))
               for size in range(1, min(len(words), 3) + 1))


@invariant('language.damaged_repeat_labels_supply_no_winner', scope='language',
           owner='core/language/how_a_game_ended.py', observational=False)
def _damaged_repeat_invariant() -> tuple:
    assert says_a_round_is_over('Attempts: 3 Accuracy: 0% TRYJAGAIN J')
    assert how_it_ended('Attempts: 3 Accuracy: 0% TRYJAGAIN J') == ''
    assert not says_a_round_is_over('If you miss, TRYJAGAIN J')
    assert not says_a_round_is_over('Your turn. PLAYAGAlN')
    return ()


#: What a screen says when the whole game is over, not a level of it: its last count, or that it is lost or won.
_THE_GAME_OVER: Final = re.compile(
    r"\b(game over|(?:your )?final score|you scored|your score (?:is|was)|out of (?:lives|time)|time'?s up|"
    r"you (?:lose|lost|died|crashed|win|won)|try again|play again)\b")


def what_says_the_game_is_over(words: str) -> str:
    """The words of a screen that say the game is over ("game over", "final score"), in small letters; "" where none
    do. A level finished is not the game over, and is not told here."""
    if says_play_is_ongoing(words):
        return ""
    for clause in _clauses(words):
        if _asserted(clause):
            found = _THE_GAME_OVER.search(clause)
            if found:
                return found.group(1)
    return ""


def how_it_ended(words: str) -> str:
    """"won", "lost", or "" where the words do not say."""
    text = _plain(words)
    if not text or says_play_is_ongoing(words):
        return ""
    clauses = [clause for clause in _clauses(words) if _asserted(clause)]
    readings = [_who_it_says_won(clause) for clause in clauses]
    if None in readings:
        return ""
    declared = set(readings) - {""}
    if len(declared) > 1:
        return ""
    if declared:
        return next(iter(declared))
    terminal = any(_ROUND_OVER.search(clause) or _OVER.search(clause) or _PRAISED.search(clause) for clause in clauses)
    sides = _SIDES.search(text)
    if terminal and sides and sides.group(3) not in ("to", "and", "points"):
        if sides.group(3) not in _THE_OPPONENT:
            return ""
        mine, theirs = int(sides.group(2)), int(sides.group(4))
        if mine != theirs:
            return "won" if mine > theirs else "lost"
    if any(_PRAISED.search(clause) for clause in clauses):
        return "won"
    if any(_OVER.search(clause) or _NOTHING_LEFT.search(clause) for clause in clauses):
        return "lost"
    return ""


_ASKS: Final = re.compile(
    r"\b(until (?:you|she|i) (?:win|beat)|and win\b|win (?:it|the game|against|a game|at least)|"
    r"beat (?:it|the game|them|the computer|the level|each|every|all)|to win\b|try to win|"
    r"until (?:you've|you have) won)"
)


def asks_to_win(request: str) -> bool:
    """Whether a request asks for the game to be won, not only played."""
    return bool(_ASKS.search(_plain(request)))


_NUMBERS: Final = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_A_COUNT: Final = r"(\d{1,3}|" + "|".join(_NUMBERS) + r")"
_ATTEMPTS: Final = re.compile(r"\b(?:play|try|make|take|do|run|give)\s+(?:(?:it|the game|this game|that game|them)\s+)?"
                              r"(?:(?:for|at most|up to)\s+)?" + _A_COUNT + r"\s+(?:attempts?|tries|times|rounds?|runs?|goes)\b")
_BEST_OF: Final = re.compile(r"\bbest (?:of|out of)\s+" + _A_COUNT + r"\b")

def _a_count_in(text: str) -> int | None:
    found = _ATTEMPTS.search(text) or _BEST_OF.search(text)
    if not found:
        return None
    word = found.group(1)
    value = _NUMBERS.get(word) if word in _NUMBERS else int(word)
    return value if value > 0 else None


def requested_attempts(request: str) -> int | None:
    """An explicit count of play attempts, independent of whether winning is asked.

    A count said in a condition on one shape of thing ("if it only keeps a
    score, play it three times") belongs to that shape
    (core/language/what_counts_as_done.py), not to every game played.
    """
    from core.language.what_counts_as_done import without_conditions

    return _a_count_in(_plain(without_conditions(request)))


#: A score as a screen writes it: "Your score: 1,250", "Points 40", "Total = 300".
#: A score as a last screen writes it; and a run measured, not counted ("TOTAL DISTANCE: 24 ft", "Height: 120 m"): LIVE
#: 2026-10-09 a hamster launch's end screen wrote its total distance, and she said no score was shown.
_A_SCORE: Final = re.compile(r"\b(?:(?:total|final|your|best)\s+)?(?:score|points|pts|total|distance|height|altitude|depth)\b"
                             r"\s*[:=]?\s*(\d[\d,]*)")
#: Progress a game can be won by, counted: "Level 2", "Stage 1", "Round 3".
_PROGRESS: Final = re.compile(r"\b(?:level|stage|round|wave|mission|world|chapter)\s*\d")


def the_score_in(words: str) -> int | None:
    """The last score a screen's words write out, or None where they write none."""
    found = list(_A_SCORE.finditer(_plain(words)))
    return int(found[-1].group(1).replace(",", "")) if found else None


#: A measure named on a last screen, whether or not its number is drawn: "YOUR SCORE:", "Your rating: LONG SHOT".
_NAMES_A_MEASURE: Final = re.compile(r"\b(?:(?:your|final|total|high|best)\s+)?(?:score|rating|rank|grade|points)\b\s*[:=]")
#: A rating as a last screen writes it, up to whatever goes on from there.
_A_RATING: Final = re.compile(r"\b(?:your\s+)?(?:rating|rank|grade)\s*[:=]\s*([a-z][a-z' !-]{1,28}?)"
                              r"(?=\s+(?:play|try|again|continue|next|menu|back|your|score|click|press)\b|\s*$)")


def names_a_measure(words: str) -> bool:
    """Whether a screen's words name a measure of the run (a score, a rating), with its number drawn or not. A game's
    last screen wrote "GAME OVER  YOUR SCORE:  YOUR RATING: CRACKED UP" with nothing drawn after the score (offline
    2026-10-12, a player that leaves a game's own number fields blank), and four finished runs were four games lost."""
    return bool(_NAMES_A_MEASURE.search(_plain(words)))


def the_rating_in(words: str) -> str:
    """The rating a last screen gives the run in words ("CRACKED UP", "LONG SHOT"), or ''."""
    found = _A_RATING.search(_plain(words))
    return found.group(1).strip(" -!").upper() if found else ""


def offers_a_win(words: str) -> bool:
    """Whether a game's words anywhere hold out a win: a winner, a goal to beat, or levels and stages to clear."""
    text = _plain(words)
    return bool(_TO_WIN.search(text) or _WINNING.search(text) or _PRAISED.search(text) or _PROGRESS.search(text))


_FOR_POINTS: Final = re.compile(
    r"\b(as many (?:points|\w+) as (?:possible|you can)|highest score|high score|beat your (?:best|score|record)|"
    r"get the most|score as (?:many|much|high)|before (?:the )?time runs out|in (?:one|two|three|\d+) minutes?|"
    r"for (?:bonus |extra |more )?points)\b"
)
_TO_WIN: Final = re.compile(r"\b(first to|wins?\b|beat (?:the|him|her|them|your opponent)|defeat|win the|to win)\b")
#: An objective a player is set, that is reached and then over: someone rescued, all of something found, a place got to.
_OBJECTIVE: Final = re.compile(
    r"\b(rescue|save (?:the|him|her|them|his|their)|escape (?:from|the)|find (?:all|every|the|his|her|their)|"
    r"collect (?:all|every)|reach the (?:end|top|exit|goal|finish|other side)|solve|"
    r"help (?:\w+ ){1,3}(?:to )?(?:find|get|escape|save|rescue|stop|reach|collect)|"
    # Something brought to a place is an end reached: "get a jawbreaker into a cup", "put the ball onto the green".
    r"(?:get|put|bring|guide|roll|drop|land) (?:\w+ ){1,3}(?:into|onto) (?:a|an|the|his|her|their|your) \w+)\b"
)
#: Words that set a player to make something, not to win: a maker, a dress-up, a paint box. Picking favourites is making
#: where there are several things to pick ("your favorite body parts"); one pick ("Pick your favorite color!" for a golf
#: ball, LIVE 2026-10-09) is a choice inside a game, and that game was left as a thing for making after ninety seconds.
_FOR_MAKING: Final = re.compile(
    r"\b((?:create|make|design|build|draw|paint|invent) your own|dress (?:up|him|her|them)|decorate|customi[sz]e|"
    r"mix and match|(?:choose|pick|select|selecting|choosing|picking) your (?:own |favou?rite )(?:\w+ )?\w+s\b|"
    r"your own (?:character|creation)|use your imagination|"
    # A thing that answers what is put to it is played once as one for making is: asked, and its answer told.
    r"type (?:in )?(?:only )?(?:your |a |any )?(?:\W?yes\W? or \W?no\W? )?questions?|"
    r"ask (?:\w+ ){0,4}(?:a|any|your) (?:\W?yes\W? or \W?no\W? )?questions?|"
    # "make the craziest, creepiest, or goofiest scenes you can think of": made things, said with how they are to be.
    r"(?:make|create|design|draw|compose|set up)(?: the| a| an| some)?(?: \w+,?){0,5} (?:scenes?|pictures?|photos?|"
    r"characters?|creations?|songs?|beats?|tunes?|music|designs?|outfits?|drawings?|comics?|stories|story|"
    r"masterpieces?|mixes?|looks?)\b)"
)


def _making_surface() -> Any:
    """The learned surface over the floor above: sentences that set a player to make something rather than win, declared
    with phrasings the floor gets wrong both ways (core/language/learned_matcher.py)."""
    from core.language.learned_matcher import LearnedMatcher
    from core.language.model_features import model_hidden_features

    return LearnedMatcher(
        name="sets_a_player_to_make",
        positives=("Create your own character by choosing a head, a body and a costume.",
                   "Use your imagination to make the craziest scenes you can think of.",
                   "Mix the beats and make your own song.", "Dress her up for the party in your favourite outfit.",
                   "Design a monster and watch it come to life.", "Set up the scene and take a photo for the album."),
        negatives=("Make it to the end of the level before time runs out.", "Collect as many coins as you can.",
                   "Help Dexter find his missing robot.", "Make the highest score you can.",
                   "Use the arrow keys to move and space to jump.", "Build a tower as high as you can before it falls."),
        features=model_hidden_features)


MAKING = _making_surface()


def _learned_it_makes(text: str) -> bool:
    """Whether the learned surface has decided a sentence of ``text`` sets a player to make something; never waits."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence.split()) < 4:
            continue
        try:
            if MAKING.decide_without_waiting(sentence):
                return True
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return False
    return False


def what_it_asks_of_a_player(words: str) -> str:
    """What a game's own words set a player to do: "win" it, run up a "score", "make" something, or "" where they do not say.

    A game that counts points against a clock has no winner to be: Toonami:
    Tunnel Rush says "you have two minutes to collect as many points as
    possible". Asked to win one, the honest end is a finished run and its
    score, not a run of losses until a clock that never declares a winner.
    And a thing for making has none either: LIVE 2026-10-07 "Create your own
    KND by selecting your favorite body parts ... click on the GENERATE
    button" was played four minutes and reported "not won".
    """
    text = _plain(words)
    if _TO_WIN.search(text):
        return "win"
    if _FOR_POINTS.search(text):
        return "score"
    made = _FOR_MAKING.search(text)
    if made:
        # What the floor is sure of teaches the surface the sentence it was said in.
        MAKING.observe(next((part for part in re.split(r"(?<=[.!?])\s+", text) if _FOR_MAKING.search(part)), text),
                       holds=True)
        return "make"
    if _learned_it_makes(text):
        return "make"
    # Set an objective ("Help Dexter find his missing robot"), a player has an end to reach.
    if _OBJECTIVE.search(text):
        return "win"
    return ""


def declares_a_win(words: str) -> bool:
    """Whether a thing's words name a winner or a win: "You win!", "Winner", "Victory"."""
    return how_it_ended(words) == "won"
