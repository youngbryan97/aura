"""What the people of a place ask of her and what they offer for it: an errand log, kept from what they say.

A game with people in it often moves forward through them. "Bring me my glasses and I'll open the gate." "If you
find the tambourine, I'll give you the pineapple." A person playing keeps a list in their head of who wants what and
what they will get for it. When they come across the thing, they know whose it is, and they take it back. The people
in the recordings of these games did just that. They read what each character said, went off to find it, and came
back with it. One errand often led to the next: the pineapple got for the tambourine was what a third person wanted.

So whatever a place's words ask of her is kept as an errand: what is wanted, what is offered, and who asked, where
the words name a speaker ("Old Man: bring me my glasses"). The thing wanted is a thing to get from then on. The
errand is said once, the way a person thinks aloud ("So he wants his glasses, and he'll open the gate for them"),
and given to what she reasons with.

Nothing here knows a game.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Errand", "errands_in"]

#: What a thing wanted or offered may be called: a few words, ended by the end of its clause.
def _a_thing(name: str) -> str:
    return (r"(?:the |a |an |my |some |your |that |this |his |her |our |their )?(?P<" + name +
            r">[a-z][a-z'-]*(?: (?!(?:to|for|and|or|in|at|on|by|from|back|somewhere|around)\b)[a-z][a-z'-]*){0,3}?)"
            r"(?=\s*(?:[.,!;?:]|$|\s(?:and|so|then|if|before|or|for|in|to|from|back|please|because|somewhere|around|"
            r"near|by|at|on|under|inside|outside|with)\b))")


_WANTS = re.compile(
    r"\b(?:bring|give|fetch|find|get|hand|return|take)\s+me\s+(?:back\s+)?" + _a_thing("a") + r"|"
    r"\b(?:i|we)\s+(?:need|want|lost|dropped|am looking for|'m looking for|can't find|cannot find)\s+" + _a_thing("b") + r"|"
    r"\b(?:can|could|will|would)\s+you\s+(?:please\s+)?(?:find|bring|get|fetch)\s+(?:me\s+)?" + _a_thing("c") + r"|"
    r"\bif\s+you\s+(?:can\s+)?(?:bring|find|get|fetch|give)\s+(?:me\s+)?" + _a_thing("d") + r"|"
    r"\b(?:give|bring|take|return)\s+" + _a_thing("e") + r"\s+(?:back\s+)?to\s+(?P<to>him|her|them|[A-Z][a-z]+(?: [A-Z][a-z]+)?)\b",
    re.I)
_OFFERS = re.compile(
    r"\b(?:i'll|i will|i can|we'll|we will|he'll|he will|she'll|she will|they'll|they will)\s+(?:give you|let you have|"
    r"trade you|swap you|hand you|reward you with)\s+" + _a_thing("given") + r"|"
    r"\b(?:i'll|i will|i can|we'll|we will|he'll|he will|she'll|she will|they'll|they will)\s+"
    r"(?P<verb>open|unlock|raise|lower|fix|mend|show you|teach you)\s+" + _a_thing("done") + r"|"
    r"\bin exchange for\s+" + _a_thing("swap"), re.I)
#: Who is speaking, where a line names them as a script does: "Old Man: ...".
_SPEAKER = re.compile(r"^\s*([A-Z][A-Za-z.' -]{1,24}?)\s*:\s+\S")
#: What is never a thing wanted: the asking itself, and how a thing is.
_NOT_A_THING = frozenset({"it", "them", "this", "that", "something", "anything", "help", "out", "here", "there", "back",
                          "up", "down", "home", "away", "more", "one", "some", "go", "you", "to", "time", "a hand"})


_ACTS_OFFERED = frozenset({"open", "unlock", "raise", "lower", "fix", "mend", "show", "teach"})


@dataclass
class Errand:
    """What is wanted, what is offered for it, who asked, and the words it was asked in."""

    want: str
    offer: str = ""
    who: str = ""
    said: str = ""

    def says(self) -> str:
        """The errand as a person thinking it over says it."""
        whose = self.who or "someone here"
        wants = "want" if whose == "they" else "wants"
        line = f"So {whose} {wants} the {self.want}"
        if self.offer and self.offer.split()[0] in _ACTS_OFFERED:
            line += f", and {'they' if whose in ('someone here', 'they') else 'he' if whose == 'he' else 'she' if whose == 'she' else whose} will {self.offer} for it"
        elif self.offer:
            line += f", and I'll get the {self.offer} for it"
        if self.want.endswith("s") and not self.want.endswith("ss"):
            line = line.replace(" for it", " for them")
        return line + "."


def _named(found: re.Match[str] | None, *names: str) -> str:
    if found is None:
        return ""
    words = next((found.group(n) for n in names if found.group(n)), "") or ""
    words = words.lower().strip(" '-")
    return "" if not words or words in _NOT_A_THING or words.split()[0] in _NOT_A_THING else words


def errands_in(sentence: str) -> list[Errand]:
    """The errands one thing said asks for: what is wanted in it, and what is offered for it, where it says."""
    text = " ".join(str(sentence or "").split())
    speaker = _SPEAKER.match(text)
    who = speaker.group(1).strip() if speaker and len(speaker.group(1).split()) <= 3 else ""
    body = text[speaker.end(1) + 1:] if who else text
    asked = _WANTS.search(body)
    wanted = _named(asked, "a", "b", "c", "d", "e")
    if not wanted:
        return []
    if not who and asked is not None and asked.group("to"):
        to = asked.group("to")
        who = to if to[:1].isupper() and to.lower() not in ("him", "her", "them") else {"him": "he", "her": "she"}.get(
            to.lower(), "they")
    offered = _OFFERS.search(body)
    offer = ""
    if offered is not None:
        thing = _named(offered, "given", "done", "swap")
        if thing and thing != wanted:
            offer = f"{offered.group('verb').lower()} the {thing}" if offered.group("verb") else thing
    return [Errand(wanted, offer, who, text[:200])]
