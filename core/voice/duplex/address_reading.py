"""Was that said to her? Her own model reads which talk the words belong to.

An open microphone hears her person talking to her, but also the television, one side of a phone call, someone else in
the room, and him talking to himself. Lists of phrases cannot tell these apart: "can you see how it thickens?" is a
cooking show, "can you pick up some bread?" is a phone call, and "the blue one" is an answer to her, and no list of
openers or pronouns knows which is which. Measured 13 November 2026 on 121 declared utterances from eleven kinds of
situation, the phrase ladder this replaces answered 46 of the 70 that were not said to her.

What does tell them apart is which conversation the words continue. Her address reader reads the words three times:

- after her conversation, as the person's next turn (``hers``);
- after what her microphone has overheard lately that was not said to her (``room``), when it has overheard anything;
- after nothing at all (``alone``).

Each reading is the mean log-probability per token the reader gives the words. Words said to her are predicted better
by her conversation than by the room's talk or by nothing; the lead of ``hers`` over the better of the other two is the
reading, and she answers when it clears a margin measured on the same reader (tools/voice/measure_address_margin.py,
artifacts/voice/address_margin.json).

The reader is not her cortex. Measured 13 November 2026, her cortex gives the end of a person's turn a probability near
one at every position inside it, so how likely it finds the person's words says nothing about them. A small chat model
that still models both sides of a conversation reads them, in a process of its own (address_process.py). Her
conversation as her live context holds it is what it reads, so her name in that conversation is what tells it who
"Aura" is.

Nothing here is an instruction to a model. The three texts are her conversation as she read it, the words her
microphone overheard, and nothing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.Voice.AddressReading")

#: Where the margin her address reader's readings are judged against is written, with how it was measured.
MARGIN_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "voice" / "address_margin.json"

#: Where a text begins for the reader when nothing comes before it: the token her tokenizer puts between documents.
DOCUMENT_BEGINS = "<|endoftext|>"

#: How much overheard talk is kept as the room's text. It bounds what the reader reads for the room on each utterance
#: (about five hundred tokens) and the cache that text holds; older talk falls off the front.
ROOM_KEEPS_CHARS = 2000

# (text, words, cache scope) -> the log-probability of each token of the words after the text, or None.
Scorer = Callable[[str, str, str], Awaitable[Sequence[float] | None]]


def spoken_words(transcript: str) -> str:
    """The words as her live context writes them into her conversation (live_context.HerLiveContext.answer)."""
    return " ".join(str(transcript or "").split())


@dataclass(frozen=True, slots=True)
class AddressReading:
    """How well each text predicts the words, in mean log-probability per token."""

    hers: float
    alone: float
    room: float | None = None
    tokens: int = 0
    #: The words after the turn's opening alone, with no conversation before it: how likely any turn of talk makes
    #: them. None for a reading taken without it, which then counts the room as it reads.
    generic: float | None = None

    @property
    def room_beyond_talk(self) -> float | None:
        """What the room's talk predicts beyond what any turn of talk would.

        Overheard lines make any conversational line more likely, whoever it is said to: measured 14 November 2026,
        once one line had been overheard in a session, "Hello? Is anyone there?" read as the room's (lead 1.17) and
        went unanswered. The room is a rival only for what it adds over talk in general, which is the room's
        reading less the turn opening's own lift over nothing.
        """
        if self.room is None:
            return None
        if self.generic is None:
            return self.room
        return self.room - (self.generic - self.alone)

    @property
    def lead(self) -> float:
        """How much better her conversation predicts the words than the room's talk or nothing does."""
        room = self.room_beyond_talk
        rival = self.alone if room is None else max(self.alone, room)
        return self.hers - rival

    @property
    def rival(self) -> str:
        room = self.room_beyond_talk
        return "the room's talk" if room is not None and room > self.alone else "nothing at all"

    def as_dict(self) -> dict[str, Any]:
        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, 3)

        return {"hers": round(self.hers, 3), "alone": round(self.alone, 3), "room": rounded(self.room),
                "generic": rounded(self.generic), "lead": round(self.lead, 3), "tokens": self.tokens}


@dataclass(frozen=True, slots=True)
class AddressMargin:
    """The lead she answers above, as measured on one model."""

    margin: float
    model: str
    measured_at: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> AddressMargin | None:
        target = path or MARGIN_PATH
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            margin = float(data["margin"])
            model = str(data["model"])
        except FileNotFoundError:
            return None  # not a failure: no margin measured yet
        except (OSError, ValueError, KeyError, TypeError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation("voice_duplex.address_reading", exc,
                               action="judged address without a measured margin", severity="warning")
            return None
        if not math.isfinite(margin) or not model:
            return None
        return cls(margin=margin, model=model, measured_at=str(data.get("measured_at") or ""))


class RoomTalk:
    """What her microphone overheard that was not said to her, kept as the room's text."""

    def __init__(self, keep_chars: int = ROOM_KEEPS_CHARS) -> None:
        self._lines: list[str] = []
        self._keep = max(1, int(keep_chars))

    def overheard(self, transcript: str) -> None:
        words = spoken_words(transcript)
        if not words:
            return
        self._lines.append(words)
        while len(self._lines) > 1 and sum(len(line) + 1 for line in self._lines) > self._keep:
            self._lines.pop(0)

    def text(self) -> str | None:
        """The room's text for the reader, or None when nothing has been overheard."""
        if not self._lines:
            return None
        return DOCUMENT_BEGINS + "\n".join(self._lines) + "\n"


def _mean(values: Sequence[float] | None) -> float | None:
    if not values:
        return None
    numbers = [float(v) for v in values]
    if not all(math.isfinite(v) for v in numbers):
        return None
    return sum(numbers) / len(numbers)


async def read_address(transcript: str, *, hers_text: str, hers_scope: str, room: RoomTalk, room_scope: str,
                       alone_scope: str, scorer: Scorer, opening: str = "",
                       opening_scope: str = "opening") -> AddressReading | None:
    """The reader's readings of the words, or None when the reading after her conversation or after nothing failed.

    ``opening`` is the turn's opening with no conversation before it; when the room has talk in it, the words are
    read after that too, so the room counts only for what it adds over talk in general.
    """
    words = spoken_words(transcript)
    if not words or not hers_text:
        logger.info("no address reading: %s", "no words" if not words else "no conversation text to read them after")
        return None
    hers_values = await scorer(hers_text, words, hers_scope)
    hers = _mean(hers_values)
    if hers is None:
        logger.info("no address reading: the words after her conversation read as %r", hers_values)
        return None
    alone_values = await scorer(DOCUMENT_BEGINS, words, alone_scope)
    alone = _mean(alone_values)
    if alone is None:
        logger.info("no address reading: the words after nothing read as %r", alone_values)
        return None
    room_text = room.text()
    room_reading = _mean(await scorer(room_text, words, room_scope)) if room_text else None
    generic = _mean(await scorer(opening, words, opening_scope)) if room_reading is not None and opening else None
    return AddressReading(hers=hers, alone=alone, room=room_reading, tokens=len(hers_values or ()), generic=generic)


class AddressReader:
    """One voice session's reader: her live context, the room's talk, and her address reader to read them with."""

    def __init__(self, live: Any, *, session_key: str, scorer: Scorer | None = None,
                 margin: AddressMargin | None = None, model_name: Callable[[], str] | None = None) -> None:
        self._live = live
        self._key = str(session_key or "voice")[:40]
        self._scorer = scorer or _read_in_her_address_process
        self._model_name = model_name or _her_address_reader_name
        self.room = RoomTalk()
        self._margin = margin if margin is not None else AddressMargin.load()
        self._mismatch_reported = False

    @property
    def margin(self) -> AddressMargin | None:
        """The margin, when it was measured on the model her conversation is read with now."""
        if self._margin is None:
            return None
        running = self._model_name()
        if running and running != self._margin.model:
            if not self._mismatch_reported:
                self._mismatch_reported = True
                from core.runtime.errors import record_degradation

                record_degradation(
                    "voice_duplex.address_reading",
                    RuntimeError(f"margin measured on {self._margin.model}, her address reader is {running}"),
                    action="judged address without readings until the margin is measured on her address reader",
                    severity="warning")
            return None
        return self._margin

    async def read(self, transcript: str) -> AddressReading | None:
        hers = await self._live.text_before(transcript) if hasattr(self._live, "text_before") else None
        if not hers:
            logger.info("no address reading: her live context gave no text for the words to follow")
            return None
        opening = str(getattr(self._live, "_opening", "") or "")
        return await read_address(transcript, hers_text=hers, hers_scope=f"hers:{self._key}", room=self.room,
                                  room_scope=f"room:{self._key}", alone_scope="alone", scorer=self._scorer,
                                  opening=opening if hers.endswith(opening) else "")

    def overheard(self, transcript: str) -> None:
        self.room.overheard(transcript)


async def _read_in_her_address_process(text: str, words: str, scope: str) -> Sequence[float] | None:
    from core.voice.duplex.address_process import read_in_her_address_process

    return await asyncio.to_thread(read_in_her_address_process, text, words, scope)


def _her_address_reader_name() -> str:
    from core.voice.duplex.address_process import her_address_reader_name

    return her_address_reader_name()


__all__ = [
    "DOCUMENT_BEGINS",
    "MARGIN_PATH",
    "ROOM_KEEPS_CHARS",
    "AddressMargin",
    "AddressReader",
    "AddressReading",
    "RoomTalk",
    "read_address",
    "spoken_words",
]
