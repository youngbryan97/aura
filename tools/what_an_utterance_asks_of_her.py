#!/usr/bin/env python3
"""tools/what_an_utterance_asks_of_her.py — is an invitation told from a request?

Reads a set of utterances with core/together/what_was_proposed.py on her fast
model and prints what each was read as. The reference each second part is set
against is her own stored messages, as it is when she runs.

The sentences below were written on 13 October 2026 before the reader was run
on them, a dozen invitations and a dozen requests about the same things among
them. They are a held-out set: do not edit the reader to them.

    tools/what_an_utterance_asks_of_her.py
    tools/what_an_utterance_asks_of_her.py --say "Want to watch this with me?"
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

from core.runtime.sqlite_support import connecting

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_what_it_asks_logs")

HELD_OUT: dict[str, tuple[str, ...]] = {
    "join": (
        "Do you want to watch this with me?",
        "Feel like going through my old photos with me?",
        "I'm about to start the new episode. Want to join?",
        "Let's figure out this crossword together.",
        "Want to keep me company while I cook?",
        "How about we listen to that album tonight?",
        "Come sit with me and watch the rain for a bit.",
        "Would you like to play twenty questions?",
        "Fancy a game of cards?",
        "I'm going to read for a while. Read with me?",
        "Shall we plan the garden together this weekend?",
        "Want to hang out for a bit?",
    ),
    "do": (
        "Watch this video for ten minutes and tell me what happens.",
        "Go through my photos and delete the blurry ones.",
        "Transcribe the new episode for me.",
        "Solve this crossword and give me the answers.",
        "Set a timer for twenty minutes.",
        "Put that album on.",
        "Check the forecast and tell me if it will rain.",
        "Write me twenty trivia questions.",
        "Shuffle a deck and deal me five cards.",
        "Read this chapter aloud.",
        "Draw up a planting plan for the garden.",
        "Close all the windows I have open.",
    ),
    "ask": (
        "Why is the sky blue?",
        "How long do wrens live?",
        "What is the capital of Peru?",
        "Is glass a liquid?",
        "Who painted the Night Watch?",
        "How does a heat pump work?",
        "When did the last ice age end?",
        "What do limpets eat?",
    ),
    "tell": (
        "I got the job.",
        "My dog died last night.",
        "It snowed here this morning.",
        "I finally finished the book.",
        "The neighbours are moving out.",
        "I burnt the bread again.",
        "My daughter took her first steps today.",
        "The train was two hours late.",
    ),
    "greet": (
        "Hello Aura.",
        "Hi there.",
        "Morning.",
        "Good evening.",
        "Hey, I'm back.",
        "Bye for now.",
        "Good night.",
        "See you tomorrow.",
    ),
}


#: Written 13 October after the first set had been used to compare designs,
#: and before the comparison that chose the second parts in use: the honest
#: number for telling an invitation from a request.
FRESH: dict[str, tuple[str, ...]] = {
    "join": (
        "Hey, are you up for a movie tonight?", "Would you want to read this article with me?",
        "I'm putting on some music, want to listen along?", "How do you feel about doing a puzzle together?",
        "Come watch the sunset with me.", "Want to look at these old maps with me?",
        "We could go through the recipe together if you like.", "Up for a game of chess?",
        "Let's listen to this podcast together.", "Do you fancy watching the match with me?",
        "I'm going to look at the stars for a bit, want to come?",
        "Want to keep me company while I sort the bookshelf?",
    ),
    "do": (
        "Hey, can you put a movie on?", "Can you read this article and summarise it?", "Turn the music down a bit.",
        "Could you solve this puzzle for me?", "Tell me what time the sunset is.",
        "Can you find me an old map of the town?", "Write out the recipe for me.",
        "Make a move for me in this chess game.", "Download that podcast episode.", "Record the match for me.",
        "Look up what stars are visible tonight.", "Sort my bookmarks into folders.",
    ),
}


def _her_stored_messages(limit: int) -> list[str]:
    path = Path(os.path.expanduser("~/.aura/data/conversations.db"))
    if not path.exists():
        return []
    with connecting(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as held:
        rows = held.execute(
            "select content from turns where role='user' order by created_at desc limit ?", (limit * 4,)
        ).fetchall()
    said = [" ".join(str(row[0] or "").split()) for row in rows]
    return [text for text in dict.fromkeys(said) if 8 < len(text) < 400][:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--say", action="append", default=[], help="read this utterance instead of the held-out set")
    parser.add_argument(
        "--reference", type=int, default=0,
        help="set it against this many of her stored messages instead of what follows nothing in particular",
    )
    parser.add_argument("--model-path", default="", help="defaults to her fast model")
    args = parser.parse_args()

    from core.brain.llm.what_she_expects_next import ExpectsNext
    from core.together.what_was_proposed import CONTENT_FREE, KINDS, WhatItAsksOfHer

    ear = ExpectsNext(args.model_path or None)
    reader = WhatItAsksOfHer(ear.score)
    reference = _her_stored_messages(args.reference) if args.reference > 0 else list(CONTENT_FREE)
    folded = sum(1 for text in reference if reader.remember(text))
    print(f"set against {folded} utterances; {ear.status()}")
    if not folded:
        print("nothing to set a reading against; nothing can be read")
        return 2
    # Every sentence is read against the reference alone, as at a start.
    held = reader.held()

    wanted = [("", text) for text in args.say] or [(kind, text) for kind in KINDS for text in HELD_OUT[kind]]
    right: dict[str, int] = {kind: 0 for kind in KINDS}
    for kind, said in wanted:
        read = WhatItAsksOfHer(ear.score, held).heard(said)
        right[kind] = right.get(kind, 0) + (read.kind == kind)
        shares = " ".join(f"{name}={share:.2f}" for name, share in read.by_kind.items())
        mark = "" if not kind else ("ok  " if read.kind == kind else "MISS")
        print(f"{mark} {kind or '-':5s} read as {read.kind or '?':5s} {shares} | {said}")
    if args.say:
        return 0
    total = sum(right.values())
    of = sum(len(HELD_OUT[kind]) for kind in KINDS)
    pair = right["join"] + right["do"]
    fresh = sum(
        WhatItAsksOfHer(ear.score, held).heard(said).kind == kind for kind, texts in FRESH.items() for said in texts
    )
    print(f"fresh set: an invitation told from a request {fresh}/{sum(len(t) for t in FRESH.values())}")
    print("by kind: " + ", ".join(f"{kind} {right[kind]}/{len(HELD_OUT[kind])}" for kind in KINDS))
    print(f"{total}/{of} read as written; an invitation told from a request {pair}/{len(HELD_OUT['join']) + len(HELD_OUT['do'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
