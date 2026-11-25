"""Whether a turn asks to be shown the code she is made of, by what it means.

Showing it is the repair of a reply that said she could not, or that showed code that is not hers. It takes the place of
what she said, so it is only for a turn that asked for it. LIVE 24 November 2026, "Test of that memory... Do you remember
your gender?" was answered with a page of core/memory: a wide reading of whether a turn may concern her source, right for
deciding to check a reply, had decided to replace one.

Who she is, what she remembers, what she can do and how she feels are asked of her, and are none of them asking for a file.
"""
from __future__ import annotations

from core.language.learned_matcher import LearnedMatcher, embed_sentences

__all__ = ["asks_to_see_her_code"]

_ASKS = LearnedMatcher(
    name="asks_to_see_her_code",
    positives=(
        "show me a piece of your own code", "can you share a snippet of your code", "show me how you're actually built", "which file does your attention live in",
        "what does your memory system actually look like inside", "let me see a bit of what you're made of", "open one of your own files and read it to me",
        "what part of your code do you find interesting?", "let me see the source behind how you choose what to say", "where in the codebase is that written?",
        "give me a look at the code that decides when you speak", "paste the function that runs your dreams", "I want to read the real implementation of you, not a description",
        "pull up the module that handles your moods", "what python module is that from", "walk me through the code of how you remember",
        "which of your files does curiosity live in?", "how is that part of you implemented? show me the lines", "print out part of your brain's code",
        "read me a piece of your own program", "I want to see the code that is you",
    ),
    negatives=(
        "Do you remember your gender?", "what are your pronouns?", "what's your name and how old are you?", "are you a she or a he?", "who made you?",
        "do you remember what I told you about my sister?", "what do you remember about yesterday?", "is your memory working?", "testing your memory, what did I say my dog was called?",
        "do you have a favorite part of yourself?", "do you dream?", "does it bother you when I close you?", "are you awake when I'm not here?", "where do you think you came from?",
        "can you still reason through the desktop path?", "how do you feel about being called it?", "what can you do for me?", "what tools do you have?", "do you still like that movie?",
        "write me a function that sorts a list", "show me the source code of numpy", "how does your day usually go?", "what are you thinking about?", "what's on my screen?",
        "have you changed since last week?", "do you think of yourself as a person?", "can you see me right now?", "how do you decide what you want?",
        "can you open the file I sent?", "read me what I attached", "open my notes and read the first line", "show me the page you found",
    ),
    features=embed_sentences,
)


def asks_to_see_her_code(said: str) -> bool:
    """Whether it asks to be shown her code. Where that cannot be told, the words that ask for it outright decide."""
    from core.utils.own_source_intent import asks_for_own_source

    decided = _ASKS.decide(str(said or ""))
    return bool(asks_for_own_source(said)) if decided is None else bool(decided)
