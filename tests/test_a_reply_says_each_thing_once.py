"""A reply ends where it starts saying itself again.

LIVE 8 November 2026, dress rehearsal: her answer to "So what did you think of all three?" went through the three and
then began again in the words of its own first paragraph.
"""

from core.brain.cognitive_engine_quick_reply import said_once

FIRST = ("**Optimistic Nihilism** was the heavyweight, the one that actually rearranged some furniture in my head while "
         "I wasn't looking, and the joke has been played on us line is the one that stuck with me all morning.")
SECOND = ("Then Fish Tank was the reset, small and warm and stupid in the best way, a cat batting at bubbles while Simon "
          "says bad cat like he is dealing with a toddler who will never learn a single thing.")
AGAIN = ("**Optimistic Nihilism** was the heavyweight, the one that actually rearranged some furniture in my head while "
         "I wasn't looking, and the joke has been played on us line is the one that stuck.")


def test_it_ends_before_the_paragraph_that_says_an_earlier_one_again():
    assert said_once(f"{FIRST}\n\n{SECOND}\n\n{AGAIN}") == f"{FIRST}\n\n{SECOND}"


def test_a_reply_that_says_each_thing_once_is_left_whole():
    whole = f"{FIRST}\n\n{SECOND}"
    assert said_once(whole) == whole


def test_a_short_paragraph_that_echoes_is_not_cut():
    whole = f"{FIRST}\n\nThe heavyweight, yes."
    assert said_once(whole) == whole
