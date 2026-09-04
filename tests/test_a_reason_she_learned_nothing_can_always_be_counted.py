"""Every reason a pair is thrown away has somewhere to be counted.

The tally was a plain dict holding the two reasons that had ever fired. Two
later reasons — the thing had stopped answering, and several acts with one
reading — were written straight into it by name, against keys that were not
there. Both raise KeyError out of the middle of the deciding step, so a branch
written to protect the learner would take the whole run down the first time it
was reached.

LIVE 2026-09-04: it had never been reached, because the count it was gated on
was never set. The first run that set it crashed on the next move.
"""

from __future__ import annotations

import inspect
import re

from core.skills import screen_pursuit


def test_every_reason_written_into_the_tally_can_be_counted():
    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    named = set(re.findall(r'dropped\[([\'"])(.+?)\1\]', source))
    assert named, "the tally is written somewhere"
    tally: object = None
    for line in source.splitlines():
        if line.strip().startswith("dropped"):
            tally = line
            break
    assert tally is not None
    for _quote, why in named:
        # A Counter answers for a key it has never seen; a dict raises.
        from collections import Counter

        counting: Counter[str] = Counter()
        counting[why] += 1
        assert counting[why] == 1


def test_the_tally_is_declared_as_one_that_cannot_miss_a_key():
    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert "dropped: Counter[str] = Counter()" in source


def test_what_she_could_not_learn_from_reads_a_tally():
    from collections import Counter

    from core.skills.screen_pursuit import _what_she_could_not_learn_from

    counting: Counter[str] = Counter()
    assert _what_she_could_not_learn_from(counting) == ""
    counting["more than one act, one reading"] += 3
    assert "3 to more than one act, one reading" in _what_she_could_not_learn_from(counting)
