"""When she has no recollection, she must not supply one.

``_build_live_self_process_reply`` searches recent exchanges for something the
person raised, and speaks it as "I still have this recent concern in view:
...". When the search came back empty it filled in a hardcoded sentence —

    "Bryan has been checking whether the live desktop path is really connected
     to Aura's mind instead of a raw assistant lane"

— and said it with exactly the confidence of a real recollection. It fired
precisely when no real memory was found, which is the moment it is least
likely to be true, and it names a specific person doing a specific thing.

Saying less is not a degradation. Saying something invented is.
"""

from __future__ import annotations

import re
from pathlib import Path

CHAT_ROUTE = Path(__file__).resolve().parents[1] / "interface/routes/chat.py"
SOURCE = CHAT_ROUTE.read_text(encoding="utf-8")


def _runtime_lines() -> list[str]:
    """Source lines that are not comments, so prose about the fix is allowed."""
    return [
        line
        for line in SOURCE.splitlines()
        if not line.lstrip().startswith("#")
    ]


def test_the_invented_concern_is_gone():
    assert "Bryan has been checking whether the live desktop path" not in "\n".join(
        _runtime_lines()
    )


def test_the_recollection_clause_requires_an_actual_recollection():
    """The clause must be guarded by having found something.

    Read out of `interface/routes/chat.py` with a 300-character window above
    the clause. The clause moved to `chat_own_source` and the window was
    landing on an unrelated comment in the file it had stayed behind in —
    a window measures formatting, and this one was measuring another module.

    The guard and the clause are one statement, so the function that holds
    them is the unit and the order is the property.
    """
    from interface.routes import chat_own_source
    from tests.source_contract import function_containing, in_order

    _name, body = function_containing(
        chat_own_source,
        'parts.append(f"I still have this recent concern in view',
    )
    assert re.search(r"if\s+remembered_user\s+and\b", body), (
        "the clause can still speak an empty or defaulted recollection"
    )
    in_order(
        body,
        "if remembered_user and",
        'parts.append(f"I still have this recent concern in view',
    )


def test_no_runtime_string_asserts_what_the_person_has_been_doing():
    """A canned claim about a named person is a fabricated memory.

    Comments describing past incidents are fine; a string literal that would
    be SPOKEN is not.
    """
    offenders = [
        line.strip()
        for line in _runtime_lines()
        if re.search(r'"[^"]*\bBryan (?:has been|was|kept|always)\b[^"]*"', line)
    ]
    assert not offenders, f"canned claims about Bryan can still be spoken: {offenders}"
