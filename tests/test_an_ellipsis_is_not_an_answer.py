"""An ellipsis is not an answer; it is the shape of one.

Four places wrote `or "…"` where a reply might be empty, and each turned "there
is no answer here" into something every downstream `if not reply` guard reads
as an answer — so the turn looks answered to everything that asks.

LIVE 2026-08-17: "what's on my screen right now?" served as a bare ellipsis
over a 172-character reply. LIVE 2026-09-07: "does the file X exist, and what
is in it?" served as a bare ellipsis over 1,155 characters that quality had
already scored confidence=high and off_topic=False. The August fix added
salvage inside the stripper; the substitution at the caller survived it, which
is why this is a rule rather than another repair.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from interface.routes.chat import _never_an_ellipsis

_SOURCE = Path("interface/routes/chat.py").read_text()

#: Sites that RECORD a turn rather than serve one — the recent-response buffer
#: and the logged exchange, with its metadata. A placeholder in a receipt is not
#: put in front of a person. The number is here so a new one has to be justified
#: by moving it, rather than by being added quietly.
_RECORDING_SITES = 6


@pytest.mark.parametrize("value", ["…", "...", "  …  ", "", None, "\n"])
def test_the_shape_of_an_answer_is_not_one(value) -> None:
    assert _never_an_ellipsis(value) == ""


@pytest.mark.parametrize(
    "value",
    ["Blue", "The file does not exist.", "…and then it finished", "5050"],
)
def test_a_real_answer_survives(value: str) -> None:
    assert _never_an_ellipsis(value) == value.strip()


def test_the_served_reply_never_becomes_an_ellipsis() -> None:
    """The one that matters: what goes in the response body."""
    start = _SOURCE.index("_final_reply = (")
    body = _SOURCE[start : start + 500]
    assert "_never_an_ellipsis" in body
    assert 'or "…"' not in body


def test_no_new_site_manufactures_one() -> None:
    """A rule, not a repair. The August fix was a repair and this came back."""
    sites = [
        line
        for line in _SOURCE.splitlines()
        if 'or "…"' in line and not line.strip().startswith(("#", "#:"))
    ]
    assert len(sites) <= _RECORDING_SITES, (
        f"{len(sites)} places still turn an empty reply into an ellipsis; only "
        f"{_RECORDING_SITES} recording sites are allowed to, and a served reply "
        f"never may:\n" + "\n".join(sites)
    )


def test_the_guards_that_check_for_it_still_exist() -> None:
    """Downstream code tests for the ellipsis; it must keep being able to."""
    assert 'reply == "…"' in _SOURCE or 'text == "…"' in _SOURCE
