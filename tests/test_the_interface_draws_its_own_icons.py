"""No pictographic emoji in text this interface authors.

The rule is already written in `interface/static/aura.js`: emoji are not this
interface's iconography, the feed draws its channels as monoline sigils, and
`stripNeuralPictographs` removes the ones the backend decorates its log lines
with. Typographic marks that read as text — the arrows, the check, the cross,
the warning triangle in its text presentation — are deliberately kept.

What the rule did not cover was the text the frontend writes itself. Four
onboarding headings carried a brain, a microphone, a bar chart and a gear;
eight messages the person is shown carried a warning triangle in emoji
presentation, a recycling symbol, a package, a phone, a galaxy and a cross
mark. A rule applied to somebody else's strings and not to your own is style
advice.

The strip function's own comment is the boundary this test enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "interface" / "static"

#: What Unicode calls emoji presentation, which is the distinction the rule is
#: actually about.
#:
#: The astral pictograph blocks default to a colour glyph. The BMP symbols —
#: the check, the cross, the warning triangle, the gear — default to TEXT
#: presentation and only become emoji when U+FE0F follows them, which is why
#: `⚙&#xFE0E;` in the tab strip is a monoline icon and `⚙️` in a heading was
#: not. Matching `Extended_Pictographic` would have failed both.
_PICTOGRAPH = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F000-\U0001F0FF\U0001F900-\U0001F9FF]"
    "|[\u2190-\u2BFF]\uFE0F"
)

#: Lines that NAME an emoji in order to say it does not belong. Removing the
#: example would remove the explanation.
_EXPLAINS_THE_RULE = ("monoline sigils", "The feed renders", "deliberately kept")


def _authored_files() -> list[Path]:
    return [
        path
        for path in sorted(STATIC.rglob("*"))
        if path.is_file()
        and path.suffix in {".html", ".js", ".css"}
        and "node_modules" not in path.parts
        and "vendor" not in path.parts
    ]


def test_there_are_files_to_check() -> None:
    """A sweep over nothing reports clean forever."""

    assert len(_authored_files()) >= 8


@pytest.mark.parametrize("path", _authored_files(), ids=lambda p: p.name)
def test_no_pictograph_in_what_the_interface_writes(path: Path) -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()[:90]}"
        for number, line in enumerate(
            path.read_text("utf-8", errors="replace").splitlines(), 1
        )
        if _PICTOGRAPH.search(line)
        and not any(marker in line for marker in _EXPLAINS_THE_RULE)
    ]
    assert not offenders, "\n".join(offenders)


def test_the_backend_strip_is_still_there() -> None:
    """The other half: what the runtime decorates its own log lines with."""

    body = (STATIC / "aura.js").read_text("utf-8")
    assert "function stripNeuralPictographs" in body
    assert "Extended_Pictographic" in body


def test_the_typographic_marks_are_kept() -> None:
    """The rule is about iconography, not about every non-ASCII character."""

    body = (STATIC / "aura.js").read_text("utf-8")
    assert "✓ Connection restored" in body
    assert "⚠ Connection lost" in body
