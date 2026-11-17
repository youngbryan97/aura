"""Text after something was taken out of it: the gaps closed, its lines and its code as they were.

A check that removes a sentence, a citation with no source, or a maths delimiter leaves two spaces where one was,
a space before a full stop, or a blank line too many. Each closed those gaps over the whole reply, and each did it
by rewriting every line: `" ".join(line.split())`, `[ \\t]{2,}` to one space, `(?m)^\\s+` to nothing. A line's
leading whitespace is its indentation, and in code it is the program. LIVE 12 and 17 November 2026, two code
answers she wrote were stored with every line flush left, a Python function among them, after a sentence elsewhere
in the reply was taken out.

Here a gap is closed only inside a line, after its first visible character; a line's indentation and every line of
a fenced code block are left as they were written.
"""

from __future__ import annotations

import re

__all__ = ["closed_up"]

_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_INNER_RUN = re.compile(r"(?<=\S)[ \t]{2,}")
_SPACE_BEFORE_MARK = re.compile(r"(?<=\S)[ \t]+(?=[.,;:!?](?:\s|$))")


def closed_up(text: str) -> str:
    """``text`` with the gaps a removal left closed: runs of spaces inside a line made one, no space before a
    sentence mark, no trailing whitespace, no more than one blank line in a row. Indentation and fenced code are kept.
    """
    out: list[str] = []
    in_code = False
    blank = 0
    for line in str(text or "").splitlines():
        if _FENCE.match(line):
            in_code = not in_code
            out.append(line.rstrip())
            blank = 0
            continue
        if in_code:
            out.append(line)
            continue
        line = _SPACE_BEFORE_MARK.sub("", _INNER_RUN.sub(" ", line)).rstrip()
        if not line.strip():
            blank += 1
            if blank > 1:
                continue
            line = ""
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip("\n")
