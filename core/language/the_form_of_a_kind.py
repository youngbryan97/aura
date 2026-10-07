"""The form of a kind of writing: its parts in order, how each is known, and how each stands; and writing fitted to it.

Whatever is written is of a kind, and the kind has a form a reader expects.
A letter opens with a greeting on a line of its own and ends with a closing
and a name, each on a line of their own; an email the same, more loosely; a
memo heads itself with To, From, Date and Subject, a line each. A person
writing one sets it that way without thinking, and a reader notices at once
when it is not. LIVE 2026-10-06 a letter she wrote in a program she built
began "Dear Future Self, I am writing this on a quiet Tuesday evening" on one
line.

The forms are data, here, beside the code that fits writing to them: a kind is
known by the words a request names it with, and each part of its form by the
shape of its words. Fitting puts each part found where its form says it stands
and changes no words of it; a part the form expects that is not there is
named as missing, never made up. Nothing here knows which program the writing
is done in, or who it is to.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["FORMS", "Fitted", "Form", "FormPart", "fitted", "the_form_for"]


@dataclass(frozen=True)
class FormPart:
    """One part of a form: what it is called, where it comes, the shape of its words, and whether the form expects it.

    ``shape`` matches a paragraph that begins with this part and may run on:
    group 1 is the part, group 2 what ran on after it (or, for a part that
    comes last, group 1 what ran before it and group 2 the part).
    """

    name: str
    where: str  # "first" | "last" | "head" (each of several lines at the top)
    shape: re.Pattern[str]
    expected: bool = True


@dataclass(frozen=True)
class Form:
    kind: str
    called: re.Pattern[str]
    parts: tuple[FormPart, ...]


_GREETING = re.compile(r"^((?:Dear|Dearest|Hi|Hello|Hey|Good (?:morning|afternoon|evening)|To)\s+[^,!?\n]{1,60}[,:])(?:\s+(\S.*))?$", re.S)
_SIGN_OFF = re.compile(r"^(?:(.*?[.!?])\s+)?((?:With [\w ]{1,30}|Warmly|Warm regards|Sincerely|Yours[\w ]{0,20}|All the best|"
                       r"Best(?: wishes| regards)?|Love|Much love|Kind regards|Regards|Cheers|Thank you|Thanks),\s+"
                       r"[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,3})\s*$", re.S)
_HEADING = re.compile(r"^((?:To|From|Date|Subject|Re|Cc):\s*[^\n]*?)(?=\s+(?:To|From|Date|Subject|Re|Cc):|$)\s*(.*)$", re.S)

#: The forms of the kinds of writing she knows. A kind is added by saying its form, not by writing code.
FORMS: tuple[Form, ...] = (
    Form("letter", re.compile(r"\bletters?\b", re.I), (
        FormPart("greeting", "first", _GREETING),
        FormPart("closing and name", "last", _SIGN_OFF),
    )),
    Form("email", re.compile(r"\b(?:e-?mails?|messages? to)\b", re.I), (
        FormPart("greeting", "first", _GREETING, expected=False),
        FormPart("closing and name", "last", _SIGN_OFF, expected=False),
    )),
    Form("memo", re.compile(r"\bmemo(?:randum|s)?\b", re.I), (
        FormPart("heading", "head", _HEADING),
    )),
)


def the_form_for(task: str) -> Form | None:
    """The form of the kind of writing ``task`` asks for, where it names one she knows."""
    return next((form for form in FORMS if form.called.search(task or "")), None)


@dataclass
class Fitted:
    paragraphs: list[str]
    found: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def fitted(paragraphs: Sequence[str], form: Form | None) -> Fitted:
    """``paragraphs`` set to ``form``: each part found on the lines its form gives it, no word changed; expected parts not found named."""
    out = [p.strip() for p in paragraphs if p.strip()]
    if form is None or not out:
        return Fitted(out)
    found: list[str] = []
    for part in form.parts:
        if part.where == "first" and (m := part.shape.match(out[0])):
            out[:1] = [m.group(1).strip(), *([m.group(2).strip()] if m.group(2) else [])]
            found.append(part.name)
        elif part.where == "last" and len(out) > 1 and not out[-2].endswith((".", "!", "?")) and part.shape.match(f"{out[-2]} {out[-1]}"):
            found.append(part.name)  # already set as the form sets it: the closing, and the name under it
        elif part.where == "last" and (m := part.shape.match(out[-1])):
            said = m.group(2).strip()
            closing, _, name = said.partition(",")
            out[-1:] = [*([m.group(1).strip()] if m.group(1) else []), f"{closing},", name.strip()]
            found.append(part.name)
        elif part.where == "head":
            heads, rest = [], out[0]
            while (m := part.shape.match(rest)) and m.group(1).strip():
                heads.append(m.group(1).strip())
                rest = m.group(2).strip()
            if heads:
                out[:1] = [*heads, *([rest] if rest else [])]
                found.append(part.name)
    missing = [part.name for part in form.parts if part.expected and part.name not in found]
    return Fitted(out, found, missing)
