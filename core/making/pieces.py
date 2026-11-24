"""The pieces a program is made from: engineered, each with checks of its own, taken in only where it is used.

A piece is a block in one of the files beside this one. Its first line names it and says what it is for, the lines after
say how it is called, ``needs`` names the pieces it stands on, and its ``//?`` lines are expressions that are true of it
(tests/test_making_pieces.py runs every one). What a model is asked to write for a program is then the program, and not
the hundredth collision, calendar, parser or chart, each a little wrong in its own way.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["Piece", "bundle", "calls", "catalogue", "near", "pieces", "used_in"]

_HERE = Path(__file__).resolve().parent / "pieces"


@dataclass(frozen=True)
class Piece:
    name: str
    for_what: str
    calls: str
    needs: tuple[str, ...]
    code: str
    checks: tuple[str, ...]
    language: str = "javascript"


def _read_python(text: str) -> list[Piece]:
    """The pieces of the file of them that tools in Python are made from, in the same form under Python's own marks."""
    found = []
    for block in re.split(r"^# ## ", text, flags=re.M)[1:]:
        lines = block.splitlines()
        name, _colon, for_what = lines[0].partition(":")
        found.append(Piece(
            name=name.strip(), for_what=for_what.strip(), calls="\n".join(line[3:].strip() for line in lines if line.startswith("#  ")),
            needs=(), code="\n".join(line for line in lines[1:] if not line.startswith(("#  ", "# ??"))).strip(),
            checks=tuple(line[5:].strip() for line in lines if line.startswith("# ?? ")), language="python"))
    return found


def _read(text: str) -> list[Piece]:
    found = []
    for block in re.split(r"^//# ", text, flags=re.M)[1:]:
        lines = block.splitlines()
        name, _colon, for_what = lines[0].partition(":")
        told = [line[4:].strip() for line in lines if line.startswith("//  ")]
        needs = next((line[len("needs:"):].strip() for line in told if line.startswith("needs:")), "")
        found.append(Piece(
            name=name.strip(), for_what=for_what.strip(),
            calls="\n".join(line for line in told if not line.startswith("needs:")),
            needs=tuple(n.strip() for n in needs.split(",") if n.strip()),
            code="\n".join(line for line in lines[1:] if not line.startswith("//")).strip(),
            checks=tuple(line[4:].strip() for line in lines if line.startswith("//? "))))
    return found


@lru_cache(maxsize=2)
def pieces(language: str = "javascript") -> dict[str, Piece]:
    """Every piece of a language, by its name."""
    held: dict[str, Piece] = {}
    read = _read if language == "javascript" else _read_python
    for path in sorted(_HERE.glob("*.js" if language == "javascript" else "tool_pieces.py.txt")):
        for piece in read(path.read_text(encoding="utf-8")):
            if piece.name in held:
                raise ValueError(f"two pieces are called {piece.name}")
            held[piece.name] = piece
    return held


def catalogue(language: str = "javascript") -> str:
    """Each piece and what it is for, a line each."""
    return "\n".join(f"{piece.name}: {piece.for_what}" for piece in pieces(language).values())


def with_needs(names: Sequence[str], language: str = "javascript") -> list[str]:
    """The pieces named, with everything they stand on, in an order that holds."""
    held, out = pieces(language), []

    def take(name: str) -> None:
        if name in out or name not in held:
            return
        for need in held[name].needs:
            take(need)
        out.append(name)

    for name in names:
        take(name)
    return out


def bundle(names: Sequence[str], language: str = "javascript") -> str:
    """The code of the pieces named and what they stand on."""
    apart = "\n" if language == "javascript" else "\n\n\n"
    return apart.join(pieces(language)[name].code for name in with_needs(names, language))


def calls(names: Sequence[str], language: str = "javascript") -> str:
    """How the pieces named are called."""
    held = pieces(language)
    return "\n".join(f"{name} ({held[name].for_what}):\n  {held[name].calls}" for name in names if name in held)


def used_in(code: str, language: str = "javascript") -> list[str]:
    """The pieces some code calls on."""
    called = r"\." if language == "javascript" else r"\s*[.(]"
    own = set() if language == "javascript" else set(re.findall(r"^(?:def|class)\s+(\w+)", code, flags=re.M))
    return [name for name in pieces(language) if name not in own and re.search(rf"(?<![\w.$]){re.escape(name)}{called}", code)]


def near(words: str, *, at_most: int = 8) -> list[str]:
    """The pieces nearest some words, by her encoder where it is ready; none where it is not."""
    from core.cognition.evidence_relevance import vectors_if_ready

    held = [*pieces().values(), *pieces("python").values()]
    vectors = vectors_if_ready([words, *(f"{piece.name}: {piece.for_what}" for piece in held)])
    if not vectors or len(vectors) != len(held) + 1:
        return []
    import numpy as np

    v = np.asarray(vectors, dtype=np.float32)
    score = v[1:] @ v[0] / (np.linalg.norm(v[1:], axis=1) * np.linalg.norm(v[0]) + 1e-9)
    return [held[int(i)].name for i in np.argsort(-score)[:at_most]]
