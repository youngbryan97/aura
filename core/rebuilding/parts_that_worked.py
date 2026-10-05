"""The parts of her earlier builds that worked, found again for a feature like theirs.

Every part kept in a build was kept because a person's checks held when done
with it. That is a library of working code for features she has built before,
and it was being thrown away: the next program with a Bold button or a Save
dialog started from nothing. A part that worked is now shown to her model for
a feature like it in any later build, of any program, to start from where it
fits (Voyager's skill library, for programs). Nothing is copied unread: the new
part is still kept only for what its own checks show.

The library is her builds themselves, each folder's program.json (its parts)
and what_it_does.json (its features); nothing else is stored, and it needs no
network.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("Rebuilding.PartsThatWorked")

__all__ = ["PartThatWorked", "parts_like", "shown", "work_area_like"]

#: How alike two features' words must be for one's part to be shown for the other.
ALIKE = 0.3

#: The most of a part's code shown to her model (all of it is used when it is reused as it is).
SHOWN_CHARS = 3500

_PLAIN = frozenset("the and for with from that this into when then what which their them they your have will are can its".split())


@dataclass(frozen=True)
class PartThatWorked:
    program: str
    feature: str
    said: str
    code: str


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", str(text or "").lower()) if w not in _PLAIN}


def _alike(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def _builds(where: Path, leaving_out: Path | None) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """(title, what it does, program) for each build kept under ``where``, newest first."""
    found = []
    for program_file in sorted(Path(where).glob("*/program.json"), key=lambda p: -p.stat().st_mtime):
        folder = program_file.parent
        if leaving_out is not None and folder.resolve() == Path(leaving_out).resolve():
            continue
        try:
            program = json.loads(program_file.read_text("utf-8"))
            does = json.loads((folder / "what_it_does.json").read_text("utf-8"))
        except (OSError, ValueError):
            continue
        found.append((str(program.get("title") or folder.name), does, program))
    return found


def parts_like(feature: Any, where: Path | None, *, leaving_out: Path | None = None, at_most: int = 1) -> list[PartThatWorked]:
    """The parts that worked in earlier builds for the features most like ``feature``, the most alike first."""
    if where is None or not Path(where).is_dir():
        return []
    wanted = _words(f"{feature.name} {feature.name} {feature.how} {feature.shows}")
    scored: list[tuple[float, PartThatWorked]] = []
    for title, does, program in _builds(Path(where), leaving_out):
        features = {str(f.get("name") or "").lower(): f for f in does.get("features") or [] if isinstance(f, dict)}
        for part in program.get("parts") or []:
            if part.get("name") == "work area" or not part.get("code"):
                continue
            for served in part.get("serves") or []:
                described = features.get(str(served).lower()) or {"name": served}
                said = f"{described.get('name', '')}: {described.get('how', '')} -> {described.get('shows', '')}"
                share = _alike(wanted, _words(f"{described.get('name', '')} {described.get('name', '')} {described.get('how', '')} {described.get('shows', '')}"))
                if share >= ALIKE:
                    scored.append((share, PartThatWorked(title, str(served), said, str(part["code"]))))
    scored.sort(key=lambda pair: -pair[0])
    seen: set[str] = set()
    best = []
    for _share, found in scored:
        if found.code not in seen:
            seen.add(found.code)
            best.append(found)
    return best[:at_most]


def work_area_like(genome: Any, where: Path | None, *, leaving_out: Path | None = None) -> PartThatWorked | None:
    """The work area of the earlier build most like this program, by what each is."""
    if where is None or not Path(where).is_dir():
        return None
    wanted = _words(f"{genome.what_it_is} {genome.work}")
    best: tuple[float, PartThatWorked | None] = (ALIKE, None)
    for title, does, program in _builds(Path(where), leaving_out):
        share = _alike(wanted, _words(f"{does.get('what_it_is', '')} {does.get('work', '')}"))
        area = next((p for p in program.get("parts") or [] if p.get("name") == "work area"), None)
        if area and share >= best[0]:
            best = (share, PartThatWorked(title, "work area", str(does.get("work") or ""), str(area.get("code") or "")))
    return best[1]


def shown(found: list[PartThatWorked]) -> str:
    """The parts found, as they are shown to her model."""
    return "".join(
        f"\nA part that worked in an earlier program of hers, {f.program}, for \"{f.said}\" (its checks held). "
        f"Start from it where it fits this program; change what does not:\n```js\n{f.code[:SHOWN_CHARS]}\n```\n"
        for f in found
    )
