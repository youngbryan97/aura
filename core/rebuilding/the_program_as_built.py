"""A rebuilt program as it stands: the frame, and the parts written for it so far.

Each part is a piece of script that plugs into the frame through ``app`` and
is kept in its own script element: a part that does not parse, or throws as
it starts, is recorded and leaves the others working. The program is one
file, which opens in any browser with nothing to install.
"""
from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["Part", "ProgramAsBuilt"]

_FRAME = Path(__file__).with_name("an_application_frame.html")


@dataclass
class Part:
    """One piece of the program: what it is for, and its code."""

    name: str
    code: str
    serves: list[str] = field(default_factory=list)

    def element(self) -> str:
        body = self.code.replace("</script", "<\\/script")
        return f"<script>\napp.part({json.dumps(self.name)}, (app) => {{\n{body}\n}});\n</script>"


@dataclass
class ProgramAsBuilt:
    """The frame with its title, look and parts."""

    title: str
    document: str = "Untitled"
    accent: str = "#2b579a"
    style: str = ""
    parts: list[Part] = field(default_factory=list)

    def page(self, *, leaving_out: str = "") -> str:
        frame = _FRAME.read_text("utf-8")
        parts = "\n".join(p.element() for p in self.parts if p.name != leaving_out)
        return (
            frame.replace("__TITLE__", html.escape(self.title))
            .replace("__DOCUMENT__", html.escape(self.document))
            .replace("__ACCENT__", self.accent if _a_colour(self.accent) else "#2b579a")
            .replace("__STYLE__", self.style.replace("</style", ""))
            .replace("__PARTS__", parts)
        )

    def with_part(self, part: Part) -> ProgramAsBuilt:
        """This program with ``part`` added, or put in place of the part of that name."""
        if any(p.name == part.name for p in self.parts):
            parts = [part if p.name == part.name else p for p in self.parts]
        else:
            parts = [*self.parts, part]
        return ProgramAsBuilt(self.title, self.document, self.accent, self.style, parts)

    def keep(self, folder: str | Path) -> Path:
        """Its parts kept beside it, so it can be opened again and changed part by part."""
        out = Path(folder) / "program.json"
        out.write_text(json.dumps(asdict(self), indent=1), "utf-8")
        return out

    @classmethod
    def kept_in(cls, folder: str | Path) -> ProgramAsBuilt:
        data = json.loads((Path(folder) / "program.json").read_text("utf-8"))
        parts = [Part(**p) for p in data.pop("parts", [])]
        return cls(**data, parts=parts)

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.page(), "utf-8")
        return out


def _a_colour(value: str) -> bool:
    return bool(value) and value.startswith("#") and len(value) in (4, 7) and all(c in "0123456789abcdefABCDEF" for c in value[1:])
