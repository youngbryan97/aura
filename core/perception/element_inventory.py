"""A screen as a list of things you can act on, each with a name you can cite.

Clean-room adoption of the structured-screen-parsing idea (the OmniParser line
of work): stop handing a model a picture or a wall of OCR text and asking it to
aim, and instead give it an INVENTORY — every interactable element, each with a
stable id, a box, where the knowledge came from, and what the thing does. The
model then says "press e7", not "click at (812, 344)", and the id either exists
in the current parse or the action does not happen.

None of their code was read; this is built on Aura's own accessibility read
(``screen_blueprint.read_window_elements``) in her own idiom.

WHY THIS PARTICULAR SHAPE MATTERS HERE. Measured live on 2026-08-04: asked to
quote the text visible in two windows, she produced two plausible strings while
the display was showing nothing at all — an independent capture came back
all-black. The turn before, asked what was BEHIND her window, she was exactly
right, because that answer came from a structured window read instead of from
generation.

The difference was never the difficulty of the question. It was whether the
answer had a structure behind it or was composed. So this module's contract is
not "describe the screen" but:

  * every element carries the SOURCE that produced it;
  * an inventory knows when it was taken and goes stale;
  * an action must cite an id present in a fresh inventory, and
    ``resolve_action_target`` refuses rather than guessing.

That last rule is the same rule as "a quotation needs a capture", applied to
clicking instead of speaking.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ElementInventory")

#: How long a parse describes the screen. Screens change; an inventory older
#: than this is a memory of a screen, not a reading of one.
INVENTORY_FRESHNESS_S = 8.0

#: Two boxes overlapping by more than this are the same thing seen twice — the
#: accessibility tree and an OCR pass both find a button and report it once
#: each. Intersection over the smaller box, so a label inside a button counts
#: as contained rather than merely adjacent.
OVERLAP_MERGE_THRESHOLD = 0.65

#: Roles that can be acted on. A static label is worth knowing about and is not
#: a click target, and conflating the two is how an agent "clicks" a caption.
INTERACTABLE_ROLES = frozenset(
    {
        "button",
        "checkbox",
        "combo box",
        "link",
        "menu button",
        "menu item",
        "pop up button",
        "radio button",
        "slider",
        "tab",
        "text area",
        "text field",
    }
)

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ScreenElement:
    """One thing on screen, and where the knowledge of it came from."""

    element_id: str
    role: str
    name: str
    x: float
    y: float
    width: float
    height: float
    #: "accessibility" | "ocr" | "window" — never blank. An element with no
    #: provenance is a guess wearing a box.
    source: str
    app: str = ""
    window: str = ""

    @property
    def interactable(self) -> bool:
        return self.role.strip().lower() in INTERACTABLE_ROLES

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def function(self) -> str:
        """What acting on this would DO, in a sentence a person would use.

        Not a caption of how it looks. "press Send", "type into Search" —
        the useful description of a control is its effect, and a model choosing
        between controls is choosing between effects.
        """
        role = self.role.strip().lower()
        label = self.name.strip() or "an unlabelled control"
        if role in {"text field", "text area", "combo box"}:
            return f"type into {label}"
        if role in {"button", "menu button", "pop up button"}:
            return f"press {label}"
        if role in {"checkbox", "radio button"}:
            return f"toggle {label}"
        if role in {"link", "menu item", "tab"}:
            return f"open {label}"
        if role == "slider":
            return f"adjust {label}"
        return f"read {label}"

    def as_line(self) -> str:
        """One line for a model to read and refer back to."""
        return (
            f"[{self.element_id}] {self.function()} "
            f"({self.role} at {int(self.x)},{int(self.y)} "
            f"{int(self.width)}x{int(self.height)}, via {self.source})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "role": self.role,
            "name": self.name,
            "box": [self.x, self.y, self.width, self.height],
            "source": self.source,
            "app": self.app,
            "window": self.window,
            "interactable": self.interactable,
            "function": self.function(),
        }


@dataclass(frozen=True)
class ElementInventory:
    """Everything actionable on screen at one moment, with an expiry."""

    elements: tuple[ScreenElement, ...] = ()
    captured_at: float = field(default_factory=time.time)
    app: str = ""
    window: str = ""
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return not self.unavailable_reason

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.captured_at)

    def is_fresh(self, *, max_age_s: float = INVENTORY_FRESHNESS_S) -> bool:
        return self.available and self.age_s <= max_age_s

    @property
    def interactable(self) -> tuple[ScreenElement, ...]:
        return tuple(e for e in self.elements if e.interactable)

    def by_id(self, element_id: str) -> ScreenElement | None:
        wanted = str(element_id or "").strip().lower()
        for element in self.elements:
            if element.element_id.lower() == wanted:
                return element
        return None

    def render(self, *, limit: int = 40) -> str:
        """The inventory as the model should see it: ids it can cite."""
        if not self.available:
            return f"I cannot read the screen's controls right now ({self.unavailable_reason})."
        live = self.interactable[:limit]
        if not live:
            return "I can read the screen, and there is nothing interactable on it."
        header = f"{len(self.interactable)} interactable element(s)"
        if self.app:
            header += f" in {self.app}"
        return header + ":\n" + "\n".join(f"· {e.as_line()}" for e in live)

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "age_s": round(self.age_s, 3),
            "app": self.app,
            "window": self.window,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "element_count": len(self.elements),
            "interactable_count": len(self.interactable),
            "elements": [e.to_dict() for e in self.elements],
        }


def _element_id(role: str, name: str, x: float, y: float) -> str:
    """Stable within a screen state, and derived from what the element IS.

    A positional counter would renumber every control whenever one appeared
    above it, so "press e7" would mean a different button between the parse and
    the click.
    """
    seed = f"{role.strip().lower()}|{name.strip().lower()}|{int(x)}|{int(y)}"
    return "e" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:6]


def _overlap_fraction(a: ScreenElement, b: ScreenElement) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    smaller = min(a.area, b.area)
    return intersection / smaller if smaller > 0 else 0.0


def merge_overlapping(elements: Sequence[ScreenElement]) -> list[ScreenElement]:
    """Collapse the same control seen by two sources into one entry.

    Accessibility wins over OCR when they collide: one reports what a control
    IS, the other reports what it looks like, and a model choosing an action
    needs the first.
    """
    ranked = sorted(
        elements,
        key=lambda e: (0 if e.source == "accessibility" else 1, -e.area),
    )
    kept: list[ScreenElement] = []
    for candidate in ranked:
        if any(
            _overlap_fraction(candidate, existing) >= OVERLAP_MERGE_THRESHOLD
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def elements_from_accessibility(payload: dict[str, Any]) -> list[ScreenElement]:
    """Turn one ``read_window_elements`` result into typed elements."""
    if not isinstance(payload, dict) or not payload.get("ok"):
        return []
    app = str(payload.get("app") or "")
    window = str(payload.get("window") or "")
    out: list[ScreenElement] = []
    for raw in payload.get("elements") or []:
        if not isinstance(raw, dict):
            continue
        try:
            x = float(raw.get("x", raw.get("left", 0)) or 0)
            y = float(raw.get("y", raw.get("top", 0)) or 0)
            width = float(raw.get("w", raw.get("width", 0)) or 0)
            height = float(raw.get("h", raw.get("height", 0)) or 0)
        except (TypeError, ValueError):
            continue
        role = str(raw.get("role") or raw.get("kind") or "").strip()
        name = str(raw.get("name") or raw.get("title") or raw.get("value") or "").strip()
        if not role:
            continue
        out.append(
            ScreenElement(
                element_id=_element_id(role, name, x, y),
                role=role,
                name=name,
                x=x,
                y=y,
                width=width,
                height=height,
                source="accessibility",
                app=app,
                window=window,
            )
        )
    return out


def build_inventory(app: str, *, reader: Any = None) -> ElementInventory:
    """Read the frontmost window's controls into a citable inventory."""
    if not str(app or "").strip():
        return ElementInventory(unavailable_reason="no app named")
    try:
        if reader is None:
            from core.perception.screen_blueprint import read_window_elements as reader
        payload = reader(app)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("element_inventory", exc, severity="warning")
        return ElementInventory(unavailable_reason=f"accessibility read failed ({exc})")

    if not isinstance(payload, dict) or not payload.get("ok"):
        reason = str((payload or {}).get("error") or "no accessibility data")
        return ElementInventory(unavailable_reason=reason)

    elements = merge_overlapping(elements_from_accessibility(payload))
    return ElementInventory(
        elements=tuple(elements),
        app=str(payload.get("app") or app),
        window=str(payload.get("window") or ""),
    )


@dataclass(frozen=True)
class TargetResolution:
    """Whether an action may proceed, and against which element."""

    resolved: bool
    element: ScreenElement | None = None
    reason: str = ""

    def as_metrics(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "element_id": self.element.element_id if self.element else None,
            "reason": self.reason,
        }


def resolve_action_target(
    inventory: ElementInventory,
    reference: str,
    *,
    max_age_s: float = INVENTORY_FRESHNESS_S,
) -> TargetResolution:
    """Map a reference to an element, or refuse.

    Refusing is the point. An agent that always produces a target will click
    something when it recognised nothing, and a wrong click is not a smaller
    version of the right one — it is a different action taken on the person's
    machine. Three ways to succeed: cite an id, name a control exactly, or name
    one whose words uniquely identify it. Anything ambiguous is refused WITH the
    candidates, so the next turn can disambiguate instead of guessing.
    """
    if not inventory.available:
        return TargetResolution(False, reason=f"no inventory ({inventory.unavailable_reason})")
    if not inventory.is_fresh(max_age_s=max_age_s):
        return TargetResolution(
            False,
            reason=f"inventory is {inventory.age_s:.1f}s old; re-read the screen before acting",
        )

    wanted = str(reference or "").strip()
    if not wanted:
        return TargetResolution(False, reason="no target named")

    direct = inventory.by_id(wanted)
    if direct is not None:
        return TargetResolution(True, direct, "cited by id")

    candidates = inventory.interactable
    lowered = wanted.lower()
    exact = [e for e in candidates if e.name.strip().lower() == lowered]
    if len(exact) == 1:
        return TargetResolution(True, exact[0], "exact name")
    if len(exact) > 1:
        return TargetResolution(
            False,
            reason=f"{len(exact)} controls are named {wanted!r}; cite an id: "
            + ", ".join(e.element_id for e in exact[:5]),
        )

    words = set(_WORD.findall(lowered))
    if not words:
        return TargetResolution(False, reason=f"nothing on screen matches {wanted!r}")
    scored = [
        (len(words & set(_WORD.findall(e.name.lower()))), e)
        for e in candidates
        if words & set(_WORD.findall(e.name.lower()))
    ]
    if not scored:
        return TargetResolution(False, reason=f"nothing on screen matches {wanted!r}")
    best = max(score for score, _ in scored)
    winners = [e for score, e in scored if score == best]
    if len(winners) == 1:
        return TargetResolution(True, winners[0], "unique word match")
    return TargetResolution(
        False,
        reason=f"{len(winners)} controls match {wanted!r}; cite an id: "
        + ", ".join(e.element_id for e in winners[:5]),
    )


def inventory_from_elements(
    elements: Iterable[ScreenElement], *, app: str = "", window: str = ""
) -> ElementInventory:
    """Build an inventory directly — for tests and for non-AX sources."""
    return ElementInventory(
        elements=tuple(merge_overlapping(list(elements))), app=app, window=window
    )


__all__ = [
    "INVENTORY_FRESHNESS_S",
    "INTERACTABLE_ROLES",
    "OVERLAP_MERGE_THRESHOLD",
    "ElementInventory",
    "ScreenElement",
    "TargetResolution",
    "build_inventory",
    "elements_from_accessibility",
    "inventory_from_elements",
    "merge_overlapping",
    "resolve_action_target",
]
