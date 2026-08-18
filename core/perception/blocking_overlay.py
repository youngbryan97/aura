"""Notice a blocking overlay, and name the safe way out of it.

The first thing a real page does is get in the way. Cookie banners, welcome
modals, newsletter popups, "choose your region", tutorial prompts, app-install
interstitials. A task that cannot get past one stalls on step one, and the
stall is invisible from inside the task: perception reports text, the action
reports success, and nothing happens because a dialog owns the screen.

Measured live 2026-08-18, opening play2048.co through her own browser
controller and reading it through her own screen perception: the board was
there at y=0.5, and so was "WELCOME TO 2048!" with "Play Tutorial" and "New
Game" at y=0.14. Keys sent to the page went nowhere useful because the modal
had focus. Nothing in the runtime could see that as an obstacle rather than
as text.

This module is deliberately about DISMISSAL and nothing else.

Closing a dialog is a reversible act on someone's own screen. Accepting terms,
granting consent, agreeing to a privacy policy or opting into tracking are
not: they create obligations and permissions in the person's name. So the
dismissive affordances are an allowlist, the accepting ones are a denylist,
and a banner offering only acceptance is reported as needing the PERSON —
never clicked. On a consent choice the least-permission option is the one
named, which is refusal, never "accept all".

Nothing here knows about any particular site. It reads positioned text and
returns a point, so it works the same on a cookie wall, an installer and a
game.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Labels that CLOSE something without agreeing to anything. Ordered by how
#: unambiguous they are: an explicit refusal beats a bare "close", which beats
#: a glyph that might be decoration.
DISMISSIVE_LABELS: tuple[tuple[str, int], ...] = (
    (r"^\s*no\s*thanks\s*$", 100),
    (r"^\s*not\s*now\s*$", 100),
    (r"^\s*maybe\s*later\s*$", 100),
    (r"^\s*decline\s*$", 95),
    (r"^\s*reject\s*all\s*$", 95),
    (r"^\s*reject\s*$", 90),
    (r"^\s*deny\s*$", 90),
    (r"^\s*only\s*(?:use\s*)?(?:strictly\s*)?necessary\s*$", 90),
    (r"^\s*essential\s*(?:cookies\s*)?only\s*$", 90),
    (r"^\s*manage\s*(?:cookie\s*)?preferences\s*$", 55),
    (r"^\s*skip(?:\s*for\s*now|\s*tutorial|\s*intro)?\s*$", 85),
    (r"^\s*dismiss\s*$", 85),
    (r"^\s*close\s*$", 80),
    (r"^\s*cancel\s*$", 70),
    (r"^\s*(?:got\s*it|ok(?:ay)?)\s*$", 60),
    (r"^\s*continue\s*without\s*$", 60),
    (r"^\s*[x✕✖×⨯]\s*$", 50),
)

#: Labels that AGREE to something. Never clicked, at any confidence.
#:
#: A dialog whose only way out is one of these is a decision for the person:
#: it binds them to terms, grants a permission or opts them into collection,
#: and none of that is reversible by closing a window afterwards.
ACCEPTING_LABELS: tuple[str, ...] = (
    r"\baccept\b",
    r"\bagree\b",
    r"\ballow\b",
    r"\bconsent\b",
    r"\bi\s*understand\b",
    r"\bsign\s*(?:in|up)\b",
    r"\bsubscribe\b",
    r"\bcontinue\s*with\s*(?:google|facebook|apple|email)\b",
    r"\benable\b",
    r"\bgrant\b",
    r"\bopt\s*in\b",
)

#: Words that suggest the thing on screen is an overlay at all, rather than
#: page content that happens to contain a button.
OVERLAY_HINTS: tuple[str, ...] = (
    r"\bcookies?\b",
    r"\bconsent\b",
    r"\bprivacy\b",
    r"\bwelcome\b",
    r"\bnewsletter\b",
    r"\bsubscribe\b",
    r"\bnotifications?\b",
    r"\btutorial\b",
    r"\bsign\s*(?:in|up)\b",
    r"\bterms\b",
    r"\bwould\s*you\s*like\b",
    r"\bget\s*started\b",
    r"\binstall\b",
)


@dataclass(frozen=True)
class OverlayVerdict:
    """What is in the way, and what may be done about it."""

    #: True when something looks like it is covering the content.
    present: bool = False
    #: A dismissal target, normalized 0..1 with a top-left origin.
    click_x: float | None = None
    click_y: float | None = None
    #: The label that will be clicked, for the receipt.
    label: str = ""
    #: How sure the label dismisses rather than agrees.
    confidence: int = 0
    #: A keystroke that dismisses this overlay without agreeing to anything.
    #:
    #: Escape is the platform-standard way out of a modal on every desktop OS,
    #: it commits to nothing, and it is reversible. It is offered when an
    #: overlay is present and no safely-labelled control was found — which is
    #: the common real case, not the exception: measured live, play2048.co's
    #: welcome modal offers only "Play Tutorial" and "New Game", neither
    #: dismissive nor accepting, so label matching alone had no answer while
    #: the universal one was available the whole time.
    suggested_key: str = ""
    #: Set when the only way forward requires the person.
    needs_person: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "click": None if self.click_x is None else [self.click_x, self.click_y],
            "label": self.label,
            "confidence": self.confidence,
            "suggested_key": self.suggested_key,
            "needs_person": self.needs_person,
            "reasons": list(self.reasons),
        }


def _is_accepting(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return any(re.search(pattern, lowered) for pattern in ACCEPTING_LABELS)


def _dismissal_score(text: str) -> int:
    lowered = str(text or "").strip().lower()
    if not lowered or _is_accepting(lowered):
        return 0
    for pattern, score in DISMISSIVE_LABELS:
        if re.search(pattern, lowered):
            return score
    return 0


def assess_overlay(observation: dict[str, Any]) -> OverlayVerdict:
    """Read one screen observation and decide whether something is in the way.

    Takes the same shape every reading already produces — `text` plus a
    `layout` of positioned runs — so any caller that can see can use this
    without new plumbing.
    """
    text = str(observation.get("text") or "")
    layout = list(observation.get("layout") or [])
    if not text and not layout:
        return OverlayVerdict()

    lowered = text.lower()
    hints = tuple(
        pattern for pattern in OVERLAY_HINTS if re.search(pattern, lowered)
    )

    best_score = 0
    best: dict[str, Any] | None = None
    accepting_seen: list[str] = []
    for region in layout:
        label = str(region.get("text") or "").strip()
        if not label or len(label) > 40:
            continue
        if _is_accepting(label):
            accepting_seen.append(label)
            continue
        score = _dismissal_score(label)
        if score > best_score:
            best_score, best = score, region

    if best is not None:
        try:
            x = float(best.get("center_x", best.get("x")))
            y = float(best.get("center_y", best.get("y")))
        except (TypeError, ValueError):
            x = y = None  # type: ignore[assignment]
        if x is not None and y is not None:
            return OverlayVerdict(
                present=True,
                click_x=round(x, 5),
                click_y=round(y, 5),
                label=str(best.get("text") or "").strip(),
                confidence=best_score,
                reasons=hints or ("dismissive_control_present",),
            )

    if best is None and hints and not accepting_seen:
        # Something is in the way and nothing on it is safe to click by name.
        # Escape is the answer that needs no label: it closes without agreeing.
        return OverlayVerdict(
            present=True,
            suggested_key="escape",
            confidence=45,
            reasons=hints + ("no_labelled_dismissal_offered",),
        )

    if accepting_seen:
        # Something is asking for agreement and offering no way to decline.
        # That is the person's call, and closing it for them would be making a
        # commitment in their name.
        return OverlayVerdict(
            present=True,
            needs_person=(
                "this dialog only offers acceptance ("
                + ", ".join(sorted(set(accepting_seen))[:3])
                + "), which is a decision for you rather than for me"
            ),
            reasons=hints or ("accepting_controls_only",),
        )

    return OverlayVerdict(present=bool(hints), reasons=hints)


__all__ = [
    "ACCEPTING_LABELS",
    "DISMISSIVE_LABELS",
    "OVERLAY_HINTS",
    "OverlayVerdict",
    "assess_overlay",
]
