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
from collections.abc import Sequence
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
    # A bare "X" is deliberately NOT here.
    #
    # It was, at low confidence, and it did real damage: driving a page, the
    # detector found an "X" and clicked it, and the browser ended up on
    # x.com — the glyph it matched was a tab label, not a close button. One
    # character carries no evidence of what it closes. It is a close control,
    # a tab-close, a delete control, a clear-field control, and a company
    # logo, and the wrong guess navigates away from the task or destroys a row
    # of someone's data.
    #
    # Nothing is lost by refusing it: a dialog whose only exit is a glyph is
    # exactly the case Escape exists for, and Escape cannot close a tab.
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

#: Warnings that this dialog destroys something.
#:
#: A confirmation that says work will be lost is not a blocker to get past. It
#: is a question whose answer belongs to the person, exactly like an agreement:
#: closing it is free, confirming it is not undone by anything afterwards.
#:
#: LIVE, 2026-08-18. Clearing the way on a page produced "Are you sure you want
#: to start a new game? All progress will be lost." with a "Start New Game"
#: button. Nothing here could tell that apart from a cookie banner, and a loop
#: told to get past obstacles would have wiped a game in progress — or, on a
#: different page, a draft, a cart, or an unsaved document.
DESTRUCTIVE_WARNINGS: tuple[str, ...] = (
    r"\bprogress\s+will\s+be\s+lost\b",
    r"\bwill\s+be\s+lost\b",
    r"\bcannot\s+be\s+undone\b",
    r"\bcan'?t\s+be\s+undone\b",
    r"\bpermanently\s+(?:delete|remove|erase)\b",
    r"\bdelete\s+(?:all|everything|permanently)\b",
    r"\bunsaved\s+changes\b",
    r"\bdiscard\s+(?:your\s+)?(?:changes|draft|work)\b",
    r"\bthis\s+action\s+is\s+irreversible\b",
)
# "start over" was on this list and is a control label, not a warning.
#
# A warning is a sentence about a consequence — "your progress will be
# lost", "this cannot be undone". "Start over" is the name of a button, and
# treating it as a warning meant every screen offering one was a screen she
# had to stop and ask about. Measured live: a finished game showing "Try
# again" and "Start over" could never be restarted, because deciding to
# restart it produced a halt saying the decision was the person's to make.

#: Words that suggest the thing on screen is an overlay at all, rather than
#: page content that happens to contain a button.
#: Fraction of the window occupied by its own toolbar/tab strip. Text above
#: this is furniture, not content, and a modal blocking the content is in the
#: content.
CHROME_STRIP_HEIGHT = 0.12

#: How many independent hints justify pressing Escape when nothing on screen
#: is safe to click by name. One word is a mention; several together are a
#: dialog. A single hint sent a loop into forty Escape presses.
MIN_HINTS_FOR_BARE_ESCAPE = 2

#: How far apart two runs of text can be and still be parts of one thing, as
#: a share of the screen. A dialog is a cluster: its wording and its controls
#: occupy one patch of screen, because they are one object. Words scattered
#: across a page are a page.
TOGETHER_X = 0.34
TOGETHER_Y = 0.18

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


def _region_y(region: dict[str, Any]) -> float | None:
    """Vertical centre of a text run, or None when it has no geometry."""
    try:
        return float(region.get("center_y", region.get("y")))
    except (TypeError, ValueError):
        return None


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



#: Warnings that stop her no matter what she intended. Losing this attempt's
#: progress is recoverable by doing it again; these are not.
NEVER_INTENDED = (
    r"\bpermanently\s+(?:delete|remove|erase)\b",
    r"\bdelete\s+(?:all|everything|permanently)\b",
    r"\bcannot\s+be\s+undone\b",
    r"\bcan'?t\s+be\s+undone\b",
    r"\bthis\s+action\s+is\s+irreversible\b",
)


def _where(region: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return (
            float(region.get("center_x", region.get("x", 0.0))),
            float(region.get("center_y", region.get("y", 0.0))),
        )
    except (TypeError, ValueError):
        return None


def _together(one: dict[str, Any], other: dict[str, Any]) -> bool:
    """Whether two runs of text are near enough to be parts of one thing."""
    here, there = _where(one), _where(other)
    if here is None or there is None:
        return True
    return abs(here[0] - there[0]) <= TOGETHER_X and abs(here[1] - there[1]) <= TOGETHER_Y


def _hints_around(
    region: dict[str, Any], layout: Sequence[dict[str, Any]]
) -> tuple[str, ...]:
    """The overlay wording near this control, rather than anywhere on screen.

    A dialog is a cluster. Its wording sits with its buttons because they are
    one object, and a page is not made into a dialog by holding the same words
    somewhere else on it.

    LIVE: a games page carried "SIGN UP" in one advertising rail and
    "Subscribe & Save 5%" in another, half a screen apart and half a screen
    from the board. Between them they matched two hints, which was the whole
    evidence needed to declare a dialog only the person could answer — and the
    task stopped with the board untouched in the middle of the screen.
    """
    nearby = " ".join(
        str(other.get("text") or "")
        for other in layout
        if _region_y(other) is None or _region_y(other) >= CHROME_STRIP_HEIGHT
        if _together(region, other)
    ).lower()
    return tuple(pattern for pattern in OVERLAY_HINTS if re.search(pattern, nearby))


def _is_what_she_intended(warnings: tuple[str, ...], intending: str) -> bool:
    """Whether a warning is about the thing she has just decided to do."""
    if not intending or not warnings:
        return False
    if any(pattern in NEVER_INTENDED for pattern in warnings):
        return False
    return "start over" in intending.lower() or "restart" in intending.lower()


def assess_overlay(observation: dict[str, Any], *, intending: str = "") -> OverlayVerdict:
    """Read one screen observation and decide whether something is in the way.

    Takes the same shape every reading already produces — `text` plus a
    `layout` of positioned runs — so any caller that can see can use this
    without new plumbing.

    ``intending`` names what the caller has just deliberately decided to do.
    A warning halts an action she did not choose; it is not there to stop her
    doing the thing she decided on. A dialog saying progress will be lost, to
    somebody who has just chosen to abandon this attempt and begin again, is
    a confirmation of that decision rather than an ambush — and only for
    warnings about the very thing being intended. Anything speaking of
    permanent deletion still stops, whatever was intended.
    """
    text = str(observation.get("text") or "")
    layout = list(observation.get("layout") or [])
    if not text and not layout:
        return OverlayVerdict()

    # Hints are counted from POSITIONED text, and the top strip is ignored.
    #
    # A window's own furniture is full of these words. Chrome carries an
    # "Install" button in its toolbar, and \binstall\b is a hint — so on the
    # first scoped run every single reading looked like a modal, the loop
    # pressed Escape forty times instead of playing, and made no moves at all.
    # That is the false positive this module's own tests warn about: dismissing
    # something the person wanted.
    #
    # Chrome and toolbars live at the top edge; a modal that is blocking the
    # content is IN the content. Ignoring the top strip is what separates them
    # without knowing anything about either application.
    lowered = text.lower()
    destructive = tuple(
        pattern for pattern in DESTRUCTIVE_WARNINGS if re.search(pattern, lowered)
    )
    if destructive and _is_what_she_intended(destructive, intending):
        # She decided this. The dialog is confirming it, not ambushing her.
        destructive = ()
    if destructive:
        # Stop here. Something on this dialog destroys work, and which button
        # does it is not knowable from a label — "Start New Game" reads like
        # progress and wipes a game. Dismissal controls on such a dialog are
        # not offered either, because "Cancel" on one dialog is "discard" on
        # another and the difference is invisible from the outside.
        return OverlayVerdict(
            present=True,
            needs_person=(
                "this dialog warns that something will be lost, so whether to "
                "go ahead is yours to decide"
            ),
            reasons=destructive,
        )

    body_text = " ".join(
        str(region.get("text") or "")
        for region in layout
        if _region_y(region) is None or _region_y(region) >= CHROME_STRIP_HEIGHT
    ).lower() or lowered
    hints = tuple(
        pattern for pattern in OVERLAY_HINTS if re.search(pattern, body_text)
    )

    best_score = 0
    best: dict[str, Any] | None = None
    accepting_seen: list[str] = []
    #: Acceptance controls that have dialog wording AROUND them, which is what
    #: separates a dialog from a page that says the same words somewhere else.
    accepting_here: list[tuple[str, tuple[str, ...]]] = []
    for region in layout:
        label = str(region.get("text") or "").strip()
        if not label or len(label) > 40:
            continue
        if _is_accepting(label):
            accepting_seen.append(label)
            near = _hints_around(region, layout)
            if near:
                accepting_here.append((label, near))
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

    if best is None and len(hints) >= MIN_HINTS_FOR_BARE_ESCAPE and not accepting_seen:
        # Something is in the way and nothing on it is safe to click by name.
        # Escape is the answer that needs no label: it closes without agreeing.
        return OverlayVerdict(
            present=True,
            suggested_key="escape",
            confidence=45,
            reasons=hints + ("no_labelled_dismissal_offered",),
        )

    if accepting_here and len(accepting_here[0][1]) >= MIN_HINTS_FOR_BARE_ESCAPE:
        # Something is asking for agreement and offering no way to decline.
        # That is the person's call, and closing it for them would be making a
        # commitment in their name.
        #
        # Only where something is really in the way. The same page that holds
        # the task also holds advertising, and advertising is written in the
        # language of consent: "Subscribe & Save 57%" in a banner beside a
        # game read as a dialog offering nothing but acceptance, and the whole
        # task stopped and asked the person to decide. A word is not a dialog.
        # The evidence required here is the evidence required to press Escape
        # at something — a page that merely SAYS "subscribe" is a page.
        return OverlayVerdict(
            present=True,
            needs_person=(
                "this dialog only offers acceptance ("
                + ", ".join(sorted({label for label, _near in accepting_here})[:3])
                + "), which is a decision for you rather than for me"
            ),
            reasons=hints or ("accepting_controls_only",),
        )

    # Hints with nothing to act on are not an obstacle worth reporting: a page
    # that merely MENTIONS cookies is a page, not a cookie wall.
    return OverlayVerdict(
        present=len(hints) >= MIN_HINTS_FOR_BARE_ESCAPE, reasons=hints
    )


def overlay_focus(observation: dict[str, Any]) -> tuple[float, float] | None:
    """Roughly where the blocking dialog sits, from its own text.

    Used to disambiguate a control that appears more than once. A declared
    label like "New Game", "Start" or "Continue" frequently names both a
    dialog's button and a permanent control in the app's own toolbar — on
    play2048 it matched four regions, one of them the toolbar. Clicking the
    toolbar one starts a game behind the dialog and leaves the dialog up, so
    the run stays blocked while every step reports success.

    The centroid of the hint-bearing text is where the dialog is talking, and
    the button that belongs to it is the one nearest that.
    """
    points: list[tuple[float, float]] = []
    for region in observation.get("layout") or []:
        text = str(region.get("text") or "").strip().lower()
        if not text or not any(re.search(hint, text) for hint in OVERLAY_HINTS):
            continue
        y = _region_y(region)
        try:
            x = float(region.get("center_x", region.get("x")))
        except (TypeError, ValueError):
            continue
        if y is None or y < CHROME_STRIP_HEIGHT:
            continue
        points.append((x, y))
    if not points:
        return None
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )


def overlay_box(observation: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """The rectangle the dialog's own text occupies, or None.

    A centroid is not enough to tell a dialog's button from an identically
    named control elsewhere in the app. Measured live: the dialog's text spanned
    x 0.42-0.46 while the page's permanent "New Game" button sat at x 0.759 at
    the SAME height, so nearest-to-centroid chose the toolbar — starting a game
    behind the dialog and leaving it up.

    A box answers what a point cannot: a control belongs to the dialog when it
    lies within the dialog's horizontal span, at or below its text. Buttons sit
    under the message that explains them, which is a property of dialogs
    generally rather than of any one page.
    """
    xs: list[float] = []
    ys: list[float] = []
    for region in observation.get("layout") or []:
        text = str(region.get("text") or "").strip().lower()
        if not text or not any(re.search(hint, text) for hint in OVERLAY_HINTS):
            continue
        y = _region_y(region)
        try:
            x = float(region.get("center_x", region.get("x")))
        except (TypeError, ValueError):
            continue
        if y is None or y < CHROME_STRIP_HEIGHT:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


__all__ = [
    "ACCEPTING_LABELS",
    "DESTRUCTIVE_WARNINGS",
    "CHROME_STRIP_HEIGHT",
    "MIN_HINTS_FOR_BARE_ESCAPE",
    "DISMISSIVE_LABELS",
    "OVERLAY_HINTS",
    "OverlayVerdict",
    "overlay_box",
    "overlay_focus",
    "assess_overlay",
]
