"""A word is not a dialog.

Asked to play a game in a browser, she stopped before her first move and
handed the task back: "this dialog only offers acceptance (Subscribe & Save
57%), which is a decision for you rather than for me." There was no dialog.
There was an advertisement beside the board, and advertising is written in
the language of consent.

The judgement that something is in the way already requires evidence before
it will press Escape at a screen. Deciding that something needs the person
takes the whole task away from them, so it cannot ask for less.
"""
from __future__ import annotations

from core.perception.blocking_overlay import MIN_HINTS_FOR_BARE_ESCAPE, assess_overlay


def _seen(*runs) -> dict:
    """One reading, in the shape every observation already produces."""
    layout = [
        {"text": text, "x": 0.5, "y": y, "width": 0.2, "height": 0.03, "center_x": 0.5, "center_y": y}
        for text, y in runs
    ]
    return {"ok": True, "text": " ".join(text for text, _y in runs), "layout": layout}


def test_an_advertisement_beside_the_task_does_not_stop_it():
    page = _seen(
        ("play2048.co", 0.08),
        ("Subscribe & Save 57%", 0.22),
        ("2048", 0.4),
        ("Join the numbers and get to the 2048 tile!", 0.52),
        ("New Game", 0.6),
    )
    verdict = assess_overlay(page)
    assert not verdict.needs_person, verdict.needs_person


def test_a_real_consent_wall_still_belongs_to_the_person():
    """Where the page really is behind a wall, agreeing on someone's behalf is
    making a commitment in their name."""
    wall = _seen(
        ("We value your privacy", 0.3),
        ("This site uses cookies to personalise content and ads.", 0.4),
        ("Accept all cookies", 0.55),
        ("Sign up for our newsletter", 0.62),
    )
    verdict = assess_overlay(wall)
    assert verdict.present
    assert verdict.needs_person
    assert "acceptance" in verdict.needs_person


def test_the_evidence_required_is_the_same_either_way():
    """Pressing Escape at a screen is the smaller move; handing the task back
    is the larger one, and it cannot ask for less evidence."""
    import inspect

    from core.perception import blocking_overlay

    source = inspect.getsource(blocking_overlay.assess_overlay)
    assert "if accepting_seen and len(hints) >= MIN_HINTS_FOR_BARE_ESCAPE:" in source
    assert MIN_HINTS_FOR_BARE_ESCAPE >= 2


def test_a_wall_offering_a_way_out_is_dismissed_rather_than_handed_over():
    """Nothing above changes what she does when there IS a way through."""
    dismissible = _seen(
        ("We use cookies to improve your experience", 0.3),
        ("This website stores data such as cookies", 0.36),
        ("Reject all", 0.5),
        ("Accept all", 0.5),
    )
    verdict = assess_overlay(dismissible)
    assert verdict.present
    assert not verdict.needs_person
    assert verdict.label == "Reject all"
