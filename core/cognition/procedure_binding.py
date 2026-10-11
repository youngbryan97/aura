"""Bind procedural variables to controls revealed by a measured interaction.

Method words describe input delivery. A named control identifies an input.
Opening a selector can expose the values of an otherwise unnamed choice slot.
That binding belongs to the observed surface and its current control inventory.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from core.verify.invariants import invariant

_METHOD_WORDS = frozenset("click clicking clicks press pressing here to open opening choose choosing select selecting "
                          "use using with the a an left right mouse pointer button buttons control controls".split())
_INPUT_WORDS = frozenset("click clicking clicks press pressing mouse pointer button buttons left right".split())
_CHOICE = re.compile(r"\b(?:choose|select|pick)\b", re.I)
_SLOT = re.compile(r"^(?:(?:the|a|an|any|one)\s+)?(?:type|kind|option|item|template|component|style|entry)\b", re.I)
_OPENS = re.compile(r"\b(?:open|expand|show|browse)\b", re.I)
_CONTAINER = re.compile(r"\b(?:library|panel|palette|catalog|catalogue|selector|list|menu|browser|chooser)\b", re.I)
_COMMAND = re.compile(r"^\W*(?:click|press|open|close|skip|delete|remove|test|run|simulate|execute|render|preview|"
                      r"start|clear|reset|undo|redo|back|menu|help|quit|exit|options|settings|pause|save|load|submit|"
                      r"done|finish|launch|play|next|continue|restart|retry|cancel|show|hide|rotate|turn|flip|zoom)\b", re.I)
_CONSUMES = re.compile(r"^\W*(?:test|run|simulate|execute|render|preview|evaluate|validate|compile|"
                       r"check\s+(?:my|the)|try\s+(?:it|this)|launch)\b", re.I)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).casefold()))


def control_of(frame: Any) -> str:
    """The named control, excluding prose that merely describes how to use it."""
    using = str(frame.using or "")
    words = _tokens(using)
    if not words or words <= _METHOD_WORDS:
        return ""
    return using


def input_method_only(frame: Any) -> bool:
    """A bare input method has no literal target name to match."""
    words = _tokens(frame.using or frame.sentence)
    return (frame.act == "click things" and not frame.thing and not frame.where
            and bool(words) and words <= _INPUT_WORDS and not re.search(r"[\"'“”]", frame.sentence))


def choice_slot(frame: Any) -> bool:
    return (frame.act == "click things" and not control_of(frame)
            and bool(_CHOICE.search(f"{frame.using} {frame.sentence}"))
            and bool(_SLOT.search(str(frame.thing or ""))))


def opens_choices(frame: Any) -> bool:
    return (frame.act == "click things" and bool(_OPENS.search(f"{frame.using} {frame.sentence}"))
            and bool(_CONTAINER.search(str(frame.thing or ""))))


def selector_for(frame: Any, choice: Any) -> bool:
    """A selector exposes the entity the choice names, or an unrestricted option slot."""
    generic = _METHOD_WORDS | frozenset("of one any you your wish want need".split())
    subject = {word.rstrip("s") for word in _tokens(_SLOT.sub("", choice.thing, count=1)) - generic}
    return opens_choices(frame) and choice_slot(choice) and (not subject or subject <= {word.rstrip("s") for word in _tokens(frame.thing)})


def consumes_artifact(label: str) -> bool:
    """An evaluation or execution control that needs an artifact in a construction task."""
    return bool(_CONSUMES.search(str(label or "")))


def selection_label(label: str) -> bool:
    return bool(str(label).strip()) and not str(label).startswith(("the shape at", "the one that stands out", "the middle")) and not _COMMAND.search(label)


@dataclass(frozen=True)
class RevealedChoices:
    surface: str
    bounds: tuple
    epoch: str
    capture_at: float
    moves: frozenset[str]
    opened_by: str
    for_step: str

    @classmethod
    def from_effect(cls, before: dict, after: dict, step_key: str, choice_key: str) -> RevealedChoices | None:
        from core.agency.what_i_can_do_here import what_is_clicked

        surface = str(after.get("surface_id") or "")
        epoch = str(after.get("capture_epoch") or "")
        earlier, later = before.get("capture_at"), after.get("capture_at")
        bounds = after.get("bounds") or ()
        if (not surface or surface != before.get("surface_id") or not epoch or epoch == before.get("capture_epoch")
                or not before.get("capture_epoch")
                or after.get("bounds") != before.get("bounds")
                or len(bounds) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or bounds[2] <= 0 or bounds[3] <= 0
                or type(earlier) not in (int, float) or type(later) not in (int, float)
                or not math.isfinite(earlier) or not math.isfinite(later) or not earlier < later):
            return None
        introduced = set(after.get("clickable") or ()) - set(before.get("clickable") or ())
        moves = frozenset(move for move in introduced if (label := what_is_clicked(move)) and selection_label(label))
        return cls(surface, tuple(bounds), epoch, later, frozenset(sorted(moves)[:128]), step_key, choice_key) if len(moves) >= 2 else None

    def offers(self, move: str, current: dict, step_key: str) -> bool:
        at = current.get("capture_at")
        return (current.get("surface_id") == self.surface and tuple(current.get("bounds") or ()) == self.bounds
                and bool(current.get("capture_epoch")) and self.for_step == step_key
                and type(at) in (int, float) and math.isfinite(at) and at >= self.capture_at
                and (current["capture_epoch"] != self.epoch or at == self.capture_at)
                and move in self.moves and move in (current.get("clickable") or ()))


@invariant("cognition.choice_bindings_require_observed_current_controls", scope="cognition",
           owner="core/cognition/procedure_binding.py", observational=False)
def _choice_binding_invariant() -> tuple:
    before = {"surface_id": "editor", "bounds": [0, 0, 400, 300], "capture_epoch": "a", "capture_at": 1., "clickable": []}
    after = {**before, "capture_epoch": "b", "capture_at": 2., "clickable": ['click "Source"', 'click "Filter"', 'click "Run"']}
    binding = RevealedChoices.from_effect(before, after, "open", "choose")
    assert binding is not None and binding.offers('click "Filter"', after, "choose")
    assert not binding.offers('click "Run"', after, "choose")
    for changed in ({"surface_id": "elsewhere"}, {"capture_at": 1.}, {"clickable": []}, {"bounds": [0, 0, 800, 600]}):
        assert not binding.offers('click "Filter"', {**after, **changed}, "choose")
    assert RevealedChoices.from_effect(after, after, "open", "choose") is None
    return ()
