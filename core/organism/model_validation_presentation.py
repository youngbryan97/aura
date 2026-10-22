"""The browser-rendered presentation check is valid only for the measured source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.organism.nothing_measured import NothingMeasured

SOURCES = (
    "interface/static/aura.js", "interface/static/conversation_visuals.js",
    "interface/static/conversation_visuals.css", "interface/static/neuron_glyph.svg",
    "core/perception/browser_presence.py", "core/perception/browser_presence.js",
    "core/capabilities/phantom_browser.py", "core/capabilities/phantom_browser_interaction.py",
    "core/perception/a_picture_of_her_page.py", "core/skills/watching_along.py",
    "core/skills/sovereign_browser_drawing.py",
)
CHECKS = frozenset({
    "numeric chart values", "source table disclosure", "idempotent rendering",
    "diagram edges", "stream completion", "missing data fallback", "metric cards",
    "user text unchanged", "code retained literally", "390px width", "background-tab click",
    "closed shadow isolation", "nested screenshot suspension", "no conversation activity notifications",
})
GRADIENT_SOURCES = ("interface/static/ambient_gradient.js", "interface/static/aura.css",
                    "interface/static/index.html")


def measured_gradient(root: Path | None = None) -> bool:
    root = root or Path(__file__).resolve().parents[2]
    for path in sorted((root / "docs/evidence").glob("ambient-gradient-*.json"), reverse=True):
        try:
            receipt = json.loads(path.read_text())
            if any(receipt.get("sources", {}).get(p) != hashlib.sha256((root / p).read_bytes()).hexdigest()
                   for p in GRADIENT_SOURCES):
                continue
        except (OSError, ValueError):
            continue
        return receipt.get("ok") is True and {
            "pixels change over time", "reduced motion stops movement", "voice mode stops background work",
        } <= set(receipt.get("checks", []))
    raise NothingMeasured("No gradient measurement matches the current source.")


def measured_presentation(root: Path | None = None) -> bool:
    root = root or Path(__file__).resolve().parents[2]
    for path in sorted((root / "docs/evidence").glob("conversation-presence-*.json"), reverse=True):
        try:
            receipt = json.loads(path.read_text())
            if receipt.get("schema") != "aura.presentation.browser_measurement.v1":
                continue
            if any(receipt.get("sources", {}).get(source) != hashlib.sha256((root / source).read_bytes()).hexdigest()
                   for source in SOURCES):
                continue
        except (OSError, ValueError):
            continue
        return receipt.get("ok") is True and CHECKS <= set(receipt.get("checks", []))
    raise NothingMeasured("No browser presentation measurement matches the current source.")


def install_presentation_claims(suite) -> None:
    from core.organism.model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        boolean_score,
    )

    statement = "The measured conversation renderer preserves chart values and source data, and its owned browser can click behind another tab."
    suite.add_test(ValidationTest(
        name="conversation_presentation_measured_in_browser", description=statement, required_capability="",
        observation=Observation(name="browser_presentation", value=True,
                                source="tests/test_conversation_visuals.py; docs/evidence/conversation-presence-2026-10-21.json"),
        predict=lambda _m: measured_presentation(),
        score=lambda p, o: boolean_score(bool(p), expected=bool(o.value), subject="browser presentation"),
        owner="interface/static/conversation_visuals.js",
    ))
    suite.add_claim(Claim(statement=statement, test="conversation_presentation_measured_in_browser",
                          owner="interface/static/conversation_visuals.js", asserted_in=__file__,
                          evidence=Evidence.MEASURED_SYNTHETIC,
                          evidence_note="Offline Chromium UI fixture using production renderers. No resident-model conversation or native desktop cursor was measured."))
    gradient_statement = "The measured chat colour field changes pixels and stops moving for reduced motion and focused voice."
    suite.add_test(ValidationTest(
        name="ambient_gradient_measured", description=gradient_statement, required_capability="",
        observation=Observation(name="ambient_gradient", value=True, source="docs/evidence/ambient-gradient-*.json"),
        predict=lambda _m: measured_gradient(),
        score=lambda p, o: boolean_score(bool(p), expected=bool(o.value), subject="ambient gradient"),
        owner="interface/static/ambient_gradient.js",
    ))
    suite.add_claim(Claim(statement=gradient_statement, test="ambient_gradient_measured",
                          owner="interface/static/ambient_gradient.js", asserted_in=__file__,
                          evidence=Evidence.MEASURED_SYNTHETIC,
                          evidence_note="Rendered Chromium fixture. This does not measure native WebKit GPU usage."))
