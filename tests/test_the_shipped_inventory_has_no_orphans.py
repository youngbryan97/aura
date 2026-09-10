"""Everything that ships has a way in.

"Every shipped capability, integration, UI control, and model" is a claim
about four registries, and each fails the same way: something declared that
nothing can reach. This repository has found that shape often enough — skills
that could never run, a proof that fired on nothing, a channel with no reader
— to stop taking the declaration as the answer.

Found the first time it ran, 2026-09-07: a ten-step first-run wizard nothing
opened, whose every settings write used a request body the API stopped
accepting; the engineering drawings surface, reachable only by typing its URL;
and the diagnostics page, the same.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inventory():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.reqproof.shipped import build

    return build()


def test_the_inventory_counted_all_four(inventory) -> None:
    """A kind that raised is a kind nobody checked."""

    assert not inventory["summary"]["could_not_count"]
    by_kind = inventory["summary"]["by_kind"]
    for kind in ("api", "model", "skill", "ui"):
        assert by_kind.get(kind, 0) > 0, f"{kind} enumerated nothing"


def test_nothing_ships_without_a_way_in(inventory) -> None:
    orphans = inventory["summary"]["orphans"]
    assert not orphans, (
        f"{len(orphans)} shipped things nothing can reach: {orphans[:12]}"
    )


def test_every_entry_says_what_reaches_it(inventory) -> None:
    silent = [
        f"{entry['kind']}:{entry['name']}"
        for entry in inventory["entries"]
        if entry["reachable"] and not entry.get("reachable_by")
    ]
    assert not silent, f"entries called reachable with nothing named: {silent[:12]}"


def test_a_settings_write_from_the_wizard_matches_the_api(inventory) -> None:
    """The wizard's request body is the one the settings route parses.

    It sent the settings object itself for months. The route reads `changes`
    and `expected_revision` off the body, so every step saved nothing and the
    wizard showed no sign of it.
    """

    wizard = (ROOT / "interface/static/first_run.js").read_text("utf-8")
    assert "expected_revision" in wizard
    assert '"changes"' in wizard or "changes," in wizard
    assert "onboarding.completed" in wizard

    from core.runtime.settings_schema import SCHEMA_BY_KEY

    assert "onboarding.completed" in SCHEMA_BY_KEY, (
        "the wizard writes a key the settings schema rejects as unknown"
    )


def test_the_first_run_offer_cannot_block_the_desktop(inventory) -> None:
    """An offer, not a gate. A failed read must leave the app as it was."""

    shell = (ROOT / "interface/static/aura.js").read_text("utf-8")
    start = shell.index("function offerFirstRunSetupOnce")
    body = shell[start:]
    assert "completed !== false" in body, (
        "the offer shows unless the answer is explicitly 'not completed'"
    )
    assert "location.href" not in body and "location.replace" not in body, (
        "the offer navigates the person somewhere instead of offering"
    )
    assert "Not now" in body, "the offer cannot be dismissed"
