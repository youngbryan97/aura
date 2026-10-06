"""A compound task reaches its declared artifact owner before desktop fallback."""
from types import SimpleNamespace

import pytest

from core.intent.capability_selection import artifact_capability_precedes_desktop


def _catalogue(owner="mend_external_source"):
    return {
        owner: SimpleNamespace(description="Fix a broken program, game, page or script in a file. "
                               "Run it, watch its behaviour, repair its bugs, and play the result.",
                               effect_scope="read_write_artifacts", enabled=True),
        "general_body": SimpleNamespace(description="Open applications and click controls on the desktop.",
                                        effect_scope="foreground_desktop_control", enabled=True),
        "own_maintenance": SimpleNamespace(description="Repair your own architecture and source code.",
                                           effect_scope="privileged_mutation", enabled=True),
    }


@pytest.mark.parametrize("owner", ["mend_external_source", "a_registered_tool_added_tomorrow"])
def test_capability_names_do_not_decide_lane_ownership(owner):
    assert artifact_capability_precedes_desktop(
        "Fix the broken game at /tmp/example.html, then play it for three attempts to show the repair works.",
        _catalogue(owner))


def test_the_live_compound_request_cannot_become_generic_os_automation(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective
    from interface.routes.chat_desktop_evidence import (
        _desktop_objective_self_sufficient_without_cognitive_text,
    )

    original = ServiceContainer.peek
    engine = SimpleNamespace(skills=_catalogue())
    monkeypatch.setattr(ServiceContainer, "peek", lambda name, default=None:
                        engine if name == "capability_engine" else original(name, default=default))
    request = "Fix the broken game at /Users/bryan/aura-demos/pong/pong.html, then play it for three attempts to show the repair works."
    assert not looks_like_desktop_objective(request)
    assert not _desktop_objective_self_sufficient_without_cognitive_text(request)
    assert looks_like_desktop_objective("Open Calculator and click its controls.")


def test_an_ambiguous_or_disabled_owner_does_not_claim_the_lane():
    skills = _catalogue()
    skills["duplicate"] = skills["mend_external_source"]
    assert not artifact_capability_precedes_desktop("Fix a broken game and play the result.", skills)
    del skills["duplicate"]
    skills["mend_external_source"].enabled = False
    assert not artifact_capability_precedes_desktop("Fix a broken game and play the result.", skills)
    assert not artifact_capability_precedes_desktop("Set system volume to 30%.", _catalogue())


def test_registered_lane_claim_measures_its_declaration():
    from core.intent.capability_selection import _artifact_lane_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _artifact_lane_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    assert any(claim.test == "artifact_lane_follows_declared_effect" for claim in suite.claims())


def test_classifying_words_cannot_initialize_a_catalogue(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    monkeypatch.setattr(ServiceContainer, "peek", lambda name, default=None: default)
    def forbidden(*args, **kwargs):
        pytest.fail("a routing check tried to initialize a service")
    monkeypatch.setattr(ServiceContainer, "get", forbidden)
    assert looks_like_desktop_objective("Open Calculator and click its controls.")
