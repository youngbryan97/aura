from __future__ import annotations

from core.runtime.os_automation_effects import (
    DesktopSnapshot,
    EffectKind,
    build_effect_contract,
    canonical_app_target,
    evaluate_effect_contract,
    extract_target_apps,
)


def _snapshot(**overrides: object) -> DesktopSnapshot:
    values: dict[str, object] = {
        "frontmost_app": "Google Chrome",
        "frontmost_window": "Aura proof",
        "window_frame": (300, 100, 1100, 800),
        "desktop_frame": (0, 0, 1920, 1080),
        "window_minimized": False,
        "focused_value_excerpt": "",
        "browser_url": "https://example.com/start",
        "screen_text": "",
        "clipboard_excerpt": "",
        "running_apps": ("Finder", "Google Chrome"),
    }
    values.update(overrides)
    return DesktopSnapshot.from_mapping(values)


def test_window_contract_rejects_generic_foreground_state() -> None:
    contract = build_effect_contract(
        "Resize the current browser window and arrange it on the left side of the screen."
    )
    before = _snapshot()
    unchanged = _snapshot()

    verdict = evaluate_effect_contract(contract, before, unchanged)

    assert contract.verifiable is True
    assert {item.kind for item in contract.requirements} == {
        EffectKind.APP_FRONTMOST,
        EffectKind.WINDOW_REGION,
    }
    assert verdict.verified is False
    assert any("expected window region" in reason for reason in verdict.failure_reasons)


def test_window_contract_verifies_requested_region_against_desktop_geometry() -> None:
    contract = build_effect_contract(
        "Resize the current browser window and arrange it on the left side of the screen."
    )
    before = _snapshot()
    partial_move = _snapshot(window_frame=(0, 24, 230, 408))
    after = _snapshot(window_frame=(0, 24, 940, 1030))

    partial_verdict = evaluate_effect_contract(contract, before, partial_move)
    verdict = evaluate_effect_contract(contract, before, after)

    assert partial_verdict.verified is False
    assert verdict.verified is True
    assert any("window_region=left_half" in evidence for evidence in verdict.evidence)


def test_text_contract_requires_visible_readback_not_clipboard_content() -> None:
    contract = build_effect_contract(
        "Open Notes and write the requested status note.",
        text_payload="Daily reliability review complete.",
    )
    before = _snapshot(frontmost_app="Finder")
    clipboard_only = _snapshot(
        frontmost_app="Notes",
        clipboard_excerpt="Daily reliability review complete.",
    )

    clipboard_verdict = evaluate_effect_contract(contract, before, clipboard_only)
    visible_verdict = evaluate_effect_contract(
        contract,
        before,
        _snapshot(
            frontmost_app="Notes",
            focused_value_excerpt="Daily reliability review complete.",
            clipboard_excerpt="Daily reliability review complete.",
        ),
    )

    assert clipboard_verdict.verified is False
    assert visible_verdict.verified is True


def test_user_app_wording_resolves_to_launchservices_identity() -> None:
    assert canonical_app_target("my Note app") == "Notes"
    contract = build_effect_contract("Open my Note app and write Hello.")
    assert {
        item.expected
        for item in contract.requirements
        if item.kind == EffectKind.APP_FRONTMOST
    } == {"Notes"}


def test_app_target_language_separates_apps_from_browser_surfaces() -> None:
    assert extract_target_apps(
        "Open the application DefinitelyNotInstalledAuraProbe."
    ) == ("DefinitelyNotInstalledAuraProbe",)
    assert extract_target_apps("Launch the app named Remote Studio.") == (
        "Remote Studio",
    )
    assert extract_target_apps("Open a tab for Google Docs.") == ()
    assert extract_target_apps("Open Google Docs and start typing.") == ()


def test_search_contract_verifies_active_browser_destination() -> None:
    contract = build_effect_contract(
        "Search Google for Aura causal verification.",
        expected_url="https://www.google.com/search?q=Aura+causal+verification",
    )
    verdict = evaluate_effect_contract(
        contract,
        _snapshot(browser_url="https://example.com/start"),
        _snapshot(browser_url="https://www.google.com/search?q=Aura%20causal%20verification"),
    )

    assert verdict.verified is True


def test_calculator_contract_requires_visible_expected_result() -> None:
    contract = build_effect_contract("Open Calculator and click 2 plus 3 equals.")
    wrong = evaluate_effect_contract(
        contract,
        _snapshot(frontmost_app="Finder"),
        _snapshot(frontmost_app="Calculator", focused_value_excerpt="4"),
    )
    correct = evaluate_effect_contract(
        contract,
        _snapshot(frontmost_app="Finder"),
        _snapshot(frontmost_app="Calculator", focused_value_excerpt="5"),
    )

    assert wrong.verified is False
    assert correct.verified is True
    assert any("calculation_result=5" in evidence for evidence in correct.evidence)


def test_named_interaction_requires_target_presence_and_visible_state_change() -> None:
    contract = build_effect_contract("Click the Continue button.")
    before = _snapshot(screen_text="Setup\nContinue")
    unchanged = _snapshot(screen_text="Setup\nContinue")
    changed = _snapshot(screen_text="Account details")

    assert evaluate_effect_contract(contract, before, unchanged).verified is False
    assert evaluate_effect_contract(contract, before, changed).verified is True


def test_unbounded_interaction_is_declared_unverifiable() -> None:
    contract = build_effect_contract("Click around until it works.")

    assert contract.verifiable is False
    assert contract.unsupported_reasons


def test_quit_app_requires_running_app_readback() -> None:
    contract = build_effect_contract("Quit the Preview app.")
    before = _snapshot(running_apps=("Finder", "Preview"))
    after = _snapshot(frontmost_app="Finder", running_apps=("Finder",))

    verdict = evaluate_effect_contract(contract, before, after)

    assert verdict.verified is True
    assert any("app_not_running=Preview" in evidence for evidence in verdict.evidence)


def test_close_window_is_not_misrepresented_as_process_termination() -> None:
    contract = build_effect_contract("Close the Preview app.")

    assert contract.verifiable is False
    assert any("window-closure" in reason for reason in contract.unsupported_reasons)


def test_generic_app_descriptor_does_not_become_a_fake_application_name() -> None:
    contract = build_effect_contract("Open a visible app and prepare a short note.")

    assert contract.verifiable is False
    assert contract.requirements == ()


def test_incidental_app_name_in_content_is_not_an_execution_target() -> None:
    contract = build_effect_contract(
        "Write a report about Chrome into Notes.",
        text_payload="Chrome release notes",
    )

    targets = {
        item.expected
        for item in contract.requirements
        if item.kind == EffectKind.APP_FRONTMOST
    }
    assert targets == {"Notes"}


def test_supported_prefix_cannot_hide_unverified_destructive_suffix() -> None:
    contract = build_effect_contract("Open Notes and delete the current note.")

    assert contract.verifiable is False
    assert any("deletion" in reason for reason in contract.unsupported_reasons)


def test_connector_variants_cannot_hide_unverified_destructive_suffix() -> None:
    for goal in (
        "Open Notes to delete the current note.",
        "Open Notes where you should delete the current note.",
        "Open Notes; please delete the current note.",
    ):
        contract = build_effect_contract(goal)
        assert contract.verifiable is False
        assert any("deletion" in reason for reason in contract.unsupported_reasons)


def test_action_target_stops_before_then_or_to_connector() -> None:
    then_contract = build_effect_contract(
        'Open Notes then type "Daily review complete".'
    )
    to_contract = build_effect_contract(
        'Open Notes to type "Daily review complete".'
    )

    assert {
        item.expected
        for item in then_contract.requirements
        if item.kind == EffectKind.APP_FRONTMOST
    } == {"Notes"}
    assert {
        item.expected
        for item in to_contract.requirements
        if item.kind == EffectKind.APP_FRONTMOST
    } == {"Notes"}
