from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_computer_use_perform_propagates_driver_rejection():
    from core.tools.computer_use import ComputerUseAction, ComputerUseSkill

    skill = ComputerUseSkill()

    async def rejecting_driver(_action):
        return {"ok": False, "error": "desktop permission denied"}

    skill.register_driver("screenshot", rejecting_driver)

    result = await skill.perform(
        ComputerUseAction(kind="screenshot", target="display:0"),
        sandbox_check=lambda _capability, _target: (True, "ok"),
        capability_grant=True,
    )

    assert result.ok is False
    assert result.output == {"ok": False, "error": "desktop permission denied"}
    assert "desktop permission denied" in result.failure_reason


@pytest.mark.asyncio
async def test_computer_use_window_detection_uses_read_capability():
    from core.tools.computer_use import ComputerUseAction, ComputerUseSkill

    skill = ComputerUseSkill()
    seen_capabilities = []

    async def window_driver(_action):
        return {"ok": True, "window_tree": "Finder"}

    def sandbox_check(capability, _target):
        seen_capabilities.append(capability)
        return True, "ok"

    skill.register_driver("detect_windows", window_driver)

    result = await skill.perform(
        ComputerUseAction(kind="detect_windows", target="display:0"),
        sandbox_check=sandbox_check,
        capability_grant=True,
    )

    assert result.ok is True
    assert seen_capabilities == ["browser.read"]


@pytest.mark.asyncio
async def test_computer_use_default_screenshot_fails_honestly_without_backend(monkeypatch):
    import core.skills._pyautogui_runtime as pyautogui_runtime
    import core.tools.computer_use as module

    records = []

    async def missing_screencapture(*_args, **_kwargs):
        message = "screencapture missing"
        raise FileNotFoundError(message)

    def unavailable_pyautogui():
        return None, "unavailable"

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", missing_screencapture)
    monkeypatch.setattr(pyautogui_runtime, "get_pyautogui", unavailable_pyautogui)
    monkeypatch.setattr(module, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    # The screen-capture privacy gate is fail-closed and runs BEFORE any
    # backend is tried, so with no readable frontmost window this refuses on
    # privacy and never reaches the backend this test is about. Give it a
    # window it is allowed to capture; the gate still runs.
    import core.security.screen_capture_policy as policy

    monkeypatch.setattr(
        policy,
        "evaluate_screen_capture_admission",
        lambda **_kw: policy.ScreenCaptureAdmission(
            allowed=True, reason=None, authority="test"
        ),
    )

    skill = module.ComputerUseSkill()

    with pytest.raises(RuntimeError, match="screen capture unavailable"):
        await skill._default_screenshot(module.ComputerUseAction(kind="screenshot", target="display:0"))

    assert records


def test_computer_use_rejects_invalid_driver_registration():
    from core.tools.computer_use import ComputerUseSkill

    skill = ComputerUseSkill()

    with pytest.raises(ValueError, match="kind must be non-empty"):
        skill.register_driver("", lambda _action: None)

    with pytest.raises(TypeError, match="driver must be callable"):
        skill.register_driver("screenshot", None)
