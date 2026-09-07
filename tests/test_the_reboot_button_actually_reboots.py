"""Pressing Reboot in the header has to reboot.

LIVE, 2026-08-22, two defects stacked on one button.

It was silent first. `confirm()` routes to the app's WKUIDelegate, and the
shell implements the media-capture handler and none of the JavaScript panels,
so WebKit answered false itself and showed nothing. Pressed twice, the runtime
kept the same pid and start time and the log recorded no request at all.

With the page asking its own question, the request went out and came back 500:
"GovernanceViolationError: subprocess_gateway.spawn:runtime_relaunch:
schedule_relaunch called outside governed context". Arranging your own
replacement is internal maintenance, and the gateway refuses maintenance that
does not declare itself.
"""

from __future__ import annotations

from pathlib import Path

UI = Path("interface/static/aura.js")
SHELL = Path("scripts/AuraLauncher.swift")
RELAUNCH = Path("core/runtime/runtime_relaunch.py")


def test_successor_captures_source_after_predecessor_exits(monkeypatch, tmp_path):
    from core.runtime import launch_provenance, runtime_relaunch

    events = []
    monkeypatch.setattr(runtime_relaunch, "wait_for_predecessor",
                        lambda *_: events.append("wait"))
    monkeypatch.setattr(runtime_relaunch.os, "chdir", lambda *_: None)
    def capture(root, **kwargs):
        assert root == str(tmp_path)
        assert kwargs == {"new_launch": True}
        events.append("capture")
    monkeypatch.setattr(launch_provenance, "bind_runtime_source_snapshot", capture)
    def execute(*_):
        events.append("exec")
        raise OSError("test does not launch a runtime")
    monkeypatch.setattr(runtime_relaunch.os, "execv", execute)
    assert runtime_relaunch.main([
        "--pid", "123", "--cwd", str(tmp_path), "--", "python", "aura_main.py"
    ]) == 1
    assert events == ["wait", "capture", "exec"]


def test_the_page_does_not_depend_on_the_shell_for_a_question():
    source = UI.read_text(encoding="utf-8")
    assert "function auraConfirm(" in source
    # No bare confirm() left driving a control.
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        assert "if (confirm(" not in stripped, stripped


def test_both_controls_ask_through_the_page():
    source = UI.read_text(encoding="utf-8")
    assert "await auraConfirm('Reboot Aura?" in source
    assert "await auraConfirm('Clear the transcript" in source


def test_the_dialog_can_be_dismissed_by_keyboard():
    source = UI.read_text(encoding="utf-8")
    block = source[source.index("function auraConfirm("):]
    block = block[: block.index("window.auraConfirm = auraConfirm;")]
    assert "'Escape'" in block
    assert "'Enter'" in block
    assert "go.focus()" in block


def test_the_shell_answers_webkits_dialogs():
    """A delegate that silently denies is a trap for whatever asks next."""
    source = SHELL.read_text(encoding="utf-8")
    for method in (
        "runJavaScriptAlertPanelWithMessage",
        "runJavaScriptConfirmPanelWithMessage",
        "runJavaScriptTextInputPanelWithPrompt",
    ):
        assert method in source, method
    # Only Aura's own runtime is answered.
    block = source[source.index("runJavaScriptConfirmPanelWithMessage"):]
    block = block[: block.index("runJavaScriptTextInputPanelWithPrompt")]
    assert "isLocalRuntimeOrigin(frame.securityOrigin)" in block


def test_arranging_a_relaunch_declares_its_governed_scope():
    source = RELAUNCH.read_text(encoding="utf-8")
    block = source[source.index("def schedule_relaunch("):]
    block = block[: block.index("def main(")]
    assert "local_internal_governed_scope(" in block
    assert block.index("local_internal_governed_scope(") < block.index("get_subprocess_gateway().spawn(")


def test_a_relaunch_that_cannot_be_arranged_refuses_rather_than_killing():
    """Nothing has been signalled yet, so the runtime is still up."""
    source = Path("interface/routes/subsystems.py").read_text(encoding="utf-8")
    block = source[source.index("Reboot requested via API"):]
    block = block[: block.index("raise_signal")]
    assert "reboot_unavailable" in block
    assert "503" in block
