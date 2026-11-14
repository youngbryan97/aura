from __future__ import annotations

from pathlib import Path

from interface.routes.settings import _schema_payload

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
HTML = (ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")


def test_runtime_controls_are_backend_mapped_not_local_storage_preferences():
    local_defaults = JS.split("const defaultSettings =", 1)[1].split("};", 1)[0]
    for dead_local_key in (
        "autolisten",
        "ttsSpeed",
        "enrichment",
        "reflection",
        "autonomy",
        "approval",
    ):
        assert dead_local_key not in local_defaults

    expected = {
        "voice.input_enabled": "setting-voice-input",
        "voice.output_enabled": "setting-voice-output",
        "voice.auto_listen": "setting-autolisten",
        "voice.output_rate": "setting-tts-speed",
        "learning.auto_enrichment_enabled": "setting-enrichment",
        "learning.reflection_enabled": "setting-reflection",
        "governance.approval_mode": "setting-approval",
    }
    for key, control_id in expected.items():
        assert f"'{key}': {{ id: '{control_id}'" in JS
        assert f'id="{control_id}"' in HTML
    assert "'autonomy.actions_enabled': { id:" not in JS
    assert 'id="setting-autonomy-status"' in HTML
    assert "autonomyStatus.textContent = 'ACTIVE'" in JS
    autonomy_schema = next(
        setting
        for setting in _schema_payload()
        if setting["key"] == "autonomy.actions_enabled"
    )
    assert autonomy_schema["default"] is True
    assert autonomy_schema["mutable"] is False
    autonomy_level_schema = next(
        setting
        for setting in _schema_payload()
        if setting["key"] == "autonomy.level"
    )
    assert autonomy_level_schema["default"] == "full"
    assert autonomy_level_schema["mutable"] is False
    controls_html = (ROOT / "interface" / "static" / "controls.html").read_text(
        encoding="utf-8"
    )
    assert 'id="seg-autonomy.level"' not in controls_html
    assert "intrinsic · active" in controls_html


def test_runtime_settings_client_has_cas_conflict_and_idempotency_contracts():
    assert "expected_revision: expectedRevision" in JS
    assert "request_id: requestId" in JS
    assert "changes: desired" in JS
    assert "payload.error === 'settings_revision_conflict'" in JS
    assert "settings_conflict_requires_review" in JS
    assert "runtimeSettingsRequestId()" in JS
    assert "await hydrateRuntimeSettings({ quiet: true, reconcileVoice: false })" in JS
    assert "if (payload.superseded === true)" in JS
    assert "settings_idempotent_replay_superseded" in JS


def test_auto_listen_runs_the_duplex_lane_and_keeps_the_server_lane_as_fallback():
    """LIVE 13 November 2026 auto-listen ran the server lane because it came first: his quiet speech fell under its
    fixed loudness gate, and what he said while she answered was dropped. The duplex lane is tried first; the server
    lane listens only when the browser cannot."""
    assert "reconcileAutoListenFromSettings" in JS
    body = JS[JS.index("async function reconcileAutoListenFromSettings("):JS.index("async function hydrateRuntimeSettings(")]
    assert "await duplex.setAmbient(true)" in body
    assert "await toggleVoice(false)" in body  # input disabled still stops the browser microphone
    assert "state.voiceSummary.server_capture === true" in body
    assert body.index("await duplex.setAmbient(true)") < body.index("state.voiceSummary.server_capture === true")
    assert "acknowledgeFrontendSetting" not in JS


def test_microphone_button_obeys_runtime_input_gate():
    assert "runtimeSettingsState.values['voice.input_enabled'] !== true" in JS
    assert "Microphone input is disabled in Runtime Settings" in JS
    assert "inputEnabled ? 'Voice unavailable' : 'Microphone input disabled'" in JS


def test_chat_confirmation_modal_retries_original_turn_without_duplicate_render():
    assert "approvalStatus === 'approval_required'" in JS
    assert "approvalStatus === 'require_fresh_user_auth'" in JS
    assert "(approval && approval.challenge_id)" in JS
    assert "void runChatRequest(item, { messageAlreadyRendered: true })" in JS
    assert "await confirmNextRuntimeAction(challengeId)" in JS
    assert "JSON.stringify({ challenge_id: String(challengeId || '') })" in JS
    assert "'/api/settings/auth/revoke'" in JS
    assert "approvalConfirmationInFlight" in JS
    assert "approvalModal.querySelectorAll" in JS
    assert 'aria-describedby="approval-modal-message"' in HTML
    assert "if (retry) void retry()" in JS
    assert "setting-confirm-next-action" not in HTML
    assert "setting-confirmation-status" not in HTML
    for element_id in (
        "approval-modal",
        "approval-modal-message",
        "approval-modal-cancel",
        "approval-modal-confirm",
    ):
        assert f'id="{element_id}"' in HTML


def test_inert_control_disclaimer_and_styling_are_gone():
    assert "setting-row-unwired" not in HTML
    assert "settings-help-unwired" not in HTML
    assert "these do not gate autonomy" not in HTML.lower()
    assert "not connected yet" not in HTML.lower()
