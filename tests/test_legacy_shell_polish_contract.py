import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _code_only(js: str) -> str:
    """Strip comments so a "we used to do X" note explaining a fix cannot
    satisfy — or trip — an assertion about what the code still does."""
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    js = re.sub(r"^\s*//.*$", "", js, flags=re.M)
    return js


def test_legacy_shell_keeps_constitutional_health_slots():
    html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")

    required_ids = [
        "brand-status-dot",
        "hud-status",
        "c-policy-mode",
        "c-fragmentation",
        "c-contradictions",
        "c-contested",
        "c-commitments",
        "c-tools-available",
        "health-flags",
        "rolling-summary",
        "phenomenal-summary",
        "tool-available-count",
        "tool-degraded-count",
        "tool-voice-state",
        "tool-last-stage",
        "tool-last-detail",
    ]

    for item in required_ids:
        assert f'id="{item}"' in html, f"legacy shell missing {item}"


def test_legacy_shell_frontend_uses_bootstrap_and_tool_catalog():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "/api/ui/bootstrap" in js
    assert "/api/tools/catalog" in js
    assert "tool_event" in js
    assert "hydrateBootstrap" in js
    assert "renderToolCatalog" in js


def test_legacy_shell_keeps_policy_deferral_neutral():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function toolEventIsDeferred(event)" in js
    assert "resultStatus === 'deferred'" in js
    assert "typeof rawResult === 'string'" in js
    assert "deferralSignals.some(" in js
    assert "const deferred = toolEventIsDeferred(event);" in js
    assert "if (!deferred && event && ['failed', 'rejected', 'degraded']" in js


def test_legacy_shell_presents_cold_standby_as_not_ready_shell_state():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function laneIsStandby" in js
    assert "if (laneIsStandby(lane)) return 'preparing';" in js
    assert "CORTEX PREPARING" in js
    assert "Aura is ready. Cortex will warm on first turn." not in js
    assert "syncSplashState(payload);" in js
    assert "Live shell is still syncing. Aura is stabilizing background channels..." in js
    assert "setTimeout(() => dismissSplash(), 8000)" not in js


def test_legacy_shell_placeholder_is_system_status_not_aura_speech():
    html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "Conversation lane initializing. Waiting for verified Aura reply path..." in html
    assert "Conversation lane initializing. Waiting for verified Aura reply path..." in js
    assert "Aura: Infinity online. Synchronizing cognitive drives" not in html
    assert "Aura: Infinity online. Synchronizing cognitive drives" not in js


def test_legacy_shell_presents_active_generation_as_working_not_unavailable():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function laneHasActiveGeneration" in js
    assert "active_generation_in_flight" in js
    assert "Number(lane.active_generations || 0) > 0" in js
    # "Thinking" is a FOREGROUND generation. A background pass of her own
    # holds the lane without owning the foreground and is not her thinking
    # about the message in the composer.
    assert "function laneHasForegroundGeneration" in js
    assert "lane.foreground_owned === false) return false;" in js
    assert "if (laneHasForegroundGeneration(lane)) return 'thinking';" in js
    assert "CORTEX THINKING" in js
    assert "lane.conversation_ready === false && !laneHasActiveGeneration(lane)" in js
    assert "lane.conversation_ready === false && !laneHasActiveGeneration(lane)" in js


def test_legacy_shell_does_not_hide_lane_failures_as_generic_warming():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function laneFailureClass" in js
    assert "memory_pressure_refused_worker_spawn" in js
    assert "projected_process_tree_rss" in js
    assert "model_load_headroom" in js
    assert "visible_conversation_probe_missing" in js
    assert "endpoint_timeout" in js
    assert "if (failureClass === 'memory_guard') return 'memory_guard';" in js
    assert "if (failureClass === 'cognitive_engine') return 'route_blocked';" in js
    assert "if (failureClass === 'timeout') return 'timeout';" in js
    assert "CORTEX MEMORY GUARD" in js
    assert "CORTEX ROUTE BLOCKED" in js
    assert "CORTEX TIMEOUT" in js


def test_legacy_shell_matches_conversation_lane_timeout_budget():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function conversationLaneRequestTimeoutMs" in js
    assert "CHAT_REQUEST_TIMEOUT_READY_MS = 335000" in js
    assert "CHAT_REQUEST_TIMEOUT_RECOVERING_MS = 395000" in js
    assert "const requestTimeoutMs = conversationLaneRequestTimeoutMs(state.conversationLane);" in js
    assert "const requestTimeoutMs = 90000;" not in js


def test_legacy_shell_has_neural_feed_backpressure_controls():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "THOUGHT_QUEUE_MAX = 160" in js
    assert "function normalizeThoughtEvent" in js
    assert "function buildThoughtFingerprint" in js
    assert "function coalesceThoughtQueueItem" in js
    assert "state.thoughtQueue.splice(0, state.thoughtQueue.length - THOUGHT_QUEUE_MAX + 1);" in js


def test_legacy_shell_has_dark_boot_recovery_guard_before_external_assets():
    html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")

    critical_style = html.index('id="aura-critical-boot-style"')
    first_external_script = html.index('<script src="/static/error_banner.js">')
    assert critical_style < first_external_script
    assert "window.__auraLegacyShellGuardInstalled" in html
    assert "window.__auraShowHardRecovery" in html
    assert "legacy_shell_load_timeout" in html
    assert "/api/ui/shell-error" in html
    assert "/api/dashboard/snapshot" in html
    assert 'href="/logs"' not in html


def test_legacy_shell_marks_ready_only_after_full_script_install():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function markLegacyShellReady()" in js
    assert "window.__auraLegacyShellReady = true;" in js
    assert "document.body.dataset.auraShell = 'ready';" in js
    assert js.rfind("markLegacyShellReady();") > js.rfind("$('regen-btn')?.addEventListener('click', regenerateResponse);")


def test_legacy_shell_neural_feed_receives_health_liveness_pulses():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "lastNeuralPulseAt" in js
    assert "lastSemanticThoughtAt" in js
    assert "lastHealthWarningPulseAt" in js
    assert "NEURAL_LIVENESS_PULSE_MS = 30000" in js
    assert "HEALTH_POLL_REMINDER_MS = 5 * 60 * 1000" in js
    assert "function queueNeuralLivenessCard" in js
    assert "function publishHealthNeuralPulse" in js
    assert "const interval = strictHealthy ? NEURAL_LIVENESS_PULSE_MS : HEALTH_POLL_REMINDER_MS;" in js
    assert "warningReminderDue = !strictHealthy" in js
    assert "if (!strictHealthy) state.lastHealthWarningPulseAt = now;" in js
    assert "function conversationPayloadReady" in js
    assert "conversationPayloadReady(payload, blockers)" in js
    assert "lane.conversation_ready === true || payload.conversation_ready === true" not in js
    assert "publishHealthNeuralPulse(payload, 'websocket_heartbeat');" in js
    assert "publishHealthNeuralPulse(d, 'health_poll');" in js
    assert "function recordHealthPollFailure" in js
    assert "health endpoint unavailable; retaining last known state" in js
    assert "endpoint recovered after" in js


def test_lane_state_is_a_key_not_the_words_shown_to_the_user():
    """The tier badge used to recover the lane state by string-comparing the
    text already rendered in the header (`laneText === 'cortex thinking'`).
    Any rewording silently collapsed every state into CORTEX WARMING, so the
    words could not be improved without breaking the badge. State and
    presentation are separate now, and must stay that way."""
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function conversationLaneStateKey" in js
    assert "const LANE_STATES = {" in js
    assert "tierEl.textContent = laneWords.tier;" in js
    # No consumer may branch on the rendered label.
    assert "laneText ===" not in _code_only(js)


def test_shell_never_prints_raw_runtime_blocker_tokens_as_status():
    """`runtimeHealthStatusText` returned `blockers.slice(0, 2).join(', ')`,
    so the first line a newcomer read could be
    `RUNTIME_REQUIRED_PROBES, PROBE:KERNEL`. The tokens are still carried —
    tooltip and data-raw — but the visible line is a sentence."""
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")

    assert "blockers.slice(0, 2).join(', ')" not in _code_only(js)
    assert "window.AuraShellLexicon" in js
    assert "statusEl.dataset.raw = rawText" in js
    assert "function statusExplanation" in js
    assert '<script src="/static/shell_lexicon.js"></script>' in html
    # The lexicon must load before the shell that consumes it.
    assert html.index("shell_lexicon.js") < html.index("/static/aura.js")


def test_hard_recovery_panel_leads_with_a_sentence_not_a_token():
    """The recovery panel is what a person sees when the window has already
    failed — the worst moment to hand them `legacy_shell_load_timeout` as the
    first line. It gets a sentence first and keeps the reason verbatim in the
    <pre> below.

    The table is inline on purpose: this guard has to work when aura.js and
    shell_lexicon.js never loaded, so it cannot depend on either."""
    html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")

    guard = _code_only(
        html[html.index("window.__auraLegacyShellGuardInstalled") : html.index("</script>")]
    )
    assert "SAID_PLAINLY" in guard
    assert "shell_lexicon" not in guard, "the boot guard must not depend on a later script"
    assert "AuraShellLexicon" not in guard

    # Every reason showRecovery can raise needs an entry.
    for reason in (
        "legacy_shell_load_timeout",
        "legacy_shell_runtime_error",
        "legacy_shell_unhandled_rejection",
        "legacy_shell_fault",
    ):
        assert f'showRecovery("{reason}"' in html or f"{reason}:" in html, reason
        assert f"{reason}:" in html, f"SAID_PLAINLY missing {reason}"

    # The raw reason and detail still reach the panel unchanged.
    assert 'root.querySelector("pre").textContent = safeReason + "\\n" + safeDetail;' in html
    assert 'root.querySelector(".aura-hard-recovery-said").textContent = plainly;' in html


def test_shell_lexicon_translates_every_blocker_the_shell_can_raise():
    """A blocker the shell can push must have an entry, or the lexicon is
    decorative and the raw token reaches the user anyway."""
    lex = (PROJECT_ROOT / "interface" / "static" / "shell_lexicon.js").read_text(encoding="utf-8")
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    for token in (
        "runtime_required_probes",
        "runtime_health_unavailable",
        "runtime_transport_only",
        "conversation_transport",
    ):
        assert f"blockers.push('{token}')" in js or f"['{token}']" in js, token
        assert token in lex, f"lexicon missing {token}"

    # Every probe group the shell requires needs a description.
    assert "REQUIRED_RUNTIME_PROBES = ['kernel', 'inference', 'memory', 'scheduler', 'tool_governance']" in js
    for group in ("kernel", "inference", "memory", "scheduler", "tool_governance"):
        assert f"'probe:{group}'" in lex, f"lexicon missing probe:{group}"

    # Unknown tokens must still degrade to a sentence, never to a bare slug.
    assert "function fallback" in lex
    assert "does not have a description for" in lex


def test_server_keeps_legacy_shell_as_default_route():
    server = (PROJECT_ROOT / "interface" / "server.py").read_text(encoding="utf-8")

    assert 'ui = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")' in server
    assert 'fallback = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")' in server
    assert '"shell": "legacy_shell" if LEGACY_UI_INDEX.exists() else "react_shell"' in server


def test_react_shell_is_opt_in_so_original_hud_stays_canonical():
    server = (PROJECT_ROOT / "interface" / "server.py").read_text(encoding="utf-8")
    system = (PROJECT_ROOT / "interface" / "routes" / "system.py").read_text(encoding="utf-8")

    assert "AURA_ENABLE_REACT_SHELL" in server
    assert "if LEGACY_UI_INDEX.exists() and not _react_shell_enabled():" in server
    assert '"canonical_shell": "legacy_shell"' in server
    assert "experimental_shell_enabled" in system
