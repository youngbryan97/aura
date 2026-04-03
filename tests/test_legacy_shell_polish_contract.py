from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def test_legacy_shell_presents_cold_standby_as_ready_shell_state():
    js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "function laneIsStandby" in js
    assert "cortex on standby" in js
    assert "CORTEX ON STANDBY" in js
    assert "Aura is ready. Cortex will warm on first turn." in js
    assert "syncSplashState(payload);" in js
    assert "Live shell is still syncing. Aura is stabilizing background channels..." in js
    assert "setTimeout(() => dismissSplash(), 8000)" not in js


def test_server_keeps_legacy_shell_as_default_route():
    server = (PROJECT_ROOT / "interface" / "server.py").read_text(encoding="utf-8")

    assert 'ui = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")' in server
    assert 'fallback = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")' in server
    assert '"shell": "legacy_shell" if LEGACY_UI_INDEX.exists() else "react_shell"' in server
