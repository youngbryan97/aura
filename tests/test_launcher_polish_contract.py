from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_launcher_exposes_desktop_window_action_and_dock_presence():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    assert "Open Aura" in swift
    assert "openDesktopWindow" in swift
    assert 'app.setActivationPolicy(.regular)' in swift
    assert "requestUserAttention" in swift
    assert '--open-gui-window' in swift
    assert "replacementReason(expectedSemver:" in swift
    assert 'split(separator: "-", maxSplits: 1)' in swift
    assert "autoOpenDesktopWindowIfNeeded" in swift
    assert "Aura is awake" in swift
    assert "let progress = snapshot.launcherReady ? 100.0 : snapshot.progress" in swift
    assert "bootMarkerIsStaleWithoutRuntime" in swift
    assert 'lockDirectory.appendingPathComponent("orchestrator.lock")' in swift
    assert "normalizedDirectCLIArguments" in swift
    assert 'case "--open-gui-window":' in swift
    assert 'return "--gui-window"' in swift
    assert 'auraMainScript = auraRoot.appendingPathComponent("aura_main.py")' in swift
    assert '["-u", auraMainScript.path, "--desktop"]' in swift
    assert "requiresProtectedFolderFallback" in swift
    assert 'desktop-terminal-launch.command' in swift
    assert 'desktop-terminal-launch.marker' in swift
    assert "terminalHandoffIsFresh" in swift
    assert "terminalHandoffIsStaleWithoutRuntime" in swift
    assert "age >= staleMarkerWithoutRuntimeWindow" in swift
    assert "AURA_SAFE_BOOT_DESKTOP" in swift
    assert "AURA_EAGER_CORTEX_WARMUP" in swift
    assert "AURA_DEFERRED_CORTEX_PREWARM" in swift
    assert "AURA_SAFE_BOOT_METAL_CACHE_RATIO" in swift
    assert "AURA_SAFE_BOOT_METAL_CACHE_CAP_GB" in swift
    assert "spawnDetachedViaShell" in swift
    assert "spawnAuraSubprocess(arguments:" in swift
    assert 'proc.executableURL = URL(fileURLWithPath: "/bin/bash")' in swift
    assert 'proc.arguments = [launchScript.path] + arguments' in swift
    assert "Force Stop" in swift
    assert "progressBelowBadge" in swift
    assert "progressBelowIcon" in swift
    assert "forceStopAura" in swift
    assert "guard let window else" in swift


def test_launch_script_supports_gui_window_mode():
    shell = (PROJECT_ROOT / "launch_aura.sh").read_text(encoding="utf-8")

    assert "--open-gui-window|--gui-window" in shell
    assert "aura_main.py --gui-window" in shell
    assert "AURA_CLEANUP_RECENT_GRACE_S:=45" in shell
    assert 'cd -P "$(dirname "$0")"' in shell
    assert "AURA_EAGER_CORTEX_WARMUP" in shell
    assert "AURA_DEFERRED_CORTEX_PREWARM" in shell
    assert "AURA_ENABLE_PERMANENT_SWARM:=0" in shell
    assert "AURA_SAFE_BOOT_METAL_CACHE_RATIO:=0.56" in shell
    assert "AURA_SAFE_BOOT_METAL_CACHE_CAP_GB:=36" in shell
    assert "resolve_launch_log()" in shell
    assert "ACTIVE_LAUNCH_LOG" in shell
    assert "aura-desktop-launch.log" in shell


def test_launcher_cleanup_shim_exists_at_repo_root():
    shim = PROJECT_ROOT / "aura_cleanup.py"
    contents = shim.read_text(encoding="utf-8")

    assert shim.exists()
    assert 'scripts" / "one_off" / "aura_cleanup.py"' in contents


def test_aura_main_supports_gui_window_mode():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert '--gui-window' in main_py
    assert "gui_actor_entry(args.port)" in main_py


def test_bundle_script_builds_regular_dock_app_and_embeds_version_metadata():
    bundle_script = (PROJECT_ROOT / "scripts" / "bundle_app.sh").read_text(encoding="utf-8")

    assert 'VERSION_FILE="${RESOURCES_DIR}/aura-version"' in bundle_script
    assert 'ROOT_DIR="$(cd -P "$(dirname "$0")/.." && pwd -P)"' in bundle_script
    assert 'VERSION_FULL_FILE="${RESOURCES_DIR}/aura-version-full"' in bundle_script
    assert 'INSTALL_PATH="${AURA_INSTALL_PATH:-}"' in bundle_script
    assert 'cp -R "${APP_DIR}" "${INSTALL_PATH}"' in bundle_script
    assert 'codesign --force --sign - "${APP_DIR}"' in bundle_script
    assert 'codesign --force --sign - "${INSTALL_PATH}"' in bundle_script
    assert "CFBundleShortVersionString" in bundle_script
    assert "LSUIElement" not in bundle_script


def test_live_shell_assets_are_unversioned_and_service_worker_skips_shell_cache():
    index_html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")
    sw = (PROJECT_ROOT / "interface" / "static" / "service-worker.js").read_text(encoding="utf-8")
    ui_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
    ui_css = (PROJECT_ROOT / "interface" / "static" / "aura.css").read_text(encoding="utf-8")

    assert '/static/aura.css"' in index_html
    assert '/static/aura.js"' in index_html
    assert '/static/manifest.json"' in index_html
    assert 'metric-guide-toggle' in index_html
    assert 'metric-guide-panel' in index_html
    assert "What it means for Aura" in index_html
    assert "LIVE_SHELL_PATHS" in sw
    assert "SKIP_WAITING" in sw
    assert "updateViaCache: 'none'" in ui_js
    assert "const METRIC_GUIDE =" in ui_js
    assert "findNearestMetricGuideSectionKey" in ui_js
    assert "SECTION_GUIDE_BY_LABEL" in ui_js
    assert "rolling-summary" in ui_js
    assert "executive_authority" in ui_js
    assert "initializeMetricGuide()" in ui_js
    assert ".metric-guide-panel" in ui_css
    assert ".metric-guide-live" in ui_css
