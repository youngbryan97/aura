from pathlib import Path
from types import SimpleNamespace

from core.runtime.resource_observation import ProcessObservation

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _launcher_observation(resource_observer, *, pid: int, create_time: float):
    executable = "/Applications/Aura.app/Contents/MacOS/aura-launcher"
    return ProcessObservation(
        provenance=resource_observer.provenance,
        pid=pid,
        ppid=1,
        create_time=create_time,
        status="running",
        name="aura-launcher",
        cmdline=(executable,),
        rss_bytes=1024,
        exe=executable,
    )


def _process_observation(
    resource_observer,
    *,
    pid: int,
    ppid: int,
    create_time: float,
    cmdline: tuple[str, ...],
    cwd: str,
    ancestor_pids: tuple[int, ...] = (),
):
    return ProcessObservation(
        provenance=resource_observer.provenance,
        pid=pid,
        ppid=ppid,
        create_time=create_time,
        status="running",
        name=Path(cmdline[0]).name,
        cmdline=cmdline,
        rss_bytes=1024,
        ancestor_pids=ancestor_pids,
        exe=cmdline[0],
        cwd=cwd,
    )


def test_launcher_exposes_desktop_window_action_and_dock_presence():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    assert "Open Aura" in swift
    assert "openDesktopWindow" in swift
    assert 'app.setActivationPolicy(.regular)' in swift
    assert "requestUserAttention" in swift
    assert "import WebKit" in swift
    assert "WKWebView" in swift
    assert "openNativeDesktopWindow" in swift
    assert "surface=native-app" in swift
    assert "desktopWindowIsVisible" in swift
    assert "claimAppInstanceLock" in swift
    assert 'desktop-app-instance.lock' in swift
    assert "activateExistingLauncherInstance" in swift
    assert "NSRunningApplication.runningApplications(withBundleIdentifier:" in swift
    assert "NSApp.terminate(nil)" in swift
    assert "releaseAppInstanceLock" in swift
    assert 'env["AURA_NATIVE_BRIDGE_PID"] = String(ProcessInfo.processInfo.processIdentifier)' in swift
    assert '--open-gui-window' in swift
    assert "replacementReason(expectedSemver:" in swift
    assert "launcherReady || systemReady || conversationOperational" in swift
    assert "recovery owns post-handoff failures" in swift
    assert "if !forceRelaunch && self.existingRuntimeIsObservable()" in swift
    assert "never spawn a second" in swift
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
    assert 'aura-launch-provenance.json' in swift
    assert 'AURA_LAUNCH_EXPECTED_COMMIT' in swift
    assert 'AURA_LAUNCH_EXPECTED_WORKSPACE_SHA256' in swift
    assert 'requiredProvenanceString("shell_assets_sha256")' in swift
    assert "shellAssetsSHA256.count == 64" in swift
    assert 'bundleIdentifier == "com.aura.desktop"' in swift
    assert '["-u", auraMainScript.path, "--desktop"]' in swift
    assert "requiresProtectedFolderFallback" in swift
    assert 'desktop-terminal-launch.command' in swift
    assert 'desktop-terminal-launch.marker' in swift
    assert 'desktop-gui-window.marker' in swift
    assert "desktopWindowLaunchInProgress" in swift
    assert "guiWindowHelperIsRunning" in swift
    assert "markGuiWindowLaunch" in swift
    assert "clearGuiWindowLaunchMarker" in swift
    assert "single-flight" in swift
    assert "terminalHandoffIsFresh" in swift
    assert "terminalHandoffIsStaleWithoutRuntime" in swift
    assert "age >= staleMarkerWithoutRuntimeWindow" in swift
    assert "AURA_LOCAL_BACKEND" in swift
    assert 'env["AURA_LOCAL_BACKEND"] = "mlx"' in swift
    assert "AURA_DESKTOP_RESOURCE_GUARD" in swift
    assert 'env["AURA_DESKTOP_RESOURCE_GUARD"] = "1"' in swift
    assert 'env["AURA_SAFE_BOOT_DESKTOP"]' not in swift
    assert 'env.removeValue(forKey: "AURA_SAFE_BOOT_DESKTOP")' in swift
    assert 'env.removeValue(forKey: "AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR")' in swift
    assert 'env["AURA_ENABLE_BACKGROUND_COGNITION"] = "1"' in swift
    assert 'env["AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM"] = "1"' in swift
    assert 'env["AURA_ALLOW_RUNTIME_SELF_MODIFICATION"] = "1"' in swift
    assert 'env["AURA_ALLOW_AUTONOMOUS_PATCH_PROMOTION"] = "1"' in swift
    assert 'env["AURA_ALLOW_REPAIR_LAB_SOURCE_PROMOTION"] = "1"' in swift
    assert 'env["HF_HUB_OFFLINE"] = "1"' in swift
    assert 'env["HF_HUB_DISABLE_TELEMETRY"] = "1"' in swift
    assert 'env["TRANSFORMERS_OFFLINE"] = "1"' in swift
    assert 'env["AURA_BACKGROUND_BOOT_GRACE_S"] = "60"' in swift
    assert 'env["AURA_EAGER_LOCAL_SENSORY_BOOT"] = "1"' in swift
    assert 'env.removeValue(forKey: "AURA_AUTO_LISTEN")' in swift
    assert 'env["AURA_AUTO_LISTEN"] = "1"' not in swift
    assert "AURA_EAGER_CORTEX_WARMUP" in swift
    assert "AURA_DEFERRED_CORTEX_PREWARM" in swift
    assert "export AURA_BACKGROUND_BOOT_GRACE_S=60" in swift
    assert "AURA_DESKTOP_METAL_CACHE_RATIO" in swift
    assert "AURA_DESKTOP_METAL_CACHE_CAP_GB" in swift
    assert "AURA_DESKTOP_MLX_MEMORY_RATIO" in swift
    assert "AURA_DESKTOP_MLX_MEMORY_CAP_GB" in swift
    assert "AURA_DESKTOP_MLX_MEMORY_FLOOR_GB" in swift
    assert "AURA_DESKTOP_PROCESS_RSS_RATIO" in swift
    assert "AURA_DESKTOP_PROCESS_RSS_CAP_GB" in swift
    assert "AURA_PROCESS_RSS_LIMIT_GB" in swift
    assert "AURA_LOCAL_RUNTIME_SINGLETON" in swift
    assert "AURA_LOCAL_PARALLEL_SLOTS" in swift
    assert "AURA_ENABLE_LOCAL_DEEP_SOLVER" in swift
    assert "AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB" in swift
    assert "AURA_MLX_32B_PROCESS_RESERVE_GB" in swift
    assert "AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB" in swift
    assert "AURA_MLX_72B_PROCESS_RESERVE_GB" in swift
    assert "AURA_FOREGROUND_CHAT_MAX_TOKENS" in swift
    assert 'env["AURA_DESKTOP_METAL_CACHE_RATIO"] = "0.16"' in swift
    assert 'env["AURA_DESKTOP_METAL_CACHE_CAP_GB"] = "10"' in swift
    assert 'env["AURA_DESKTOP_MLX_MEMORY_RATIO"] = "0.54"' in swift
    assert 'env["AURA_DESKTOP_MLX_MEMORY_CAP_GB"] = "34"' in swift
    assert 'env["AURA_DESKTOP_MLX_MEMORY_FLOOR_GB"] = "18"' in swift
    assert 'env["AURA_PROCESS_RSS_LIMIT_GB"] = "40"' in swift
    assert 'env["AURA_MEMWATCH_LETHAL_MB"] = "43008"' in swift
    assert 'env["AURA_MEMORY_SENTINEL_INTERVAL_S"] = "0.5"' in swift
    assert 'env["AURA_GOVERNOR_PRUNE_MB"] = "37888"' in swift
    assert 'env["AURA_GOVERNOR_UNLOAD_MB"] = "39936"' in swift
    assert 'env["AURA_GOVERNOR_CRITICAL_MB"] = "41984"' in swift
    assert 'env["AURA_FOREGROUND_CHAT_MAX_TOKENS"] = "2048"' in swift
    assert 'env["AURA_EAGER_CORTEX_WARMUP"] = "0"' in swift
    assert 'env["AURA_DEFERRED_CORTEX_PREWARM"] = "1"' in swift
    assert 'env["AURA_LOCAL_RUNTIME_SINGLETON"] = "1"' in swift
    assert 'env["AURA_LOCAL_PARALLEL_SLOTS"] = "1"' in swift
    assert 'env["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "0"' in swift
    assert 'env["AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB"] = "24"' in swift
    assert 'env["AURA_MLX_32B_PROJECTED_FOOTPRINT_GB"] = "auto"' in swift
    assert 'env["AURA_MLX_32B_PROCESS_RESERVE_GB"] = "3"' in swift
    assert 'env["AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB"] = "52"' in swift
    assert 'env["AURA_MLX_72B_PROJECTED_FOOTPRINT_GB"] = "auto"' in swift
    assert 'env["AURA_MLX_72B_PROCESS_RESERVE_GB"] = "5"' in swift
    assert "AURA_EXTERNAL_GUI_OWNER" in swift
    assert "export AURA_EXTERNAL_GUI_OWNER=1" in swift
    assert "spawnDetachedViaShell" in swift
    assert "spawnAuraSubprocess(arguments:" in swift
    assert 'proc.executableURL = URL(fileURLWithPath: "/bin/bash")' in swift
    assert 'proc.arguments = [launchScript.path] + arguments' in swift
    assert "Force Stop" in swift
    assert "progressBelowBadge" in swift
    assert "progressBelowIcon" in swift
    assert "forceStopAura" in swift
    assert "guard let target = desktopWindow ?? window else" in swift
    assert "nativeDesktopBridgeCommandRequiresMainThread" in swift
    assert 'command == "request_screen" || command == "request_accessibility"' in swift
    assert "DispatchQueue.main.sync" in swift
    assert "bridgeActivateForPermissionPrompt()" in swift
    assert "NSRunningApplication.current.activate" in swift


def test_launcher_quit_reaps_every_direct_runtime_child_with_grace_period():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")
    termination_hook = swift.split("func applicationWillTerminate", 1)[1].split(
        "private func configurePaths",
        1,
    )[0]

    assert "terminateSpawnedProcesses()" in termination_hook
    assert "if explicitStopInProgress" not in termination_hook
    assert "AURA_LAUNCHER_CHILD_TERMINATION_GRACE_S" in swift
    assert "?? 18.0" in swift
    assert "kill(proc.processIdentifier, SIGKILL)" in swift


def test_launch_script_supports_gui_window_mode():
    shell = (PROJECT_ROOT / "launch_aura.sh").read_text(encoding="utf-8")

    assert "--open-gui-window|--gui-window" in shell
    assert "aura_main.py --gui-window" in shell
    assert "AURA_CLEANUP_RECENT_GRACE_S:=45" in shell
    assert 'cd -P "$(dirname "$0")"' in shell
    assert "AURA_EAGER_CORTEX_WARMUP" in shell
    assert "AURA_DEFERRED_CORTEX_PREWARM" in shell
    assert "export AURA_LOCAL_BACKEND=mlx" in shell
    assert "AURA_BACKGROUND_BOOT_GRACE_S:=60" in shell
    assert "export AURA_BACKGROUND_BOOT_GRACE_S" in shell
    assert "AURA_ENABLE_PERMANENT_SWARM:=0" in shell
    assert "AURA_EXTERNAL_GUI_OWNER:=1" in shell
    assert "AURA_DESKTOP_METAL_CACHE_RATIO:=0.16" in shell
    assert "AURA_DESKTOP_METAL_CACHE_CAP_GB:=10" in shell
    assert "AURA_DESKTOP_MLX_MEMORY_RATIO:=0.54" in shell
    assert "AURA_DESKTOP_MLX_MEMORY_CAP_GB:=34" in shell
    assert "AURA_DESKTOP_MLX_MEMORY_FLOOR_GB:=18" in shell
    assert "AURA_PROCESS_RSS_LIMIT_GB:=40" in shell
    assert "AURA_MEMWATCH_LETHAL_MB:=43008" in shell
    assert "AURA_MEMORY_SENTINEL_INTERVAL_S:=0.5" in shell
    assert "AURA_GOVERNOR_PRUNE_MB:=37888" in shell
    assert "AURA_GOVERNOR_UNLOAD_MB:=39936" in shell
    assert "AURA_GOVERNOR_CRITICAL_MB:=41984" in shell
    assert "AURA_LOCAL_RUNTIME_SINGLETON:=1" in shell
    assert "AURA_LOCAL_PARALLEL_SLOTS:=1" in shell
    assert "AURA_ENABLE_LOCAL_DEEP_SOLVER:=0" in shell
    assert "AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB:=24" in shell
    assert "AURA_MLX_32B_PROJECTED_FOOTPRINT_GB:=auto" in shell
    assert "AURA_MLX_32B_PROCESS_RESERVE_GB:=3" in shell
    assert "AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB:=52" in shell
    assert "AURA_MLX_72B_PROJECTED_FOOTPRINT_GB:=auto" in shell
    assert "AURA_MLX_72B_PROCESS_RESERVE_GB:=5" in shell
    assert "AURA_FOREGROUND_CHAT_MAX_TOKENS:=2048" in shell
    assert "export AURA_ENABLE_LOCAL_DEEP_SOLVER" in shell
    assert "unset AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR" in shell
    assert "AURA_DESKTOP_FORCE_DISABLE_SECONDARY_MODEL_REPAIR" in shell
    assert "resolve_launch_log()" in shell
    assert "ACTIVE_LAUNCH_LOG" in shell
    assert "aura-desktop-launch.log" in shell
    assert "cleanup_attached_launcher()" in shell
    assert "trap 'cleanup_attached_launcher 130' INT" in shell
    assert '"$PYTHON_CMD" aura_cleanup.py >/dev/null 2>&1 || true' in shell


def test_bundle_app_prefers_stable_local_codesign_without_timestamp_by_default():
    bundle = (PROJECT_ROOT / "scripts" / "bundle_app.sh").read_text(encoding="utf-8")

    assert 'AURA_AUTO_USE_LOCAL_CODESIGN="${AURA_AUTO_USE_LOCAL_CODESIGN:-1}"' in bundle
    assert 'sed -n \'s/.*"\\(Aura Local Code Signing[^"]*\\)".*/\\1/p\'' in bundle
    assert 'CODESIGN_ARGS+=(--options runtime)' in bundle
    assert 'if [ "${AURA_CODESIGN_TIMESTAMP:-0}" = "1" ]; then' in bundle
    assert 'CODESIGN_ARGS+=(--timestamp)' in bundle


def test_installed_launcher_binds_the_repository_python_across_worktrees():
    bundle = (PROJECT_ROOT / "scripts" / "bundle_app.sh").read_text(encoding="utf-8")
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    assert 'PYTHON_RUNTIME_FILE="${RESOURCES_DIR}/aura-python-path"' in bundle
    assert 'ENV_PATH_FILE="${RESOURCES_DIR}/aura-env-path"' in bundle
    assert 'MODELS_PATH_FILE="${RESOURCES_DIR}/aura-models-path"' in bundle
    assert 'FUSED_MODEL_PATH_FILE="${RESOURCES_DIR}/aura-fused-model-path"' in bundle
    assert "git rev-parse --path-format=absolute --git-common-dir" in bundle
    assert '"${PRIMARY_ROOT}/.venv/bin/python3"' in bundle
    assert "import sys, httpx" in bundle
    assert 'printf \'%s\\n\' "${PYTHON_RUNTIME}" > "${PYTHON_RUNTIME_FILE}"' in bundle
    assert 'printf \'%s\\n\' "${MODELS_ROOT}" > "${MODELS_PATH_FILE}"' in bundle
    assert 'printf \'%s\\n\' "${FUSED_MODEL_ROOT}" > "${FUSED_MODEL_PATH_FILE}"' in bundle
    assert '"${PRIMARY_ROOT:+${PRIMARY_ROOT}/.env}"' in bundle
    assert "resolvePythonExecutable(resourcesURL: resourcesURL)" in swift
    resolver = swift.split(
        "private func resolvePythonExecutable(resourcesURL: URL) throws -> URL {",
        1,
    )[1].split("private func baseAuraEnvironment()", 1)[0]
    assert 'appendingPathComponent("aura-python-path")' in resolver
    assert ".resolvingSymlinksInPath()" not in resolver
    assert "/opt/homebrew/bin/python3.12" not in resolver
    assert "/usr/local/bin/python3.12" not in resolver
    assert 'appendingPathComponent("aura-env-path")' in swift
    assert 'launchProvenanceEnvironment["AURA_ENV_FILE"] = envURL.path' in swift
    assert 'appendingPathComponent("aura-models-path")' in swift
    assert 'launchProvenanceEnvironment["AURA_MODELS_DIR"] = modelsURL.path' in swift
    assert 'appendingPathComponent("aura-fused-model-path")' in swift
    assert 'launchProvenanceEnvironment["AURA_FUSED_MODEL_ROOT"] = fusedModelURL.path' in swift


def test_runtime_loads_the_signed_external_environment_before_module_boot():
    main = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    early_loader = main.split("# Early .env loading", 1)[1].split(
        "# 1. Path Resolution",
        1,
    )[0]

    assert 'os.environ.get("AURA_ENV_FILE", "")' in early_loader
    assert "Path(_env_override).expanduser().absolute()" in early_loader
    assert "_env_path.is_file()" in early_loader
    assert "_load_dotenv(_env_path, override=False)" in early_loader


def test_launcher_cleanup_shim_exists_at_repo_root():
    shim = PROJECT_ROOT / "aura_cleanup.py"
    target = PROJECT_ROOT / "scripts" / "one_off" / "aura_cleanup.py"
    contents = shim.read_text(encoding="utf-8")

    assert shim.exists()
    assert target.exists()
    assert 'scripts" / "one_off" / "aura_cleanup.py"' in contents


def test_cleanup_preserves_verified_live_runtime_unless_forced():
    cleanup = (PROJECT_ROOT / "scripts" / "one_off" / "aura_cleanup.py").read_text(
        encoding="utf-8"
    )

    assert "def _verified_live_runtime_pid()" in cleanup
    assert 'AURA_CLEANUP_FORCE' in cleanup
    assert "skipping aggressive pre-launch process cleanup" in cleanup
    assert "preserving lock directory" in cleanup
    assert "def _kill_stale_native_launchers()" in cleanup
    assert "Aura.app/Contents/MacOS/aura-launcher" in cleanup
    assert "pid in {current_pid, parent_pid}" in cleanup
    assert "preserving native Aura.app launcher bridge" in cleanup
    assert "AURA_CLEANUP_RECENT_GRACE_S" in cleanup
    assert "Preserving recent Aura native launcher" in cleanup
    assert "select_script_process_tree" in cleanup
    assert "pkill" not in cleanup


def test_cleanup_recognizes_native_launcher_process():
    from scripts.one_off import aura_cleanup

    launcher = SimpleNamespace(
        info={
            "exe": "/Applications/Aura.app/Contents/MacOS/aura-launcher",
            "cmdline": ["/Applications/Aura.app/Contents/MacOS/aura-launcher"],
            "name": "aura-launcher",
        },
    )
    unrelated = SimpleNamespace(
        info={
            "exe": "/Applications/Notes.app/Contents/MacOS/Notes",
            "cmdline": ["/Applications/Notes.app/Contents/MacOS/Notes"],
            "name": "Notes",
        },
    )

    assert aura_cleanup._is_native_launcher_process(launcher) is True
    assert aura_cleanup._is_native_launcher_process(unrelated) is False


def test_cleanup_preserves_native_launcher_when_live_runtime_verified(monkeypatch):
    from scripts.one_off import aura_cleanup

    monkeypatch.setattr(aura_cleanup, "_verified_live_runtime_pid", lambda: 1234)

    aura_cleanup._kill_stale_native_launchers()


def test_cleanup_preserves_recent_native_launcher_without_force(
    monkeypatch,
    resource_observer,
):
    import time

    from scripts.one_off import aura_cleanup

    resource_observer.configure_processes(
        [
            _launcher_observation(
                resource_observer,
                pid=4321,
                create_time=time.time() - 2.0,
            )
        ]
    )
    monkeypatch.setattr(aura_cleanup, "_verified_live_runtime_pid", lambda: None)
    monkeypatch.setenv("AURA_CLEANUP_RECENT_GRACE_S", "45")
    monkeypatch.delenv("AURA_CLEANUP_FORCE", raising=False)

    aura_cleanup._kill_stale_native_launchers()

    assert resource_observer.process(4321) is not None


def test_cleanup_terminates_old_native_launcher(monkeypatch, resource_observer):
    import sys
    import time

    from scripts.one_off import aura_cleanup

    terminated: list[int] = []

    class FakeProc:
        def __init__(self, pid: int):
            self.pid = pid

        def terminate(self):
            terminated.append(self.pid)

        def wait(self, timeout):
            return 0

    class FakePsutil:
        Error = Exception
        TimeoutExpired = TimeoutError

        @staticmethod
        def Process(pid):  # noqa: N802 - psutil-compatible test action handle.
            return FakeProc(pid)

    resource_observer.configure_processes(
        [
            _launcher_observation(
                resource_observer,
                pid=4321,
                create_time=time.time() - 120.0,
            )
        ]
    )
    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)
    monkeypatch.setattr(aura_cleanup, "_verified_live_runtime_pid", lambda: None)
    monkeypatch.setenv("AURA_CLEANUP_RECENT_GRACE_S", "45")
    monkeypatch.delenv("AURA_CLEANUP_FORCE", raising=False)

    aura_cleanup._kill_stale_native_launchers()

    assert terminated == [4321]


def test_cleanup_treats_missing_lock_pid_as_stale(monkeypatch):
    from scripts.one_off import aura_cleanup
    monkeypatch.setattr(aura_cleanup, "read_instance_lock_pid", lambda _name: 999999)

    assert aura_cleanup._verified_live_runtime_pid() is None


def test_cleanup_targets_exact_runtime_tree_without_matching_shell_text(
    monkeypatch,
    resource_observer,
):
    import sys
    import time

    from scripts.one_off import aura_cleanup

    created_at = time.time() - 120.0
    terminated: list[int] = []

    class FakeProc:
        def __init__(self, pid: int):
            self.pid = pid

        def create_time(self):
            return created_at

        def terminate(self):
            terminated.append(self.pid)

        def kill(self):
            raise AssertionError("responsive exact runtime should not require SIGKILL")

    class FakePsutil:
        Error = Exception

        @staticmethod
        def Process(pid):  # noqa: N802 - psutil-compatible test action handle.
            return FakeProc(pid)

        @staticmethod
        def wait_procs(processes, timeout):
            del timeout
            return list(processes), []

    runtime = _process_observation(
        resource_observer,
        pid=4322,
        ppid=111,
        create_time=created_at,
        cmdline=(sys.executable, str(PROJECT_ROOT / "aura_main.py"), "--desktop"),
        cwd=str(PROJECT_ROOT),
    )
    worker = _process_observation(
        resource_observer,
        pid=4323,
        ppid=4322,
        create_time=created_at,
        cmdline=(sys.executable, "-c", "from multiprocessing.spawn import spawn_main"),
        cwd=str(PROJECT_ROOT),
        ancestor_pids=(4322, 111),
    )
    shell_probe = _process_observation(
        resource_observer,
        pid=4324,
        ppid=111,
        create_time=created_at,
        cmdline=(
            "/bin/zsh",
            "-lc",
            "pgrep -fl 'aura_main.py --desktop' | head -2",
        ),
        cwd=str(PROJECT_ROOT),
    )
    resource_observer.configure_processes([runtime, worker, shell_probe])
    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)
    monkeypatch.setattr(aura_cleanup, "_verified_live_runtime_pid", lambda: None)

    aura_cleanup._kill_stale_processes()

    assert terminated == [4322, 4323]


def test_cleanup_revalidates_process_identity_before_signaling(monkeypatch, resource_observer):
    import sys
    import time

    from scripts.one_off import aura_cleanup

    original_created_at = time.time() - 120.0
    reused_created_at = original_created_at + 60.0
    runtime = _process_observation(
        resource_observer,
        pid=5322,
        ppid=111,
        create_time=original_created_at,
        cmdline=(sys.executable, str(PROJECT_ROOT / "aura_main.py"), "--desktop"),
        cwd=str(PROJECT_ROOT),
    )
    reused = _process_observation(
        resource_observer,
        pid=5322,
        ppid=111,
        create_time=reused_created_at,
        cmdline=(sys.executable, str(PROJECT_ROOT / "aura_main.py"), "--desktop"),
        cwd=str(PROJECT_ROOT),
    )

    class ChangingObserver:
        def process_table(self):
            return SimpleNamespace(available=True, error="", processes=(runtime,))

        def process(self, pid):
            assert pid == 5322
            return reused

    class ForbiddenPsutil:
        class Error(Exception):
            pass

        @staticmethod
        def Process(_pid):  # noqa: N802 - psutil-compatible test action handle.
            raise AssertionError("a reused PID must never become a signal handle")

        @staticmethod
        def wait_procs(processes, timeout):
            del timeout
            assert processes == []
            return [], []

    monkeypatch.setattr(aura_cleanup, "get_resource_observer", ChangingObserver)
    monkeypatch.setattr(aura_cleanup, "_verified_live_runtime_pid", lambda: None)
    monkeypatch.setitem(sys.modules, "psutil", ForbiddenPsutil)

    aura_cleanup._kill_stale_processes()


def test_aura_main_supports_gui_window_mode():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert '--gui-window' in main_py
    assert "gui_actor_entry(args.port)" in main_py
    assert 'acquire_instance_lock(lock_name="desktop_gui_window")' in main_py
    assert '"--gui-window",' in main_py
    assert '"--watchdog",' in main_py
    assert "helper_modes" in main_py
    assert 'AURA_EXTERNAL_GUI_OWNER' in main_py
    assert 'AURA_LAUNCHED_FROM_APP' in main_py
    assert "launch_gui=None" in main_py


def test_gui_window_process_is_dockless_single_visible_aura_app():
    gui_actor = (PROJECT_ROOT / "interface" / "gui_actor.py").read_text(encoding="utf-8")

    assert "NSApplication.sharedApplication()" in gui_actor
    assert "setActivationPolicy_(1)" in gui_actor
    assert "NSApplicationActivationPolicyAccessory" in gui_actor
    assert "continued GUI boot after dockless activation policy setup failed" in gui_actor


def test_full_runtime_status_exposes_background_cognition():
    system_route = (PROJECT_ROOT / "interface" / "routes" / "system.py").read_text(
        encoding="utf-8"
    )
    index_html = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    ui_js = (PROJECT_ROOT / "interface" / "static" / "aura.js").read_text(
        encoding="utf-8"
    )

    assert "background_cognition" in system_route
    assert "subjective_choice" in system_route
    assert "ambient_life_director" in system_route
    assert "background_activity_reason(" in system_route
    assert "background_loop_start_reason(" in system_route
    assert "running_required_count" in system_route
    assert "registered_required_count" in system_route
    assert 'id="fr-background"' in index_html
    assert "protected_full_desktop" in ui_js
    assert "Background cognition live:" in ui_js
    assert "work_defer_reason" in ui_js


def test_boot_sensory_services_do_not_escalate_optional_io_to_fail_closed():
    boot_sensory = (
        PROJECT_ROOT / "core" / "orchestrator" / "mixins" / "boot" / "boot_sensory.py"
    ).read_text(encoding="utf-8")

    helper = boot_sensory.split("def _register_sensory_service", 1)[1].split(
        "async def _maybe_await",
        1,
    )[0]
    assert "required: bool = False" in helper
    assert 'failure_policy: str = "degrade_with_receipt"' in helper
    assert "required=required" in helper
    assert "failure_policy=failure_policy" in helper


def test_desktop_api_server_bounds_uvicorn_connection_drain_on_shutdown():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert "AURA_UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_S" in main_py
    assert "timeout_graceful_shutdown=graceful_shutdown_s" in main_py
    # Zenflow e7c2256b deliberately raised the default drain 2s→8s (float-tolerant).
    assert 'os.environ.get("AURA_UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_S", "8")' in main_py


def test_packaged_launcher_parses_structured_runtime_lock():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    assert "private func parseRuntimeLockPID" in swift
    assert "JSONSerialization.jsonObject" in swift
    assert 'payload["pid"]' in swift
    assert "guard let pid = parseRuntimeLockPID(text)" in swift


def test_packaged_launcher_rejects_explicitly_stale_locked_runtime():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    assert "private func existingRuntimeIsObservable()" in swift
    assert "fetchBootSnapshotSynchronously" in swift
    assert "snapshot.staleRuntimeFailureReason" in swift
    assert "runtimeHasUserVisibleHandoff" in swift
    assert "the launcher is an observer" in swift
    assert "forceStopAuraProcess(preserveResidentLauncher: true)" in swift
    forced_relaunch_body = swift.split("private func beginForcedRelaunch(reason: String)", 1)[1].split(
        "private func renderSnapshot",
        1,
    )[0]
    assert "forceStopAuraProcess(preserveResidentLauncher: true)" in forced_relaunch_body
    assert "launchAuraIfNeeded(forceRelaunch: false)" in forced_relaunch_body
    assert "launchAuraIfNeeded(forceRelaunch: true)" not in forced_relaunch_body
    assert "AURA_STOP_PRESERVE_RESIDENT_LAUNCHER" in swift
    assert 'checks["running"]' in swift
    assert "important:mind_tick" in swift
    assert "contract/important:mind_tick" in swift
    # zombie handling evolved: the launcher now names the dead-mind-tick +
    # no-live-lane verdict explicitly before replacing the session
    assert "replacing the zombie session" in swift
    assert "important:event_loop_monitor" in swift
    assert "only an" in swift and "explicit boot contract failure" in swift


def test_packaged_launcher_bounds_stop_helpers_during_runtime_refresh():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")
    stop_body = swift.split(
        "private func forceStopAuraProcess(preserveResidentLauncher: Bool = false)",
        1,
    )[1].split("@objc private func openLogs", 1)[0]

    assert "func runTool(arguments: [String], timeout: TimeInterval = 45.0)" in stop_body
    assert "AURA_STOP_GRACE_SECONDS" in stop_body
    assert '"18"' in stop_body
    assert "Date().addingTimeInterval(timeout)" in stop_body
    assert "Aura stop helper timed out" in stop_body
    assert "proc.terminate()" in stop_body
    assert "kill(proc.processIdentifier, SIGKILL)" in stop_body
    assert "runTool(arguments: [cleanupScript.path], timeout: 20.0)" in stop_body


def test_packaged_launcher_tracks_spawned_runtime_children_for_teardown():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    assert "private var spawnedProcesses: [Process] = []" in swift
    assert "private let spawnedProcessesLock = NSLock()" in swift
    assert "terminateSpawnedProcesses()" in swift
    assert "func applicationWillTerminate" in swift
    assert "trackSpawnedProcess(proc)" in swift
    assert "proc.terminationHandler" in swift
    assert "kill(proc.processIdentifier, SIGKILL)" in swift


def test_packaged_launcher_uses_native_app_window_for_default_desktop_surface():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    open_body = swift.split("@objc private func openDesktopWindow()", 1)[1].split(
        "@objc private func openBrowser()",
        1,
    )[0]
    auto_body = swift.split("private func autoOpenDesktopWindowIfNeeded()", 1)[1].split(
        "\n    }\n}",
        1,
    )[0]

    assert "openNativeDesktopWindow()" in open_body
    assert 'spawnAuxiliaryAura(arguments: ["--open-gui-window"])' not in open_body
    assert "openNativeDesktopWindow()" in auto_body
    assert 'spawnAuxiliaryAura(arguments: ["--open-gui-window"])' not in auto_body
    assert "WKWebView(frame:" in swift
    assert "desktop.contentView = webView" in swift
    assert 'desktop.title = "Aura Zenith"' in swift


def test_packaged_launcher_reopens_primary_desktop_without_resurrecting_monitor():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    primary_body = swift.split("private func frontPrimaryWindow()", 1)[1].split(
        "func applicationShouldHandleReopen",
        1,
    )[0]
    reopen_body = swift.split("func applicationShouldHandleReopen", 1)[1].split(
        "func applicationShouldTerminateAfterLastWindowClosed",
        1,
    )[0]

    assert "desktopWindow == nil && runtimeLockIndicatesLiveProcess()" in primary_body
    assert "openNativeDesktopWindow()" in primary_body
    assert "desktopWindow ?? window" in primary_body
    assert "target.isMiniaturized" in primary_body
    assert "target.deminiaturize" in primary_body
    assert "target.makeKeyAndOrderFront" in primary_body
    assert "frontPrimaryWindow()" in reopen_body
    assert "window.makeKeyAndOrderFront" not in reopen_body
    assert "showPrimaryWindowNotification" in swift
    assert "handleShowPrimaryWindowNotification" in swift
    assert "DistributedNotificationCenter.default().postNotificationName" in swift


def test_packaged_launcher_repairs_a_lost_window_on_application_activation():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    active_body = swift.split("func applicationDidBecomeActive", 1)[1].split(
        "func applicationShouldTerminateAfterLastWindowClosed",
        1,
    )[0]
    assert "visibleInteractiveSurfaceExists" in active_body
    assert "frontPrimaryWindow()" in active_body
    assert "bubblePanel?.isVisible" not in swift.split(
        "private func visibleInteractiveSurfaceExists", 1
    )[1].split("private func frontPrimaryWindow", 1)[0]


def test_packaged_launcher_publishes_real_windows_to_accessibility_and_active_space():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    publish_body = swift.split("private func publishAccessiblePrimaryWindow", 1)[1].split(
        "private func frontPrimaryWindow",
        1,
    )[0]
    assert "setAccessibilityElement(true)" in publish_body
    assert "setAccessibilityRole(.window)" in publish_body
    assert "setAccessibilitySubrole(.standardWindow)" in publish_body
    assert "NSApp.setAccessibilityWindows([primary])" in publish_body
    assert "NSApp.setAccessibilityMainWindow(primary)" in publish_body
    assert "NSApp.setAccessibilityFocusedWindow(primary)" in publish_body
    assert "window.collectionBehavior.insert(.moveToActiveSpace)" in swift
    assert "desktop.collectionBehavior.insert(.moveToActiveSpace)" in swift


def test_packaged_launcher_retains_its_weak_app_delegate_for_the_event_loop():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    main_body = swift.split("let app = NSApplication.shared", 1)[1]
    assert "app.delegate = delegate" in main_body
    assert "withExtendedLifetime(delegate)" in main_body
    assert main_body.index("withExtendedLifetime(delegate)") < main_body.index("app.run()")


def test_packaged_launcher_readiness_is_single_window_handoff():
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    open_body = swift.split("private func openNativeDesktopWindow()", 1)[1].split(
        "private func bootMarkerAge()",
        1,
    )[0]
    ready_body = swift.split("if snapshot.launcherReady", 1)[1].split(
        "return",
        1,
    )[0]

    assert open_body.index("hideLauncherWindow()") < open_body.index(
        "desktopWindow?.makeKeyAndOrderFront"
    )
    assert ".activateAllWindows" not in open_body
    assert "scheduleCloseIfNeeded" not in ready_body
    assert "private func scheduleCloseIfNeeded" not in swift


def test_aura_main_acquires_singleton_lock_before_port_cleanup_and_reaper_boot():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert main_py.index("bootstrap_lock(skip_lock=args.watchdog)") < main_py.index("kill_port(args.port)")
    assert main_py.index("bootstrap_lock(skip_lock=args.watchdog)") < main_py.index("reaper_proc = multiprocessing.Process(")
    assert "stop_aura()" in main_py
    assert "if not args.cli and not args.gui_window and not args.watchdog:" in main_py
    # The reaper-boot guard also refuses after a latched shutdown (monotonic
    # shutdown, 76e5a71c) — pin the strengthened condition, not the old text.
    assert (
        "if not args.gui_window and not args.watchdog and not is_shutdown_requested():"
        in main_py
    )
    assert "AURA_REAPER_MANIFEST" in main_py


def test_stop_aura_signals_parent_before_touching_child_actors():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    legacy_helper = main_py.split("def _disable_legacy_launchagent(", 1)[1].split(
        "def _is_reapable_aura_process_command(",
        1,
    )[0]
    stop_body = main_py.split("def stop_aura():", 1)[1].split("# ---------------------------------------------------------------------------", 1)[0]

    assert "p.send_signal(signal.SIGTERM)" in stop_body
    assert "AURA_STOP_GRACE_SECONDS" in stop_body
    assert "p.wait(timeout=stop_grace_s)" in stop_body
    assert "def stop_native_desktop_launchers()" in stop_body
    assert "Aura.app/Contents/MacOS/aura-launcher" in stop_body
    assert "AURA_STOP_PRESERVE_RESIDENT_LAUNCHER" in stop_body
    assert "Preserving resident Aura.app launcher bridge PID" in stop_body
    assert "parent_pid = os.getppid()" in stop_body
    assert "pid in {current_pid, parent_pid}" in stop_body
    assert '"--desktop", "--gui-window"' in stop_body
    assert '"--stop" not in cmdline' in stop_body
    assert "_disable_legacy_launchagent(" in stop_body
    assert "reason=\"explicit_stop\"" in stop_body
    assert '["launchctl", "bootout", f"gui/{uid}", str(plist_path)]' in legacy_helper
    assert '["launchctl", "disable", f"gui/{uid}/com.aura.sovereign"]' in legacy_helper
    assert '["launchctl", "unload", "-w", str(plist_path)]' in legacy_helper
    assert "core.orchestrator.main" in legacy_helper
    assert "legacy_launchagents" in legacy_helper
    assert "AURA_KEEP_LEGACY_LAUNCHAGENT" in legacy_helper
    assert "Stopped native launcher session(s)" in stop_body
    assert "Stopped post-shutdown revived Aura session(s)" in stop_body
    assert "reason=\"modern_desktop_launch\"" in main_py
    first_signal = stop_body.index("p.send_signal(signal.SIGTERM)")
    assert "for child in p.children(recursive=True):" not in stop_body[:first_signal]


def test_desktop_runtime_preserves_outer_signal_owner():
    orchestrator_main = (PROJECT_ROOT / "core" / "orchestrator" / "main.py").read_text(
        encoding="utf-8"
    )

    assert "desktop_signal_owner" in orchestrator_main
    assert "Desktop shutdown signal owner preserved in aura_main" in orchestrator_main
    assert "AURA_LAUNCHED_FROM_APP" in orchestrator_main
    assert "AURA_EXTERNAL_GUI_OWNER" in orchestrator_main


def test_watchdog_mode_remains_supervision_only():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    watchdog_slice = main_py.split("async def run_watchdog(", 1)[1].split("# ---------------------------------------------------------------------------", 1)[0]

    assert "create_orchestrator" not in watchdog_slice
    assert "bootstrap_aura(orchestrator)" not in watchdog_slice
    assert "await orchestrator.start()" not in watchdog_slice
    assert 'logger.info("🛡️ Watchdog supervisor active (supervision-only mode).")' in watchdog_slice
    assert "_watchdog_child_args(args)" in watchdog_slice
    assert "get_subprocess_gateway().spawn_async(" in watchdog_slice
    assert 'local_internal_governed_scope(' in watchdog_slice
    assert '"environment_action:watchdog_supervisor"' in watchdog_slice
    assert "[_launcher_python_executable(), __file__, *child_args]" in watchdog_slice
    assert "asyncio.create_subprocess_exec" not in watchdog_slice


def test_gui_reaper_spawn_is_governed_environment_action():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    gui_slice = main_py.split("async def _gui_reaper_loop():", 1)[1].split(
        "logger.info(\"🎨 GUI Process Started",
        1,
    )[0]

    assert "get_subprocess_gateway().spawn_async(" in gui_slice
    assert "local_internal_governed_scope(" in gui_slice
    assert '"environment_action:gui_actor_reaper"' in gui_slice
    assert 'domain="environment_action"' in gui_slice


def test_watchdog_preserves_requested_restart_mode_and_port_cleanup_is_pattern_limited():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert "def _watchdog_child_args(" in main_py
    assert "asyncio.run(run_watchdog(args))" in main_py
    assert 'child_args.append("--desktop")' in main_py
    assert 'child_args.append("--server")' in main_py
    assert 'child_args.append("--headless")' in main_py
    assert 'child_args.append("--cli")' in main_py
    assert "force_all_ports = {10003}" in main_py
    assert "shared_ports = {8000}" in main_py
    assert "Leaving non-Aura process" in main_py


def test_aura_main_routes_bootstrap_background_tasks_through_task_tracker():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert 'tracker.create_task(mem_monitor.start(), name="memory_monitor.start")' in main_py
    assert 'tracker.create_task(orchestrator.run(), name="OrchestratorMainLoop")' in main_py
    assert 'tracker.create_task(_gui_reaper_loop(), name="gui_reaper")' in main_py
    assert 'get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")' in main_py


def test_aura_main_long_running_streams_are_shutdown_bounded():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    philosophy_slice = main_py.split("async def run_philosophy_stream", 1)[1].split("async def run_server_async", 1)[0]
    desktop_slice = main_py.split("async def run_desktop", 1)[1].split("def _watchdog_child_args", 1)[0]

    assert "while not is_shutdown_requested():" in philosophy_slice
    assert '_env_float("AURA_PHILOSOPHY_STREAM_INTERVAL"' in philosophy_slice
    assert "while True" not in philosophy_slice
    assert "while line := await stream.readline():" in desktop_slice
    assert 'logger.error("[GUI] %s", decoded)' in desktop_slice
    assert 'logger.debug("[GUI] %s", decoded)' in desktop_slice
    assert "while True" not in desktop_slice


def test_aura_main_boot_wait_accepts_launchable_warming_payload():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    wait_slice = main_py.split("async def _wait_for_server_http", 1)[1].split(
        "def _native_launcher_owns_gui",
        1,
    )[0]

    assert "launcher_ready = bool(data.get(\"launcher_ready\"))" in wait_slice
    assert "conversation_warming" in wait_slice
    assert "conversation_recovering" in wait_slice
    assert "conversation_failed" in wait_slice
    assert "GUI launchable while boot_phase" in wait_slice


def test_aura_main_uses_shared_runtime_boot_helper_across_cli_server_and_desktop():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert "async def _boot_runtime_orchestrator(" in main_py
    assert "async def boot_aura_runtime(" in main_py
    assert "if not _RUNTIME_LOCK_CLAIMED:" in main_py
    assert "bootstrap_lock(skip_lock=False)" in main_py
    assert main_py.count("create_orchestrator()") == 1
    assert main_py.count("await bootstrap_aura(orchestrator)") == 1
    assert main_py.count("ServiceContainer.lock_registration()") == 1
    assert main_py.count("boot_aura_runtime(") >= 4
    assert 'orchestrator = await boot_aura_runtime(profile=profile, ready_label="CLI")' in main_py


def test_desktop_shell_is_render_fault_tolerant_and_bootstrap_normalized():
    main_py = (PROJECT_ROOT / "aura_main.py").read_text(encoding="utf-8")
    shell_app = (PROJECT_ROOT / "interface" / "static" / "shell" / "src" / "App.jsx").read_text(
        encoding="utf-8"
    )
    shell_main = (PROJECT_ROOT / "interface" / "static" / "shell" / "src" / "main.jsx").read_text(
        encoding="utf-8"
    )
    shell_css = (PROJECT_ROOT / "interface" / "static" / "shell" / "src" / "shell.css").read_text(
        encoding="utf-8"
    )
    system_routes = (PROJECT_ROOT / "interface" / "routes" / "system.py").read_text(encoding="utf-8")

    assert "function normalizeBootstrap(payload)" in shell_app
    assert "safeArray(raw.tools)" in shell_app
    assert "setBootstrap(normalized)" in shell_app
    assert "setTelemetry(normalized.telemetry)" in shell_app
    assert "makeId(" in shell_app
    assert "class ShellErrorBoundary extends React.Component" in shell_main
    assert "Aura shell render failure" in shell_main
    assert "window.location.reload()" in shell_main
    assert 'window.open("/api/health/boot", "_blank")' in shell_main
    assert ".shell-crash" in shell_css
    assert ".message-content" in shell_css
    assert "overflow-wrap: anywhere" in shell_css
    assert ".feed-content" in shell_css
    assert "overflow: visible" in shell_css
    assert '@router.post("/ui/shell-error")' in system_routes
    assert "Desktop shell render fault recovered" in system_routes
    assert 'ready_label="Desktop"' in main_py
    assert 'ready_label="Server"' in main_py


def test_retired_3d_launcher_is_not_referenced_by_runtime_paths():
    retired_launcher = PROJECT_ROOT / "scripts" / "one_off" / "launch_aura_3d.py"

    assert not retired_launcher.exists()
    for path in (
        PROJECT_ROOT / "aura_main.py",
        PROJECT_ROOT / "launch_aura.sh",
        PROJECT_ROOT / "scripts" / "AuraLauncher.swift",
    ):
        assert "launch_aura_3d.py" not in path.read_text(encoding="utf-8")


def test_bundle_script_builds_regular_dock_app_and_embeds_version_metadata():
    bundle_script = (PROJECT_ROOT / "scripts" / "bundle_app.sh").read_text(encoding="utf-8")

    assert 'VERSION_FILE="${RESOURCES_DIR}/aura-version"' in bundle_script
    assert 'ROOT_DIR="$(cd -P "$(dirname "$0")/.." && pwd -P)"' in bundle_script
    assert 'VERSION_FULL_FILE="${RESOURCES_DIR}/aura-version-full"' in bundle_script
    assert 'PROVENANCE_FILE="${RESOURCES_DIR}/aura-launch-provenance.json"' in bundle_script
    assert 'INSTALL_PATH="${AURA_INSTALL_PATH:-}"' in bundle_script
    assert 'ENTITLEMENTS_PLIST="${DIST_DIR}/aura.entitlements"' in bundle_script
    assert 'DEFAULT_CODESIGN_IDENTITY="-"' in bundle_script
    assert "run_with_timeout()" in bundle_script
    assert "Aura Local Code Signing" in bundle_script
    assert 'AURA_AUTO_USE_LOCAL_CODESIGN:-1' in bundle_script
    assert "macOS TCC" in bundle_script
    assert "AURA_CODESIGN_PROBE_TIMEOUT_S:-8" in bundle_script
    assert "AURA_CODESIGN_TIMEOUT_S:-45" in bundle_script
    assert "aura-codesign-probe" in bundle_script
    assert 'run_with_timeout "${AURA_CODESIGN_PROBE_TIMEOUT_S:-8}" codesign --force --sign "${LOCAL_AURA_IDENTITY}" "${SIGN_PROBE_DIR}/probe"' in bundle_script
    assert "using ad-hoc signing" in bundle_script
    assert 'CODESIGN_IDENTITY="${AURA_CODESIGN_IDENTITY:-${DEFAULT_CODESIGN_IDENTITY}}"' in bundle_script
    assert "info_plist_overrides" in bundle_script
    assert "write_entitlements_plist" in bundle_script
    assert '"NSHighResolutionCapable": True' in bundle_script
    assert 'payload.update(info_plist_overrides())' in bundle_script
    assert 'cp -R "${APP_DIR}" "${INSTALL_PATH}"' in bundle_script
    assert 'CODESIGN_ARGS=(--force --sign "${CODESIGN_IDENTITY}" --entitlements "${ENTITLEMENTS_PLIST}")' in bundle_script
    assert 'sign_bundle "${APP_DIR}"' in bundle_script
    assert 'sign_bundle "${INSTALL_PATH}"' in bundle_script
    assert 'run_with_timeout "${timeout_s}" codesign "${CODESIGN_ARGS[@]}" "${target}"' in bundle_script
    assert 'codesign --verify --deep --strict --verbose=2' in bundle_script
    assert 'AURA_REQUIRE_STABLE_CODESIGN' in bundle_script
    assert '-m core.runtime.launch_provenance emit' in bundle_script
    assert 'ln -sfn "${ROOT_DIR}"' not in bundle_script
    assert "CFBundleShortVersionString" in bundle_script
    assert "NSAppleEventsUsageDescription" not in bundle_script
    assert "LSUIElement" not in bundle_script


def test_shell_launcher_help_is_non_destructive():
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    result = get_subprocess_gateway().run(
        [str(PROJECT_ROOT / "launch_aura.sh"), "--help"],
        cwd=PROJECT_ROOT,
        read_only=True,
        capture_output=True,
        timeout=5,
        source="test_launcher_polish_contract.help",
        accelerator_capability="none",
    )

    assert result.returncode == 0
    assert "Usage: ./launch_aura.sh" in result.stdout
    assert "Cleaning up existing instances" not in result.stdout
    assert "Starting Aura Desktop" not in result.stdout


def test_legacy_installer_uses_stable_bundle_manifest():
    installer = (PROJECT_ROOT / "scripts" / "install_to_applications.py").read_text(
        encoding="utf-8"
    )

    assert 'sys.path.insert(0, str(PROJECT_ROOT))' in installer
    assert "info_plist_overrides" in installer
    assert "write_entitlements_plist" in installer
    assert '"CFBundleIdentifier": "com.aura.desktop"' in installer
    assert "com.aura.sovereign" not in installer
    assert "payload.update(info_plist_overrides())" in installer


def test_local_codesign_identity_helper_exists_for_stable_tcc_identity():
    helper = PROJECT_ROOT / "scripts" / "create_local_codesign_identity.sh"
    source = helper.read_text(encoding="utf-8")

    assert helper.exists()
    assert "extendedKeyUsage = codeSigning" in source
    assert "Aura Local Code Signing" in source
    assert "PBE-SHA1-3DES" in source
    assert "security import" in source
    assert "trust_identity_for_codesign" in source
    assert "security add-trusted-cert" in source
    assert "-p codeSign" in source
    assert "verify_identity_can_sign" in source
    assert 'codesign --force --sign "${IDENTITY_NAME}" "${probe}"' in source
    assert "Existing identity cannot sign" in source


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


def test_memory_ui_uses_packaged_fallback_and_visible_error_panel():
    memory_ui = (PROJECT_ROOT / "interface" / "memory_ui.py").read_text(encoding="utf-8")
    panel = (PROJECT_ROOT / "interface" / "static" / "memory_panel.html").read_text(
        encoding="utf-8"
    )

    root_handler = memory_ui.split("async def serve_memory_root():", 1)[1].split(
        "async def serve_memory_ui():",
        1,
    )[0]
    assert 'interface" / "static" / "memory_panel.html"' in root_handler
    assert "AURA_MEMORY_DEV_UI" in root_handler
    assert root_handler.index("memory_panel.html") < root_handler.index("AURA_MEMORY_DEV_UI")
    assert root_handler.index("memory_panel.html") < root_handler.index("static\" / \"memory\" / \"dist\"")
    assert "source Vite entry is not a valid packaged desktop fallback" in root_handler
    assert "function renderMemoryLoadError(error)" in panel
    assert "const AURA_API_BASE = 'http://127.0.0.1:8000'" in panel
    assert "function apiUrl(path)" in panel
    assert "window.location.protocol === 'file:'" in panel
    assert "fetch(apiUrl('/api/memory/semantic?limit=100'))" in panel
    assert "window.addEventListener('unhandledrejection'" in panel
    assert "semantic memory request failed" in panel
    assert "escapeHtml(text)" in panel
    assert "addActionButton(actions, 'Edit'" in panel


def test_webviews_answer_webkits_microphone_prompt():
    """LIVE DEFECT 2026-08-10: voice mode showed ERROR and heard nothing.

    The app holds the microphone TCC grant and the Python capture path used it
    fine — the wake-word loop was transcribing whole sentences at the time. But
    voice mode runs in the WKWebView and calls getUserMedia, and WebKit routes
    that to the app's WKUIDelegate. No web view had one, and with no delegate
    to answer, WebKit denies: "NotAllowedError: Permission denied". A
    permission the user had already granted could not reach the page needing
    it.
    """
    swift = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")

    assert "WKUIDelegate" in swift, "no delegate means WebKit denies capture outright"
    assert "requestMediaCapturePermissionFor" in swift

    # Every web view that can host the voice UI must route the prompt.
    assert swift.count("uiDelegate = self") >= 3, (
        "desktop, bubble and companion web views each need the delegate"
    )

    # The grant is scoped: the app's microphone is not handed to any page.
    assert "isLocalRuntimeOrigin" in swift
    assert "decisionHandler(.deny)" in swift
    assert "decisionHandler(.grant)" in swift
