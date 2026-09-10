import json
import logging
import os
import site
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Setup Path Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime.errors import record_degradation  # noqa: E402
from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS  # noqa: E402
from core.runtime.network_gateway import get_network_gateway  # noqa: E402


def _inject_project_venv_site_packages() -> None:
    """Mirror aura_main.py so GUI subprocesses see venv-installed deps."""
    venv_path = PROJECT_ROOT / ".venv"
    if not venv_path.exists():
        venv_path = PROJECT_ROOT / ".venv_aura"
    if not venv_path.exists():
        return

    lib_dir = venv_path / "lib"
    if not lib_dir.exists():
        return

    current_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for py_dir in lib_dir.glob("python3.*"):
        if py_dir.name != current_version:
            continue
        site_packages = py_dir / "site-packages"
        if site_packages.exists() and str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))
            site.addsitedir(str(site_packages))


_inject_project_venv_site_packages()

logger = logging.getLogger("Aura.GUI")

_BOOT_WAITING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aura – Starting</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0010;color:#e0d0f0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       display:flex;align-items:center;justify-content:center;height:100vh;overflow:hidden}
  .card{text-align:center;padding:48px 56px;border-radius:24px;
        background:rgba(140,80,220,0.08);border:1px solid rgba(140,80,220,0.18);
        backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px)}
  .orb-wrap{position:relative;width:96px;height:96px;margin:0 auto 32px}
  .orb{width:96px;height:96px;border-radius:50%;
       background:radial-gradient(circle at 40% 35%,#c084fc,#7c3aed 50%,#4c1d95);
       box-shadow:0 0 60px rgba(139,92,246,0.5),0 0 120px rgba(124,58,237,0.25);
       animation:pulse 2.4s ease-in-out infinite}
  @keyframes pulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.08);opacity:.82}}
  .orb-ring{position:absolute;inset:-8px;border-radius:50%;border:2px solid rgba(196,132,252,0.25);
            animation:ring-spin 6s linear infinite}
  @keyframes ring-spin{to{transform:rotate(360deg)}}
  h1{font-size:28px;font-weight:700;letter-spacing:6px;margin-bottom:8px;
     background:linear-gradient(135deg,#c084fc,#e0d0f0);-webkit-background-clip:text;
     -webkit-text-fill-color:transparent}
  .subtitle{font-size:14px;color:rgba(196,168,228,0.8);margin-bottom:28px}
  .bar-track{width:220px;height:4px;border-radius:2px;background:rgba(140,80,220,0.15);margin:0 auto 20px;overflow:hidden}
  .bar-fill{width:30%;height:100%;border-radius:2px;
            background:linear-gradient(90deg,#7c3aed,#c084fc);animation:slide 1.8s ease-in-out infinite}
  @keyframes slide{0%{transform:translateX(-100%)}50%{transform:translateX(250%)}100%{transform:translateX(-100%)}}
  #status{font-size:12px;color:rgba(196,168,228,0.55);min-height:18px}
</style>
</head>
<body>
<div class="card">
  <div class="orb-wrap"><div class="orb"></div><div class="orb-ring"></div></div>
  <h1>AURA</h1>
  <p class="subtitle">Connecting to runtime\u2026</p>
  <div class="bar-track"><div class="bar-fill"></div></div>
  <p id="status">Initializing</p>
</div>
</body>
</html>
""".strip()

_GUI_RECOVERABLE_ERRORS = (
    ImportError,
    RuntimeError,
    OSError,
    ValueError,
)

def _flush_logs_before_forced_exit() -> None:
    try:
        logging.shutdown()
    except (RuntimeError, OSError):
        pass


def _set_macos_accessory_activation_policy(stage: str) -> None:
    if sys.platform != "darwin":
        return
    try:
        import AppKit

        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory
    except _GUI_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "gui_actor",
            exc,
            action=f"continued GUI boot after dockless activation policy setup failed ({stage})",
            severity="warning",
        )


def _heartbeat_response_healthy(resp: Any) -> bool:
    """Accept only the canonical runtime readiness heartbeat as healthy."""
    if getattr(resp, "status_code", None) != 200:
        return False
    try:
        payload = resp.json()
    except (AttributeError, TypeError, ValueError):
        return False
    if not bool(payload.get("healthy") is True and payload.get("status") == "healthy"):
        return False
    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or blockers:
        return False
    probes = payload.get("required_probes")
    if not isinstance(probes, dict) or not bool(probes.get("all_passed", False)):
        return False
    for group, expected_components in REQUIRED_HEALTH_PROBE_GROUPS.items():
        probe = probes.get(group)
        if not isinstance(probe, dict) or not bool(probe.get("ok", False)):
            return False
        components = probe.get("components")
        if not isinstance(components, dict):
            return False
        if any(components.get(component) is not True for component in expected_components):
            return False
    return True


def _heartbeat_response_state(resp: Any) -> str:
    """Classify the canonical heartbeat without weakening healthy semantics.

    ``warming`` is explicitly not healthy. It means the runtime probes are
    alive, the shell can stay loaded, and the only blocker is the foreground
    conversation lane waiting for demand-driven Cortex warmup.
    """
    if _heartbeat_response_healthy(resp):
        return "healthy"
    try:
        payload = resp.json()
    except (AttributeError, TypeError, ValueError):
        return "unhealthy"

    status_code = int(getattr(resp, "status_code", 0) or 0)
    if status_code not in {200, 503}:
        return "unhealthy"

    blockers = payload.get("blockers")
    probes = payload.get("required_probes")
    if not isinstance(blockers, list) or not isinstance(probes, dict):
        return "unhealthy"
    blocker_set = {str(blocker) for blocker in blockers}
    conversation_warming_only = blocker_set and blocker_set <= {"conversation_ready"}
    if (
        bool(payload.get("runtime_probe_healthy", False))
        and bool(probes.get("all_passed", False))
        and payload.get("conversation_ready") is False
        and str(payload.get("boot_phase") or "").startswith("conversation_")
        and conversation_warming_only
    ):
        return "warming"
    # Degraded-but-conversational: every required probe passes and the
    # conversation lane is ready, but an *important*-tier background loop
    # (mind_tick, event_loop_monitor, unified_runtime_pressure) is degraded,
    # so `healthy` is False. The user can still chat — the watchdog must NOT
    # rip the live UI away and revert to "Connecting to runtime" over a
    # background loop (observed 2026-07-06: mind_tick dead → reconnect surface
    # while conversation worked).
    if (
        bool(payload.get("runtime_probe_healthy", False))
        and bool(probes.get("all_passed", False))
        and payload.get("conversation_ready") is True
    ):
        return "degraded_ready"
    return "unhealthy"


def _gateway_heartbeat_healthy(response: dict[str, Any]) -> bool:
    return _gateway_heartbeat_state(response) == "healthy"


def _gateway_heartbeat_state(response: dict[str, Any]) -> str:
    content = response.get("content") or b""
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
    try:
        payload = json.loads(text or "{}")
    except json.JSONDecodeError:
        return "unhealthy"

    class _Response:
        status_code = int(response.get("status_code") or 0)

        @staticmethod
        def json() -> dict[str, Any]:
            return payload

    return _heartbeat_response_state(_Response())


def gui_actor_entry(port: int, token: str = None):
    """Entry point for the GUI process."""
    logger.info("🚀 Aura GUI Actor starting on port %d...", port)
    # The intentional RuntimeError has been removed to allow boot.

    # 1. Standardize Environment for macOS/WebKit
    if sys.platform == "darwin":
        os.environ["OPENCV_VIDEOIO_AVFOUNDATION_USE_FRAME_RECEIVER"] = "0"
        os.environ["PYAV_SKIP_AVF_FRAME_RECEIVER"] = "1"
        os.environ["WEBKIT_DISABLE_COMPOSITING_MODE"] = "1"
        _set_macos_accessory_activation_policy("pre_webview")
    
    # 1.5 Set Proxy Mode Flag (though mostly unused now)
    os.environ["AURA_GUI_PROXY"] = "1"

    # 2. Setup Logging for the process
    from core.config import config
    from core.observability.logging_config import setup_logging
    setup_logging(log_dir=config.paths.log_dir)
    
    logger.info(f"🎨 GUI Actor initiating Pure WebView (Port: {port})")

    # 4. Launch webview
    try:
        import webview

        from core.utils.port_check import wait_for_port
        
        app_url = f"http://127.0.0.1:{port}"
        
        # Wait for the Kernel API to be ready
        # Increased to 60.0s for slow Silicon model loads
        if wait_for_port(port, timeout=60.0):
            logger.info(f"📡 API Server (Kernel) detected online on port {port}. Launching WebView...")
        else:
            logger.warning(f"⚠️ API Server (Kernel) NOT detected on port {port} after 60s. Attempting window creation anyway...")
        
        # No URL in create_window to prevent race conditions during startup
        # The app is called Aura. "Zenith" is the release label — see LABEL in
        # core/runtime/version.py — and it belongs in the version chip, which
        # already shows "2026.4.20-Zenith". The bundle has said "Aura" since
        # signing; only this window disagreed.
        window = webview.create_window(
            "Aura", 
            width=1280, height=820, min_size=(800, 600)
        )
        _set_macos_accessory_activation_policy("post_window_create")
        shutdown_event = threading.Event()

        # ISSUE #14 - window closure delay race condition
        def _on_closed():
            logger.info("🎨 Window closed. Forcing GUI process termination.")
            shutdown_event.set()
            _flush_logs_before_forced_exit()
            os._exit(0)
            
        window.events.closed += _on_closed

        def _on_shown():
            logger.info("🎨 WebView Window Shown. Loading boot screen...")
            _set_macos_accessory_activation_policy("window_shown")
            try:
                window.load_html(_BOOT_WAITING_HTML)
            except _GUI_RECOVERABLE_ERRORS as e:
                logger.warning("Failed to load boot waiting page: %s", e)

            max_wait = 180
            backoff = 1.0
            max_backoff = 5.0
            elapsed = 0.0
            attempt = 0

            while elapsed < max_wait:
                attempt += 1
                try:
                    resp = get_network_gateway().request(
                        "GET",
                        f"http://127.0.0.1:{port}/",
                        timeout=5,
                        source="gui_actor.boot_wait",
                        read_only=True,
                        suppress_degradation=True,
                    )
                    status_code = int(resp.get("status_code", 0) or 0)
                    # A chat-capable runtime must not stay hidden behind the
                    # splash. Root returns 503 while the runtime is merely
                    # DEGRADED (e.g. a background subsystem in failure-lockdown),
                    # even though conversation works — that pinned the desktop on
                    # "Connecting to runtime" indefinitely (2026-07-04). If the
                    # conversation lane is ready, load the real UI regardless.
                    conversation_ready = False
                    if not (0 < status_code < 500):
                        try:
                            boot = get_network_gateway().request(
                                "GET",
                                f"http://127.0.0.1:{port}/api/health/boot",
                                timeout=5,
                                source="gui_actor.boot_wait_conversation",
                                read_only=True,
                                suppress_degradation=True,
                            )
                            raw = boot.get("content") or b""
                            if isinstance(raw, (bytes, bytearray)):
                                raw = raw.decode("utf-8", "replace")
                            payload = json.loads(raw) if raw else {}
                            # Trust the boot contract's own user-facing verdict:
                            # `ready` is True whenever the surface should
                            # connect — including conversation_working, where
                            # the lane is BUSY serving turns and
                            # conversation_ready flips False. Re-deriving from
                            # conversation_ready alone pinned the desktop on
                            # "Connecting to runtime" over a mind that was
                            # actively answering (2026-07-12, during a soak).
                            conversation_ready = bool(
                                payload.get("conversation_ready") is True
                                or payload.get("ready") is True
                                or str(payload.get("status") or "")
                                in {"ready", "working", "degraded"}
                            )
                        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                            conversation_ready = False
                    if (0 < status_code < 500) or conversation_ready:
                        logger.info(
                            "🔄 API ready (attempt %d, %.1fs, status=%s, conversation_ready=%s). Loading GUI: %s",
                            attempt, elapsed, status_code, conversation_ready, app_url,
                        )
                        try:
                            window.load_url(app_url)
                        except _GUI_RECOVERABLE_ERRORS as e:
                            record_degradation('gui_actor', e)
                            logger.error("Failed to load URL in WebView: %s", e)
                        return
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as e:
                    logger.debug("GUI boot wait probe failed on attempt %d: %s", attempt, e)

                try:
                    window.evaluate_js(
                        f'document.getElementById("status").textContent = "Attempt {attempt} \u2013 waiting for runtime\u2026";'
                    )
                except _GUI_RECOVERABLE_ERRORS:
                    pass

                time.sleep(backoff)
                elapsed += backoff
                backoff = min(backoff * 1.5, max_backoff)

            logger.warning(
                "⚠️ API not confirmed after %.0fs / %d attempts. Loading URL as last resort.",
                elapsed, attempt,
            )
            try:
                window.load_url(app_url)
            except _GUI_RECOVERABLE_ERRORS as e:
                record_degradation('gui_actor', e)
                logger.error("Failed to load URL in WebView: %s", e)

        # Watchdog: Periodically check if the UI is responsive
        _boot_completed = False

        def _watchdog():
            nonlocal _boot_completed
            logger.info("🐕 GUI Watchdog active.")
            consecutive_failures = 0
            start_time = time.monotonic()
            last_warming_log = 0.0
            while not shutdown_event.wait(20):
                try:
                    resp = get_network_gateway().request(
                        "GET",
                        f"{app_url}/api/health/heartbeat",
                        timeout=5,
                        source="gui_actor.watchdog",
                        read_only=True,
                        suppress_degradation=True,
                    )
                    heartbeat_state = (
                        "healthy"
                        if _gateway_heartbeat_healthy(resp)
                        else _gateway_heartbeat_state(resp)
                    )
                    if heartbeat_state == "healthy":
                        if not _boot_completed:
                            _boot_completed = True
                            logger.info("✅ [GUI WATCHDOG] First healthy heartbeat received.")
                        consecutive_failures = 0
                    elif heartbeat_state == "warming":
                        consecutive_failures = 0
                        now = time.monotonic()
                        if now - last_warming_log > 120.0:
                            logger.info(
                                "[GUI WATCHDOG] Runtime probes pass; conversation lane is warming "
                                "and will be demand-warmed by the next foreground turn."
                            )
                            last_warming_log = now
                    elif heartbeat_state == "degraded_ready":
                        # Conversation works; only a background loop is degraded.
                        # Keep the live UI — do NOT count this toward the
                        # reconnect-surface threshold. The user can still talk to
                        # her while the runtime self-heals the background loop.
                        if not _boot_completed:
                            _boot_completed = True
                        consecutive_failures = 0
                        now = time.monotonic()
                        if now - last_warming_log > 120.0:
                            logger.info(
                                "[GUI WATCHDOG] Runtime degraded on a background loop but "
                                "conversation is ready; keeping the live UI."
                            )
                            last_warming_log = now
                    else:
                        consecutive_failures += 1
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                    consecutive_failures += 1

                if (
                    not _boot_completed
                    and consecutive_failures > 0
                    and (time.monotonic() - start_time) > 90
                ):
                    logger.warning("🚨 [GUI WATCHDOG] Boot not completed after 90s. Forcing reload.")
                    try:
                        window.load_url(app_url)
                    except _GUI_RECOVERABLE_ERRORS as _exc:
                        record_degradation('gui_actor', _exc)
                        logger.warning("GUI watchdog boot reload failed: %s", _exc)
                    _boot_completed = True

                if consecutive_failures == 3:
                    logger.warning(
                        "[GUI WATCHDOG] Kernel heartbeat missed three times; preserving the "
                        "current window while the runtime recovers."
                    )

                # A foreground MLX generation can hold native resources long
                # enough for several heartbeat probes to time out even though
                # the kernel process is alive.  Reloading every two misses used
                # to destroy in-flight UI state, and exiting after six misses
                # made Aura appear to quit during a long chat turn.  Keep the
                # window alive and move to a reconnect surface only after a
                # sustained outage; normal websocket/bootstrap polling will
                # restore the live UI when the server returns.
                if consecutive_failures == 6:
                    logger.warning(
                        "[GUI WATCHDOG] Kernel heartbeat unavailable for a sustained interval; "
                        "showing the reconnect surface without exiting the desktop process."
                    )
                    try:
                        window.load_html(_BOOT_WAITING_HTML)
                    except _GUI_RECOVERABLE_ERRORS as _exc:
                        record_degradation("gui_actor", _exc)
                        logger.warning("GUI watchdog reconnect surface failed: %s", _exc)

        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()

        # Permission checks during boot must be passive.  Prompting on every
        # window launch caused a duplicate Accessibility dialog even when the
        # user had already granted Aura.app access, because this process is a
        # Python helper with a different TCC identity.  Explicit settings-page
        # actions own all permission requests.

        # In Zenith, we use the functional start to load the URL after initialization
        webview.start(func=_on_shown, debug=False)
        
    except _GUI_RECOVERABLE_ERRORS as e:
        record_degradation('gui_actor', e)
        logger.error(f"❌ WebView Failure: {e}")
        time.sleep(5)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        gui_actor_entry(int(sys.argv[1]))
    else:
        gui_actor_entry(8000)
