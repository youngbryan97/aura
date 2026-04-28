from core.runtime.errors import record_degradation
import os
import sys
import logging
import site
import threading
import time
from pathlib import Path
from typing import Optional

# Setup Path Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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

def gui_actor_entry(port: int, token: str = None):
    """Entry point for the GUI process."""
    logger.info("🚀 Aura GUI Actor starting on port %d...", port)
    # The intentional RuntimeError has been removed to allow boot.
    
    # 1. Standardize Environment for macOS/WebKit
    if sys.platform == "darwin":
        os.environ["OPENCV_VIDEOIO_AVFOUNDATION_USE_FRAME_RECEIVER"] = "0"
        os.environ["PYAV_SKIP_AVF_FRAME_RECEIVER"] = "1"
        os.environ["WEBKIT_DISABLE_COMPOSITING_MODE"] = "1"
    
    # 1.5 Set Proxy Mode Flag (though mostly unused now)
    os.environ["AURA_GUI_PROXY"] = "1"

    # 2. Setup Logging for the process
    from core.logging_config import setup_logging
    from core.config import config
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
        window = webview.create_window(
            "Aura Zenith", 
            width=1280, height=820, min_size=(800, 600)
        )

        # ISSUE #14 - window closure delay race condition
        def _on_closed():
            logger.info("🎨 Window closed. Forcing GUI process termination.")
            os._exit(0)
            
        window.events.closed += _on_closed

        def _on_shown():
            logger.info("🎨 WebView Window Shown. Initiating load...")
            time.sleep(1.0) # Grace period for WebKit
            try:
                window.load_url(app_url)
                logger.info(f"🔄 GUI Loaded: {app_url}")
            except Exception as e:
                record_degradation('gui_actor', e)
                logger.error(f"Failed to load URL in WebView: {e}")

        # Watchdog: Periodically check if the UI is responsive
        def _watchdog():
            # ISSUE #13 - requests missing import swallows watchdog
            try:
                import requests
            except ImportError:
                logger.error("🚨 [GUI WATCHDOG] 'requests' module not found. Watchdog disabled.")
                return
                
            logger.info("🐕 GUI Watchdog active.")
            consecutive_failures = 0
            while True:
                time.sleep(20)
                try:
                    resp = requests.get(f"{app_url}/api/health", timeout=5)
                    if resp.status_code == 200:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                except Exception:
                    consecutive_failures += 1
                
                if consecutive_failures >= 3:
                    logger.warning("🚨 [GUI WATCHDOG] Kernel API unreachable. Attempting reload.")
                    try:
                        window.load_url(app_url)
                    except Exception as _exc:
                        record_degradation('gui_actor', _exc)
                        logger.debug("Suppressed Exception: %s", _exc)

                if consecutive_failures >= 6:
                    logger.critical("🛑 [GUI WATCHDOG] Kernel API unavailable for too long. Exiting stale WebView.")
                    os._exit(1)

        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()

        # In Zenith, we use the functional start to load the URL after initialization
        webview.start(func=_on_shown, debug=False)
        
    except Exception as e:
        record_degradation('gui_actor', e)
        logger.error(f"❌ WebView Failure: {e}")
        time.sleep(5)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        gui_actor_entry(int(sys.argv[1]))
    else:
        gui_actor_entry(8000)
