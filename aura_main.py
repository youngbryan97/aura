#!/usr/bin/env python3
"""
Aura Main Entry Point (v13.5 Unified)
------------------------------------
Standardized, single-entry launcher for CLI, Server, Desktop, and Watchdog modes.
Replaces: aura_launcher.py, aura_desktop.py, run_aura.py, run_aura_loop.py, and reboot.py.
"""

import argparse
import asyncio
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, List

# 1. Path Resolution
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 2. Bootstrap Configuration & Logging
try:
    from core.config import config
    from core.logging_config import setup_logging
    # Centralized logging setup - always include log_dir for persistence
    setup_logging(log_dir=config.paths.log_dir)
    logger = logging.getLogger("Aura.Main")
except Exception as e:
    # Minimal fallback logging if core is broken
    import traceback
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("Aura.Main")
    logger.error("❌ BOOTSTRAP FAILURE: Could not load core configuration.")
    logger.error(traceback.format_exc())
    config = None # Ensure NameError is avoided

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def check_environment():
    """Verify system readiness."""
    logger.info("🔍 Verifying Environment Integrity...")
    logger.info("📍 RUNTIME PATH Diagnostic:")
    logger.info("   • __file__: %s", __file__)
    logger.info("   • sys.executable: %s", sys.executable)
    logger.info("   • sys.path: %s", sys.path)
    try:
        import core
        logger.info("   • core.__file__: %s", core.__file__)
    except Exception as e:
        logger.error("   • core import failed: %s", e)
    
    if sys.version_info < (3, 9):
        logger.error("Aura requires Python 3.9+")
        sys.exit(1)
        
    if config is None:
        logger.error("❌ Environment check aborted: Configuration not loaded.")
        sys.exit(1)

    # Ensure home directory exists
    config.paths.create_directories()
    
    # Check for Ollama (optional warning)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", 11434)) != 0:
            logger.warning("⚠️  Ollama (11434) not detected. Cognitive functions may be limited.")

def kill_port(port: int, pattern: str = "aura"):
    """Force kill any process on a specific port matching a pattern."""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil missing - skipping advanced port cleanup.")
        return

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    cmd_str = " ".join(proc.cmdline() or []).lower()
                    if pattern in cmd_str or pattern in proc.name().lower():
                        logger.info("Terminating stale process %s on port %s...", proc.pid, port)
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except psutil.TimeoutExpired:
                            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def clean_artifacts():
    """Purge stale bytecode and temporary caches."""
    logger.info("🧹 Purging runtime artifacts...")
    for p in PROJECT_ROOT.rglob("__pycache__"):
        try: shutil.rmtree(p)
        except Exception: pass
    for p in PROJECT_ROOT.rglob("*.pyc"):
        try: p.unlink()
        except Exception: pass

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

async def run_console():
    """Interactive CLI Mode"""
    from core.main import conversation_loop
    await conversation_loop()

def run_server(host: str, port: int, reload: bool = False):
    """API Server Mode (FastAPI + Uvicorn)"""
    import uvicorn
    logger.info("🚀 Starting API Server on %s:%s", host, port)
    
    # In frozen bundles, uvicorn can't import by string path.
    # Import the app object directly.
    if getattr(sys, 'frozen', False):
        from interface.server import app
        uvicorn.run(app, host=host, port=port, log_level="info")
    else:
        uvicorn.run("interface.server:app", host=host, port=port, reload=reload, log_level="info")

def run_desktop(port: int):
    """GUI Mode (WebView + in-process server)"""
    import uvicorn
    
    # Import the FastAPI app directly (works in both dev and frozen)
    from interface.server import app as fastapi_app
    
    # Run uvicorn in a background daemon thread
    from core.config import config
    host = "127.0.0.1" if config.security.internal_only_mode else "0.0.0.0"
    
    server_config = uvicorn.Config(
        fastapi_app, host=host, port=port, log_level="info"
    )
    server = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready (up to 15s)
    # Use 127.0.0.1 for the wait check as it's always valid locally
    for i in range(30):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    logger.info("Server ready on %s:%s", host, port)
                    break
        except Exception:
            pass
        time.sleep(0.5) # Synchronous is okay here as we're preparing the environment before GUI start
    else:
        logger.error("Server failed to start on %s:%s within 15 seconds", host, port)
    
    # Try webview first, fall back to browser
    try:
        import webview
        logger.info("🎨 Initializing Desktop GUI...")
        webview.create_window("Aura", f"http://127.0.0.1:{port}", width=1280, height=820, min_size=(800, 600))
        webview.start()
    except ImportError:
        logger.warning("PyWebView missing. Opening in browser instead.")
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}")
        # Keep alive so the server thread doesn't die
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    logger.info("Desktop mode shutting down...")
    server.should_exit = True

async def run_watchdog():
    """Stability Watchdog Loop (Async)."""
    restart_count = 0
    while restart_count < 5:
        logger.info("🛡️  Watchdog: Launching Aura (Attempt %s)", restart_count+1)
        start_time = time.time()
        
        # Use asyncio.create_subprocess_exec for non-blocking wait
        proc = await asyncio.create_subprocess_exec(sys.executable, __file__, "--cli")
        await proc.wait()
        
        # If ran for > 10 mins, reset counter
        if time.time() - start_time > 600:
            restart_count = 0
        
        if proc.returncode == 0:
            logger.info("Clean shutdown detected. Watchdog exiting.")
            break
            
        restart_count += 1
        logger.warning("Crash detected (Code: %s). Restarting in 5s...", proc.returncode)
        await asyncio.sleep(5)

# ---------------------------------------------------------------------------
_SINGLETON_FD = None

def acquire_singleton_lock(skip_lock: bool = False):
    """Ensure only one Aura instance runs at a time using a file lock.

    H-15 FIX: Lock file uses 0o600 (owner-only) instead of 0o666.
    M-07 FIX: fcntl import is platform-guarded for Windows compatibility.
    """
    if skip_lock: return

    global _SINGLETON_FD
    import tempfile
    lock_file = Path(tempfile.gettempdir()) / "aura_singleton.lock"
    try:
        import fcntl
    except ImportError:
        # M-07 FIX: fcntl not available on Windows
        logger.warning("File locking unavailable on this platform. Singleton check skipped.")
        return
    try:
        _SINGLETON_FD = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(_SINGLETON_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("Aura is already running. Exiting to prevent multiple instances.")
        print("⚠️  Aura is already running in another window. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.warning("Failed to acquire single-instance lock: %s", e)

# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aura Unified Entry Point")
    parser.add_argument("--cli", action="store_true", help="Interactive Console Mode")
    parser.add_argument("--server", action="store_true", help="API Server Mode")
    parser.add_argument("--desktop", action="store_true", help="Desktop GUI Mode")
    parser.add_argument("--watchdog", action="store_true", help="Watchdog / Keep-alive Mode")
    parser.add_argument("--reboot", action="store_true", help="Force cleanup and restart (Standardize)")
    parser.add_argument("--port", type=int, default=8000, help="Port for Server/GUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Server")
    
    args = parser.parse_known_args()[0]
    
    # Standardize: Reboot behavior
    if args.reboot:
        logger.info("🔄 REBOOT SEQUENCE ACTIVATED")
        kill_port(args.port)
        clean_artifacts()
        # Default to desktop if no other mode specified
        if not (args.cli or args.server or args.desktop):
            args.desktop = True

    check_environment()
    
    # Do not acquire lock for watchdog, since the watchdog itself runs the child process
    acquire_singleton_lock(skip_lock=args.watchdog)

    try:
        if args.server:
            # Dynamic host selection if default was used
            host = args.host
            if host == "127.0.0.1" and not config.security.internal_only_mode:
                host = "0.0.0.0"
            run_server(host, args.port)
        elif args.desktop:
            run_desktop(args.port)
        elif args.watchdog:
            asyncio.run(run_watchdog())
        elif args.cli:
            asyncio.run(run_console())
        else:
            # Default fallback: Desktop if double-clicked, else CLI if terminal
            if sys.stdin and sys.stdin.isatty():
                asyncio.run(run_console())
            else:
                logger.info("Initializing in Desktop/Autonomy mode...")
                run_desktop(args.port)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    except Exception as e:
        logger.critical("FATAL BOOT ERROR: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
