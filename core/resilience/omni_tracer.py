"""core/resilience/omni_tracer.py
Omni-Tracer: A robust, deep systemic error capturer for Aura.

This module intercepts all unhandled exceptions across:
1. The main Python thread (sys.excepthook)
2. All background threads (threading.excepthook)
3. All asyncio tasks (loop.set_exception_handler)
4. All CRITICAL / ERROR logging events

It dumps them instantly into a unified, timestamped trace file, along with
system resource context (RAM, CPU, PID). This ensures that when a systemic 
cascade (like the [REAPER] or SEPSIS crash) occurs during chat, the exact 
root causes are preserved in one place.
"""

import sys
import threading
import asyncio
import logging
import json
import time
import os
import traceback
import psutil
from pathlib import Path
from typing import Any, Dict

_TRACE_FILE = Path.home() / ".aura" / "run" / "omni_trace.jsonl"
_OMNI_LOCK = threading.Lock()
_HOOKED = False

def _ensure_trace_dir():
    _TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)

def _get_system_context() -> Dict[str, Any]:
    try:
        vm = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        return {
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "cpu_percent": psutil.cpu_percent(),
            "mem_percent": vm.percent,
            "proc_mem_mb": proc.memory_info().rss / (1024 * 1024),
            "open_fds": proc.num_fds() if hasattr(proc, "num_fds") else 0,
        }
    except OSError:
        return {"pid": os.getpid()}

def write_trace(source: str, error_type: str, message: str, trace: str = ""):
    _ensure_trace_dir()
    event = {
        "ts": time.time(),
        "source": source,
        "type": error_type,
        "message": message,
        "traceback": trace,
        "context": _get_system_context()
    }
    with _OMNI_LOCK:
        try:
            with open(_TRACE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
                f.flush()
        except OSError:
            pass

    # [UI Integration] Forward to the Neural Stream / Terminal UI
    try:
        from core.health.degraded_events import record_degraded_event
        # Only forward actual crashes to the UI stream to prevent log noise
        if error_type != "System" and not source.startswith("log_info") and not source.startswith("log_warning"):
            record_degraded_event(
                subsystem=f"omni_{source}",
                reason=error_type,
                detail=f"{message}\n{trace}"[:800], # Keep it concise for the UI
                severity="critical",
                classification="system_crash",
            )
    except ImportError:
        pass

class OmniLogHandler(logging.Handler):
    """Intercepts high-severity logs and dumps them to the Omni-Trace."""
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            try:
                msg = self.format(record)
                trace = ""
                if record.exc_info:
                    trace = "".join(traceback.format_exception(*record.exc_info))
                write_trace(f"log_{record.levelname.lower()}", record.name, msg, trace)
            except OSError:
                pass

def _sys_excepthook(exc_type, exc_value, exc_traceback):
    trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    write_trace("sys_excepthook", exc_type.__name__, str(exc_value), trace)
    # Call the original excepthook if it exists and isn't ours
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

def _threading_excepthook(args):
    trace = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    write_trace("threading_excepthook", args.exc_type.__name__ if args.exc_type else "Unknown", str(args.exc_value), trace)
    if threading.__excepthook__ != _threading_excepthook:
         threading.__excepthook__(args)

def _asyncio_exception_handler(loop, context):
    msg = context.get("message", "Unknown Asyncio Error")
    exc = context.get("exception")
    trace = ""
    error_type = "AsyncioError"
    if exc:
        error_type = type(exc).__name__
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    write_trace("asyncio_handler", error_type, msg, trace)
    loop.default_exception_handler(context)

def hook_omni_tracer():
    global _HOOKED
    if _HOOKED:
        return
    
    # 1. Sys excepthook
    sys.excepthook = _sys_excepthook
    
    # 2. Threading excepthook
    threading.excepthook = _threading_excepthook
    
    # 3. Asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_asyncio_exception_handler)
    except RuntimeError:
        pass # Not in an async context yet, the orchestrator boot can apply it later
        
    # 4. Global Logging
    root_logger = logging.getLogger()
    has_omni = any(isinstance(h, OmniLogHandler) for h in root_logger.handlers)
    if not has_omni:
        root_logger.addHandler(OmniLogHandler())
        
    _HOOKED = True
    write_trace("omni_tracer", "System", "Omni-Tracer Online. Hooks attached.")
