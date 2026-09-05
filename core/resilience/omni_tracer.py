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

import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from core.governance_context import local_internal_governed_scope
from core.runtime import resource_psutil as psutil
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root

_TRACE_FILE = state_root() / "run" / "omni_trace.jsonl"
_OMNI_LOCK = threading.Lock()
logger = logging.getLogger("Aura.OmniTracer")


def _classify_forwarded_log(
    source: str,
    message: str,
    severity: str,
    error_type: str = "",
) -> tuple[str, str]:
    lowered_source = str(source or "").lower()
    lowered_message = str(message or "").lower()
    lowered_error_type = str(error_type or "").lower()
    final_severity = str(severity or "critical").lower()
    # [FIX] Intercepted Python log messages are degradation handler output,
    # NOT actual system crashes.  Classifying every CRITICAL log as
    # "system_crash" (weight 1.0) created a feedback amplification loop:
    #   health check → CRITICAL log → omni_log_critical degraded event →
    #   failure pressure spike → more health failures → more CRITICAL logs →
    #   pressure spiral to 0.88 → mind_tick blocked → orchestrator UNHEALTHY
    #   → meta-evolution aborted → chat repair failure → cascade.
    # Only genuine unhandled exceptions from exception hooks (sys_excepthook,
    # asyncio_handler, threading_excepthook) warrant "system_crash".  Log-
    # intercepted messages are at worst "background_degraded".
    if lowered_source.startswith("log_"):
        classification = "background_degraded"
    else:
        classification = "system_crash" if final_severity == "critical" else "background_degraded"

    if lowered_source.startswith(("log_error", "log_critical")) and (
        " - info - " in lowered_message
        or " | info " in lowered_message
        or "level=info" in lowered_message
    ):
        return "info", "non_critical_fallback"

    if (
        "aura.healthcontract" in lowered_error_type
        or "aura.healthcontract" in lowered_message
        or "health contract:" in lowered_message
    ):
        if "health contract: dead" in lowered_message:
            return "critical", "foreground_blocking"
        if "health contract: critical" in lowered_message:
            return "error", "background_degraded"
        if "health:" in lowered_message:
            return "error", "background_degraded"
        if "health contract: degraded" in lowered_message:
            return "warning", "background_degraded"
        return "error", "background_degraded"

    if "brain.gemini" in lowered_source and any(
        marker in lowered_message
        for marker in ("permission_denied", "api key", "leaked", " 403", "error 403")
    ):
        return "warning", "background_degraded"

    optional_dependency_sources = (
        "aura.voiceengine",
        "aura.socialmedia",
        "twitteradapter",
        "redditadapter",
    )
    optional_dependency_markers = (
        "not installed",
        "tts unavailable",
        "stt unavailable",
        "adapter disabled",
        "connection disabled",
        "no credentials configured",
        "incomplete credentials",
    )
    if any(source in lowered_source for source in optional_dependency_sources) and any(
        marker in lowered_message for marker in optional_dependency_markers
    ):
        return "warning", "background_degraded"

    if "generation deadline reached" in lowered_message and "llm.mlx" in lowered_source:
        return "warning", "foreground_blocking"

    if "local inference paths exhausted" in lowered_message and "aura.inferencegate" in lowered_source:
        return "warning", "foreground_blocking"

    if "responsegeneration phase timeout" in lowered_message or "unitaryresponsephase timed out" in lowered_message:
        return "warning", "foreground_blocking"

    return final_severity, classification
_OMNI_THREAD: threading.Thread | None = None
_OMNI_STOP = False
_OMNI_BUFFER = []
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

def _omni_writer_loop():
    global _OMNI_BUFFER
    while not _OMNI_STOP:
        batch = []
        try:
            with _OMNI_LOCK:
                if not _OMNI_BUFFER:
                    # Release lock and sleep if nothing to do
                    pass
                else:
                    batch = _OMNI_BUFFER
                    _OMNI_BUFFER = []
                    
            if not batch:
                time.sleep(0.5)
                continue
            
            _ensure_trace_dir()
            with local_internal_governed_scope(
                "resilience.omni_tracer.trace",
                receipt_prefix="omni-tracer-append",
            ):
                get_file_write_gateway().append_text(
                    _TRACE_FILE,
                    "".join(line + "\n" for line in batch),
                    source="resilience.omni_tracer.trace",
                )
            del batch
        except (OSError, IOError):
            time.sleep(1)

def write_trace(source: str, error_type: str, message: str, trace: str = "", severity: Optional[str] = None):
    global _OMNI_THREAD
    if _OMNI_THREAD is None:
        with _OMNI_LOCK:
            if _OMNI_THREAD is None:
                _OMNI_THREAD = threading.Thread(target=_omni_writer_loop, daemon=True, name="OmniTracerWriter")
                _OMNI_THREAD.start()

    event = {
        "ts": time.time(),
        "source": source,
        "type": error_type,
        "message": message,
        "traceback": trace,
        "severity": severity,
        "context": _get_system_context()
    }
    line = json.dumps(event)
    with _OMNI_LOCK:
        _OMNI_BUFFER.append(line)

    # [UI Integration] Forward to the Neural Stream / Terminal UI
    try:
        from core.health.degraded_events import record_degraded_event
        # Only forward actual crashes to the UI stream to prevent log noise
        if error_type != "System" and not source.startswith("log_info") and not source.startswith("log_warning"):
            # Determine severity: use provided, or infer from source/type
            final_severity = severity
            if not final_severity:
                if error_type == "EventLoopLag":
                    final_severity = "warning"
                else:
                    final_severity = "critical"
            final_severity, classification = _classify_forwarded_log(
                source,
                message,
                final_severity,
                error_type=error_type,
            )

            record_degraded_event(
                subsystem=f"omni_{source}",
                reason=error_type,
                detail=f"{message}\n{trace}"[:800], # Keep it concise for the UI
                severity=final_severity,
                classification=classification,
            )
    except (ImportError, RuntimeError, AttributeError) as _exc:
        logging.getLogger("Aura.OmniTracer").debug(
            "Suppressed %s in core.resilience.omni_tracer: %s",
            type(_exc).__name__,
            _exc,
        )

class OmniLogHandler(logging.Handler):
    """Intercepts high-severity logs and dumps them to the Omni-Trace."""
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            try:
                msg = self.format(record)
                trace = ""
                if record.exc_info:
                    trace = "".join(traceback.format_exception(*record.exc_info))
                write_trace(
                    f"log_{record.levelname.lower()}",
                    record.name,
                    msg,
                    trace,
                    severity=record.levelname.lower(),
                )
            except OSError as _exc:
                logging.getLogger("Aura.OmniTracer").debug(
                    "Suppressed %s in core.resilience.omni_tracer: %s",
                    type(_exc).__name__,
                    _exc,
                )

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

def _install_loop_handler(loop: asyncio.AbstractEventLoop) -> None:
    loop.set_exception_handler(_asyncio_exception_handler)


def install_asyncio_exception_handler(loop: Optional[asyncio.AbstractEventLoop] = None) -> bool:
    """Attach the Omni async exception sink to the active loop when available."""
    try:
        _install_loop_handler(loop or asyncio.get_running_loop())
        return True
    except RuntimeError:
        return False


def hook_omni_tracer():
    global _HOOKED
    if _HOOKED:
        return
    
    # 1. Sys excepthook
    sys.excepthook = _sys_excepthook
    
    # 2. Threading excepthook
    threading.excepthook = _threading_excepthook
    
    # 3. Asyncio
    install_asyncio_exception_handler()
        
    # 4. Global Logging
    root_logger = logging.getLogger()
    has_omni = any(isinstance(h, OmniLogHandler) for h in root_logger.handlers)
    if not has_omni:
        root_logger.addHandler(OmniLogHandler())
        
    _HOOKED = True
    write_trace("omni_tracer", "System", "Omni-Tracer Online. Hooks attached.")
