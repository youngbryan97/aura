import atexit
import json
import logging
import logging.handlers
import os
import queue
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from re import Pattern
from typing import Any

import structlog
from structlog.dev import ConsoleRenderer

from core.runtime.state_ownership import state_root

# ── Redaction Patterns ─────────────────────────────────────────

_REDACT_PATTERNS: list[tuple[Pattern[str], str]] = [
    (re.compile(r'(sk-[A-Za-z0-9\-_]{20,})', re.IGNORECASE), "[REDACTED_API_KEY]"),
    (re.compile(r'(Bearer\s+)[A-Za-z0-9\-_\.=]{10,}', re.IGNORECASE), r"\1[REDACTED_BEARER]"),
    (re.compile(r'(password["\s:=]+)[^\s"\']+', re.IGNORECASE), r"\1[REDACTED_PASS]"),
    (re.compile(r'(token["\s:=]+)[^\s"\']+', re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
]

def redact_text(text: str) -> str:
    """Apply every redaction pattern to a rendered log line."""
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def _redact_processor(_: Any, __: Any, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to redact sensitive patterns in the event dict."""
    for key, value in event_dict.items():
        if isinstance(value, str):
            for pattern, replacement in _REDACT_PATTERNS:
                event_dict[key] = pattern.sub(replacement, event_dict[key])
    return event_dict


class JsonLineFormatter(logging.Formatter):
    """Render every record — structlog or plain stdlib — as one redacted JSON object per line.

    Structlog events arrive pre-rendered as JSON strings and pass through with
    logger/level/timestamp back-filled; anything else (third-party libraries,
    bare ``logging`` calls) is wrapped in the same envelope so the file sink
    stays machine-parseable end to end.
    """

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = str(record.msg)
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        payload: dict[str, Any] | None = None
        if message.startswith("{"):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    payload = parsed
            except ValueError:
                payload = None
        if payload is None:
            payload = {"event": message}
        payload.setdefault("logger", record.name)
        payload.setdefault("level", record.levelname.lower())
        payload.setdefault("timestamp", timestamp)
        if record.exc_text:
            payload.setdefault("exc_info", record.exc_text)

        try:
            line = json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            line = json.dumps({
                "event": message, "logger": record.name,
                "level": record.levelname.lower(), "timestamp": timestamp,
            }, ensure_ascii=False, default=str)
        return redact_text(line)

# ── Non-blocking log transport ───────────────────────────────
#
# Every slow sink (stdout pipe to the launcher, rotating JSON file) lives
# behind ONE QueueListener daemon thread. Emitting a record from any thread —
# including the event loop — is a lock-free put_nowait. This is the fix for a
# whole class of observed live stalls: a logger.info inside the resilience
# engine blocked the event loop for 6s because the file/stdout write stalled
# under disk contention. Logging must never be able to stall the organism.

_LOG_QUEUE_CAPACITY = 20_000
_dropped_log_records = 0


class _DropNewestOnOverflowQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that never blocks the emitting thread.

    On overflow the OLDEST queued record is discarded to make room for the
    newest one (the record being logged right now is the one describing the
    current incident — it matters most). Drops are counted, never silent.
    """

    def enqueue(self, record: logging.LogRecord) -> None:
        global _dropped_log_records
        try:
            self.queue.put_nowait(record)
            return
        except queue.Full:
            pass
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        _dropped_log_records += 1
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass


def get_dropped_log_count() -> int:
    """Records discarded under log-queue overflow (telemetry surface)."""
    return _dropped_log_records


def _stop_queue_listener() -> None:
    """Flush queued records into the real sinks at interpreter exit."""
    global _queue_listener
    listener = _queue_listener
    _queue_listener = None
    if listener is not None:
        try:
            listener.stop()
        except (RuntimeError, ValueError):
            pass  # interpreter teardown: sinks may already be closed


# ── Main Entry-Point ─────────────────────────────────────────

_initialised: bool = False
_queue_listener: logging.handlers.QueueListener | None = None


def _resolve_log_dir(log_dir: Path | None) -> Path:
    """Explicit argument wins, then AURA_LOG_DIR (test/CI hermeticity), then ~/.aura/logs."""
    if log_dir is not None:
        return Path(log_dir)
    env_log_dir = os.environ.get("AURA_LOG_DIR")
    if env_log_dir:
        return Path(env_log_dir)
    return state_root() / "logs"

def setup_logging(
    name: str = "Aura",
    level: str | int = logging.INFO,
    log_dir: Path | None = None,
    max_bytes: int = 100 * 1024 * 1024, # 100MB
    backup_count: int = 10,
) -> Any:
    """Configure structured logging and return a bound logger."""
    global _initialised, _queue_listener

    if _initialised:
        return structlog.get_logger(name)

    # 1. Stdlib handlers for local file backup (structured JSON)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_dir = _resolve_log_dir(log_dir)

    file_handler = None
    for candidate in (Path(log_dir), Path(tempfile.gettempdir()) / "aura-logs"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                candidate / "aura_json.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            break
        except OSError:
            continue

    if file_handler is not None:
        file_handler.setFormatter(JsonLineFormatter())
        handlers.append(file_handler)

    # 2. Configure stdlib logging bridge
    root_logger = logging.getLogger()

    # Remove only handlers this function would duplicate (stdout/stderr
    # console handlers and a previous aura_json.log file handler). A blanket
    # clear also destroyed FOREIGN handlers — pytest's log-capture handler,
    # host-app handlers — whenever the first Aura import happened mid-run,
    # which surfaced as order-dependent test failures and silent log loss.
    for handler in list(root_logger.handlers):
        stream = getattr(handler, "stream", None)
        is_own_console = (
            type(handler) is logging.StreamHandler
            and stream in (sys.stdout, sys.stderr)
        )
        is_stale_aura_file = (
            isinstance(handler, logging.handlers.RotatingFileHandler)
            and Path(getattr(handler, "baseFilename", "") or "").name == "aura_json.log"
        )
        is_stale_queue = isinstance(handler, _DropNewestOnOverflowQueueHandler)
        if is_own_console or is_stale_aura_file or is_stale_queue:
            root_logger.removeHandler(handler)

    # All slow sinks live behind one QueueListener daemon: emitting a log
    # record from ANY thread — including the event loop — is a non-blocking
    # put_nowait. A stalled stdout pipe or a contended disk can no longer
    # stall the caller (observed live: 6s event-loop stall inside a
    # logger.info when the file sink blocked).
    log_queue: queue.Queue = queue.Queue(maxsize=_LOG_QUEUE_CAPACITY)
    queue_handler = _DropNewestOnOverflowQueueHandler(log_queue)
    _queue_listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    _queue_listener.start()
    atexit.register(_stop_queue_listener)

    root_logger.addHandler(queue_handler)
    root_logger.setLevel(level)

    # 3. Structlog configuration
    from core.config import Environment, config
    
    # Zenith HUD consumes JSON, but developers prefer human-readable console output
    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    # Force JSON if explicitly requested or if we are in a production/silent environment
    if os.environ.get("AURA_LOG_JSON") == "1":
        renderer = structlog.processors.JSONRenderer()
    elif config.env == Environment.DEV and is_tty:
        renderer = ConsoleRenderer(colors=True)
    elif is_tty:
        renderer = ConsoleRenderer(colors=False) # Human-readable but no escape codes
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_processor,
            renderer
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Silence noisy libs
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if file_handler is None:
        logging.getLogger("Aura.Logging").warning(
            "File logging unavailable; continuing with stdout-only logging."
        )

    _initialised = True
    return structlog.get_logger(name)

def get_logger(name: str) -> Any:
    """Return a module-level bound logger."""
    return structlog.get_logger(name)
