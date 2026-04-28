import logging
import logging.handlers
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Pattern, Union, Optional
import structlog
from structlog.dev import ConsoleRenderer

# ── Redaction Patterns ─────────────────────────────────────────

_REDACT_PATTERNS: list[tuple[Pattern[str], str]] = [
    (re.compile(r'(sk-[A-Za-z0-9\-_]{20,})', re.IGNORECASE), "[REDACTED_API_KEY]"),
    (re.compile(r'(Bearer\s+)[A-Za-z0-9\-_\.=]{10,}', re.IGNORECASE), r"\1[REDACTED_BEARER]"),
    (re.compile(r'(password["\s:=]+)[^\s"\']+', re.IGNORECASE), r"\1[REDACTED_PASS]"),
    (re.compile(r'(token["\s:=]+)[^\s"\']+', re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
]

def _redact_processor(_: Any, __: Any, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to redact sensitive patterns in the event dict."""
    for key, value in event_dict.items():
        if isinstance(value, str):
            for pattern, replacement in _REDACT_PATTERNS:
                event_dict[key] = pattern.sub(replacement, event_dict[key])
    return event_dict

# ── Main Entry-Point ─────────────────────────────────────────

_initialised: bool = False

def setup_logging(
    name: str = "Aura",
    level: Union[str, int] = logging.INFO,
    log_dir: Optional[Path] = None,
    max_bytes: int = 100 * 1024 * 1024, # 100MB
    backup_count: int = 10,
) -> Any:
    """Configure structured logging and return a bound logger."""
    global _initialised
    
    if _initialised:
        return structlog.get_logger(name)

    # 1. Stdlib handlers for local file backup (structured JSON)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if log_dir is None:
        log_dir = Path.home() / ".aura" / "logs"
    
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
        handlers.append(file_handler)

    # 2. Configure stdlib logging bridge
    root_logger = logging.getLogger()
    
    # If handlers already exist, we might be in a re-init or partial init.
    # Clear existing handlers to ensure our configuration is the single source of truth.
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    # Use basicConfig or manual handler addition to root
    for h in handlers:
        root_logger.addHandler(h)
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
