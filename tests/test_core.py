################################################################################

"""
tests/test_core.py
──────────────────
Verify core infrastructure: MemoryEvent, Logging redaction.
"""

import logging
import time
from core.memory.base import MemoryEvent
from core.logging_config import _redact_processor

def test_memory_event():
    # Defaults
    e = MemoryEvent(event_type="test")
    assert e.timestamp > 0
    assert e.cost == 0.0
    assert e.metadata == {}
    
    # Custom
    t = time.time()
    e2 = MemoryEvent("chat", timestamp=t, goal="reply", outcome={"ok": True}, cost=0.5)
    assert e2.is_failure is False
    assert e2.to_dict()['t'] == t

def test_logging_redaction():
    fake_key = "sk-" + "1234567890abcdef12345"
    # Mock structlog event
    event = {
        "event": f"My secret {fake_key} is here",
        "nested": "Bearer abcdef1234567890 token"
    }
    
    processed = _redact_processor(None, None, event)
    
    assert fake_key not in processed["event"]
    assert "[REDACTED_API_KEY]" in processed["event"]
    assert "Bearer [REDACTED_BEARER]" in processed["nested"]
    assert "abcdef1234567890" not in processed["nested"]


##
