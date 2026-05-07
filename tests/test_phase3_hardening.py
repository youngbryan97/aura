import pytest
import logging
import asyncio
from core.utils.sanitizer import PIIFilter, sanitize_string
from core.resilience.resilience import SmartCircuitBreaker, PROMETHEUS_AVAILABLE

def test_pii_sanitization():
    fake_key = "sk-" + "1234567890abcdef1234567890"
    raw = (
        "My email is test@example.com and my server is at 192.168.1.1. "
        f"API_KEY='{fake_key}'"
    )
    sanitized = sanitize_string(raw)
    
    assert "test@example.com" not in sanitized
    assert "[EMAIL_REDACTED]" in sanitized
    assert "192.168.1.1" not in sanitized
    assert "[IP_REDACTED]" in sanitized
    assert fake_key not in sanitized
    assert "********" in sanitized
