"""Logging integrity: the JSON file sink must stay machine-parseable and redacted.

Covers the three guarantees added in the logging-integrity pass:
1. Every record written through JsonLineFormatter is one valid JSON object per
   line — including plain stdlib records from third-party libraries.
2. Redaction applies to every sink path, not just structlog events.
3. Log-dir resolution honors AURA_LOG_DIR so test runs never pollute the live
   ~/.aura/logs of a running instance.
"""
import json
import logging
import time
from pathlib import Path

import pytest

from core.observability.logging_config import JsonLineFormatter, _resolve_log_dir, redact_text


def _record(msg: str, *, level: int = logging.INFO, name: str = "test.logger",
            exc_info=None, args=()) -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg=msg, args=args, exc_info=exc_info,
    )


class TestJsonLineFormatter:
    def test_plain_stdlib_record_becomes_json_envelope(self):
        line = JsonLineFormatter().format(_record("Belief Updated: AURA_SELF"))
        payload = json.loads(line)
        assert payload["event"] == "Belief Updated: AURA_SELF"
        assert payload["logger"] == "test.logger"
        assert payload["level"] == "info"
        assert payload["timestamp"].endswith("+00:00") or "T" in payload["timestamp"]

    def test_percent_args_are_interpolated(self):
        line = JsonLineFormatter().format(_record("loaded %d beliefs", args=(8,)))
        assert json.loads(line)["event"] == "loaded 8 beliefs"

    def test_structlog_json_line_passes_through_with_backfill(self):
        inner = json.dumps({"event": "will decision", "receipt_id": "r-123"})
        line = JsonLineFormatter().format(_record(inner, level=logging.WARNING))
        payload = json.loads(line)
        assert payload["event"] == "will decision"
        assert payload["receipt_id"] == "r-123"
        # Backfilled from the stdlib record without clobbering existing keys.
        assert payload["level"] == "warning"
        assert payload["logger"] == "test.logger"

    def test_existing_level_key_is_not_clobbered(self):
        inner = json.dumps({"event": "x", "level": "error"})
        line = JsonLineFormatter().format(_record(inner, level=logging.INFO))
        assert json.loads(line)["level"] == "error"

    def test_non_dict_json_message_is_wrapped(self):
        line = JsonLineFormatter().format(_record('{"not closed'))
        payload = json.loads(line)
        assert payload["event"] == '{"not closed'

    def test_exception_info_is_captured(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = _record("failed", level=logging.ERROR, exc_info=sys.exc_info())
        payload = json.loads(JsonLineFormatter().format(record))
        assert "ValueError: boom" in payload["exc_info"]

    def test_stdlib_record_is_redacted(self):
        secret = "sk-" + "a" * 24
        line = JsonLineFormatter().format(_record(f"calling api with {secret}"))
        assert secret not in line
        assert "[REDACTED_API_KEY]" in line

    def test_structlog_payload_values_are_redacted(self):
        inner = json.dumps({"event": "auth", "header": "Bearer abcdefghijklmnop"})
        line = JsonLineFormatter().format(_record(inner))
        assert "abcdefghijklmnop" not in line
        assert "[REDACTED_BEARER]" in line

    def test_every_output_line_parses_as_json(self):
        formatter = JsonLineFormatter()
        for msg in ("plain text", '{"event": "structured"}', "{broken json",
                    "unicode: résumé 🧠", ""):
            json.loads(formatter.format(_record(msg)))


class TestRedactText:
    @pytest.mark.parametrize("raw,marker", [
        ("key sk-" + "x" * 30, "[REDACTED_API_KEY]"),
        ("Authorization: Bearer 0123456789abcdef", "[REDACTED_BEARER]"),
        ('password: "hunter2secret"', "[REDACTED_PASS]"),
        ('token="deadbeefcafe"', "[REDACTED_TOKEN]"),
    ])
    def test_patterns(self, raw, marker):
        assert marker in redact_text(raw)


class TestLogDirResolution:
    def test_explicit_argument_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "env"))
        assert _resolve_log_dir(tmp_path / "explicit") == tmp_path / "explicit"

    def test_env_override_used_when_no_argument(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "env"))
        assert _resolve_log_dir(None) == tmp_path / "env"

    def test_default_is_this_runtimes_own_log_dir(self, monkeypatch):
        """Logs land under state_root(), not a hard-coded ~/.aura.

        This asserted ``Path.home() / ".aura" / "logs"`` from before runtime
        profiles existed. A TEST runtime resolves to ``.aura-test``, so the old
        assertion demanded that a test run default its logs into the LIVE
        instance's directory — the exact thing state_ownership.py exists to
        prevent, and the thing the sibling test below checks conftest avoids.

        state_root() is the same ~/.aura/logs for a live runtime, so this keeps
        the original guarantee while no longer requiring the wrong one.
        """
        from core.runtime.state_ownership import state_root

        monkeypatch.delenv("AURA_LOG_DIR", raising=False)
        assert _resolve_log_dir(None) == state_root() / "logs"

    def test_conftest_redirects_test_logs_away_from_live_home(self):
        import os
        configured = os.environ.get("AURA_LOG_DIR", "")
        assert configured, "conftest must set AURA_LOG_DIR for suite hermeticity"
        live = Path.home() / ".aura" / "logs"
        assert Path(configured).resolve() != live.resolve()

class TestQueueHandlerOverflow:
    """The UI broadcast buffer alarm must fire only for warning+ rotations."""

    @pytest.fixture()
    def server_module(self):
        pytest.importorskip("fastapi")
        import interface.server as server
        return server

    @pytest.fixture()
    def handler(self, server_module):
        h = server_module._QueueHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        # Reset per-instance counters (class attrs are shadowed on first write).
        h._dropped_count = 0
        h._dropped_warn_count = 0
        h._last_reported_warn_drops = 0
        h._last_overflow_warning_at = 0.0
        return h

    @pytest.fixture()
    def isolated_queue(self, server_module):
        from interface.websocket_manager import log_queue
        saved = list(log_queue)
        log_queue.clear()
        yield log_queue
        log_queue.clear()
        log_queue.extend(saved)

    def _fill(self, queue, level: str):
        while len(queue) < queue.maxlen:
            queue.append({"type": "log", "message": "x", "level": level,
                          "timestamp": time.time(), "module": "t"})

    def test_info_rotation_does_not_alarm(self, handler, isolated_queue, caplog):
        self._fill(isolated_queue, "info")
        with caplog.at_level(logging.WARNING, logger="Aura.Server"):
            handler.emit(_record("new warning arrives", level=logging.WARNING))
        assert not [r for r in caplog.records if "UI log buffer" in r.getMessage()]
        assert handler._dropped_count == 1
        assert handler._dropped_warn_count == 0

    def test_warning_rotation_alarms_with_accurate_delta(self, handler, isolated_queue, caplog):
        self._fill(isolated_queue, "warning")
        with caplog.at_level(logging.WARNING, logger="Aura.Server"):
            handler.emit(_record("any record", level=logging.INFO))
        alarms = [r for r in caplog.records if "UI log buffer" in r.getMessage()]
        assert len(alarms) == 1
        assert "rotated out 1 warning+ records" in alarms[0].getMessage()

    def test_alarm_throttled_to_once_per_minute(self, handler, isolated_queue, caplog):
        self._fill(isolated_queue, "error")
        with caplog.at_level(logging.WARNING, logger="Aura.Server"):
            for _ in range(5):
                handler.emit(_record("r", level=logging.INFO))
        alarms = [r for r in caplog.records if "UI log buffer" in r.getMessage()]
        assert len(alarms) == 1
        assert handler._dropped_warn_count == 5


class TestATokenIsAValueNotAWord:
    """LIVE 2026-09-15: 1,708 lines of one log carried [REDACTED_TOKEN] and none
    of them held a secret. The rule keyed on the space after "token", so the
    ceiling in "first-token ceiling 90.0s" and the word after a claim named
    "..._ruled_out_token:" were both redacted."""

    @pytest.mark.parametrize("raw", [
        "⏱️ [MLX] first-token ceiling 90.0s for a 5-char prompt (1 max tokens)",
        "verifier [warning] claims.x @ endogenous_bias_cannot_promote_a_ruled_out_token: trained on 3 seeds",
        "the token budget is 4096 tokens",
        "token count 512",
    ])
    def test_prose_around_the_word_token_is_left_alone(self, raw):
        assert redact_text(raw) == raw

    @pytest.mark.parametrize("raw", [
        "Capability: Generated token 3f2a9b1c for tools: ['think']",
        "access_token=eyJhbGciOiJIUzI1NiJ9.abc",
        'token: "abcdefghij12345678"',
    ])
    def test_a_token_value_is_still_redacted(self, raw):
        assert "[REDACTED_TOKEN]" in redact_text(raw)
