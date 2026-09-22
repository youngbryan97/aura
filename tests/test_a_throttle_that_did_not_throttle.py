"""A cap that did not apply must not be announced as if it had.

LIVE, 2026-09-21, on a turn asking how far the bird flies:

    🧠 [ANSWER BUDGET] 258 tokens fit this turn's clock at the measured rate
    🔥 CRITICAL METABOLIC PANIC: Arousal=0.11, RAM=71.3%, CPU=99.6%,
       GovThrottle=1.00. Parameter throttle ENABLED (max_tokens capped at 128).

Nothing was capped. ``_cap`` returns early for a foreground turn — which is
right, a person's turn keeps the budget it was priced for — and the three
log lines announced the cap regardless. The reply came back empty for its
own reasons and this line was the first place to look, which cost the time
it takes to read a throttle that had not fired.
"""

from __future__ import annotations

import logging

from core.brain.llm import somatic_throttle


def _options():
    return {"max_tokens": 258, "temperature": 0.7}


class _Loaded:
    """A host reported as critically loaded, whatever this one is doing."""

    @staticmethod
    def cpu_percent(interval=0):
        return 99.0

    @staticmethod
    def virtual_memory():
        class _M:
            percent = 99.0

        return _M()


def test_a_foreground_turn_keeps_its_budget(monkeypatch):
    monkeypatch.setattr(somatic_throttle, "psutil", _Loaded())
    options = somatic_throttle.SomaticComputeSentinel().adjust_generation_options(
        _options(), foreground=True
    )
    assert options["max_tokens"] == 258, "a person's turn keeps what it was priced at"


def test_the_line_does_not_claim_a_cap_it_did_not_apply(caplog, monkeypatch):
    monkeypatch.setattr(somatic_throttle, "psutil", _Loaded())
    with caplog.at_level(logging.INFO, logger=somatic_throttle.logger.name):
        somatic_throttle.SomaticComputeSentinel().adjust_generation_options(
            _options(), foreground=True
        )
    said = "\n".join(record.getMessage() for record in caplog.records)
    assert "capped at" not in said, (
        "a foreground turn is never capped, so no line may say it was:\n" + said
    )


def test_a_background_turn_is_capped_and_says_so(caplog, monkeypatch):
    monkeypatch.setattr(somatic_throttle, "psutil", _Loaded())
    with caplog.at_level(logging.INFO, logger=somatic_throttle.logger.name):
        options = somatic_throttle.SomaticComputeSentinel().adjust_generation_options(
            _options(), foreground=False
        )
    assert options["max_tokens"] < 258, "background work is capped under real pressure"
    said = "\n".join(record.getMessage() for record in caplog.records)
    assert "capped at" in said or "Sampling capped" in said, (
        "a cap that applied must be named:\n" + said
    )
