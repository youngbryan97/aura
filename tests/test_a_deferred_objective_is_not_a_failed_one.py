"""A background objective held back for admission is deferred, not failed.

LIVE 2026-09-16, 05:27 and 05:32 UTC and every five minutes before: the
narrative journal asked the engine to write while a chat turn held the
foreground. The response phase suppressed it (foreground_generation_active),
the thinking loop then charged the objective with friction as if it had been
tried and had failed, and after a few rounds "objective repeatedly unresolved:
You are writing Aura" was a degradation, a fault, frustration=1.00 and
depletion=1.00 in the resilience engine, and a metabolism throttle — off a
journal entry that was never attempted.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_response_phase_marks_a_suppression_on_the_state() -> None:
    source = (ROOT / "core/phases/response_generation.py").read_text(encoding="utf-8")
    at = source.index("suppressing background objective for origin=%s (%s).")
    window = source[at : at + 900]
    assert 'state.response_modifiers["background_suppression"] = str(reason)' in window
    assert window.index('["background_suppression"]') < window.index("return state")


def test_the_thinking_loop_reads_the_mark_before_it_charges_friction() -> None:
    from source_support import inlined_function_source

    loop = inlined_function_source(
        ROOT / "core/brain/cognitive_engine_thinking_loop.py", "_RunsTheThinkingLoop._run_thinking_loop"
    )
    read = loop.index('response_modifiers.get("background_suppression")')
    charged = loop.index("experience_friction(friction_key, 0.45)")
    assert read < charged, "friction was charged before the deferral was read"
    deferred = loop[read : read + 700]
    assert '_empty_thought(mode, "background_deferred")' in deferred
    assert not re.search(r"record_degradation\(", deferred), "a deferral is not a degradation"
