"""One warning killed the resident model and then every tool she has.

LIVE 2026-08-19, one turn, read from the neural stream in order:

  inference_gate CRITICAL: nonlocal text from 'resident-cortex-recovery'
      published without provider attribution (error=none)
  llm_health_router: CRITICAL SERVICE FAILURE — subsystem 'inference_gate'
      failed with failure policy 'fail-closed'
  RECOVERY: primary 32B cortex is dead, background respawn 1/5
  deliberate_action: her reasoning produced nothing
  frustration 0.47, 0.61, 0.74, 0.87
  CRITICAL EXISTENTIAL STAKES — threat 0.84
  WILL REFUSED: tool_execution — survival_inhibition
  ActionExecutor refused host_automation.ensure_screenshot_directory

Nothing was wrong. The recovery path writes its own text and therefore has
no provider to attribute it to; refusing to call that a verified success is
the check working. Recording a degradation on top of it made a designed
condition into a fault, and inference_gate is fail-closed, so a warning
became a critical service failure.

A provider that answered and did not attribute itself is a different thing,
and is still reported.
"""
from __future__ import annotations

import inspect

from core.brain.inference_gate import InferenceGate


def _finalizer_source() -> str:
    return inspect.getsource(InferenceGate._finalize_nonlocal_user_facing_text)


def test_text_we_wrote_ourselves_is_not_reported_as_a_provider_failure():
    source = _finalizer_source()
    assert "wrote_it_ourselves" in source
    assert "not wrote_it_ourselves" in source


def test_it_is_still_refused_as_a_verified_success():
    """The check's actual job survives: unattributed text is not a success."""
    source = _finalizer_source()
    assert "success = bool(cleaned) and not provider_error and attributed" in source
    where = source.index("if success:")
    assert "_record_user_generation_endpoint" in source[where : where + 300]


def test_a_provider_that_failed_to_attribute_itself_is_still_reported():
    source = _finalizer_source()
    where = source.index("nonlocal text from")
    guard = source[max(0, where - 400) : where]
    assert "not wrote_it_ourselves" in guard
    assert "generation_metadata is None" in source


def test_a_required_subsystem_is_fail_closed_so_a_warning_there_is_never_cosmetic():
    """Why the severity mattered rather than being cosmetic.

    A required service carries the fail-closed policy, which is what turned
    an informational warning inside inference_gate into CRITICAL SERVICE
    FAILURE and took the resident model down with it.
    """
    from core.container import ServiceDescriptor

    required = ServiceDescriptor(name="inference_gate", factory=lambda: None, required=True)
    optional = ServiceDescriptor(name="something_optional", factory=lambda: None, required=False)
    assert required.failure_policy == "fail-closed"
    assert optional.failure_policy == "degrade_with_receipt"


def test_a_worker_that_is_already_gone_is_not_a_gate_failure():
    """LIVE, mid-game: "ValueError: process object is closed" during idle
    client recycling became CRITICAL SERVICE FAILURE and took the inference
    gate's maintenance loop down.

    The handle refers to a process that has exited, which is the outcome the
    recycling pass is trying to reach — not a failure to reach it.
    """
    source = inspect.getsource(InferenceGate)
    where = source.index("continued recycling other idle local clients")
    guard = source[max(0, where - 900) : where]
    assert "process object is closed" in guard
    assert "continue" in guard, "an already-dead worker must not be recorded as a fault"


def test_deciding_without_language_is_not_recorded_as_a_degradation():
    """LIVE: 31 of them in half an hour opened a runtime concern incident
    about deliberate_action while she played perfectly well.

    Deciding without language is the designed path — the whole point of being
    able to act while the model reloads.
    """
    from core.agency import deliberate_action

    source = inspect.getsource(deliberate_action.deliberate)
    where = source.index("spoke = False")
    guard = source[max(0, where - 700) : where]
    assert "record_degradation(" not in guard, "a working fallback is being counted as a fault"
    assert "logger.info(" in guard, "and it should still be visible"
