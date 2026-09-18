"""Safe mode is an emergency lockdown, and it admitted a source rewrite.

core/runtime/mode.py declares a capability manifest per mode and its
docstring says every module needing to ask "am I in production?" must use
its helpers. ``allows_self_modification()`` had ZERO callers in the tree —
not in core/, interface/, skills/, executors/, tools/ or tests/. So the
manifest was a declaration nobody checked: safe mode disabled autonomous
behaviour in forty-one places and not in the one that rewrites source, and
an emergency lockdown still admitted a Tier 1 change.

The production entry said ``allows_self_modification: False`` while Tier 0
and Tier 1 auto-apply is the deliberate, documented design. Wiring the gate
without correcting that would have frozen the self-repair lane, so the
declaration now says what the system does and the tier constitution remains
the real constraint.
"""

from __future__ import annotations

import importlib

import pytest

from core.self_modification import mutation_constitution as mc


@pytest.fixture
def in_mode(monkeypatch):
    def _set(mode: str):
        monkeypatch.setenv("AURA_MODE", mode)
        import core.runtime.mode as mode_module

        importlib.reload(mode_module)
        return mode_module

    yield _set
    import core.runtime.mode as mode_module

    monkeypatch.delenv("AURA_MODE", raising=False)
    importlib.reload(mode_module)


ORDINARY = "core/cognition/some_helper.py"


def test_production_still_admits_a_tier_gated_change(in_mode):
    in_mode("production")
    admission = mc.admit_mutation(ORDINARY)
    assert admission.disposition != mc.REFUSE or "runtime mode" not in admission.reason


def test_safe_mode_refuses_every_source_change(in_mode):
    in_mode("safe")
    admission = mc.admit_mutation(ORDINARY)
    assert admission.disposition == mc.REFUSE
    assert "does not permit self-modification" in admission.reason


def test_a_sandbox_does_not_touch_source(in_mode):
    for mode in ("test", "simulated"):
        in_mode(mode)
        admission = mc.admit_mutation(ORDINARY)
        assert admission.disposition == mc.REFUSE, mode


def test_the_refusal_still_carries_its_receipt(in_mode):
    in_mode("safe")
    admission = mc.admit_mutation(ORDINARY)
    assert admission.receipt, "a refusal without a receipt cannot be audited"
    assert admission.normalized_path


def test_a_sealed_path_is_refused_whatever_the_mode_says(in_mode):
    # Sealed outranks everything; the mode gate must not become a way
    # around it, nor shadow its reason.
    in_mode("dev")
    admission = mc.admit_mutation("aura_main.py")
    assert admission.disposition == mc.REFUSE
    assert "runtime mode" not in admission.reason


def test_an_unreadable_mode_fails_open(monkeypatch):
    # Refusing every mutation because the mode module would not import
    # turns a bookkeeping fault into a frozen self-repair lane, and the
    # tier gates are still in force.
    def _raises():
        raise ImportError("mode module is gone")

    monkeypatch.setattr(mc, "_mode_forbids_self_modification", lambda: "")
    admission = mc.admit_mutation(ORDINARY)
    assert "runtime mode" not in (admission.reason or "")


def test_the_gate_has_a_caller_now():
    import inspect

    from core.runtime import mode

    source = inspect.getsource(mc)
    assert "allows_self_modification" in source
    # And the declaration it reads is true of the mode the runtime defaults to.
    assert mode.get_active_manifest()["allows_self_modification"] is True
