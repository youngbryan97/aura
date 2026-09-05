"""There is one answer to "how is Aura", and the rest are views of it.

SelfObject, AuraNow.self_state, the identity engine, the continuity engine,
workspace ownership and the substrate's self-representation were six separate
answers to "who am I", each internally coherent, none of them the same, with
nothing deciding between them. The fix is not to delete five: each knows
something. It is that they read the authoritative value instead of forming
their own.

These tests hold that. A view that computes its own number is not a view.
"""

from __future__ import annotations

import pytest

from core.canonical.channels import Domain, in_domain
from core.canonical.state import get_canonical_state


@pytest.fixture
def canonical():
    state = get_canonical_state()
    state.clear()
    yield state
    state.clear()


def test_self_object_reports_the_canonical_self_not_its_own(canonical):
    from core.identity.self_object import SelfObject

    for producer in ("a", "b", "c"):
        canonical.estimate("self.continuity", 0.23, confidence=0.9, producer=producer)
    snapshot = SelfObject().snapshot()
    assert snapshot.self_channels["self.continuity"] == pytest.approx(0.23, abs=1e-6)


def test_a_defaulted_self_channel_says_it_is_a_default(canonical):
    """A neutral value is not a finding about her, and a reader must know."""
    from core.identity.self_object import SelfObject

    snapshot = SelfObject().snapshot()
    for spec in in_domain(Domain.SELF):
        assert snapshot.self_channels[f"{spec.id}.defaulted"] == 1.0


def test_self_object_affect_comes_from_the_fused_value_not_one_engine(canonical):
    """Reading the wheel directly puts one estimator's answer in the self-model."""
    from core.identity.self_object import SelfObject

    for producer in ("wheel", "interiority", "welfare"):
        canonical.estimate("affect.valence", -0.42, confidence=0.8, producer=producer)
    affect = SelfObject().snapshot().affect
    assert affect["affect.valence"] == pytest.approx(-0.42, abs=1e-6)


def test_the_continuity_hash_moves_when_the_self_state_moves(canonical):
    from core.identity.self_object import SelfObject

    obj = SelfObject()
    first = obj.snapshot().continuity_hash
    for producer in ("a", "b"):
        canonical.estimate("self.continuity", 0.05, confidence=0.9, producer=producer)
        canonical.estimate("affect.valence", -0.9, confidence=0.9, producer=producer)
    assert obj.snapshot().continuity_hash != first


def test_no_self_representation_recomputes_a_canonical_number():
    """The ratchet, asserted here so a regression fails a test and not only a gate."""
    import json
    import pathlib
    import subprocess
    import sys

    baseline = json.loads(
        pathlib.Path("config/state_ownership_baseline.json").read_text(encoding="utf-8")
    )
    result = subprocess.run(
        [sys.executable, "tools/check_state_ownership.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert baseline["count"] <= 5, (
        "the private-copy baseline went up; it may only shrink"
    )


def test_attentional_coherence_is_not_named_as_self_coherence():
    """Same word, different quantity, is the duplication that hides."""
    import pathlib

    source = pathlib.Path("core/consciousness/attention_schema.py").read_text(
        encoding="utf-8"
    )
    assert "self.topic_coherence" in source
    assert "self.coherence" not in source, (
        "attention schema holds an attribute named like the canonical "
        "self-coherence channel while measuring whether attention stayed on "
        "one subject"
    )
    # The published key stays, because readers take it by name and a
    # `.get("coherence", default)` would silently take the default.
    assert '"coherence": round(self.get_topic_coherence(), 3)' in source


def test_the_subsystems_that_answered_who_am_i_are_estimators_or_views():
    """Each of these either contributes to the canonical self or reads it."""
    import pathlib

    for path in (
        "core/identity/self_object.py",
        "core/consciousness/unified_field.py",
        "core/being/welfare_state.py",
        "core/interiority/service.py",
    ):
        source = pathlib.Path(path).read_text(encoding="utf-8")
        assert "core.canonical" in source, f"{path} still answers on its own"
