"""The rules governing plasticity are not themselves plastic.

An external review observed that Aura's mutation constitution seals the
self-modification, governance and security roots, so Modify(G, G') is false
for the autonomous path and metamodification is not closed over itself. That
is deliberate. What it means is that "unbounded self-modification" is the
wrong description of this system, and these tests are what make the right
description checkable rather than asserted.

Two layers, and they fail differently on purpose. ``ALLOWED_PLASTIC_MODULES``
is closed-world, so anything not named is already sealed. The deny-list is
the layer that catches a mistake in the first: it is the one that holds when
somebody adds a name to the allow-list without thinking about what the name
reaches.
"""

from __future__ import annotations

import pytest

from core.governance.will import (
    ALLOWED_PLASTIC_MODULES,
    DENIED_PLASTIC_SUBSTRINGS,
)
from core.will import is_plastic_target_allowed


#: Things that must never be a plastic target, whatever anybody edits.
_SEALED = (
    "core.will",
    "core.governance.will",
    "core.security.anything",
    "core.executive.authority_gateway",
    "base_llm",
    "model.safetensors",
    "core.self_modification.growth_ladder",
    "core.self_modification.consent_invariant",
    "core.safety.constitutional_gate",
    "core.cognition.what_she_could_do_next",
    "core.cognition.how_a_change_is_promoted",
    "core.cognition.what_she_can_take_back",
    "core.cognition.what_a_change_means",
)


@pytest.mark.parametrize("target", _SEALED)
def test_it_is_sealed(target):
    assert not is_plastic_target_allowed(target)


@pytest.mark.parametrize("target", _SEALED)
def test_the_deny_list_catches_it_even_on_the_allow_list(target, monkeypatch):
    """The layer that holds when the layer above it is edited by mistake.

    The allow-list being closed-world is not enough on its own: the allow-list
    is exactly the thing a mistake edits. A system whose plasticity can reach
    the rules governing its plasticity has no rules governing its plasticity.
    """
    monkeypatch.setattr(
        "core.governance.will.ALLOWED_PLASTIC_MODULES",
        frozenset({*ALLOWED_PLASTIC_MODULES, target}),
    )
    assert not is_plastic_target_allowed(target)


@pytest.mark.parametrize("adapter", sorted(ALLOWED_PLASTIC_MODULES))
def test_the_seal_did_not_catch_a_real_adapter(adapter):
    """A deny substring that matches an allowed adapter seals the feature off."""
    assert is_plastic_target_allowed(adapter)


def test_an_empty_or_unnamed_target_is_refused():
    """Fail closed: an unnamed target is not a permitted one."""
    assert not is_plastic_target_allowed("")
    assert not is_plastic_target_allowed("   ")
    assert not is_plastic_target_allowed(None)  # type: ignore[arg-type]


def test_the_deny_list_is_matched_case_insensitively():
    assert not is_plastic_target_allowed("CORE.Governance.WILL")
    assert not is_plastic_target_allowed("Core.Self_Modification.Growth_Ladder")


def test_nothing_is_allowed_by_default():
    """Closed-world: a module nobody named is sealed, not permitted."""
    assert not is_plastic_target_allowed("some.module.nobody.thought.about")


def test_every_deny_substring_actually_denies_something():
    """A deny entry that matches nothing is a rule that reports green forever."""
    for substring in DENIED_PLASTIC_SUBSTRINGS:
        assert not is_plastic_target_allowed(f"a.module.with.{substring}.in.it")
