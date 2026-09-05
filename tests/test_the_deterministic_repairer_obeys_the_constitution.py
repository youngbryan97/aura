"""A second way into Aura's own source that the seal did not cover.

The model-driven repair path earns its promotions: proposal validation, sandbox
evidence, a clean branch, a non-LLM promotion harness, architecture checks, a
repository parse, a ghost boot for core, backup and rollback, commit, merge.
It honours the mutation constitution throughout.

The deterministic repairer does not go through any of that. It scans the
repository broadly, rewrites source through the write gateway, checks the
result with py_compile, and consulted the constitution nowhere at all — while
both paths are reachable from the same RSI code-refinement action. So "critical
control-plane source is sealed from every self-modification path" was true of
one path and said of all of them.

The line drawn here is the constitution's own: the tiers it marks as applicable
automatically. Propose-only and sealed are refused whatever the repair would
have been, because a deterministic rewrite is still a rewrite.
"""

from __future__ import annotations

import pytest

from core.self_modification.mutation_tiers import (
    MutationTier,
    classify_mutation_path,
)
from core.self_modification.structural_improver import (
    _may_be_repaired_deterministically,
)

SEALED = [
    "core/will.py",
    "core/constitution.py",
    "core/governance_context.py",
    "core/security/permissions.py",
    "core/self_modification/mutation_tiers.py",
]


@pytest.mark.parametrize("where", SEALED)
def test_sealed_source_is_refused_here_too(where):
    allowed, why, _wanted = _may_be_repaired_deterministically(where)
    assert not allowed, f"{where} could be rewritten by the deterministic path"
    assert why, "it refused without saying why"


@pytest.mark.parametrize("where", SEALED)
def test_those_really_are_sealed_by_the_constitution(where):
    """The test above means nothing if these paths are not the sealed ones."""
    assert classify_mutation_path(where).tier is MutationTier.SEALED


def test_ordinary_source_is_still_repairable():
    """Closing the hole must not turn the repairer into a mechanism that cannot fire."""
    allowed, _why, _wanted = _may_be_repaired_deterministically(
        "core/utils/text_helpers.py"
    )
    assert allowed


def test_a_path_the_constitution_will_not_classify_is_not_permission():
    """Refusing to answer is not a yes."""
    import core.self_modification.structural_improver as improver

    def will_not_say(_path):
        raise ValueError("no")

    was = improver.classify_mutation_path
    improver.classify_mutation_path = will_not_say
    try:
        allowed, why, _wanted = _may_be_repaired_deterministically("core/anything.py")
    finally:
        improver.classify_mutation_path = was
    assert not allowed
    assert "classify" in why


def test_what_the_tier_asked_for_is_carried_back():
    """A compile standing in for a shadow validation should at least say so."""
    _allowed, _why, wanted = _may_be_repaired_deterministically(
        "core/brain/cognitive_engine.py"
    )
    assert isinstance(wanted, tuple)


def test_every_tier_is_decided_the_way_the_constitution_decides_it():
    """This must not become a second policy that drifts from the first."""
    for where in (
        "core/will.py",
        "core/config.py",
        "core/utils/text_helpers.py",
        "core/brain/cognitive_engine.py",
    ):
        decision = classify_mutation_path(where)
        allowed, _why, _wanted = _may_be_repaired_deterministically(where)
        assert allowed is decision.auto_apply_allowed, where
