"""What is atomic about a tick, held against the code that decides it.

ARCHITECTURE.md called the tick "Aura's atomic unit of cognition" and said a
tick "commits the result atomically", two lines above an invariant that says
"Phase fails → tick continues". Both statements were written honestly and they
pull in opposite directions: the first invites a reader to expect transaction
semantics over the whole tick, the second describes a pipeline where phases 1
and 2 reach the eventual commit even when phase 3 fails.

The real boundary is the committed AuraState version. Effects outside that
object — tools, files, queued tasks, memory stores — already happened and have
no undo.

This file is the reason the corrected paragraph can be trusted: it holds the
prose and the code to each other, so the doc cannot drift back and the code
cannot quietly acquire (or lose) the guarantee the doc describes.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

_ARCHITECTURE = Path(__file__).resolve().parents[1] / "ARCHITECTURE.md"


@pytest.fixture(scope="module")
def architecture_text() -> str:
    return _ARCHITECTURE.read_text(encoding="utf-8")


def test_doc_scopes_atomicity_to_the_state_version(architecture_text: str) -> None:
    assert "The committed AuraState version is." in architecture_text
    assert "not \"the tick is a\ntransaction\"" in architecture_text or (
        "not \"the tick is a transaction\"" in architecture_text
    )


def test_doc_says_a_failing_phase_does_not_roll_back_its_predecessors(
    architecture_text: str,
) -> None:
    assert "does not roll back the phases before it" in architecture_text


def test_doc_says_effects_outside_state_are_not_covered(architecture_text: str) -> None:
    assert "Effects outside the state object" in architecture_text
    assert "no undo" in architecture_text


def test_doc_no_longer_claims_the_tick_is_the_atomic_unit(architecture_text: str) -> None:
    """The phrase may appear, but only as the thing being corrected.

    Checking for absence would be wrong: the paragraph that fixes this has to
    quote what it is fixing, or a later reader has no idea which reading was
    ruled out. So every occurrence must be inside quotation marks.
    """

    unquoted = [
        line
        for line in architecture_text.splitlines()
        if "atomic unit of cognition" in line and '"atomic unit of cognition"' not in line
    ]
    assert unquoted == [], (
        f"ARCHITECTURE.md asserts the tick is the atomic unit here: {unquoted}"
    )


# ── and now the code the prose describes ───────────────────────────────────


def test_a_failing_phase_does_not_abort_the_tick() -> None:
    """The behaviour the doc's second column describes.

    ``continue`` in the phase-failure handler is the whole claim: the tick
    keeps going, so earlier phases' transformations survive into the commit.
    """

    from core.kernel.aura_kernel import AuraKernel

    # The whole tick implementation. tick's body was extracted into
    # _tick_body, and this gate's own failure message anticipated exactly
    # that: "the handler moved and this gate needs to follow it".
    from tools.find_extraction_seam import implementation_source

    source = textwrap.dedent(implementation_source(AuraKernel, "tick"))
    tree = ast.parse(source)

    handlers_that_continue = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body_text = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        if "Continue" in body_text and "skipped phase" in ast.unparse(node):
            handlers_that_continue += 1
    assert handlers_that_continue >= 1, (
        "no exception handler in AuraKernel.tick continues past a failed phase; "
        "either the tick now aborts (and ARCHITECTURE.md §1 is wrong) or the "
        "handler moved and this gate needs to follow it"
    )


def test_the_commit_is_the_thing_that_is_all_or_nothing() -> None:
    """Constitutional admission, then a version guard, then one transaction."""

    from core.state.state_repository import StateRepository

    source = inspect.getsource(StateRepository._process_commit_transaction)
    assert "approve_state_mutation" in source, "commits are constitutionally admitted"
    assert "new_state.version <= current.version" in source, "stale writes are rejected"
    assert "return False" in source, "a refused commit publishes nothing"


def test_commit_refusal_does_not_raise_into_the_tick() -> None:
    """Invariant 3: vault commit failure is non-fatal.

    Which is also why the tick is not a transaction — a failed commit leaves
    the tick's external effects in place and returns a response anyway.
    """

    from core.state.state_repository import StateRepository

    source = inspect.getsource(StateRepository._process_commit_transaction)
    assert "raise" not in source.split("async with self.lock")[0], (
        "the pre-lock admission path must return False rather than raise"
    )


# ── the mixin surface ──────────────────────────────────────────────────────


#: RobustOrchestrator's base count when this gate landed. It only shrinks.
#:
#: The criticism is fair and the right response is not a rewrite: fifteen
#: mixins share an implicit attribute contract — a mixin can use
#: ``self.state_repo`` or ``self.reply_queue`` because some other mixin or the
#: boot path put it there — and that is harder to reason about than a typed
#: interface. But a mass extraction of fifteen behaviours out of the class that
#: serves every live turn is a large risk for a structural benefit, and the
#: class has already begun moving past the worst version of the design with
#: explicit AgencyCoordinator, MemoryCoordinator and AffectCoordinator members.
#:
#: So: stop adding to the inheritance surface. New behaviour becomes a
#: coordinator or a runtime service. Old mixins thin out over time.
_ORCHESTRATOR_MIXIN_CEILING = 15


def test_orchestrator_mixin_surface_only_shrinks() -> None:
    from core.orchestrator.main import RobustOrchestrator

    bases = [base for base in RobustOrchestrator.__bases__ if base is not object]
    assert len(bases) <= _ORCHESTRATOR_MIXIN_CEILING, (
        f"RobustOrchestrator now has {len(bases)} bases. New behaviour goes in a "
        "typed coordinator or a runtime service, not another mixin — the "
        "implicit shared-attribute contract is what makes this class hard to "
        "reason about, and every base widens it."
    )


def test_orchestrator_already_owns_typed_coordinators() -> None:
    """The direction of travel, asserted so it cannot quietly reverse."""

    import core.orchestrator.main as orchestrator_module

    source = inspect.getsource(orchestrator_module)
    for coordinator in ("AgencyCoordinator", "MemoryCoordinator", "AffectCoordinator"):
        assert coordinator in source, (
            f"{coordinator} is how behaviour is supposed to arrive now; if it "
            "has gone, the mixin ceiling above is guarding the wrong thing"
        )
