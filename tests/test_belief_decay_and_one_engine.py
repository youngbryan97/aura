"""Beliefs age, and there is exactly one engine that holds them.

Two problems, one root. Aura had two belief-revision engines:
core/epistemics/belief_revision.py, started at boot and registered as the
`belief_revision_engine` service, and core/cognition/belief_revision.py, which
exposed get_belief_engine() and was reachable from nothing.

The unreachable one declared four key behaviours and implemented one — its
class docstring claimed "contradictions are detected via semantic similarity +
LLM judgment" with no such code in the file. The canonical one genuinely lacked
only decay: it recorded ``last_updated`` on every belief and never read it, so
a belief formed once from a passing remark was asserted at its original
confidence indefinitely.

So decay went into the engine that runs, and the duplicate was retired.
"""

from __future__ import annotations

import time

import pytest

from core.epistemics.belief_revision import Belief, BeliefRevisionEngine
from tests.source_contract import family_text_at

pytestmark = pytest.mark.unit

DAY = 86400.0


@pytest.fixture
def engine(tmp_path) -> BeliefRevisionEngine:
    return BeliefRevisionEngine(db_path=str(tmp_path / "beliefs.json"))


def _belief(engine, content, confidence=0.9, evidence=1.0, source="conversation", age=0.0):
    b = Belief(
        id=content.replace(" ", "_"),
        content=content,
        confidence=confidence,
        domain="world",
        source=source,
        evidence_count=evidence,
        last_updated=time.time() - age,
    )
    engine.beliefs.append(b)
    return b


# --------------------------------------------------------------------------
# Decay, in the engine that actually runs
# --------------------------------------------------------------------------


def test_an_unreinforced_belief_loses_confidence(engine):
    b = _belief(engine, "the build is green", confidence=0.95, age=30 * DAY)
    assert engine.apply_decay() == 1
    assert b.confidence < 0.95


def test_a_fresh_belief_does_not_decay(engine):
    b = _belief(engine, "just observed", confidence=0.9)
    assert engine.apply_decay() == 0
    assert b.confidence == 0.9


def test_decay_stops_at_agnosticism_not_disbelief(engine):
    """Silence is absence of evidence, not evidence of absence."""
    b = _belief(engine, "very old", confidence=0.95, age=3650 * DAY)
    engine.apply_decay()
    assert b.confidence >= engine.DECAY_FLOOR - 1e-9


def test_decay_never_manufactures_confidence(engine):
    """A weak belief must not drift UP toward the floor and become credible."""
    b = _belief(engine, "doubted", confidence=0.10, age=365 * DAY)
    engine.apply_decay()
    assert b.confidence <= 0.10 + 1e-9


def test_well_evidenced_beliefs_fade_more_slowly(engine):
    """The point of damping by evidence mass.

    Decaying a conclusion drawn from twenty observations at the same rate as an
    offhand remark is the opposite of epistemic hygiene.
    """
    thin = _belief(engine, "heard once", confidence=0.9, evidence=1.0, age=60 * DAY)
    thick = _belief(engine, "seen often", confidence=0.9, evidence=20.0, age=60 * DAY)
    engine.apply_decay()
    assert thick.confidence > thin.confidence


def test_axioms_do_not_decay(engine):
    """Held by construction, so there is nothing for time to erode."""
    axiom = _belief(engine, "i should not deceive", confidence=1.0,
                    source="axiom", age=3650 * DAY)
    engine.apply_decay()
    assert axiom.confidence == 1.0


def test_decay_is_clock_injectable(engine):
    b = _belief(engine, "claim", confidence=0.9)
    assert engine.apply_decay(now=time.time() + 40 * DAY) == 1
    assert b.confidence < 0.9


def test_decay_reports_how_many_moved(engine):
    _belief(engine, "old one", confidence=0.9, age=40 * DAY)
    _belief(engine, "old two", confidence=0.8, age=40 * DAY)
    _belief(engine, "fresh", confidence=0.8)
    assert engine.apply_decay() == 2


# --------------------------------------------------------------------------
# One engine
# --------------------------------------------------------------------------


def test_the_duplicate_no_longer_defines_its_own_engine():
    import ast
    from pathlib import Path

    source = Path("core/cognition/belief_revision.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "BeliefRevisionEngine" not in defined, (
        "a second belief engine reappeared in core.cognition"
    )


def test_the_retired_module_re_exports_the_canonical_engine():
    from core.cognition.belief_revision import BeliefRevisionEngine as ReExported
    from core.epistemics.belief_revision import BeliefRevisionEngine as Canonical

    assert ReExported is Canonical


def test_the_legacy_accessor_returns_the_canonical_engine():
    """Handing back a private engine is how the second one stayed alive."""
    from core.cognition.belief_revision import get_belief_engine

    engine = get_belief_engine()
    assert isinstance(engine, BeliefRevisionEngine)


def test_the_legacy_accessor_tolerates_its_old_signature():
    from core.cognition.belief_revision import get_belief_engine

    assert get_belief_engine(knowledge_graph=object(), brain=object()) is not None


def test_the_system_prompt_claim_about_beliefs_has_a_live_producer():
    """context_assembler tells the model its beliefs carry a confidence.

    That instruction is about the canonical engine. Pinned because it was
    accurate for the wired engine and would have been quietly wrong had the
    unreachable duplicate been the one adopted.
    """
    from pathlib import Path

    prompt = family_text_at(Path('core/brain/llm/context_assembler.py'))
    assert "carry a confidence" in prompt
    boot = family_text_at(Path('core/orchestrator/mixins/boot/boot_autonomy.py'))
    assert "from core.epistemics.belief_revision import" in boot
