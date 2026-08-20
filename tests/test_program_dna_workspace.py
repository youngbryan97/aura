"""Generated programs stay out of the source tree.

The engine's own standards review asserts
``sandbox_workspace_is_separate_from_original_and_from_aura_runtime``. Before
2026-08-19 the default scaffold root was the project itself, so that assertion
described something untrue.
"""

from __future__ import annotations

from pathlib import Path

from core.self_improvement.program_dna import ProgramDNAReconstructionEngine


def test_the_default_workspace_is_not_the_source_tree() -> None:
    engine = ProgramDNAReconstructionEngine(project_root=Path(__file__).resolve().parents[1])
    workspace = engine._generated_workspace()
    assert not str(workspace).startswith(str(engine.project_root) + "/core")
    assert workspace != engine.project_root


def test_a_target_that_is_a_sentence_gets_a_readable_name() -> None:
    engine = ProgramDNAReconstructionEngine()
    slug = engine._slug("m going to ask you about it later in this conversation and in between i")
    assert len(slug) <= 48
    assert slug.count("-") <= 5


def test_a_blank_target_still_names_something() -> None:
    engine = ProgramDNAReconstructionEngine()
    assert engine._slug("") == "program"
    assert engine._slug("!!!") == "program"
