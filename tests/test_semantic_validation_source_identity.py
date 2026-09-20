"""A saved score cannot silently survive a different decoder or scorer."""

from dataclasses import replace

import pytest

from core.learning import semantic_validation_checkpoint as cache
from core.learning.semantic_program_compositional_campaign import select_compositional_program_candidate
from tests.test_semantic_relation_graph_learning import model_examples


def test_source_inventory_changes_with_edits_additions_and_removals(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_CORE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        cache.validation_implementation_identity()
    source = tmp_path / "decoder.py"
    source.write_text("answer = 1\n")
    first = cache.validation_implementation_identity()
    source.write_text("answer = 2\n")
    second = cache.validation_implementation_identity()
    assert second != first
    dependency = tmp_path / "operator.py"
    dependency.write_text("value = 3\n")
    assert cache.validation_implementation_identity() != second
    dependency.unlink()
    assert cache.validation_implementation_identity() == second


def test_same_coefficients_cannot_reuse_scores_after_source_change(tmp_path, monkeypatch):
    model, examples = model_examples()
    selected = (replace(examples[0], split="validation"),)
    monkeypatch.setattr(cache, "validation_implementation_identity", lambda: "a" * 64)
    path = tmp_path / "validation.json"
    first = select_compositional_program_candidate({"base": model}, selected,
        incumbent="base", checkpoint_path=path)
    assert first["implementation_source_sha256"] == "a" * 64
    again = select_compositional_program_candidate({"base": model}, selected,
        incumbent="base", checkpoint_path=path)
    assert first == again
    monkeypatch.setattr(cache, "validation_implementation_identity", lambda: "b" * 64)
    with pytest.raises(ValueError, match="identity or checksum"):
        select_compositional_program_candidate({"base": model}, selected,
            incumbent="base", checkpoint_path=path)


def test_midrun_source_change_cannot_produce_a_selection(monkeypatch):
    model, examples = model_examples()
    versions = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(cache, "validation_implementation_identity", lambda: next(versions))
    with pytest.raises(ValueError, match="changed during"):
        select_compositional_program_candidate({"base": model},
            (replace(examples[0], split="validation"),), incumbent="base")
