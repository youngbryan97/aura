"""An expected migration and a forged authority must not share a word.

`resolve_active_generation` returned "invalid" for a descriptor mismatch, which
is what a model migration looks like from the steering side. The runtime then
logged an error on every attach, so the line that will one day mean corruption
became the line that always appears.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm.model_bound_steering import (
    CHECKPOINT_INCOMPATIBLE,
    EXPECTED_DETACHED_STATUSES,
    SteeringGenerationResolution,
    resolve_active_generation,
)


def _spec(descriptor: str, *, migration: dict | None = None):
    return SimpleNamespace(
        descriptor_sha256=descriptor,
        migration_contract=lambda: migration,
    )


# ── The distinction ─────────────────────────────────────────────────────


def test_a_descriptor_mismatch_is_a_migration_not_an_invalid_authority(monkeypatch):
    from core.brain.llm import model_registry

    monkeypatch.setattr(
        model_registry, "get_active_cortex_spec", lambda **k: _spec("b" * 64)
    )
    resolution = resolve_active_generation(
        descriptor_sha256="a" * 64, model_cache_root=Path("/tmp")
    )
    assert resolution.status == CHECKPOINT_INCOMPATIBLE
    assert resolution.status != "invalid"
    assert resolution.reason == "active_cortex_descriptor_mismatch"
    assert resolution.migration_pending is True
    assert resolution.expected_detachment is True


def test_a_missing_migration_contract_is_still_invalid(monkeypatch):
    # Same checkpoint, no contract: the authority genuinely does not hold up.
    from core.brain.llm import model_registry

    monkeypatch.setattr(
        model_registry,
        "get_active_cortex_spec",
        lambda **k: _spec("a" * 64, migration=None),
    )
    resolution = resolve_active_generation(
        descriptor_sha256="a" * 64, model_cache_root=Path("/tmp")
    )
    assert resolution.status == "invalid"
    assert resolution.migration_pending is False
    assert resolution.expected_detachment is False


def test_an_absent_cortex_is_neither(monkeypatch):
    from core.brain.llm import model_registry

    monkeypatch.setattr(
        model_registry, "get_active_cortex_spec", lambda **k: None
    )
    resolution = resolve_active_generation(
        descriptor_sha256="a" * 64, model_cache_root=Path("/tmp")
    )
    assert resolution.status == "unmanaged"
    assert resolution.reason == "active_cortex_absent"
    assert resolution.migration_pending is False


# ── The vocabulary ──────────────────────────────────────────────────────


def test_expected_detachment_covers_the_three_correct_outcomes():
    assert EXPECTED_DETACHED_STATUSES == {
        "deferred",
        "retired",
        CHECKPOINT_INCOMPATIBLE,
    }


def test_invalid_is_never_an_expected_detachment():
    assert SteeringGenerationResolution("invalid").expected_detachment is False


def test_qualified_is_not_a_detachment_at_all():
    assert SteeringGenerationResolution("qualified").expected_detachment is False
    assert SteeringGenerationResolution("qualified").migration_pending is False


def test_unmanaged_is_not_reported_as_a_migration():
    # No promoted cortex is a configuration state, not a migration in progress.
    assert SteeringGenerationResolution("unmanaged").migration_pending is False


# ── What the runtime records ────────────────────────────────────────────


def test_a_migration_records_a_visible_capability_state_not_a_fault():
    """Status truth, not suppression: it stays reported, at info severity."""
    source = Path(__file__).resolve().parents[1] / "core/consciousness/affective_steering.py"
    text = source.read_text()
    assert 'self._model_info["steering_capability_state"] = (' in text
    assert '"migration_pending"' in text
    # The expected branch logs at info; the invalid branch keeps its error.
    expected_branch = text[text.index("if steering_resolution.expected_detachment:"):]
    expected_branch = expected_branch[: expected_branch.index("return False")]
    assert "logger.info(" in expected_branch
    assert "logger.error(" not in expected_branch


def test_the_invalid_branch_keeps_its_alarm():
    source = Path(__file__).resolve().parents[1] / "core/consciousness/affective_steering.py"
    text = source.read_text()
    invalid_branch = text[text.index('if steering_resolution_status == "invalid":'):]
    invalid_branch = invalid_branch[: invalid_branch.index("return False")]
    assert "logger.error(" in invalid_branch
    assert '"authority_invalid"' in invalid_branch


def test_a_migration_never_silently_enables_the_old_vectors():
    """Every non-qualified status returns before the library is opened."""
    source = Path(__file__).resolve().parents[1] / "core/consciousness/affective_steering.py"
    text = source.read_text()
    attach = text.index("if steering_resolution.expected_detachment:")
    library = text.index("self._library = SteeringVectorLibrary(", attach)
    between = text[attach:library]
    # Both non-qualified branches must return before any vector is loaded.
    assert between.count("return False") >= 2


def test_the_live_resolution_is_a_migration_not_a_fault():
    """Against the installed pointer, if there is one."""
    from core.brain.llm.model_registry import get_active_cortex_spec

    try:
        spec = get_active_cortex_spec()
    except Exception:  # noqa: BLE001
        pytest.skip("no readable active cortex pointer")
    if spec is None:
        pytest.skip("no active cortex pointer in this environment")
    stale = "0" * 64
    resolution = resolve_active_generation(
        descriptor_sha256=stale, model_cache_root=Path("/tmp")
    )
    assert resolution.status == CHECKPOINT_INCOMPATIBLE
    assert resolution.expected_detachment is True
