"""MTP is reported unsupported for reasons, not as a default.

A config that declares `mtp_num_hidden_layers` reads like a decode accelerator
waiting to be enabled. For the installed checkpoints it is a declaration with
no weights behind it, and each blocker is checked separately so that repairing
one does not read as the capability arriving.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.llm.mtp_capability import (
    MTPAcceptanceTelemetry,
    compatible_draft_models,
    detect,
    loader_discards_mtp,
)

#: The checkout this test is running from. Derived rather than written
#: down: a literal install path reads another checkout's artifacts when
#: the suite runs in a worktree, and names one machine's account.
INSTALL = Path(__file__).resolve().parents[1]


def _write(directory: Path, config: dict, weights: dict | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    if weights is not None:
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weights})
        )
    return directory


def test_the_installed_loader_really_does_discard_mtp_weights():
    """Read from mlx_lm, not asserted, so a future release flips it here."""
    pytest.importorskip("mlx_lm.models.qwen3_5")
    assert loader_discards_mtp("qwen3_5") is True


def test_delegation_is_followed_when_finding_the_discard():
    # Model.sanitize delegates to the inner text model's. Inspecting only the
    # outer class reported False while the keys were still being dropped.
    pytest.importorskip("mlx_lm.models.qwen3_5")
    import inspect

    from mlx_lm.models import qwen3_5

    outer = inspect.getsource(qwen3_5.Model.sanitize)
    assert '"mtp." not in k' not in outer
    assert loader_discards_mtp("qwen3_5") is True


def test_an_architecture_without_a_loader_is_not_claimed_to_discard():
    assert loader_discards_mtp("not_a_real_architecture") is False


def test_a_checkpoint_with_no_mtp_tensors_is_unsupported(tmp_path):
    model = _write(
        tmp_path / "m",
        {"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1,
                                                  "vocab_size": 248320}},
        {"language_model.lm_head.weight": "a"},
    )
    capability = detect(model)
    assert capability.native_supported is False
    assert "checkpoint_carries_no_mtp_tensors" in capability.native_blockers
    assert capability.declares_mtp_layers == 1
    assert capability.mtp_tensor_count == 0


def test_even_with_tensors_present_no_supported_api_reaches_the_head(tmp_path):
    model = _write(
        tmp_path / "m2",
        {"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1,
                                                  "vocab_size": 248320}},
        {"mtp.layers.0.weight": "a", "language_model.lm_head.weight": "b"},
    )
    capability = detect(model)
    assert capability.mtp_tensor_count == 1
    assert "checkpoint_carries_no_mtp_tensors" not in capability.native_blockers
    # The remaining blockers stand on their own.
    assert capability.native_supported is False
    assert "no_supported_api_reaches_an_internal_head" in capability.native_blockers


def test_a_draft_must_share_the_vocabulary_exactly(tmp_path):
    target_vocab = 248320
    good = _write(tmp_path / "good", {"model_type": "qwen3_5",
                                      "text_config": {"vocab_size": 248320,
                                                      "num_hidden_layers": 32}})
    bad = _write(tmp_path / "bad", {"model_type": "qwen2",
                                    "text_config": {"vocab_size": 152064,
                                                    "num_hidden_layers": 28}})
    found = compatible_draft_models(
        target_vocab, "qwen3_5", {"good": good, "bad": bad}
    )
    assert [d["name"] for d in found] == ["good"]
    assert found[0]["same_family"] is True


def test_no_compatible_draft_means_no_draft_speculation(tmp_path):
    model = _write(
        tmp_path / "t",
        {"model_type": "qwen3_5", "text_config": {"vocab_size": 248320}},
        {},
    )
    capability = detect(model, {})
    assert capability.draft_speculation_supported is False
    assert capability.compatible_draft_models == ()


def test_the_benefit_is_always_listed_as_unmeasured(tmp_path):
    model = _write(
        tmp_path / "u",
        {"model_type": "qwen3_5", "text_config": {"vocab_size": 248320}},
        {},
    )
    capability = detect(model, {})
    assert "draft_acceptance_rate" in capability.unmeasured
    assert "end_to_end_speedup" in capability.unmeasured
    assert "output_equivalence_under_speculation" in capability.unmeasured


def test_acceptance_rate_is_none_before_anything_is_drafted():
    telemetry = MTPAcceptanceTelemetry()
    # Zero would be indistinguishable from a draft model that is always wrong,
    # and the two call for opposite actions.
    assert telemetry.acceptance_rate is None
    assert telemetry.receipt()["measured"] is False


def test_acceptance_rate_is_derived_from_what_was_recorded():
    telemetry = MTPAcceptanceTelemetry()
    telemetry.record(drafted=4, accepted=3)
    telemetry.record(drafted=4, accepted=1)
    assert telemetry.acceptance_rate == 0.5
    receipt = telemetry.receipt()
    assert receipt["drafted_tokens"] == 8
    assert receipt["accepted_tokens"] == 4
    assert receipt["rejections"] == 4
    assert receipt["measured"] is True


def test_accepting_more_than_was_drafted_is_refused():
    telemetry = MTPAcceptanceTelemetry()
    with pytest.raises(ValueError):
        telemetry.record(drafted=2, accepted=3)


def test_the_active_checkpoint_reports_unsupported_with_every_blocker():
    manifest = INSTALL / "training/fused-model/active.json"
    if not manifest.exists():
        pytest.skip("no active model manifest")
    model = Path(json.loads(manifest.read_text())["active_model_path"])
    if not (model / "config.json").exists():
        pytest.skip("active checkpoint is not installed")
    capability = detect(model)
    assert capability.native_supported is False
    assert set(capability.native_blockers) == {
        "checkpoint_carries_no_mtp_tensors",
        "loader_discards_mtp_tensors",
        "no_supported_api_reaches_an_internal_head",
    }
