"""Learned recurrent-update admission, calibration, and causal wiring."""

from __future__ import annotations

import hashlib
import zipfile

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.brain.llm.latent_cortex.update_gate import (  # noqa: E402
    LEARNED,
    PASSTHROUGH,
    UpdateGateRuntime,
    extract_update_features,
)
from core.learning.update_acceptance import (  # noqa: E402
    FEATURE_NAMES,
    MAX_HEAD_ARTIFACT_BYTES,
    MAX_HEAD_UNCOMPRESSED_BYTES,
    UpdateAcceptanceHead,
    VerifiedTransitionExample,
    fit_update_acceptance_head,
)

HIDDEN = 64
PROMPT = [5, 9, 17, 3, 42, 7, 11, 23, 2, 88]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _features(signal: float, *, evidence_available: float = 1.0):
    values = {name: 0.0 for name in FEATURE_NAMES}
    values.update(
        {
            "proposal_residual": 0.4,
            "anchor_alignment_delta": signal,
            "evidence_alignment_delta": 0.8 * signal,
            "anchor_distance_improvement": 0.6 * signal,
            "evidence_distance_improvement": 0.5 * signal,
            "proposal_previous_cosine": 0.9,
            "evidence_available": evidence_available,
        }
    )
    return values


def _examples(prefix: str, count: int):
    rows = []
    for index in range(count):
        improved = index % 2 == 0
        signal = (0.8 + 0.01 * (index % 5)) * (1.0 if improved else -1.0)
        rows.append(
            VerifiedTransitionExample.from_values(
                example_id=f"{prefix}-{index}",
                features=_features(signal),
                improved=improved,
                verifier_receipt_sha256=_digest(f"verifier:{prefix}:{index}"),
            )
        )
    return rows


@pytest.fixture(scope="module")
def tiny_model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=HIDDEN,
        num_hidden_layers=8,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _config(*, update_gate=None):
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=7),
        recurrence=RecurrenceConfig(
            max_steps=4,
            min_steps=1,
            convergence_eps=1e-9,
            fixed_depth=True,
        ),
        branches=BranchConfig(n_branches=1),
        latent_opt=LatentOptConfig(enabled=False),
        decode_max_tokens=4,
        update_gate=update_gate,
    )


def test_head_is_actually_fitted_calibrated_and_round_trips(tmp_path):
    head = fit_update_acceptance_head(
        _examples("train", 64),
        _examples("calibration", 40),
    )
    assert head.calibrated is True
    assert head.calibration["auc"] == pytest.approx(1.0)
    assert head.probability(_features(0.9)) > head.threshold
    assert head.probability(_features(-0.9)) < head.threshold
    with pytest.raises(ValueError, match="end in .npz"):
        head.save(tmp_path / "update-head.bin")

    path = tmp_path / "update-head.npz"
    digest = head.save(path)
    loaded = UpdateAcceptanceHead.load(path, expected_sha256=digest)
    assert loaded.to_manifest() == head.to_manifest()
    assert loaded.probability(_features(0.9)) == pytest.approx(
        head.probability(_features(0.9))
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        UpdateAcceptanceHead.load(path, expected_sha256="0" * 64)

    link = tmp_path / "linked-head.npz"
    link.symlink_to(path)
    with pytest.raises(OSError):
        UpdateAcceptanceHead.load(link, expected_sha256=digest)

    oversized = tmp_path / "oversized-head.npz"
    oversized.write_bytes(b"x" * (MAX_HEAD_ARTIFACT_BYTES + 1))
    with pytest.raises(ValueError, match="size/type"):
        UpdateAcceptanceHead.load(
            oversized,
            expected_sha256=hashlib.sha256(oversized.read_bytes()).hexdigest(),
        )

    expansion_bomb = tmp_path / "expansion-bomb.npz"
    with zipfile.ZipFile(
        expansion_bomb,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for member in (
            "means.npy",
            "scales.npy",
            "weights.npy",
            "bias.npy",
            "manifest.npy",
        ):
            content = (
                b"\0" * (MAX_HEAD_UNCOMPRESSED_BYTES + 1)
                if member == "means.npy"
                else b"x"
            )
            archive.writestr(member, content)
    bomb_digest = hashlib.sha256(expansion_bomb.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="expands beyond"):
        UpdateAcceptanceHead.load(
            expansion_bomb,
            expected_sha256=bomb_digest,
        )


def test_update_artifact_uses_a_governed_atomic_writer(tmp_path, monkeypatch):
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    head = fit_update_acceptance_head(_examples("train", 64), _examples("calibration", 40))
    gateway = get_file_write_gateway()
    write_bytes = gateway.write_bytes
    writes = []
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")

    def write(path, payload, **kwargs):
        token = get_active_governance()
        assert token is not None and token.domain == "state_mutation"
        writes.append((path, payload, kwargs["source"]))
        write_bytes(path, payload, **kwargs)

    monkeypatch.setattr(gateway, "write_bytes", write)
    path = tmp_path / "new" / "update-head.npz"
    digest = head.save(path)
    assert len(writes) == 1 and writes[0][0] == path
    assert writes[0][2] == "update_acceptance.save"
    assert writes[0][1] == path.read_bytes()
    assert digest == hashlib.sha256(writes[0][1]).hexdigest()
    assert path.stat().st_mode & 0o777 == 0o600
    assert UpdateAcceptanceHead.load(path, expected_sha256=digest).to_manifest() == head.to_manifest()


def test_an_oversized_update_artifact_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    head = fit_update_acceptance_head(_examples("train", 64), _examples("calibration", 40))
    path = tmp_path / "update-head.npz"
    digest = head.save(path)
    previous = path.read_bytes()
    manifest = head.to_manifest()
    with monkeypatch.context() as patch:
        patch.setattr(head, "to_manifest", lambda: {**manifest, "padding": "x" * MAX_HEAD_ARTIFACT_BYTES})
        with pytest.raises(ValueError, match="size|bound"):
            head.save(path)
    assert path.read_bytes() == previous
    assert UpdateAcceptanceHead.load(path, expected_sha256=digest).to_manifest() == manifest


def test_train_and_calibration_examples_must_be_disjoint():
    rows = _examples("shared", 40)
    with pytest.raises(ValueError, match="overlap"):
        fit_update_acceptance_head(rows, rows)


def test_transition_examples_cannot_bypass_constructor_validation():
    with pytest.raises(ValueError, match="non-empty"):
        VerifiedTransitionExample(
            example_id=" ",
            features=tuple(0.0 for _ in FEATURE_NAMES),
            improved=True,
            verifier_receipt_sha256=_digest("receipt"),
        )
    with pytest.raises(ValueError, match="boolean"):
        VerifiedTransitionExample(
            example_id="bad-label",
            features=tuple(0.0 for _ in FEATURE_NAMES),
            improved=1,  # type: ignore[arg-type]
            verifier_receipt_sha256=_digest("receipt"),
        )
    with pytest.raises(ValueError, match="wrong width"):
        VerifiedTransitionExample(
            example_id="bad-width",
            features=(0.0,),
            improved=True,
            verifier_receipt_sha256=_digest("receipt"),
        )


def test_feature_extractor_is_evidence_conditioned_and_finite():
    previous = mx.ones((1, 2, HIDDEN))
    proposal = previous * 1.1
    anchor = previous * 0.9
    evidence = -mx.ones((1, 1, HIDDEN))
    features, delta = extract_update_features(
        previous,
        proposal,
        anchor,
        evidence_state=evidence,
        previous_residual=0.2,
        previous_delta=proposal - previous,
    )
    assert tuple(features) == FEATURE_NAMES
    assert features["evidence_available"] == 1.0
    assert all(abs(value) <= 32.0 for value in features.values())
    assert tuple(delta.shape) == tuple(previous.shape)


def _fit_observed_transition_head(rows):
    positive = rows[0]["features"]
    negative = rows[-1]["features"]

    def split(prefix: str, count: int):
        examples = []
        for index in range(count):
            improved = index % 2 == 0
            source = positive if improved else negative
            features = dict(source)
            jitter = (index % 7 - 3) * 1e-5
            features["proposal_residual"] += jitter
            examples.append(
                VerifiedTransitionExample.from_values(
                    example_id=f"{prefix}-{index}",
                    features=features,
                    improved=improved,
                    verifier_receipt_sha256=_digest(
                        f"observed:{prefix}:{index}"
                    ),
                )
            )
        return examples

    return fit_update_acceptance_head(split("train", 64), split("cal", 40))


def test_fitted_gate_rejects_a_real_tiny_qwen_update_and_changes_downstream_logits(
    tiny_model,
    tmp_path,
):
    baseline = LatentCortexEngine(tiny_model, config=_config()).reason(
        token_ids=PROMPT,
        budget=ComputeBudget(),
    )
    baseline_rows = baseline.receipt.update_acceptance["branches"][0][
        "transitions"
    ]
    assert len(baseline_rows) >= 2
    assert baseline.receipt.update_acceptance["mode"] == PASSTHROUGH

    head = _fit_observed_transition_head(baseline_rows)
    path = tmp_path / "observed-update-head.npz"
    digest = head.save(path)
    learned = LatentCortexEngine(
        tiny_model,
        config=_config(
            update_gate={
                "mode": LEARNED,
                "head_path": str(path),
                "head_sha256": digest,
            }
        ),
    ).reason(token_ids=PROMPT, budget=ComputeBudget())

    gate = learned.receipt.update_acceptance
    assert learned.ok is True
    assert gate["accepted"] >= 1
    assert gate["rejected"] >= 1
    assert gate["causal_rejections"] >= 1
    assert gate["head_was_causal"] is True
    rejected = next(
        row
        for row in gate["branches"][0]["transitions"]
        if row["accepted"] is False
    )
    assert (
        rejected["admitted_hypothesis_sha256"]
        == rejected["prior_hypothesis_sha256"]
    )
    assert (
        rejected["proposal_hypothesis_sha256"]
        != rejected["admitted_hypothesis_sha256"]
    )
    assert (
        learned.receipt.first_logits_digest
        != baseline.receipt.first_logits_digest
    )


def test_learned_runtime_refuses_unpinned_or_missing_artifact(tmp_path):
    with pytest.raises(ValueError, match="head_sha256"):
        UpdateGateRuntime.from_config(
            {"mode": LEARNED, "head_path": str(tmp_path / "missing.npz")}
        )
    problems = _config(
        update_gate={"mode": LEARNED, "head_path": "head.npz"}
    ).validate()
    assert any("head_sha256" in problem for problem in problems)
