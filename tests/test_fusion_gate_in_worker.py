"""The worker's user-surface steering gate reads the certificate, not a constant.

For a long time this gate returned zero with a comment naming a void A/B as the
reason. These say what reopens it and what keeps it shut.
"""

from __future__ import annotations

import pytest

from core.brain.llm import mlx_worker
from core.consciousness import fusion_certificate
from core.consciousness.fusion_certificate import FusionCertificate, write_certificate


def _point_the_lookup_at(monkeypatch, root):
    """Certificates are found from the repository, not from the cwd."""
    monkeypatch.setattr(fusion_certificate, "_REPO_ROOT", root)


@pytest.fixture
def certified(tmp_path, monkeypatch):
    """A checkpoint that has earned steering, with the lookup pointed at tmp."""
    identity = "c" * 64
    certificate = FusionCertificate(
        model_identity=identity,
        model_name="test-model",
        alpha=0.2,
        distribution_shift=0.29,
        control_shift=0.26,
        state_separation=1.98,
        prompts_that_change=8,
        median_step_of_change=5,
        quality_delta=0.0,
        control_quality_delta=0.0,
        margin_delta=0.009,
        control_margin_delta=-0.006,
        quality_scale="forced-choice accuracy over 10 items",
        prompts=8,
        steps=24,
    )
    _point_the_lookup_at(monkeypatch, tmp_path)
    write_certificate(certificate, root=tmp_path)
    monkeypatch.setattr(mlx_worker, "_FUSION_MODEL_IDENTITY", identity)
    monkeypatch.delenv("AURA_USER_SURFACE_STEERING_ALPHA", raising=False)
    return certificate


def test_an_uncertified_checkpoint_gets_no_steering(monkeypatch, tmp_path):
    _point_the_lookup_at(monkeypatch, tmp_path)
    monkeypatch.setattr(mlx_worker, "_FUSION_MODEL_IDENTITY", "d" * 64)
    monkeypatch.delenv("AURA_USER_SURFACE_STEERING_ALPHA", raising=False)
    assert mlx_worker._surface_control_alpha({}, None) == 0.0


def test_a_worker_that_never_attached_gets_no_steering(monkeypatch, tmp_path):
    _point_the_lookup_at(monkeypatch, tmp_path)
    monkeypatch.setattr(mlx_worker, "_FUSION_MODEL_IDENTITY", "")
    monkeypatch.delenv("AURA_USER_SURFACE_STEERING_ALPHA", raising=False)
    assert mlx_worker._surface_control_alpha({}, None) == 0.0


def test_a_certified_checkpoint_steers_at_the_measured_alpha(certified):
    assert mlx_worker._surface_control_alpha({}, None) == pytest.approx(certified.alpha)


def test_the_job_may_still_ask_for_less(certified):
    alpha = mlx_worker._surface_control_alpha({"clean_user_surface_steering_alpha": 0.05}, None)
    assert alpha == pytest.approx(0.05)


def test_the_governor_still_caps_it(certified):
    """The certificate says what is allowed, never what is required."""
    assert mlx_worker._surface_control_alpha({}, 0.01) == pytest.approx(0.01)


def test_a_failing_certificate_keeps_the_channel_shut(monkeypatch, tmp_path):
    identity = "e" * 64
    write_certificate(
        FusionCertificate(
            model_identity=identity,
            model_name="test-model",
            alpha=0.2,
            distribution_shift=0.0001,
            control_shift=0.0001,
            state_separation=0.01,
            prompts_that_change=0,
            median_step_of_change=-1,
            quality_delta=-0.2,
            control_quality_delta=0.0,
            margin_delta=-0.5,
            control_margin_delta=0.0,
            quality_scale="forced-choice accuracy over 10 items",
            prompts=8,
            steps=24,
        ),
        root=tmp_path,
    )
    _point_the_lookup_at(monkeypatch, tmp_path)
    monkeypatch.setattr(mlx_worker, "_FUSION_MODEL_IDENTITY", identity)
    monkeypatch.delenv("AURA_USER_SURFACE_STEERING_ALPHA", raising=False)
    assert mlx_worker._surface_control_alpha({}, None) == 0.0


def test_attaching_remembers_which_checkpoint_is_loaded(monkeypatch):
    monkeypatch.setattr(mlx_worker, "_FUSION_MODEL_IDENTITY", "")
    mlx_worker._remember_fusion_identity({"descriptor_sha256": "f" * 64})
    assert mlx_worker._FUSION_MODEL_IDENTITY == "f" * 64


def test_attaching_without_an_identity_leaves_the_channel_shut(monkeypatch):
    monkeypatch.setattr(mlx_worker, "_FUSION_MODEL_IDENTITY", "f" * 64)
    mlx_worker._remember_fusion_identity(None)
    assert mlx_worker._FUSION_MODEL_IDENTITY == ""


def test_self_certification_will_not_run_without_hooks():
    class Engine:
        def active_hooks(self):
            return []

    assert mlx_worker._self_certify_fusion(object(), object(), Engine()) is False


def test_self_certification_reads_the_hooks_an_engine_actually_publishes():
    """`_hooks` is not the surface, and this test used to say it was.

    The attribute exists on the steering engine and is not what a caller is
    meant to read; `_active_steering_hooks` carries a whole docstring about the
    last time that distinction cost something. Because the fake published
    `_hooks` and nothing else, the certification could decline on every real
    engine and this suite stayed green. The live 27B declined for two days.
    """
    probed: list[str] = []

    class Engine:
        _hooks: list = []

        def active_hooks(self):
            probed.append("asked")
            return [object()]

        def set_alpha(self, alpha):
            return None

    mlx_worker._self_certify_fusion(object(), object(), Engine())

    assert probed, "the certification never asked the engine what its hooks are"


def test_a_missing_precondition_is_said_once(monkeypatch, caplog):
    """A mechanism that declines every minute has to say why, and say it once."""
    monkeypatch.setattr(mlx_worker, "_FUSION_SELF_CERTIFY_REFUSAL_LOGGED", False)
    monkeypatch.setattr(mlx_worker, "_FUSION_MODEL_IDENTITY", "")

    class Engine:
        def active_hooks(self):
            return []

    with caplog.at_level("INFO", logger=mlx_worker.logger.name):
        assert mlx_worker._self_certify_fusion(None, None, Engine()) is False
        first = len([r for r in caplog.records if "cannot be measured" in r.getMessage()])
        assert mlx_worker._self_certify_fusion(None, None, Engine()) is False
        second = len([r for r in caplog.records if "cannot be measured" in r.getMessage()])

    assert first == 1, "the reason it declined was never written down"
    assert second == 1, "the reason is written on every idle tick"


def test_self_certification_skips_a_checkpoint_that_already_has_one(certified, monkeypatch):
    class Engine:
        def active_hooks(self):
            return [object()]

        def set_alpha(self, alpha):
            raise AssertionError("must not probe a checkpoint that is already certified")

    assert mlx_worker._self_certify_fusion(object(), object(), Engine()) is False
