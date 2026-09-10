"""What it takes to let her substrate steer a person's turn.

The channel these tests guard is the one place her state reaches the model's
computation rather than its prompt. It is fail-closed by construction, and these
say so in each of the four ways it can be refused.
"""

from __future__ import annotations

import json

import pytest

from core.consciousness.fusion_certificate import (
    MIN_DISTRIBUTION_SHIFT,
    MIN_STATE_SEPARATION,
    FusionCertificate,
    certificate_for,
    certified_alpha,
    load_certificates,
    symmetric_kl,
    write_certificate,
)


def _certificate(**overrides) -> FusionCertificate:
    """A certificate that holds, so each test can break exactly one thing."""
    fields = {
        "model_identity": "a" * 64,
        "model_name": "test-model",
        "alpha": 0.2,
        "distribution_shift": 0.29,
        "control_shift": 0.26,
        "state_separation": 1.98,
        "prompts_that_change": 8,
        "median_step_of_change": 5,
        "quality_delta": 0.0,
        "control_quality_delta": 0.0,
        "margin_delta": 0.009,
        "control_margin_delta": -0.006,
        "quality_scale": "forced-choice accuracy over 10 items",
        "prompts": 8,
        "steps": 24,
    }
    fields.update(overrides)
    return FusionCertificate(**fields)


def test_a_full_certificate_holds():
    certificate = _certificate()
    assert certificate.holds
    assert certificate.why_not() == ""


def test_unchanged_answers_refuse_it():
    """The void A/B's own observation, kept as a refusal rather than a verdict."""
    certificate = _certificate(prompts_that_change=2)
    assert not certificate.arrives
    assert not certificate.holds
    assert "2 of 8 probes" in certificate.why_not()


def test_a_shift_too_small_to_matter_refuses_it():
    certificate = _certificate(distribution_shift=MIN_DISTRIBUTION_SHIFT / 2)
    assert not certificate.arrives
    assert not certificate.holds


def test_a_direction_worse_than_an_arbitrary_one_refuses_it():
    certificate = _certificate(margin_delta=-0.05, control_margin_delta=-0.01)
    assert certificate.arrives
    assert not certificate.beats_noise
    assert not certificate.holds
    assert "arbitrary displacement" in certificate.why_not()


def test_a_channel_that_carries_no_state_refuses_it():
    """A constant offset moves the model. It does not carry what she feels."""
    certificate = _certificate(state_separation=MIN_STATE_SEPARATION / 2)
    assert certificate.arrives
    assert certificate.beats_noise
    assert not certificate.carries_content
    assert not certificate.holds
    assert "constant offset" in certificate.why_not()


def test_a_channel_that_costs_answers_refuses_it():
    certificate = _certificate(quality_delta=-0.1)
    assert not certificate.costs_nothing
    assert not certificate.holds
    assert "cannot cost answers" in certificate.why_not()


def test_the_first_missing_thing_is_the_one_reported():
    certificate = _certificate(prompts_that_change=0, quality_delta=-0.5)
    assert "probes" in certificate.why_not()


def test_no_certificate_means_no_steering(tmp_path):
    assert certified_alpha("nothing-was-ever-measured-here", root=tmp_path) == 0.0
    assert certificate_for("nothing-was-ever-measured-here", root=tmp_path) is None


def test_a_failing_certificate_means_no_steering(tmp_path):
    certificate = _certificate(prompts_that_change=0)
    write_certificate(certificate, root=tmp_path)
    assert certified_alpha(certificate.model_identity, root=tmp_path) == 0.0


def test_a_holding_certificate_opens_the_channel(tmp_path):
    certificate = _certificate(alpha=0.15)
    write_certificate(certificate, root=tmp_path)
    assert certified_alpha(certificate.model_identity, root=tmp_path) == pytest.approx(0.15)


def test_a_certificate_is_bound_to_one_checkpoint(tmp_path):
    """Evidence earned by one set of weights says nothing about another."""
    certificate = _certificate(model_identity="a" * 64)
    write_certificate(certificate, root=tmp_path)
    assert certified_alpha("b" * 64, root=tmp_path) == 0.0


def test_an_unreadable_certificate_means_no_steering(tmp_path):
    directory = tmp_path / "artifacts" / "fusion"
    directory.mkdir(parents=True)
    (directory / "broken.json").write_text("{not json", encoding="utf-8")
    assert load_certificates(root=tmp_path) == {}


def test_a_round_trip_keeps_the_verdict(tmp_path):
    certificate = _certificate()
    path = write_certificate(certificate, root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["holds"] is True
    assert payload["beats_noise"] is True
    reloaded = load_certificates(root=tmp_path)[certificate.model_identity]
    assert reloaded.holds
    assert reloaded.alpha == pytest.approx(certificate.alpha)


def test_alpha_is_clamped_to_the_unit_interval(tmp_path):
    write_certificate(_certificate(alpha=7.0), root=tmp_path)
    assert certified_alpha("a" * 64, root=tmp_path) == 1.0


def test_symmetric_kl_is_zero_for_identical_distributions():
    assert symmetric_kl([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == pytest.approx(0.0)


def test_symmetric_kl_is_symmetric():
    left = [0.7, 0.2, 0.1]
    right = [0.1, 0.3, 0.6]
    assert symmetric_kl(left, right) == pytest.approx(symmetric_kl(right, left))


def test_the_measured_reflex_certificate_is_on_disk():
    """The 1.5B was measured, and what it earned is a fact in the repo.

    If this file goes, the claim that the channel was ever demonstrated goes
    with it.
    """
    certificates = [item for item in load_certificates().values() if item.holds]
    assert certificates, "no fusion certificate holds; the channel is shut everywhere"
    for certificate in certificates:
        assert certificate.prompts_that_change * 2 > certificate.prompts
        assert certificate.state_separation >= MIN_STATE_SEPARATION
        assert certificate.quality_delta >= 0.0


def test_certificates_are_found_from_the_repository_not_the_working_directory(
    tmp_path, monkeypatch
):
    """The worker is a forked child of a desktop app and its cwd is not this repo.

    A relative path would write the certificate somewhere and look for it
    somewhere else, and the channel would stay shut with no error anywhere.
    """
    from core.consciousness import fusion_certificate

    home = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(fusion_certificate, "_REPO_ROOT", home)
    monkeypatch.chdir(elsewhere)

    write_certificate(_certificate(model_identity="9" * 64))
    assert certified_alpha("9" * 64) > 0.0
    assert (home / "artifacts" / "fusion").exists()
    assert not (elsewhere / "artifacts").exists()


def test_the_repository_root_holds_the_package():
    """A wrong number of parents would point at a directory that always reads empty."""
    from core.consciousness import fusion_certificate

    assert (fusion_certificate._REPO_ROOT / "core" / "consciousness").is_dir()
