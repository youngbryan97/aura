"""Every consumer learns from distinct decisions under the current schema."""

from dataclasses import replace
import sqlite3

import numpy as np
import pytest

from core.ontogeny.authority import AuthorityLedger, AuthorityStage
from core.ontogeny.calibration import OPERATIONAL_SHADOW, track_records
from core.ontogeny.experience import Episode, ExperienceSpine, Outcome, OutcomeKind, Provenance
from core.ontogeny.features import FeatureSchema
from core.ontogeny.heads import PredictionHead
from core.ontogeny.service import ControlPoint, OntogenyCore
from core.ontogeny.trainer import TrainingResult


@pytest.fixture
def organ(tmp_path):
    core = OntogenyCore(
        spine=ExperienceSpine(tmp_path / "experience.db", autoflush=False),
        authority=AuthorityLedger(tmp_path / "authority.json"), units=4, autostart=False,
    )
    core.register(ControlPoint("cp", FeatureSchema("cp", ("x",), outcome_contract="v2"), ("a", "b")))
    yield core
    core.stop()


def record(core, index, *, schema=None, kind=OutcomeKind.SUCCESS, repeats=1):
    cp = core._control_points["cp"]
    row = Episode(
        control_point="cp", features={"x": float(index)}, decision="a",
        feature_schema=schema or cp.schema.schema_id, provenance=Provenance.TEST,
        decided_at=float(index), repeat_count=repeats,
        shadow={"a": 0.8}, shadow_version=1, context={"runtime_revision": "r1"},
    )
    core._spine.record(row)
    core._spine.resolve(row.episode_id, Outcome(kind, resolved_at=float(index + 1)))
    return row


def test_failed_read_preserves_head_and_authority(organ, monkeypatch):
    cp = organ._control_points["cp"]
    cp.heads["a"].version = 9
    before = cp.heads["a"].w.copy()
    organ.authority.set_stage("cp", AuthorityStage.AUTHORITY, reason="measured")
    monkeypatch.setattr(organ._spine, "stats", lambda *a, **kw: {"available": True, "evidence_rows": 500})
    def unavailable():
        raise sqlite3.OperationalError("disk read unavailable")
    monkeypatch.setattr(organ._spine, "_connect", unavailable)
    assert organ.train("cp") == {}
    assert cp.heads["a"].version == 9
    np.testing.assert_array_equal(cp.heads["a"].w, before)
    assert organ.authority.stage("cp") is AuthorityStage.AUTHORITY
    with pytest.raises(RuntimeError, match="evidence unavailable"):
        organ._spine.episodes("cp")


def test_corpus_consumption_includes_washout_and_unfitted_actions(organ, monkeypatch):
    for i in range(90):
        record(organ, i)
    for i in range(90, 250):
        record(organ, i, schema="retired")
    organ._spine.flush()
    assert organ._spine.stats("cp")["evidence_rows"] == 250
    assert organ._new_evidence() == 90
    monkeypatch.setattr(organ._trainer, "train", lambda *a, **kw: TrainingResult(
        "cp", fitted=True, samples=20, temperature_samples=5, holdout_samples=5))
    organ.train("cp")
    assert organ._control_points["cp"].evidence_at_last_fit == 90
    assert organ._new_evidence() == 0
    record(organ, 251)
    organ._spine.flush()
    assert organ._new_evidence() == 1


def test_observation_window_filters_schema_before_limit_and_cache(organ):
    for i in range(55):
        record(organ, i)
    for i in range(55, 155):
        record(organ, i, schema="retired", kind=OutcomeKind.UNOBSERVED)
    organ._spine.flush()
    assert organ._spine.observation_stats("cp", recent_limit=50)["observation_rate"] == 0
    schema = organ._control_points["cp"].schema.schema_id
    current = organ._spine.observation_stats("cp", recent_limit=50, feature_schema=schema)
    assert current["closed"] == 50
    assert current["observation_rate"] == 1
    organ.authority.set_stage("cp", AuthorityStage.AUTHORITY, reason="measured")
    assert organ._enforce_authority_observation() == ()
    assert organ.authority.has_authority("cp")


def test_rehydration_uses_current_schema_and_independent_decisions(organ):
    record(organ, 1, repeats=10_000)
    for i in range(2, 60):
        record(organ, i, schema="retired", kind=OutcomeKind.FAILURE)
    organ._spine.flush()
    organ.rehydrate_track_records()
    track = organ.track_record("cp", "a")
    assert (track.successes, track.failures) == (1, 0)
    assert organ.rehydrate_operational_calibration()["cp"] == 1
    row = replace(record(organ, 100), outcome=Outcome(OutcomeKind.SUCCESS), repeat_count=10_000)
    assert track_records([row, row])["a"].successes == 1


def test_schema_change_clears_in_memory_heads_and_delayed_outcomes(organ):
    cp = organ._control_points["cp"]
    cp.heads["a"].version = 7
    cp.evidence_at_last_fit = 99
    episode = record(organ, 1)
    organ._remember_episode(episode)
    organ._track.observe("cp", "a", OutcomeKind.SUCCESS)
    organ._operational_calibration.observe(
        "cp", confidence=0.8, correct=True, episode_id="old", provenance=OPERATIONAL_SHADOW)
    cp.schema = replace(cp.schema, outcome_contract="v3")
    organ.register(cp)
    assert all(head.version == 0 for head in cp.heads.values())
    assert cp.evidence_at_last_fit == 0
    assert organ.track_record("cp", "a") is None
    organ._note_resolution(episode.episode_id, Outcome(OutcomeKind.SUCCESS))
    assert organ.track_record("cp", "a") is None
    report = organ._operational_calibration.report("cp")
    assert report is None or report.samples == 0


def test_late_old_schema_episode_cannot_enter_new_track_record(organ):
    episode = record(organ, 1, schema="retired")
    organ._remember_episode(episode)
    organ._note_resolution(episode.episode_id, Outcome(OutcomeKind.SUCCESS))
    assert organ.track_record("cp", "a") is None


def test_training_weights_do_not_multiply_collapsed_repeats(organ, monkeypatch):
    for i in range(400):
        record(organ, i, repeats=10_000 if i == 80 else 1)
    organ._spine.flush()
    observed = []
    def fit(self, rows, labels, *, weights, **kwargs):
        observed.extend(weights)
        return {"fitted": False, "reason": "test capture"}
    monkeypatch.setattr(PredictionHead, "fit", fit)
    monkeypatch.setattr("core.ontogeny.trainer.time.time", lambda: 1000.0)
    organ.train("cp")
    assert observed and max(observed) <= 1
    assert max(observed) / min(observed) < 1.001


def test_failed_rehydration_does_not_erase_track_record(organ, monkeypatch):
    organ._track.observe("cp", "a", OutcomeKind.SUCCESS)
    def unavailable(*args, **kwargs):
        raise RuntimeError("episode evidence unavailable")
    monkeypatch.setattr(organ._spine, "episodes", unavailable)
    assert organ.rehydrate_track_records().get("cp") is None
    assert organ.track_record("cp", "a").successes == 1


def test_compaction_does_not_hide_new_evidence(organ):
    for i in range(3):
        record(organ, i)
    organ._spine.flush()
    organ._control_points["cp"].evidence_at_last_fit = 3
    with organ._spine._using_the_store() as conn:
        conn.execute("DELETE FROM episodes WHERE control_point = 'cp'")
    record(organ, 4)
    organ._spine.flush()
    assert organ._new_evidence() == 1
    assert organ._spine.stats("cp")["evidence_rows"] == 1
    organ._spine._init_schema()
    assert organ._new_evidence() == 1


def test_head_restart_retains_consumed_evidence_revision(organ):
    cp = organ._control_points["cp"]
    cp.evidence_at_last_fit = 91
    cp.heads["a"].version = 3
    organ._save_head(cp)
    cp.evidence_at_last_fit = 0
    cp.heads["a"].version = 0
    organ._load_heads()
    assert cp.evidence_at_last_fit == 91
    assert cp.heads["a"].version == 3


def test_replayed_decision_cannot_erase_outcome_or_increment_revision(organ):
    row = record(organ, 1)
    organ._spine.flush()
    organ._spine.record(row)
    organ._spine.resolve(row.episode_id, Outcome(OutcomeKind.FAILURE))
    organ._spine.flush()
    assert organ._spine.episodes("cp")[0].outcome.kind is OutcomeKind.SUCCESS
    assert organ._spine.stats("cp")["evidence_revision"] == 1
