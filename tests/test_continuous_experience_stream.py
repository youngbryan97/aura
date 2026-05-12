from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace

from core.consciousness.continuous_experience import (
    ContinuousExperienceStream,
    ExperienceFrame,
    reset_continuous_experience_stream,
)
from core.container import ServiceContainer
from core.environment.outcome_attribution import OutcomeAssessment
from core.memory.procedural import ProceduralMemoryStore
from core.state.aura_state import AuraState
from core.unity.runtime import UnityRuntime
from core.unity.unity_state import BoundContent, UnityState


def _frame(summary: str, *, timestamp: float = 1.0, **kwargs) -> ExperienceFrame:
    return ExperienceFrame(
        frame_id="",
        sequence=0,
        timestamp=timestamp,
        scene_id="",
        summary=summary,
        focus=kwargs.pop("focus", summary),
        objective=kwargs.pop("objective", "ship safely"),
        source=kwargs.pop("source", "test"),
        **kwargs,
    )


def test_continuous_experience_hash_chain_persists_and_replays(tmp_path):
    path = tmp_path / "stream.json"
    stream = ContinuousExperienceStream(persist_path=path)
    now = time.time()

    first = stream.append_frame(_frame("first moment", timestamp=now))
    second = stream.append_frame(_frame("second moment", timestamp=now + 1))

    assert second.previous_hash == first.frame_hash
    assert stream.validate_replay()["valid"] is True

    restored = ContinuousExperienceStream(persist_path=path)
    assert restored.validate_replay()["valid"] is True
    assert restored.current_frame.frame_hash == second.frame_hash


def test_unity_runtime_commits_movie_like_experience_frame(tmp_path):
    reset_continuous_experience_stream()
    ServiceContainer.set(
        "continuous_experience_stream",
        ContinuousExperienceStream(persist_path=tmp_path / "unity_stream.json"),
        required=False,
    )
    state = AuraState()
    state.cognition.current_objective = "answer with continuity"

    UnityRuntime().apply_to_state(state, objective=state.cognition.current_objective, tick_id="tick_movie")

    payload = state.response_modifiers["experience_stream"]
    assert payload["frame_id"].startswith("exp_")
    assert payload["frame_hash"]
    assert payload["safe_to_act"] in {True, False}


def test_learning_context_snapshots_frame_deque_before_reversed_iteration():
    class MutatesOnReversed(deque):
        def __reversed__(self):
            iterator = super().__reversed__()
            first = True
            for item in iterator:
                if first:
                    first = False
                    self.append(item)
                yield item

    stream = ContinuousExperienceStream(autosave=False)
    stream.append_frame(
        _frame(
            "prediction mismatch",
            timestamp=time.time(),
            lesson="slow down and verify the live lane",
            transfer_tags=("prediction_mismatch",),
            surprise=0.8,
        )
    )
    with stream._lock:
        stream._frames = MutatesOnReversed(stream._frames, maxlen=stream.max_frames)

    context = stream.learning_context(tags=("prediction_mismatch",))

    assert context["transfer_lessons"]
    assert context["current_frame"]["frame_id"].startswith("exp_")


def test_compounding_error_guard_switches_to_observe_stabilize_replay():
    stream = ContinuousExperienceStream(autosave=False)
    now = time.time()
    for idx in range(4):
        stream.append_frame(
            _frame(
                f"bad outcome {idx}",
                timestamp=now + idx,
                outcome_score=0.1,
                harm_score=0.3,
                surprise=0.8,
                lesson="prediction mismatch: gather evidence before repeating this action",
                transfer_tags=("browser", "prediction_mismatch", "harm"),
                repair_needed=True,
                repair_reasons=("low_outcome_score",),
            )
        )

    report = stream.learning_context(target_domain="browser")
    assert report["safe_to_act"] is False
    assert report["recommended_mode"] == "observe_stabilize_replay"
    assert report["transfer_lessons"]


def test_unity_outcome_lessons_transfer_across_domains():
    stream = ContinuousExperienceStream(autosave=False)
    unity = UnityState(
        contents=[
            BoundContent(
                content_id="content_1",
                modality="visual",
                source="browser",
                summary="unknown download link",
                salience=0.9,
                confidence=0.5,
                timestamp=1.0,
                ownership="world",
                action_relevance=0.8,
                affective_charge=0.2,
            )
        ],
        global_focus_id="content_1",
        level="fragmented",
        unity_score=0.3,
        fragmentation_score=0.7,
        repair_needed=True,
        repair_reasons=["ownership_ambiguity"],
    )
    outcome = OutcomeAssessment(
        action="use",
        expected_effect="download_complete",
        observed_events=["failure"],
        success_score=0.0,
        harm_score=0.4,
        information_gain=0.0,
        surprise=0.9,
        lesson="prediction mismatch: gather evidence before repeating this action",
    )

    stream.append_from_unity(unity, outcome=outcome, objective="inspect download")
    lessons = stream.transfer_lessons(target_domain="api", tags=("prediction_mismatch",))

    assert lessons
    assert "gather evidence" in lessons[0]["lesson"]


def test_privacy_retention_deletes_and_redacts_private_frames():
    stream = ContinuousExperienceStream(
        autosave=False,
        private_retention_s=1000,
        standard_retention_s=10000,
    )
    now = time.time()
    stream.append_frame(_frame("private secret context", timestamp=now - 20, privacy_tier="private"))
    stream.append_frame(_frame("public work context", timestamp=now - 10, privacy_tier="standard"))

    redacted = stream.export_reel(redacted=True)["frames"][0]
    assert redacted["summary"].startswith("[private:")

    removed = stream.enforce_retention(now=now + 2000)
    assert removed == 1
    assert stream.validate_replay()["valid"] is True
    assert stream.current_frame.summary == "public work context"


def test_now_moment_appends_as_experiential_frame():
    stream = ContinuousExperienceStream(autosave=False)
    moment = SimpleNamespace(
        timestamp=time.time(),
        attentional_focus="the user's nap note",
        interior_text="Quietly keeping the thread warm.",
        substrate=SimpleNamespace(valence=0.2, arousal=0.4, texture_word="steady"),
        affect=SimpleNamespace(dominant_emotion="focused"),
    )

    frame = stream.append_now_moment(moment, objective="continue the work")

    assert frame.source == "stream_of_being"
    assert "emotion:focused" in frame.transfer_tags
    assert frame.summary == "Quietly keeping the thread warm."


def test_procedural_store_records_failures_and_replays_atomic_envelope(tmp_path):
    path = tmp_path / "procedural.json"
    store = ProceduralMemoryStore(path)
    store.record_outcome(
        environment_family="browser",
        context_signature="unknown download",
        action="use",
        parameters={"target": "link"},
        observed_events=["failure"],
        success=False,
        outcome_score=0.0,
        risk_score=0.8,
        failure_conditions=["prediction mismatch"],
    )
    store.save()

    restored = ProceduralMemoryStore(path)
    records = restored.retrieve(
        environment_family="browser",
        context_signature="unknown download",
        goal="use",
    )

    assert records
    assert records[0].failure_count == 1
    assert records[0].risk_score == 0.8
