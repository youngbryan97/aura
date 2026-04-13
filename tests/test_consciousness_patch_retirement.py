from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace

from core.agency_core import AgencyCore
from core.consciousness.apply_patches import apply_consciousness_patches
from core.consciousness.phenomenological_experiencer import PhenomenologicalExperiencer


class _DummyMonitor:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True


class _DummyContinuity:
    def __init__(self) -> None:
        self._moments = deque()
        self._thread = ""
        self._episode_count = 3

    @property
    def current_thread(self) -> str:
        return self._thread

    def seed(self, thread: str) -> None:
        self._thread = thread

    def get_episode_summary(self) -> dict[str, object]:
        return {
            "dominant_domain": "cognitive",
            "dominant_tone": "focused",
            "attention_stability": 0.82,
        }


def test_apply_consciousness_patches_is_native_compatibility_hook(monkeypatch):
    import core.consciousness.loop_monitor as loop_monitor_module

    monitor = _DummyMonitor()
    monkeypatch.setattr(loop_monitor_module, "get_loop_monitor", lambda orchestrator: monitor)

    save_fn = PhenomenologicalExperiencer._save_phenomenal_memory
    load_fn = PhenomenologicalExperiencer._load_phenomenal_memory
    pathway_fn = AgencyCore._pathway_self_development

    orchestrator = SimpleNamespace()
    apply_consciousness_patches(orchestrator)

    assert PhenomenologicalExperiencer._save_phenomenal_memory is save_fn
    assert PhenomenologicalExperiencer._load_phenomenal_memory is load_fn
    assert AgencyCore._pathway_self_development is pathway_fn
    assert orchestrator.loop_monitor is monitor
    assert monitor.started is True


def test_load_phenomenal_memory_restores_waking_thread_and_moment_tail(tmp_path: Path):
    save_path = tmp_path / "phenomenal_memory.json"
    save_path.write_text(
        json.dumps(
            {
                "psm_reports": ["report"],
                "psm_witness": "witness",
                "psm_present": "present",
                "continuity_thread": "deeply following a proof",
                "continuity_moments": [
                    {
                        "timestamp": time.time() - 10,
                        "focal_object": "the proof",
                        "focal_quality": "sharp",
                        "domain": "cognitive",
                        "attention_intensity": 0.9,
                        "narrative_thread": "deeply following a proof",
                        "emotional_tone": "focused",
                        "substrate_velocity": 0.3,
                        "brief": "tracking the proof",
                    }
                ],
                "last_emotion": "curious",
                "saved_at": time.time() - 3600,
                "session_dominant_domain": "cognitive",
                "session_dominant_tone": "focused",
                "session_attention_stability": 0.82,
            }
        )
    )

    dummy = SimpleNamespace(
        save_dir=tmp_path,
        psm=SimpleNamespace(
            _phenomenal_reports=[],
            _witness_observation="",
            _present_description="",
        ),
        continuity=_DummyContinuity(),
        _current_emotion="neutral",
    )
    dummy._seed_continuity_from_memory = MethodType(
        PhenomenologicalExperiencer._seed_continuity_from_memory,
        dummy,
    )

    PhenomenologicalExperiencer._load_phenomenal_memory(dummy)

    assert dummy.psm._phenomenal_reports == ["report"]
    assert dummy.psm._witness_observation == "witness"
    assert dummy.psm._present_description == "present"
    assert dummy._current_emotion == "curious"
    assert len(dummy.continuity._moments) == 1
    assert "Returning after" in dummy.continuity.current_thread
    assert "deeply following a proof" in dummy.continuity.current_thread


def test_agency_core_self_development_prefers_native_audit_targeting(monkeypatch):
    import core.agency.self_development_patch as self_dev_patch_module
    import core.agency_core as agency_core_module

    monkeypatch.setattr(
        self_dev_patch_module,
        "_derive_initiatives_from_audit",
        lambda: [
            {
                "skill": "attention_deepening",
                "message": "Follow the strongest thread until it fully ignites.",
                "theory": "GWT",
            }
        ],
    )
    monkeypatch.setattr(
        agency_core_module,
        "get_audit_suite",
        lambda: SimpleNamespace(get_trend=lambda n=5: {"latest_index": 0.72, "index_trend": "stable"}),
    )

    dummy = SimpleNamespace(
        state=SimpleNamespace(
            initiative_energy=0.8,
            last_skill_use=0.0,
        )
    )

    action = AgencyCore._pathway_self_development(dummy, now=8000.0, idle_seconds=1200.0)

    assert action is not None
    assert action["skill"] == "attention_deepening"
    assert action["audit_driven"] is True
    assert action["theory_target"] == "GWT"
    assert dummy.state.last_skill_use == 8000.0
