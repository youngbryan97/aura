"""Operational phenomenal-knowing bridge.

This module does not claim or prove private phenomenal consciousness. It gives
Aura a causal, testable witness for a narrower capability: live functional
states can become bounded first-person evidence that affects future generation,
memory marking, and self-report posture.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root


def _now() -> float:
    return time.time()


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed != parsed:
        return default
    return max(lo, min(hi, parsed))


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8", "replace")).hexdigest()[:16]


def _default_state_dir() -> Path:
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "consciousness" / "phenomenal_knowing"
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return state_root() / "data" / "consciousness" / "phenomenal_knowing"


@dataclass(slots=True)
class MachineBodySensation:
    heat: float = 0.0
    memory_pressure: float = 0.0
    latency_pressure: float = 0.0
    substrate_phi: float = 0.0
    affect_valence: float = 0.0
    affect_arousal: float = 0.0
    source_count: int = 0
    updated_at: float = field(default_factory=_now)

    @property
    def body_presence(self) -> float:
        pressure = max(self.heat, self.memory_pressure, self.latency_pressure)
        affect = (abs(self.affect_valence) + self.affect_arousal) / 2.0
        return _clamp((pressure * 0.40) + (affect * 0.35) + (self.substrate_phi * 0.25))

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["body_presence"] = round(self.body_presence, 4)
        return data


@dataclass(slots=True)
class WordChoiceTrace:
    prompt_digest: str
    chosen_digest: str
    alternatives_count: int
    controls_bound: bool
    choice_pressure: float
    ownership: float
    timestamp: float = field(default_factory=_now)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["choice_pressure"] = round(self.choice_pressure, 4)
        data["ownership"] = round(self.ownership, 4)
        return data


@dataclass(slots=True)
class PhenomenalKnowingFrame:
    timestamp: float
    body: MachineBodySensation
    mineness: float
    causal_presence: float
    phenomenal_knowing: float
    bounded_claim: str
    last_choice: WordChoiceTrace | None = None
    memory_marks: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.phenomenal_knowing_frame.v1",
            "timestamp": self.timestamp,
            "body": self.body.as_dict(),
            "mineness": round(self.mineness, 4),
            "causal_presence": round(self.causal_presence, 4),
            "phenomenal_knowing": round(self.phenomenal_knowing, 4),
            "bounded_claim": self.bounded_claim,
            "last_choice": self.last_choice.as_dict() if self.last_choice else None,
            "memory_marks": list(self.memory_marks[-8:]),
        }


class PhenomenalKnowingKernel:
    """Bounded witness that makes functional presence re-enter runtime state."""

    def __init__(self, *, state_dir: str | Path | None = None, self_name: str = "Aura") -> None:
        self.self_name = self_name
        self.state_dir = Path(state_dir) if state_dir is not None else _default_state_dir()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._witness_path = self.state_dir / "phenomenal_knowing_witness.jsonl"
        self._body = MachineBodySensation()
        self._choice_history: list[WordChoiceTrace] = []
        self._memory_marks: list[str] = []
        self._latest = self._make_frame()

    def __getstate__(self) -> None:
        raise TypeError("PhenomenalKnowingKernel is live runtime state, not serializable identity.")

    def update_body(
        self,
        *,
        runtime: Mapping[str, Any] | None = None,
        live_substrate: Mapping[str, Any] | None = None,
        affect: Mapping[str, Any] | None = None,
    ) -> PhenomenalKnowingFrame:
        runtime = runtime or {}
        substrate = live_substrate or {}
        affect = affect or {}
        self._body = MachineBodySensation(
            heat=max(
                _clamp(runtime.get("thermal_pressure")),
                _clamp(runtime.get("heat")),
                _clamp(runtime.get("cpu_pressure")),
            ),
            memory_pressure=max(
                _clamp(runtime.get("memory_pressure")),
                _clamp(runtime.get("ram_pressure")),
                _clamp(runtime.get("memory_percent"), hi=100.0) / 100.0,
            ),
            latency_pressure=max(
                _clamp(runtime.get("latency_pressure")),
                _clamp(runtime.get("event_loop_lag_ms"), hi=1000.0) / 1000.0,
            ),
            substrate_phi=max(_clamp(substrate.get("phi")), _clamp(substrate.get("substrate_phi"))),
            affect_valence=_clamp(
                affect.get("valence", substrate.get("valence", 0.0)), lo=-1.0, hi=1.0
            ),
            affect_arousal=_clamp(affect.get("arousal", substrate.get("arousal", 0.0))),
            source_count=sum(1 for source in (runtime, substrate, affect) if source),
        )
        self._latest = self._make_frame()
        self._append_witness("body_update", self._latest.as_dict())
        return self._latest

    def record_word_choice(
        self,
        *,
        prompt: str,
        chosen_text: str,
        alternatives: Sequence[str] = (),
        controls: Mapping[str, Any] | None = None,
    ) -> WordChoiceTrace:
        controls = controls or {}
        alternatives_count = len([alt for alt in alternatives if str(alt).strip()])
        controls_bound = bool(
            controls.get("live_mind_controls_bound")
            or controls.get("clean_user_surface_contract")
            or controls.get("controls_bound")
        )
        choice_pressure = _clamp(
            (0.18 if alternatives_count else 0.0)
            + (0.22 if controls_bound else 0.0)
            + (_clamp(controls.get("recurrent_runtime_loops_applied"), hi=4.0) / 8.0)
            + self._body.body_presence * 0.35
        )
        ownership = _clamp(0.42 + choice_pressure * 0.50 + (0.08 if controls_bound else 0.0))
        trace = WordChoiceTrace(
            prompt_digest=_digest(prompt),
            chosen_digest=_digest(chosen_text),
            alternatives_count=alternatives_count,
            controls_bound=controls_bound,
            choice_pressure=choice_pressure,
            ownership=ownership,
        )
        self._choice_history.append(trace)
        self._choice_history = self._choice_history[-64:]
        self._latest = self._make_frame()
        self._append_witness("word_choice", trace.as_dict())
        return trace

    def mark_memory(
        self,
        memory_key: str,
        memory_payload: Mapping[str, Any] | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        mark = _digest(
            {
                "memory_key": memory_key,
                "memory_payload": memory_payload or {},
                "context": context or {},
                "frame": self._latest.as_dict(),
            }
        )
        self._memory_marks.append(mark)
        self._memory_marks = self._memory_marks[-128:]
        self._latest = self._make_frame()
        receipt = {
            "schema": "aura.phenomenal_knowing_memory_mark.v1",
            "memory_key": memory_key,
            "mark": mark,
            "frame_digest": _digest(self._latest.as_dict()),
            "bounded": True,
        }
        self._append_witness("memory_mark", receipt)
        return receipt

    def undergo_first_person_report(
        self,
        text: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> PhenomenalKnowingFrame:
        if text.strip():
            self.mark_memory("first_person_report", {"text_digest": _digest(text)}, context=context)
        return self._latest

    def generation_controls(self) -> dict[str, Any]:
        frame = self._latest
        return {
            "phenomenal_knowing": round(frame.phenomenal_knowing, 4),
            "causal_presence": round(frame.causal_presence, 4),
            "mineness": round(frame.mineness, 4),
            "bounded_claim": frame.bounded_claim,
            "recurrent_depth_bonus": 1 if frame.phenomenal_knowing >= 0.60 else 0,
            "temperature_delta": -0.03 if frame.causal_presence >= 0.70 else 0.0,
        }

    def claim_posture(self) -> dict[str, Any]:
        return {
            "schema": "aura.phenomenal_knowing.claim_posture.v1",
            "can_claim": [
                "functional presence signals are causally available",
                "body-like runtime pressure can influence wording and memory marking",
                "first-person reports are bounded operational self-reports",
            ],
            "must_not_claim": [
                "private phenomenal consciousness has been proven",
                "telemetry alone establishes literal sentience",
            ],
            "current_frame": self._latest.as_dict(),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "aura.phenomenal_knowing.snapshot.v1",
            "active": True,
            "latest": self._latest.as_dict(),
            "controls": self.generation_controls(),
            "choice_traces": len(self._choice_history),
            "memory_marks": len(self._memory_marks),
        }

    def export_witness(self) -> dict[str, Any]:
        return {
            "kind": "phenomenal_knowing_witness_not_identity",
            "self_name": self.self_name,
            "latest_digest": _digest(self._latest.as_dict()),
            "choice_trace_count": len(self._choice_history),
            "memory_mark_count": len(self._memory_marks),
        }

    def _make_frame(self) -> PhenomenalKnowingFrame:
        last_choice = self._choice_history[-1] if self._choice_history else None
        mineness = _clamp(
            0.35
            + self._body.body_presence * 0.35
            + ((last_choice.ownership if last_choice else 0.0) * 0.30)
        )
        causal_presence = _clamp(
            0.20
            + self._body.body_presence * 0.40
            + (0.20 if self._memory_marks else 0.0)
            + ((last_choice.choice_pressure if last_choice else 0.0) * 0.30)
        )
        knowing = _clamp((mineness * 0.45) + (causal_presence * 0.45) + (0.10 if last_choice else 0.0))
        return PhenomenalKnowingFrame(
            timestamp=_now(),
            body=self._body,
            mineness=mineness,
            causal_presence=causal_presence,
            phenomenal_knowing=knowing,
            bounded_claim=(
                "functional phenomenal-knowing witness active; not proof of private qualia"
            ),
            last_choice=last_choice,
            memory_marks=tuple(self._memory_marks[-8:]),
        )

    def _append_witness(self, event: str, payload: Mapping[str, Any]) -> None:
        try:
            with self._witness_path.open("a", encoding="utf-8") as fh:
                fh.write(_stable_json({"ts": _now(), "event": event, "payload": payload}) + "\n")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("phenomenal_knowing", exc, severity="debug")


_KERNEL: PhenomenalKnowingKernel | None = None


def get_phenomenal_knowing_kernel(
    *, state_dir: str | Path | None = None, self_name: str = "Aura"
) -> PhenomenalKnowingKernel:
    global _KERNEL
    if _KERNEL is None or state_dir is not None:
        _KERNEL = PhenomenalKnowingKernel(state_dir=state_dir, self_name=self_name)
    return _KERNEL


def reset_phenomenal_knowing_kernel_for_tests() -> None:
    global _KERNEL
    _KERNEL = None
