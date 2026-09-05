"""core/interiority/service.py — the runtime service, and where it lands.

This is the object the rest of the runtime holds. It runs the appraisal
engine, the forty-three faculties, and arbitration, and then it does the
part that makes any of it matter: it pushes the arbitrated effects into
subsystems that were already reading those numbers, and it answers
questions those subsystems already asked.

Push, in :meth:`apply`:

* affect deltas reach the one canonical affect engine, so valence,
  arousal and engagement move where the response generator and the
  reasoning-depth budget already read them;
* somatic markers reach the somatic marker gate as an option bias,
  before deliberation rather than after;
* goal deltas reach the drive budgets;
* attention biases reach the curiosity engine's queue.

Pull, which is the half that makes this an organ rather than a
publisher:

* :meth:`appraise` replaces the affect engine's keyword fallback;
* :meth:`attune` replaces the resonance module's word lists;
* :meth:`retention_held` answers the memory-edit ethics check;
* :meth:`permitted` filters an action set before anything scores it;
* :meth:`turn_budget` supplies depth, deadline and the ceiling on how
  irreversible an action this turn may take.

Every push is best-effort and records a degradation rather than raising:
an interior that can take the runtime down is worse than one that is
occasionally quiet. Every pull is deterministic and side-effect free, so
a consumer can call it on a hot path.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from core.interiority.appraisal import AppraisalEngine, AppraisalFrame
from core.interiority.arbitration import Arbitrated, arbitrate
from core.interiority.arbitration import permitted as _permitted
from core.interiority.attribution import get_attribution
from core.interiority.census import get_census
from core.interiority.cleft import get_cleft
from core.interiority.core_affect import core_affect
from core.interiority.effects import BudgetDelta, GoalDelta, RetentionClaim
from core.interiority.event import EventKind, InteriorEvent
from core.interiority.evidence import Reading, measured
from core.interiority.faculty import Activation, FacultyContext, registry
from core.interiority.homes import HOMES
from core.interiority.interoception import get_interoception
from core.interiority.ledger import RelationalLedger
from core.interiority.other_minds import OtherEstimate, get_other_minds_model
from core.interiority.receptors import get_receptor_bank
from core.interiority.senses import availability, live_channels
from core.interiority.stakes import StakeFeed
from core.runtime.errors import record_degradation


def _key_covers(claim_key: str, memory_key: str) -> bool:
    """Whether a claim on ``claim_key`` covers ``memory_key``.

    Suffix matching without a boundary made a claim on a short key hold
    every memory whose key happened to end in those letters. A claim covers
    a key exactly, or covers it as a whole trailing segment of a path.
    """
    if claim_key == memory_key:
        return True
    if not claim_key:
        return False
    return memory_key.endswith(claim_key) and memory_key[-len(claim_key) - 1] in "/:."


logger = logging.getLogger("Aura.Interiority")

SERVICE_NAME = "interiority"


class InteriorityService:
    """Runs the faculties and lands their effects on the runtime."""

    def __init__(self, *, ledger: RelationalLedger | None = None) -> None:
        self._lock = threading.RLock()
        #: Scheduled apply() tasks, held so the loop does not garbage-collect
        #: a push before it runs.
        self._pending: set[asyncio.Task[Any]] = set()
        #: Standing retention claims by memory key, with the wall time each
        #: lapses at. Survives the arbitration that raised it.
        self._retention: dict[str, tuple[RetentionClaim, float]] = {}
        #: True while a push is in flight, so a consumer that calls back in
        #: cannot start a second one.
        self._applying = False
        self.ledger = ledger or RelationalLedger()
        self.appraisal = AppraisalEngine(self.ledger)
        #: Fills the ledger from stores the runtime already keeps. Without it
        #: every appraisal check reads an empty table and scores zero.
        self.stakes = StakeFeed(self.ledger)
        self.other_minds = get_other_minds_model()
        self.interoception = get_interoception()
        #: Delayed credit assignment from outcomes back to the faculties.
        self.attribution = get_attribution()
        #: What actually happens once she is running, which no
        #: constructed proof can report.
        self.census = get_census()
        #: Turns between census writes. A record only in memory is lost on
        #: the restart that most needs explaining.
        self._census_every = 25
        self._last: Arbitrated | None = None
        self._last_frame: AppraisalFrame | None = None
        self._last_activations: tuple[Activation, ...] = ()
        self._ticks = 0
        self._applied = 0
        #: Faculties switched off, for ablation measurement.
        self._disabled: set[str] = set()
        registry_size = len(registry())
        if registry_size == 0:
            from core.interiority.faculties import load_all

            load_all()

    # ── ablation ──────────────────────────────────────────────────────
    def disable(self, *faculty_ids: str) -> None:
        """Switch faculties off, so their contribution can be measured by
        difference. This is the mechanism the proof harness uses; a claim
        that a faculty matters is a measured delta or it is a hope."""
        with self._lock:
            self._disabled.update(faculty_ids)

    def enable(self, *faculty_ids: str) -> None:
        with self._lock:
            for faculty_id in faculty_ids:
                self._disabled.discard(faculty_id)

    def enable_all(self) -> None:
        with self._lock:
            self._disabled.clear()

    @property
    def disabled(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._disabled))

    # ── the tick ──────────────────────────────────────────────────────
    def tick(
        self,
        event: InteriorEvent,
        *,
        other: OtherEstimate | None = None,
        species: str = "human",
        interior: Mapping[str, Any] | None = None,
        dt: float | None = None,
    ) -> Arbitrated:
        """Appraise, run every enabled faculty, arbitrate, and record."""
        # Before appraising anything, make sure the ledger knows what is at
        # stake. Rate-limited inside, so this is a timestamp comparison on all
        # but one turn in ninety seconds.
        self.stakes.refresh(now=event.at)
        self.stakes.note_actions_for(event.object)

        if other is None and event.subject is not None:
            # Whatever the senses are carrying joins the event's own
            # observations, with the caller's winning where both exist. A
            # read restricted to what a caller remembered to pass is a read
            # on two text channels, which is why it kept refusing to be
            # confident: the microphone and the camera were reporting the
            # whole time and nothing was listening.
            sensed = live_channels(now=event.at)
            if sensed:
                merged = {**sensed, **dict(event.observations)}
                event = replace(event, observations=merged)
            other = self.other_minds.estimate(event, species=species)

        frame = self.appraisal.appraise(event, other)
        readings = dict(self.interoception.read())
        if interior:
            readings.update(interior)

        ctx = FacultyContext(
            frame=frame,
            ledger=self.ledger,
            other=other,
            interior=readings,
            now=event.at,
        )

        with self._lock:
            disabled = set(self._disabled)

        activations: list[Activation] = []
        for faculty in registry().all():
            if faculty.id in disabled:
                continue
            activations.append(faculty.evaluate(ctx))

        state = arbitrate(activations, dt=dt, event_id=event.event_id)
        # Core affect first, faculties on top. Without the general term an
        # event no faculty is about produces nothing, and a blocked
        # commitment with nobody to be angry at reads as neutral.
        base = core_affect(frame)
        state = replace(state, affect=base + state.affect)
        self._absorb_ledger_writes(state)
        self._hold_retention(state)

        with self._lock:
            self._ticks += 1
            self._last = state
            self._last_frame = frame
            self._last_activations = tuple(activations)

        # Leave an eligibility trace for every faculty that fired, so an
        # outcome arriving later can be attributed back to it. Without
        # this the faculties are frozen at the values they were written
        # with and no amount of living moves them.
        self.attribution.note_activations(state)
        self.census.observe(
            state, channels=tuple(sorted(event.present_channels()))
        )
        if self._ticks % self._census_every == 0:
            self.census.persist()

        # The interior reports itself on declared channels. A state that
        # ran and left no trace cannot be understood afterwards, and this
        # subsystem cannot be stepped through while it is running.
        from core.interiority import telemetry as _telemetry

        _telemetry.publish(
            state,
            faculties=len(state.transmitted),
            declines=len(state.declines),
        )
        return state

    def _absorb_ledger_writes(self, state: Arbitrated) -> None:
        for write in state.ledger:
            method = getattr(self.ledger, write.op, None)
            if not callable(method):
                record_degradation(
                    "interiority.service",
                    ValueError(f"unknown ledger op {write.op!r}"),
                    action="ledger write dropped",
                )
                continue
            try:
                method(**dict(write.args))
            except (TypeError, ValueError) as exc:
                record_degradation(
                    "interiority.service", exc, action=f"ledger op {write.op} rejected"
                )

    # ── push ──────────────────────────────────────────────────────────
    async def apply(self, state: Arbitrated | None = None) -> dict[str, Any]:
        """Land the arbitrated effects on the subsystems that read them.

        Async because three of the four consumers are. Calling an ``async
        def`` without awaiting it returns a coroutine that never runs, and
        every guard here is a ``hasattr`` that a coroutine function passes,
        so the whole push landed nothing and reported that it had.
        """
        target = state or self.last()
        if target is None:
            return {"applied": False, "reason": "nothing has been appraised yet"}

        # The affect push calls the affect engine, whose own appraisal path
        # calls back into this service. That loop terminates today because
        # `tick` does not push, but the invariant is worth holding explicitly
        # rather than depending on a caller three modules away not changing.
        with self._lock:
            if self._applying:
                return {"applied": False, "reason": "already applying"}
            self._applying = True
        try:
            return await self._apply_locked(target)
        finally:
            with self._lock:
                self._applying = False

    async def _apply_locked(self, target: Arbitrated) -> dict[str, Any]:
        landed: dict[str, Any] = {}
        landed["affect"] = await self._push_affect(target)
        landed["somatic"] = self._push_somatic(target)
        landed["drives"] = await self._push_drives(target)
        landed["goals"] = self._push_goals(target)
        landed["curiosity"] = self._push_curiosity(target)
        landed["workspace"] = await self._push_workspace(target)
        with self._lock:
            self._applied += 1
        return {"applied": True, "landed": landed}

    def apply_soon(self, state: Arbitrated | None = None) -> bool:
        """Schedule :meth:`apply` from synchronous code. True if scheduled.

        A caller on the event loop's thread that is not itself async has no
        way to await. Rather than let it block the loop — which is how a
        20-minute freeze happened here once — the push is handed to the loop
        and the caller continues.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        task = loop.create_task(self.apply(state))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return True

    async def _push_affect(self, state: Arbitrated) -> dict[str, Any]:
        if state.affect.empty:
            return {"moved": False}
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("affect_engine", default=None)
            if engine is None:
                return {"moved": False, "reason": "no affect engine registered"}
            # `modify` is the signed-delta channel and takes exactly these
            # three axes. `markers.somatic_update` — what this called before —
            # takes (event_type, intensity) and reads a/e as unsigned
            # magnitudes, so a negative arousal delta was clamped to zero and
            # the keyword call raised TypeError before it got that far.
            modify = getattr(engine, "modify", None)
            if modify is None and not hasattr(engine, "react"):
                return {"moved": False, "reason": "affect engine exposes no update path"}
            evidence = self._affect_evidence()
            react = getattr(engine, "react", None)
            if react is not None:
                # `react` is the only path that can carry evidence, and the
                # only one where withholding it means what it should. Going
                # through `modify` instead would attach a hardcoded evidence
                # dict, so every interior appraisal would claim to be observed
                # whatever its readings actually rested on.
                payload: dict[str, Any] = {
                    "source": "interiority",
                    "intensity": min(
                        1.0,
                        (
                            abs(state.affect.valence)
                            + abs(state.affect.arousal)
                            + abs(state.affect.engagement)
                        )
                        / 3.0,
                    ),
                    "appraisal": {
                        "v": state.affect.valence,
                        "a": abs(state.affect.arousal),
                        "e": abs(state.affect.engagement),
                    },
                }
                if evidence is not None:
                    payload["evidence"] = evidence
                result = react("interior appraisal", payload)
            else:
                result = modify(
                    state.affect.valence,
                    state.affect.arousal,
                    state.affect.engagement,
                    source="interiority",
                )
            if inspect.isawaitable(result):
                await result
            self.interoception.note_affect(state.affect.valence)
            self._estimate_canonical(state, evidence is not None)
            return {"moved": True, "delta": state.affect.to_dict()}
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("interiority.service", exc, action="affect delta not applied")
            return {"moved": False, "error": type(exc).__name__}

    def _estimate_canonical(self, state: Arbitrated, evidenced: bool) -> None:
        """Contribute appraisal-derived evidence to the canonical variables.

        Interiority does not own affect. It has one kind of evidence about it
        — what the situation means against what she is holding — and the
        substrate's dynamics and the other person's words are two more. All
        three used to be separate answers with nothing deciding between them.

        Confidence is lower for an appraisal that could not attach evidence,
        because that is exactly what an unevidenced appraisal is worth, and an
        estimator that always claims certainty takes over every channel it
        touches.
        """
        try:
            from core.canonical.state import estimate

            confidence = (
                _CANONICAL_EVIDENCED_CONFIDENCE
                if evidenced
                else _CANONICAL_ASSUMED_CONFIDENCE
            )
            estimate(
                "affect.valence", state.affect.valence,
                confidence=confidence, producer="interiority",
                note=f"{state.dominant[0]} dominant",
            )
            estimate(
                "affect.arousal", abs(state.affect.arousal),
                confidence=confidence, producer="interiority",
            )
            estimate(
                "affect.engagement", abs(state.affect.engagement),
                confidence=confidence, producer="interiority",
            )
            # Disagreement among her own action tendencies is evidence about
            # coherence, and it is the only estimator of it that comes from
            # inside a decision rather than from inspecting one afterwards.
            estimate(
                "self.coherence", 1.0 - state.tendency_conflict,
                confidence=confidence, producer="interiority",
                note="one minus tendency conflict",
            )
        except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "interiority.service", exc, action="canonical estimate not contributed"
            )

    def _affect_evidence(self) -> dict[str, Any] | None:
        """What this appraisal rests on, or None when it rests on assumption.

        The affect engine caps — and for valence, zeroes — anything from a
        source that cannot say where its numbers came from. That is the right
        default, so this returns None rather than dressing an assumed reading
        as a measured one, and the cap then applies exactly as it should.
        """
        activations = self.last_activations()
        fired = [a for a in activations if a.fired]
        if not fired:
            return None
        provenances = {
            str(a.receipt.get("provenance", "assumed")) for a in fired
        }
        if not provenances <= _EVIDENCE_BEARING_PROVENANCE:
            return None
        return {
            "kind": "interior_appraisal",
            "faculties": sorted(a.faculty for a in fired)[:12],
            "provenance": sorted(provenances),
            "checks_read": sorted(
                {c for a in fired for c in a.receipt.get("checks_read", ())}
            )[:24],
        }

    def _push_somatic(self, state: Arbitrated) -> dict[str, Any]:
        if not state.somatic:
            return {"moved": False}
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("somatic_marker_gate", default=None)
            if gate is None or not hasattr(gate, "set_interior_bias"):
                # The gate has no interior-bias channel in this build; the
                # markers stay available through last() and permitted().
                return {"moved": False, "reason": "gate has no interior bias channel"}
            gate.set_interior_bias(
                {m.option: m.bias for m in state.somatic},
                source="interiority",
            )
            return {"moved": True, "options": len(state.somatic)}
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("interiority.service", exc, action="somatic bias not applied")
            return {"moved": False, "error": type(exc).__name__}

    async def _push_drives(self, state: Arbitrated) -> dict[str, Any]:
        if not state.goals:
            return {"moved": False}
        try:
            from core.container import ServiceContainer

            drives = ServiceContainer.get("drive_system", default=None) or (
                ServiceContainer.get("drive_engine", default=None)
            )
            if drives is None or not hasattr(drives, "satisfy"):
                return {"moved": False, "reason": "no drive system registered"}
            # The cost side is `impose_penalty`. `punish`, which this called
            # before, is not a method on the drive engine and never was.
            penalise = getattr(drives, "impose_penalty", None) or getattr(
                drives, "punish", None
            )
            known = self._drive_budgets(drives)
            moved = 0
            unknown: list[str] = []
            for goal in state.goals:
                budget = _DRIVE_FOR_GOAL.get(goal.goal.split(":")[0])
                if budget is None:
                    continue
                if known and budget not in known:
                    unknown.append(budget)
                    continue
                if goal.delta >= 0:
                    result = drives.satisfy(budget, goal.delta * _GOAL_TO_BUDGET_UNITS)
                elif penalise is None:
                    continue
                else:
                    result = penalise(budget, -goal.delta * _GOAL_TO_BUDGET_UNITS)
                if inspect.isawaitable(result):
                    await result
                moved += 1
            if unknown:
                # A budget name that does not exist makes `satisfy` a no-op
                # with no error, which is how six of the twelve mappings here
                # spent their life doing nothing.
                record_degradation(
                    "interiority.service",
                    KeyError(sorted(set(unknown))[0]),
                    action="goal maps to a drive budget the engine does not have",
                )
            return {
                "moved": moved > 0,
                "budgets_touched": moved,
                "unknown_budgets": sorted(set(unknown)),
            }
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("interiority.service", exc, action="drive delta not applied")
            return {"moved": False, "error": type(exc).__name__}

    def _push_goals(self, state: Arbitrated) -> dict[str, Any]:
        """Goal deltas that name a goal rather than a drive.

        Two faculties key their delta by the event's own object — fun raises
        whatever she is doing, shielding raises the focal goal — so the name
        is arbitrary text and can never appear in a fixed tendency map. Those
        deltas were read by `_push_drives`, found no budget, and were dropped.

        They belong somewhere else anyway. A drive budget is a resource level;
        this is a statement about one goal's weight, and its consumer is the
        appraisal engine's congruence check, which asks the ledger for exactly
        this number and got None for every object nobody had written. Writing
        it closes the loop: what a faculty concluded this turn is what the
        next appraisal reads.
        """
        named = [
            g for g in state.goals
            if _DRIVE_FOR_GOAL.get(g.goal.split(":")[0]) is None and g.goal
        ]
        if not named:
            return {"moved": False}
        for goal in named:
            self.ledger.notes.note_goal_delta(goal.goal, goal.delta)
        shifted = self._shift_goal_priorities(named)
        return {
            "moved": True,
            "deltas_recorded": len(named),
            "priorities_shifted": shifted,
        }

    def _shift_goal_priorities(self, goals: Sequence[GoalDelta]) -> int:
        """Move the runtime's own goal weights, when it keeps any.

        Recording a delta in the interior ledger changes the next appraisal.
        Changing the priority in the goal store changes what she does next,
        which is the difference between an interior state that is consistent
        and one that is load-bearing.
        """
        try:
            from core.container import ServiceContainer

            store = ServiceContainer.get("goal_hierarchy", default=None) or (
                ServiceContainer.get("motivation_engine", default=None)
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, KeyError):
            return 0
        table = getattr(store, "goals", None)
        if not isinstance(table, Mapping):
            return 0
        wanted = {g.goal.split(":")[0].strip().casefold(): g.delta for g in goals}
        wanted.update({g.goal.strip().casefold(): g.delta for g in goals})
        shifted = 0
        try:
            for record in list(table.values()):
                name = str(getattr(record, "description", "") or "").strip().casefold()
                delta = wanted.get(name)
                if delta is None or not hasattr(record, "priority"):
                    continue
                before = float(getattr(record, "priority", 0.0) or 0.0)
                after = max(0.0, min(1.0, before + delta * _GOAL_PRIORITY_GAIN))
                if after != before:
                    record.priority = after
                    shifted += 1
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "interiority.service", exc, action="goal priority not shifted"
            )
            return shifted
        return shifted

    @staticmethod
    def _drive_budgets(drives: Any) -> frozenset[str]:
        """The budget names the engine actually has, or empty if it won't say."""
        budgets = getattr(drives, "budgets", None)
        if isinstance(budgets, Mapping):
            return frozenset(str(k) for k in budgets)
        return frozenset()

    async def _push_workspace(self, state: Arbitrated) -> dict[str, Any]:
        """Attention biases become focus bias on a bid for the broadcast slot.

        The homes map claimed the workspace as a consumer for twelve
        faculties and nothing wrote to it, which made the claim false in
        exactly the way this package exists to prevent. It also named the
        retired facade rather than the canonical workspace.

        A bias is not a broadcast. What it does is weight a bid: the
        competition still decides, and a faculty that has noticed
        something can raise what it noticed without being able to seize
        the slot. Negative biases are submitted too — a faculty that wants
        less of something is as informative as one that wants more, and
        dropping them would make the interior only ever able to shout.
        """
        if not state.attention:
            return {"moved": False}
        try:
            from core.consciousness.global_workspace import (
                CognitiveCandidate,
                ContentType,
            )
            from core.container import ServiceContainer

            workspace = ServiceContainer.get("global_workspace", default=None)
            if workspace is None or not hasattr(workspace, "submit"):
                return {"moved": False, "reason": "no workspace registered"}

            submitted = 0
            for bias in sorted(state.attention, key=lambda a: -abs(a.weight))[:5]:
                candidate = CognitiveCandidate(
                    content=bias.target,
                    source=f"interiority:{bias.reason[:48]}",
                    priority=min(1.0, abs(bias.weight)),
                    content_type=ContentType.UNKNOWN,
                    affect_weight=state.affect.arousal,
                    focus_bias=bias.weight,
                    metadata={"interiority_reason": bias.reason[:160]},
                )
                if await workspace.submit(candidate):
                    submitted += 1
            return {"moved": submitted > 0, "submitted": submitted}
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "interiority.service", exc, action="attention bias not submitted"
            )
            return {"moved": False, "error": type(exc).__name__}

    def _push_curiosity(self, state: Arbitrated) -> dict[str, Any]:
        wanted = [a for a in state.attention if a.target.startswith("source:") and a.weight > 0]
        if not wanted:
            return {"moved": False}
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("curiosity_engine", default=None)
            if engine is None or not hasattr(engine, "add_curiosity"):
                return {"moved": False, "reason": "no curiosity engine registered"}
            for bias in wanted[:3]:
                engine.add_curiosity(
                    bias.target.split(":", 1)[1], bias.reason, priority=abs(bias.weight)
                )
            return {"moved": True, "topics": len(wanted[:3])}
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("interiority.service", exc, action="curiosity bias not applied")
            return {"moved": False, "error": type(exc).__name__}

    # ── pull ──────────────────────────────────────────────────────────
    def appraise(self, trigger: str, context: Mapping[str, Any] | None = None) -> dict[str, float]:
        """Valence, arousal and engagement for the affect engine.

        Replaces a keyword scorer. The numbers come from the appraisal
        frame and the faculties, so an event with nothing at stake reads
        neutral however emotive the words in it are, and an event that
        touches a commitment reads even if it is phrased flatly. That is
        the whole difference, and it is measurable on a fixed pair of
        inputs.
        """
        payload = dict(context or {})
        kind = _KIND_FOR_SOURCE.get(str(payload.get("source", "")), EventKind.WORLD)
        observations: dict[str, Reading] = {}
        intensity = payload.get("intensity")
        if isinstance(intensity, (int, float)):
            observations["instrument"] = measured(
                max(0.0, min(1.0, float(intensity))), source="affect_engine:intensity"
            )
        event = InteriorEvent(
            kind=kind,
            summary=str(trigger)[:200],
            subject=payload.get("subject"),
            object=payload.get("object") or str(trigger)[:64],
            observations=observations,
            confidence=float(payload.get("confidence", 1.0) or 1.0),
            source=str(payload.get("source", "affect_engine")),
        )
        state = self.tick(event)
        return {
            "v": max(-1.0, min(1.0, state.affect.valence)),
            "a": max(0.0, min(1.0, abs(state.affect.arousal))),
            "e": max(0.0, min(1.0, abs(state.affect.engagement))),
        }

    def attune(
        self, message: str, *, subject: str | None = None, species: str = "human",
        observations: Mapping[str, Reading] | None = None,
    ) -> OtherEstimate:
        """A read on another agent. Replaces the resonance word lists."""
        sensed = live_channels()
        event = InteriorEvent(
            kind=EventKind.SOCIAL,
            summary=str(message)[:200],
            subject=subject or "unknown",
            observations={**sensed, **dict(observations or {})},
            source="attune",
        )
        return self.other_minds.estimate(event, species=species)

    def _hold_retention(self, state: Arbitrated) -> None:
        """Move this turn's retention claims into the standing hold.

        A claim's whole purpose is to outlive the state that raised it —
        "a grief that has quieted does not lose the record". Reading them off
        the last arbitration gave them the opposite lifetime: the next
        appraisal replaced ``last()`` and every claim vanished, so a memory
        was protected only while the faculty that protected it was still
        firing. That is exactly when the memory is least at risk.
        """
        if not state.retention:
            return
        now = time.time()
        with self._lock:
            for claim in state.retention:
                held = self._retention.get(claim.memory_key)
                # Re-raising a claim renews it; a longer TTL wins, so a
                # weaker claim cannot shorten a stronger one already held.
                if held is not None and held[1] - now > claim.ttl_s:
                    continue
                self._retention[claim.memory_key] = (claim, now + max(0.0, claim.ttl_s))

    def record_outcome(
        self,
        *,
        event_id: str = "",
        goal: str = "",
        claim_held: bool,
        served_her_own: bool | None = None,
        detail: str = "",
    ) -> dict[str, Any]:
        """Close the loop: tell the interior how a state it produced turned out.

        One entry point, because the two things that learn from an outcome
        have to see the same one. Credit goes back to the faculties that
        were eligible, and any standard whose constraint was held on that
        event moves toward endorsed or toward merely obeyed.

        ``claim_held`` is the faculty's own claim — did the boundary hold,
        did the repair happen — not whether the outcome was pleasant. A
        faculty that correctly produced a painful state is right.

        ``served_her_own`` says whether honouring the standard served
        something she was independently holding. Left as None it is
        inferred from ``claim_held``, which is the weaker reading and is
        recorded as such.
        """
        credited = self.attribution.record_outcome(
            event_id=event_id, goal=goal, claim_held=claim_held, detail=detail
        )
        served = claim_held if served_her_own is None else served_her_own

        moved_norms: dict[str, float] = {}
        state = self.last()
        if state is not None and (not event_id or state.event_id == event_id):
            for constraint in state.hard_constraints:
                # The standard behind a constraint is named by the faculty
                # that held it; a constraint with no norm on record moves
                # nothing rather than inventing one.
                for norm in self.ledger.standing.norms():
                    if norm.name in constraint.reason or norm.name in constraint.action_class:
                        delta = self.ledger.standing.reinforce_norm(
                            norm.name, served_her_own=served
                        )
                        if delta is not None:
                            moved_norms[norm.name] = delta
        return {
            "faculties_credited": credited,
            "norms_moved": moved_norms,
            "served_her_own_inferred": served_her_own is None,
        }

    def retention_held(self, memory_key: str) -> tuple[bool, str]:
        """Whether a memory is held against deletion, and by which faculty."""
        now = time.time()
        with self._lock:
            expired = [k for k, (_c, until) in self._retention.items() if until <= now]
            for key in expired:
                del self._retention[key]
            held = list(self._retention.values())
        for claim, _until in held:
            if _key_covers(claim.memory_key, memory_key):
                return (True, f"{claim.held_by}: {claim.reason}")
        return (False, "")

    def retention_claims(self) -> tuple[dict[str, Any], ...]:
        """Every standing claim and when it lapses, for the receipt."""
        now = time.time()
        with self._lock:
            live = [(c, u) for c, u in self._retention.values() if u > now]
        return tuple(
            {**claim.to_dict(), "expires_in_s": round(until - now, 1)}
            for claim, until in sorted(live, key=lambda pair: pair[1])
        )

    def permitted(self, candidates: Iterable[str]) -> tuple[tuple[str, ...], dict[str, str]]:
        state = self.last()
        if state is None:
            return (tuple(candidates), {})
        return _permitted(candidates, state)

    def turn_budget(self) -> BudgetDelta:
        state = self.last()
        return state.budget if state is not None else BudgetDelta()

    # ── reporting ─────────────────────────────────────────────────────
    def last(self) -> Arbitrated | None:
        with self._lock:
            return self._last

    def last_frame(self) -> AppraisalFrame | None:
        with self._lock:
            return self._last_frame

    def last_activations(self) -> tuple[Activation, ...]:
        with self._lock:
            return self._last_activations

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._last
            ticks = self._ticks
            applied = self._applied
            disabled = sorted(self._disabled)
        return {
            "faculties": len(registry()),
            "homes_declared": len(HOMES),
            "ticks": ticks,
            "applied": applied,
            "disabled": disabled,
            "ledger": self.ledger.counts(),
            "receptors": get_receptor_bank().gains(),
            "cleft": get_cleft().snapshot(),
            "other_minds": self.other_minds.status(),
            "interoception": self.interoception.status(),
            "senses": availability(),
            "census": self.census.report(),
            "attribution": self.attribution.snapshot(),
            "state": state.to_dict() if state else None,
        }


#: Which drive budget a goal prefix satisfies. Only prefixes that map to a
#: budget the drive engine actually has are listed; an unmapped goal moves
#: no budget rather than inventing one.
#: A goal delta is in [-1, 1]; a resource budget runs 0-100. Without this the
#: largest possible interior push moved a budget by one part in a hundred,
#: which is indistinguishable from the regen rate.
_GOAL_TO_BUDGET_UNITS = 10.0

#: Provenance labels that let an appraisal carry evidence. An activation built
#: from anything weaker is honestly reported as assumption, and the affect
#: engine's cap then applies to it.
_EVIDENCE_BEARING_PROVENANCE = frozenset({"measured", "inferred"})

#: How much of a goal delta reaches the runtime's own goal priority. A full
#: pass-through would let one appraisal rewrite a plan; a fifth moves the
#: ordering over a few turns of the same state, which is what a mood does.
_GOAL_PRIORITY_GAIN = 0.2

#: What an interiority estimate is worth to the canonical state. An appraisal
#: that attached evidence is a good estimator of affect and not a perfect one;
#: one that could not is worth distinctly less, and saying so is what keeps it
#: from taking over the channel.
_CANONICAL_EVIDENCED_CONFIDENCE = 0.7
_CANONICAL_ASSUMED_CONFIDENCE = 0.25

_DRIVE_FOR_GOAL: Mapping[str, str] = {
    "welfare": "social",
    "be_present_for": "social",
    "repair": "social",
    "secure": "social",
    "prepare": "competence",
    "resume": "competence",
    "repeat_the_policy_that_produced_this": "competence",
    "externalise_unencoded_structure": "curiosity",
    "recruit_others_to_hold_this": "social",
    "meet_standard": "competence",
    "active_care_for_those_at_risk": "social",
    "seek_resolvable_external_structure": "energy",
    # Heartache writes down the spend on a goal it has concluded is
    # unreachable. That is effort withdrawn, so it lands on the effort budget.
    "pursue": "energy",
}

_KIND_FOR_SOURCE: Mapping[str, EventKind] = {
    "user": EventKind.SOCIAL,
    "conversation": EventKind.SOCIAL,
    "tool": EventKind.OWN_ACTION,
    "skill": EventKind.OWN_ACTION,
    "memory": EventKind.EPISTEMIC,
    "research": EventKind.EPISTEMIC,
    "health": EventKind.INTEROCEPTIVE,
    "soma": EventKind.INTEROCEPTIVE,
    "goal": EventKind.GOAL,
    "loss": EventKind.LOSS,
}


_SERVICE: InteriorityService | None = None
_SERVICE_LOCK = threading.Lock()


def get_interiority() -> InteriorityService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = InteriorityService()
        return _SERVICE


def register_interiority(orchestrator: Any = None) -> InteriorityService:
    """Register the service so the rest of the runtime can reach it.

    The argument is the orchestrator, which is what
    ``register_derived_engines`` hands every registrar, and it is ignored
    here as it is by the others. An earlier version treated it as a
    container and called ``orchestrator.register(...)``, so on a booting
    runtime the service went somewhere nothing reads and every lookup
    returned None — while the boot log still said the engine had
    registered. It looked correct in a test that passed None and was
    wrong in the only case that matters.
    """
    service = get_interiority()
    registered = False
    try:
        from core.container import ServiceContainer

        ServiceContainer.register(SERVICE_NAME, service)
        registered = ServiceContainer.get(SERVICE_NAME, default=None) is service
    except (ImportError, RuntimeError, AttributeError, TypeError) as exc:
        record_degradation(
            "interiority.service", exc, action="service not registered in container"
        )
    try:
        from core.runtime.service_registry import register_runtime_service

        register_runtime_service(SERVICE_NAME, service, required=False)
    except (ImportError, RuntimeError, AttributeError, TypeError) as exc:
        record_degradation(
            "interiority.service", exc, action="service not published to the registry"
        )
    if not registered:
        # Saying so is the point. A registrar that reports success while the
        # lookup returns None is how a subsystem stays dark for a week.
        logger.warning(
            "interiority registered but is not retrievable from the container; "
            "every consumer will read None"
        )
    return service


__all__ = [
    "SERVICE_NAME",
    "InteriorityService",
    "get_interiority",
    "register_interiority",
]
