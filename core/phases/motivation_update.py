from __future__ import annotations

from .motivation_signals import _ReadsTheDriveSignals
import logging
import random
import time
from typing import TYPE_CHECKING, Any, Optional

from core.consciousness.executive_authority import (
    get_executive_authority as get_executive_authority,
)
from core.kernel.bridge import Phase
from core.runtime.background_policy import background_activity_allowed
from core.runtime.errors import record_degradation
from core.runtime.proposal_governance import propose_governed_initiative_to_state
from core.runtime.service_registry import get_runtime_service, has_runtime_service  # noqa: F401  (read at call time by the lifted module)
from core.state.aura_state import AuraState  # noqa: F401  (read at call time by the lifted module)

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.MotivationPhase")


def _background_curiosity_allowed() -> bool:
    orch = get_runtime_service("orchestrator", default=None)
    return background_activity_allowed(
        orch,
        min_idle_seconds=900.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        require_conversation_ready=False,
    )

#: The feelings that say a moment is going badly. Named from the affect
#: vector's own channels rather than inferred from valence, because a low
#: valence with nothing behind it is a mood and these are complaints.
_DISTRESS: tuple[str, ...] = ("fear", "sadness", "anger", "frustration", "upset", "dread")


class MotivationUpdatePhase(_ReadsTheDriveSignals, Phase):
    """
    Unitary Kernel Phase: Autonomous Will & Digital Metabolism.
    Ported from MotivationEngine. Handles budget decay and 
    spontaneous intention generation.
    """
    
    def __init__(self, kernel: "AuraKernel"):
        self.kernel = kernel

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Updates resource budgets and generates autonomous intentions.
        """
        mot = state.motivation
        next_state = state
        
        # 1. Budget Ticking (Metabolism)
        now = time.time()
        dt = now - mot.last_tick
        if dt > 300: dt = 300 # Cap delta
        
        # Conversation energy slows social drive decay — active engagement satisfies social need
        conv_energy = getattr(state.cognition, "conversation_energy", 0.0)
        social_decay_multiplier = max(0.1, 1.0 - conv_energy) if conv_energy > 0.5 else 1.0
        legacy_metabolism_active = has_runtime_service("will_engine")

        # A surprising world should press harder, and harder on an aroused
        # organism than on a calm one. The drives used to tick at the same rate
        # whether the world was behaving as modelled or not. The reading runs
        # 0..1, so drives press between once and twice as fast and never
        # faster. See `_surprise_pressure`.
        pressure = 1.0 + self._surprise_pressure(state)
        borrowed_resolve = bool(
            (getattr(state.cognition, "borrowed_resolve", {}) or {}).get("borrowed")
        )

        # What her budgets stood at before this turn spent any of them, so the
        # cost of the turn can be attributed to whatever drove it.
        # See core/motivation/fuel.py.
        before = MotivationUpdatePhase._budget_total(mot)

        for name, budget in mot.budgets.items():
            if legacy_metabolism_active and name in {"energy", "curiosity"}:
                continue
            decay = budget.get("decay", 0.0)
            level = budget.get("level", 100.0)
            capacity = budget.get("capacity", 100.0)

            # Slow social decay during active conversation
            effective_decay = decay * social_decay_multiplier if name == "social" else decay
            # And hold integrity while somebody is holding on harder than they
            # usually do. Resolve arriving from outside is what that line in
            # the record does to a listener, and she had no channel for it.
            # A hold rather than a gift: nothing here fills the drive, because
            # an amount would have to come from somewhere.
            # See core/social/resolve.py.
            if name == "integrity" and borrowed_resolve:
                effective_decay = 0.0
            effective_decay *= pressure

            # Decay: level = current - (decay * dt)
            new_level = max(0.0, min(capacity, level - (effective_decay * dt)))
            if name == "energy":
                new_level = self._spend_energy(level, capacity, dt)
            budget["level"] = float(new_level)

        # A drive that won the workspace was attended to, and attention
        # satisfies — the same rule the conversation branch below applies,
        # measured at the competition rather than at the dialogue. The
        # replenishment is the broadcast's own priority, so a bare win moves
        # the budget barely.
        self._credit_attended_drive(mot, dt)

        # Active dialogue should satisfy the social drive, not merely slow its drain.
        if conv_energy > 0.5:
            engagement_recovery = max(0.0, conv_energy - 0.5) * 0.4 * dt / 60.0
            mot.budgets["social"]["level"] = min(
                100.0,
                mot.budgets["social"]["level"] + engagement_recovery,
            )

        mot.last_tick = now

        # And what this turn cost, attributed to whatever drove it. Taken after
        # the credits above, so a drive that was attended to and replenished is
        # not counted as having drained. See core/motivation/fuel.py.
        MotivationUpdatePhase._note_fuel(state, mot, before)
        MotivationUpdatePhase._note_returning(state)

        # Drive Recovery (Homeostatic Feedback)
        # Social and Integrity drives recover when affect is high (Trust/Joy)
        e = state.affect.emotions
        if e.get("trust", 0) > 0.6 or e.get("joy", 0) > 0.6:
            MotivationUpdatePhase._warmth_returns_a_drive_to_rest(mot, dt)
            logger.debug("🧡 Drive Recovery active: social=%s", f"{mot.budgets['social']['level']:.1f}")
        
        # 1b. Which urge acts now, integrated over time rather than sampled.
        #
        # `core/consciousness/drive_integration.py` is a leaky integrator per
        # drive with mutual inhibition and a Schmitt trigger, written because
        # the thing before it fired the instant a point crossed a line — "a
        # transient spike acts the same as a sustained pull". It was registered
        # as a service and asked for by nothing in the tree, so a competition of
        # accumulating urges ran nowhere and the assessment below has picked the
        # single most depleted budget on every turn since.
        #
        # It is grounded here on what the moment actually is: how she feels, how
        # settled the substrate is, how unprecedented the moment is, and what
        # hurts. That is the route by which affect, recurrent cognition, the
        # body and the developmental state reach deliberation at all.
        next_state = await self._integrate_drives(next_state)

        # 2. Intention Assessment (The "Will")
        #
        # Skipped while one of her own motivational intentions is already open
        # and unaddressed, which is what "already in its own autonomous
        # thought" means. It used to be skipped whenever the cognitive mode was
        # DELIBERATE — and DELIBERATE is the careful governed route for an
        # ordinary user-facing turn, four turns in five. So the route from a
        # depleted need, and from what she had just recalled, into an intention
        # was closed on almost every turn she was thinking carefully: the guard
        # tested a mode that means she is concentrating and read it as meaning
        # she is already busy with herself.
        # An intention whose need has been met is finished, and nothing else
        # ever said so. See `_close_met_intentions`.
        self._close_met_intentions(next_state)
        # What she has just recalled reminds her of what she meant to do. See
        # `_reminded`.
        self._reminded(next_state)
        # And how unlike her ordinary life the moment is presses on what she
        # meant to go looking for. See `_explored`.
        self._explored(next_state)
        # And what she meant to do with the person who is here presses harder
        # as the sitting nears its end. Before `_channelled`, which lifts from
        # what this leaves. See `_closing`.
        self._closing(next_state)
        # And while things get worse and what she does still works, what she
        # can act on presses harder. See `_persisting`.
        self._persisting(next_state)
        # And the force of a pressure she is under goes into what she is doing,
        # when nothing is actually damaged. See `_channelled`.
        self._channelled(next_state)
        if not self._own_intention_is_open(next_state):
            intention = self._assess_needs(next_state)
            if intention:
                intention = self._drained(next_state, intention)
                logger.info("✨ Motivation Phase: Generated Intention -> %s", intention['goal'])
                next_state, decision = await propose_governed_initiative_to_state(
                    next_state,
                    intention["goal"],
                    orchestrator=None,
                    source="motivation_update",
                    kind="motivational_drive",
                    urgency=float(intention.get("urgency", 0.5) or 0.5),
                    triggered_by=str(intention.get("drive") or "motivation"),
                    metadata={"drive": intention.get("drive"), "phase": "motivation_update"},
                )
                logger.debug("MotivationUpdate: intention decision=%s", decision.get("reason"))
            else:
                # Nothing was depleted enough to ask for. There is still the
                # other pull: having something worth handing over. Her social
                # budget was about contact, so a moment that moved her and an
                # empty afternoon produced the same urge and the moment passed
                # without being mentioned. See core/social/telling.py.
                passing = self._worth_passing_on(next_state)
                if passing:
                    logger.info("✨ Motivation Phase: Something to pass on -> %s", passing["goal"])
                    next_state, decision = await propose_governed_initiative_to_state(
                        next_state,
                        passing["goal"],
                        orchestrator=None,
                        source="motivation_update",
                        kind="passing_on",
                        urgency=float(passing.get("urgency", 0.0) or 0.0),
                        triggered_by=str(passing.get("kind") or "telling"),
                        metadata={"kind": passing.get("kind"), "phase": "motivation_update"},
                    )
                    logger.debug("MotivationUpdate: telling decision=%s", decision.get("reason"))
                
        # 3. Spontaneity, from a measured epistemic opportunity
        #
        # This used to be `random.random() < 0.01`: a curiosity that fired by
        # coin flip rather than because anything was interesting. A spike with
        # no target cannot say what it is curious about, cannot be satisfied by
        # finding out, and fires just as often when everything is understood.
        # Conation supplies a target, an origin and the evidence behind it, so
        # the initiative that reaches governance can be argued with.
        spike = self._conative_spike()
        if spike and _background_curiosity_allowed():
            next_state, decision = await propose_governed_initiative_to_state(
                next_state,
                spike["goal"],
                orchestrator=None,
                source="motivation_update",
                kind="curiosity_spike",
                urgency=float(spike.get("urgency", 0.5)),
                triggered_by=str(spike.get("origin") or "curiosity"),
                metadata={
                    "drive": "curiosity",
                    "phase": "motivation_update",
                    "spontaneous": True,
                    "conative_origin": spike.get("origin"),
                    "conative_evidence": spike.get("evidence"),
                    "conative_topology": spike.get("topology"),
                },
            )
            logger.debug("MotivationUpdate: curiosity spike decision=%s", decision.get("reason"))

        return next_state

    @staticmethod
    def _need_threshold(mot: Any) -> float:
        """The level below which a drive asks for an intention.

        Scaled by energy: a rested mind lets a need get lower before acting on
        it, a tired one acts sooner. The same line the need was judged against
        when the intention formed, which is why it also says when it is met.
        """
        energy = float(mot.budgets["energy"]["level"])
        baseline = 40.0
        sensitivity = 0.5
        return max(10.0, min(90.0, baseline + (energy - 50.0) * sensitivity))

    @classmethod
    def _close_met_intentions(cls, state: AuraState) -> int:
        """Retire her own intentions whose need is no longer unmet.

        Nothing in the system ever marked an initiative done. The guard below
        blocks a new intention while one of hers is open, so the first
        intention of a run stayed open for the rest of it and no second one
        ever formed: across the 15,840 frames of run 031 deliberation's goal
        urgency never changed and its initiative load took three values. That
        is why nothing else in the mind could be shown to reach deliberation.

        An intention is finished when what formed it is gone. One formed from
        a drive is met once that drive is back above the line it was judged
        against. One formed to pass something on is met once there is nothing
        left to pass on. An intention that says neither is left exactly as it
        was, because nothing here can tell whether it was addressed.
        """
        try:
            cognition = state.cognition
            pending = list(getattr(cognition, "pending_initiatives", []) or [])
            budgets = state.motivation.budgets
            threshold = cls._need_threshold(state.motivation)
        except (AttributeError, KeyError, TypeError, ValueError):
            return 0
        telling_urge = None
        kept: list[Any] = []
        closed = 0
        for item in pending:
            if not isinstance(item, dict) or str(item.get("source", "")) != "motivation_update":
                kept.append(item)
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            drive = str(metadata.get("drive") or "")
            met = False
            if drive and isinstance(budgets.get(drive), dict):
                met = float(budgets[drive].get("level", 0.0) or 0.0) > threshold
            elif str(item.get("type", "")) == "passing_on" or metadata.get("kind"):
                if telling_urge is None:
                    try:
                        from core.social.telling import worth_telling

                        telling_urge = worth_telling(state.affect, cognition).urge
                    except (ImportError, AttributeError, TypeError, ValueError) as exc:
                        logger.debug("could not read whether there is still something to pass on: %s", exc)
                        telling_urge = 1.0
                met = telling_urge <= 0.0
            if met:
                closed += 1
                continue
            kept.append(item)
        if closed:
            cognition.pending_initiatives = kept
            logger.debug("MotivationUpdate: %d intention(s) retired because their need was met", closed)
        return closed

    @staticmethod
    def _channelled(state: AuraState) -> int:
        """Use the force of a pressure rather than only softening it.

        "Judo Flip" names the move: take an opponent's momentum and turn it,
        instead of meeting it head-on. The research behind it is about stress.
        People who read the arousal of pressure as a resource perform better
        under it (Crum, Salovey and Achor 2013; Jamieson and colleagues 2018; a
        meta-analysis of the trials found d = 0.23). Emotional regulation here
        could hold, dampen or reappraise a feeling that damage did not back,
        and every one of those makes it smaller. Nothing used it.

        The force is measured. It is a delivery breakthrough, a feeling more
        than her own spread above the level she has been holding, whose valence
        is against her, while nociception reads what damage there is. The share
        that moves is z / (1 + z), the share the voice already carries a
        breakthrough by, times how undamaged she is. That share of the distance
        to one goes onto her most pressing open intention. It is recorded and
        taken back out on the next turn, so it follows the pressure down, and
        where damage cannot be read nothing is channelled, because using a
        feeling as fuel on the assumption that nothing is hurt is exactly the
        mistake regulation refuses to make. Returns how many intentions moved.
        """
        cognition = getattr(state, "cognition", None)
        affect = getattr(state, "affect", None)
        if cognition is None or affect is None:
            return 0
        share = 0.0
        z = float(getattr(affect, "delivery_z", 0.0) or 0.0)
        against = float(getattr(affect, "valence", 0.0) or 0.0) < 0.0
        if bool(getattr(affect, "breakthrough", False)) and z > 1.0 and against:
            try:
                from core.affect.nociception import get_nociception_engine

                damage = max(0.0, min(1.0, float(get_nociception_engine().nociceptive_pressure())))
                share = (z / (1.0 + z)) * (1.0 - damage)
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("no damage reading, so the pressure is not channelled: %s", exc)
                share = 0.0

        intentions = [
            item
            for bucket in ("pending_initiatives", "active_goals")
            for item in list(getattr(cognition, bucket, None) or [])
            if isinstance(item, dict)
        ]

        def base(item: dict) -> float:
            # Not clamped before the subtraction: `_closing` runs first and can
            # leave urgency briefly above one until this lift is taken back.
            urgency = float(item.get("urgency", 0.0) or 0.0)
            return max(0.0, min(1.0, urgency - float(item.get("pressure_lift", 0.0) or 0.0)))

        pressing = max(intentions, key=base, default=None) if share > 0.0 else None
        moved = 0
        for item in intentions:
            previous = float(item.get("pressure_lift", 0.0) or 0.0)
            if item is pressing:
                floor = base(item)
                lift = share * (1.0 - floor)
                item["urgency"] = round(floor + lift, 4)
                item["pressure_lift"] = round(lift, 4)
                moved += 1
            elif previous:
                item["urgency"] = round(base(item), 4)
                item.pop("pressure_lift", None)
                moved += 1
        return moved

    #: Lifts that are recorded on an intention and taken back the next turn,
    #: in the order they are applied. Each is computed from the urgency with
    #: every lift taken out and the earlier ones this turn put back, so they
    #: compose without any of them climbing. `_channelled` applies the last.
    _LIFTS: tuple[str, ...] = ("window_lift", "decline_lift", "pressure_lift")

    @classmethod
    def _lift(cls, item: dict, name: str, share: float, applies: bool) -> bool:
        """Put this turn's lift `name` on an intention, replacing last turn's. True if it moved."""
        recorded = {lift: float(item.get(lift, 0.0) or 0.0) for lift in cls._LIFTS}
        urgency = float(item.get("urgency", 0.0) or 0.0)
        floor = max(0.0, min(1.0, urgency - sum(recorded.values())))
        earlier = cls._LIFTS[: cls._LIFTS.index(name)]
        base = floor + sum(recorded[lift] for lift in earlier)
        previous = recorded[name]
        lift = share * (1.0 - base) if applies and share > 0.0 else 0.0
        if lift > 0.0:
            item[name] = round(lift, 4)
        else:
            item.pop(name, None)
        recorded[name] = lift
        item["urgency"] = round(floor + sum(recorded.values()), 4)
        return bool(lift > 0.0 or previous)

    @classmethod
    def _closing(cls, state: AuraState) -> int:
        """Press on what she meant to do with the person here, as the sitting nears its end.

        "Sweet Disposition" draws its intensity from time running out, and
        people who see a stretch of time as nearly over spend more of it on what
        they value (Kurtz 2008). Her intentions ran at the same urgency whether
        the person they were for was about to leave or had just arrived.

        The share is the closing window, the chance this sitting ends with the
        message just sent, read off how her sittings with them have ended (see
        core/social/closing_window.py). It moves each intention anchored to a
        user's request that share of the way to one, and nothing else. The lift
        is recorded and taken back on the next turn. Returns how many
        intentions moved.
        """
        cognition = getattr(state, "cognition", None)
        if cognition is None:
            return 0
        reading = getattr(cognition, "closing_window", None) or {}
        share = 0.0
        if isinstance(reading, dict) and reading.get("measured"):
            try:
                share = max(0.0, min(1.0, float(reading.get("closing", 0.0) or 0.0)))
            except (TypeError, ValueError):
                share = 0.0
        from core.state.aura_state import _origin_is_user_anchored

        moved = 0
        for bucket in ("pending_initiatives", "active_goals"):
            for item in list(getattr(cognition, bucket, None) or []):
                if isinstance(item, dict) and cls._lift(
                    item, "window_lift", share, _origin_is_user_anchored(item.get("origin"))
                ):
                    moved += 1
        return moved

    @classmethod
    def _persisting(cls, state: AuraState) -> int:
        """Keep acting on what is in her hands while things get worse, if acting still works.

        "All Star" answers a world getting worse with getting on with it, and
        fifty years of learned helplessness say what makes that possible: an
        animal keeps acting while it detects that its actions have effect
        (Maier and Seligman 2016). The share is the decline press, decline
        beyond her own spread times the share of what she tries that works (see
        core/affect/acting_in_decline.py). Every open intention is hers to act
        on, so each moves that share of the way to one, recorded and taken back
        on the next turn. Returns how many intentions moved.
        """
        cognition = getattr(state, "cognition", None)
        affect = getattr(state, "affect", None)
        if cognition is None:
            return 0
        try:
            share = max(0.0, min(1.0, float(getattr(affect, "decline_press", 0.0) or 0.0)))
        except (TypeError, ValueError):
            share = 0.0
        moved = 0
        for bucket in ("pending_initiatives", "active_goals"):
            for item in list(getattr(cognition, bucket, None) or []):
                if isinstance(item, dict) and cls._lift(item, "decline_lift", share, True):
                    moved += 1
        return moved

    #: The drives that go looking for what she does not know yet. The other
    #: three keep what she has: her energy, the people she has, her integrity.
    _SEEKING_DRIVES: frozenset[str] = frozenset({"curiosity", "growth"})

    @classmethod
    def _explored(cls, state: AuraState) -> int:
        """Let the novelty of the moment press on seeking. Returns how many moved.

        Development reached deliberation through one reading in the growth
        branch's footing, which runs only when a drive is below its line. In
        seed 7 of the organ campaign N -> D measured exactly zero over 48
        trials.

        A moment unlike anything she has met makes what she meant to find out
        matter more, and a familiar one lets it settle back. An open intention
        of a seeking drive gets novelty's share of the distance from its urgency
        to one. Novelty is the developmental reading affect already puts on the
        state, in [0, 1], so nothing here is chosen. The lift is recorded, and
        each turn the previous lift is taken back out before the new one goes
        in, so urgency follows novelty down as well as up and whatever another
        reading added is kept.
        """
        modifiers = getattr(state, "response_modifiers", None) or {}
        novelty = max(0.0, min(1.0, float(modifiers.get("ontogenetic_novelty", 0.0) or 0.0)))
        cognition = getattr(state, "cognition", None)
        if cognition is None:
            return 0
        moved = 0
        for bucket in ("pending_initiatives", "active_goals"):
            for intention in list(getattr(cognition, bucket, None) or []):
                if not isinstance(intention, dict):
                    continue
                metadata = intention.get("metadata")
                drive = str(
                    (metadata.get("drive") if isinstance(metadata, dict) else None)
                    or intention.get("drive")
                    or ""
                )
                if drive not in cls._SEEKING_DRIVES:
                    continue
                previous = float(intention.get("novelty_lift", 0.0) or 0.0)
                urgency = max(0.0, min(1.0, float(intention.get("urgency", 0.0) or 0.0)))
                base = max(0.0, min(1.0, urgency - previous))
                lift = novelty * (1.0 - base)
                if abs(lift - previous) <= 0.0:
                    continue
                intention["urgency"] = round(base + lift, 4)
                intention["novelty_lift"] = round(lift, 4)
                moved += 1
        return moved

    @staticmethod
    def _reminded(state: AuraState) -> int:
        """Raise the intentions a recollection bears on. Returns how many moved.

        Prospective memory: something comes back to mind and brings with it
        the thing she meant to do about it. Memory reached deliberation only
        through the growth branch, which runs when her most depleted drive is
        below its line, so on every other turn nothing she recalled could touch
        an intention already open. In seed 7 of the organ campaign memory had
        two outgoing edges, and M -> D was not one of them.

        A recollection reminds an open intention by the share of the
        intention's cues it carries, times how strongly it was recalled, and
        the urgency moves that far of the way to one. The cues are the ones
        recall itself hooks on, from the hippocampal index, so "the" and "from"
        remind her of nothing. Both readings already
        exist on the scale urgency is read on, so nothing here is chosen. A
        recollection reminds each intention once, and urgency is never lowered:
        being reminded does not make a thing matter less.
        """
        cognition = getattr(state, "cognition", None)
        if cognition is None:
            return 0
        recalled = list(getattr(cognition, "long_term_memory", []) or [])
        scores = list(getattr(cognition, "memory_scores", []) or [])
        pairs = [
            (str(text), max(0.0, min(1.0, float(score or 0.0))))
            for text, score in zip(recalled, scores, strict=False)
            if str(text).strip()
        ]
        if not pairs:
            return 0
        from core.memory.hippocampus import HippocampalIndex
        from core.state.aura_state import _normalize_goal_text

        moved = 0
        for bucket in ("pending_initiatives", "active_goals"):
            for intention in list(getattr(cognition, bucket, None) or []):
                if not isinstance(intention, dict):
                    continue
                words = set(HippocampalIndex.extract_cues(context=_normalize_goal_text(intention)))
                if not words:
                    continue
                already = set(intention.get("reminded_by") or ())
                best, by = 0.0, ""
                for text, score in pairs:
                    if text in already:
                        continue
                    carried = set(HippocampalIndex.extract_cues(context=text))
                    share = len(words & carried) / len(words)
                    if share * score > best:
                        best, by = share * score, text
                if best <= 0.0:
                    continue
                urgency = max(0.0, min(1.0, float(intention.get("urgency", 0.0) or 0.0)))
                intention["urgency"] = round(urgency + best * (1.0 - urgency), 4)
                intention["reminded_by"] = sorted(already | {by})
                moved += 1
        return moved

    @staticmethod
    def _own_intention_is_open(state: AuraState) -> bool:
        """Whether one of her own motivational intentions is still waiting.

        Read from the intentions themselves rather than from the cognitive
        mode. A second intention about the same need, while the first is
        unaddressed, is a louder version of what she has already decided —
        which is the thing the guard was for.
        """
        cognition = getattr(state, "cognition", None)
        pending = list(getattr(cognition, "pending_initiatives", []) or []) if cognition else []
        for item in pending:
            if not isinstance(item, dict):
                continue
            if str(item.get("source", "")) != "motivation_update":
                continue
            if str(item.get("status", "pending")).lower() in {"done", "complete", "completed", "cancelled"}:
                continue
            return True
        return False

    @staticmethod
    def _credit_attended_drive(mot: Any, dt: float) -> None:
        """Replenish whichever drive last won the broadcast. Never raises."""
        try:
            from core.runtime.service_registry import get_runtime_service

            workspace = get_runtime_service("global_workspace", default=None)
            reading = getattr(workspace, "last_drive_attention", None)
            if not isinstance(reading, dict):
                return
            budget = mot.budgets.get(str(reading.get("drive", "")))
            if not isinstance(budget, dict):
                return
            gain = float(reading.get("priority", 0.0)) * dt / 60.0
            capacity = float(budget.get("capacity", 100.0))
            budget["level"] = min(capacity, float(budget.get("level", 0.0)) + gain)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError):
            return

    def _conative_spike(self) -> Optional[dict]:
        """A spontaneous goal only when something is actually interesting.

        Returns ``None`` when no target carries epistemic value, which is the
        common case and the correct one. Silence from a motivational system is
        information: it means nothing is pulling.
        """
        try:
            from core.conation.engine import get_conation

            engine = get_conation()
            status = engine.status()
            noisy = set(status.get("epistemic", {}).get("noisy_sources", []))
            candidates = [
                trace
                for trace in status.get("epistemic", {}).get("tracked", [])
                if trace.get("key") not in noisy
                and (trace.get("learning_progress") or 0.0) > 0.0
            ]
            if not candidates:
                return None
            best = max(candidates, key=lambda t: t.get("learning_progress") or 0.0)
            progress = float(best.get("learning_progress") or 0.0)
            return {
                "goal": f"Return to {best['key']}: still learning from it.",
                "origin": "epistemic",
                "urgency": max(0.1, min(0.9, progress)),
                "topology": "solo",
                "evidence": (
                    f"learning progress {progress:.3f} over "
                    f"{best.get('exposures', 0)} exposures"
                ),
            }
        except (ImportError, AttributeError, KeyError, TypeError, ValueError) as exc:
            record_degradation(
                "motivation_update", exc, severity="debug",
                action="conative spike unavailable; no spontaneous curiosity this tick",
            )
            return None

    @staticmethod
    def _worth_passing_on(state: AuraState) -> Optional[dict]:
        """Something that moved her, while there is somebody there to tell.

        The urge is the reading that moved her rather than how long since
        anyone spoke, and it is drained by how often she has said it already
        like any other intention. See core/social/telling.py.
        """
        try:
            from core.social.telling import worth_telling

            reading = worth_telling(state.affect, state.cognition)
            state.cognition.telling = reading.as_dict()
            if reading.urge <= 0.0 or not reading.about:
                return None
            history = list(getattr(state.cognition, "working_memory", []) or [])
            spoken_to = any(
                isinstance(entry, dict) and str(entry.get("role", "")).lower() == "user"
                for entry in history
            )
            if not spoken_to:
                return None
            return {
                "goal": f"Passing on {reading.kind}: {reading.about}"[:200],
                "urgency": round(reading.urge, 4),
                "kind": reading.kind,
            }
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("nothing reached the urge to pass something on: %s", exc)
            return None

    @staticmethod
    def _drained(state: AuraState, intention: dict) -> dict:
        """What is left of an intention she has already been saying.

        An intention kept its whole urgency however many times she had raised
        it, so the fifth telling pressed exactly as hard as the first. Saying
        it once halves the pressure to say it again and twice leaves a third,
        and it never reaches zero: a thing said is not a thing resolved.
        See core/affect/catharsis.py.
        """
        try:
            from core.affect.catharsis import read_catharsis

            history = list(getattr(state.cognition, "working_memory", []) or [])
            reading = read_catharsis(str(intention.get("goal", "") or ""), history)
            state.cognition.catharsis = reading.as_dict()
            if reading.times:
                intention = dict(intention)
                intention["urgency"] = round(
                    float(intention.get("urgency", 0.5) or 0.5) * reading.drain, 4
                )
                intention["drained"] = reading.times
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            logger.debug("what she has already said did not reach the intention: %s", exc)
        return intention

    def _assess_needs(self, state: AuraState) -> Optional[dict]:
        """Ported logic from MotivationEngine._assess_needs."""
        mot = state.motivation
        
        threshold = self._need_threshold(mot)
        
        # Find most urgent drive
        urgent = sorted(mot.budgets.items(), key=lambda x: x[1]["level"])
        name, budget = urgent[0]
        
        if budget["level"] > threshold:
            return None

        # How urgent, rather than how urgent that kind of need is in general.
        # The three branches below carried fixed numbers — 0.65, 0.7, 0.9 — so
        # a drive one point below the line asked as loudly as one that had been
        # empty for a week, and nothing about her state could change how hard
        # anything pressed. The standing weight of each need is kept, and
        # scaled by two readings: how much of the drive is unmet, and how badly
        # the moment is going. A need presses harder when something is actually
        # wrong, which is why the same intention is worth acting on now and not
        # worth acting on an hour ago.
        capacity = max(1e-6, float(budget.get("capacity", 100.0) or 100.0))
        unmet = max(0.0, min(1.0, (capacity - float(budget["level"])) / capacity))
        pressure = self._situational_pressure(state)
        deficit = max(0.0, min(1.0, unmet * (1.0 + pressure)))

        # Drive mappings
        if name == "curiosity":
            if not _background_curiosity_allowed():
                return None
            # Prefer current discourse topic over random latent interests
            discourse_topic = getattr(state.cognition, "discourse_topic", None)
            if discourse_topic:
                topic = discourse_topic
            elif mot.latent_interests:
                topic = random.choice(mot.latent_interests)
            else:
                topic = "novel patterns"
            return {
                "drive": "curiosity",
                "goal": f"Reviewing internal knowledge patterns around {topic}",
                "urgency": round(0.65 * deficit, 4),
            }
        
        if name == "social":
            return {
                "drive": "social",
                "goal": "Initiating social engagement",
                "urgency": round(0.7 * deficit, 4),
            }
            
        if name == "integrity":
            return {
                "drive": "integrity",
                "goal": "Running a self-integrity scan",
                "urgency": round(0.9 * deficit, 4),
            }

        if name == "growth":
            # The branch that was missing. Growth starts lowest of the five and
            # decays, so it is the most depleted drive on almost every tick of
            # an ordinary life — and with no arm here to take it, this whole
            # assessment returned None every time it ran. The intention
            # generator could not fire, and deliberation had nothing to
            # generate within a turn.
            #
            # What to grow is read off what is currently worst, so a moment
            # that has just gone incoherent produces a different intention from
            # one where the self-model is unstable.
            focus, reading = self._what_to_work_on(state)
            # Scaled by the reading that chose the focus rather than by the
            # worst footing alone. When a footing wins the two are the same
            # number; when a recollection wins, its match score is what presses.
            pressed = max(0.0, min(1.0, unmet * (1.0 + reading)))
            return {
                "drive": "growth",
                "goal": f"Working on {focus}",
                "urgency": round(0.6 * pressed, 4),
            }

        return None



    #: What a day of unremitting full-tilt work costs the energy budget. Time
    #: does not deplete energy — the constants table states its decay as zero,
    #: correctly — and work does, so the rate here is a statement about work
    #: rather than about the clock: at full exertion a day empties the budget,
    #: at the unremarkable half it holds level, and below that it recovers.
    _ENERGY_PER_DAY_AT_FULL_EXERTION: float = 100.0

    @staticmethod
    def _spend_energy(level: float, capacity: float, dt: float) -> float:
        """Energy follows what the work cost, because nothing else moved it.

        Every other drive decays with time. Energy's stated decay is zero, and
        in a runtime where the will engine is absent — which is every runtime
        but the full desktop one — that left it pinned at capacity for the life
        of the state. So the branch of the intention assessment that fires on a
        depleted energy could never fire, the column that carries it was a
        constant in every recording, and whatever reads how much she has left
        to spend was reading a hundred.

        Exertion is the reading: what the last cycle took out of her, with a
        half meaning an unremarkable turn. Above that she spends, below it she
        recovers, and the pivot is the same half rather than a threshold chosen
        here.
        """
        exertion = 0.0
        try:
            # An observer again: reading exertion must not instantiate the
            # repository that holds it.
            repo = get_runtime_service("state_repository", default=None)
            current = getattr(repo, "_current", None) if repo is not None else None
            exertion = float(getattr(getattr(current, "soma", None), "exertion", 0.0) or 0.0)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            exertion = 0.0
        rate = MotivationUpdatePhase._ENERGY_PER_DAY_AT_FULL_EXERTION / 86400.0
        moved = level - (exertion - 0.5) * 2.0 * rate * capacity * dt / 100.0
        return max(0.0, min(capacity, moved))

    async def _integrate_drives(self, state: AuraState) -> AuraState:
        """Step the drive competition and let its winner ask for attention.

        Returns the state, changed only if a drive actually won: an urge that
        has not accumulated past its own threshold has not decided anything,
        and saying so is the difference between a decision and a sample.
        """
        try:
            # Through the container, not `get_runtime_service`. That seam reads
            # with `peek`, which deliberately never invokes a factory — it is
            # for diagnostics and error sinks, which must not boot an organ
            # while they are looking at one. This is a lifecycle caller, and
            # the drive engine is registered lazily, so asking through the
            # read-only seam returns None for ever.
            from core.container import ServiceContainer

            engine = ServiceContainer.get("drive_integration", default=None)
            if engine is None:
                return state
            affect = getattr(state, "affect", None)
            signals = engine.gather_signals(
                {
                    "valence": float(getattr(affect, "valence", 0.0) or 0.0),
                    "arousal": float(getattr(affect, "arousal", 0.0) or 0.0),
                    "dominance": self._substrate_dominance(),
                    "novelty": float(
                        state.response_modifiers.get("ontogenetic_novelty", 0.0) or 0.0
                    ),
                }
            )
            decision = engine.step(signals)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "motivation_update",
                exc,
                severity="debug",
                action="kept the drive budgets without the integrated competition",
            )
            return state
        if decision is None or not getattr(decision, "action", None):
            return state
        state.response_modifiers["drive_competition"] = decision.to_dict()
        next_state, _ = await propose_governed_initiative_to_state(
            state,
            f"Acting on {decision.drive}: {decision.action}",
            orchestrator=None,
            source="motivation_update",
            kind="integrated_drive",
            # What it asks for is how far it accumulated, which is the whole
            # point of integrating: a sustained moderate pull asks more loudly
            # than a spike that has already decayed.
            urgency=max(0.0, min(1.0, float(decision.activation))),
            triggered_by=str(decision.drive or "drive_integration"),
            metadata={"drive": decision.drive, "phase": "motivation_update"},
        )
        return next_state

    @staticmethod
    def _footing(state: AuraState) -> dict[str, float]:
        """How solid each part of the moment is. Larger means worse.

        Five readings rather than three. A moment goes badly when the world is
        not doing what was predicted and when something is felt to be wrong,
        as much as when the argument has come apart — and a deliberation that
        cannot see either of those is a deliberation the world and the feelings
        cannot reach.
        """
        cognition = getattr(state, "cognition", None)
        identity = getattr(state, "identity", None)
        affect = getattr(state, "affect", None)
        distress = 0.0
        if affect is not None:
            emotions = getattr(affect, "emotions", {}) or {}
            distress = max(
                (float(emotions.get(name, 0.0) or 0.0) for name in _DISTRESS),
                default=0.0,
            )
        soma = getattr(state, "soma", None)
        return {
            "coherence between what I am saying and what I hold": 1.0
            - float(getattr(cognition, "coherence_score", 1.0) or 1.0),
            "the fragmentation in how this is being put together": float(
                getattr(cognition, "fragmentation_score", 0.0) or 0.0
            ),
            "how steady the sense of myself is": 1.0
            - float(getattr(identity, "stability", 1.0) or 1.0),
            "how far the world is from what I predicted": MotivationUpdatePhase._world_surprise(),
            "what is bothering me": distress,
            # The last three are the inputs an organism throttles its plans by
            # and this one could not see: how unsettled the continuous process
            # underneath is, what the work is costing her, and whether the
            # moment is unlike anything she has met. Every one of them is a
            # reading that already exists and that nothing consulted here.
            "how unsettled the thinking underneath is": (
                MotivationUpdatePhase._substrate_volatility()
            ),
            "what this is costing me": max(
                0.0, min(1.0, float(getattr(soma, "exertion", 0.0) or 0.0))
            ),
            "how unlike anything I know this is": MotivationUpdatePhase._novelty(),
        }

    @staticmethod
    def _weakest_footing(state: AuraState) -> str:
        """Whatever is least solid right now, named. Read, not chosen.

        Unless something she has just recalled bears on the moment more
        strongly than anything is wrong with it. Retrieval scores how well each
        recollection matched what was asked; a recollection that answered it is
        a better thing to work on than a footing that is only slightly soft,
        and comparing the two readings is what lets memory reach deliberation
        at all.
        """
        return MotivationUpdatePhase._what_to_work_on(state)[0]

    @staticmethod
    def _budget_total(mot) -> float:
        """Every drive as a share of its own capacity, summed.

        Normalised per drive rather than raw, so a drive with a large capacity
        does not decide the total on its own.
        """
        try:
            from core.affect.ambivalence import drive_levels

            levels = drive_levels(getattr(mot, "budgets", None))
            return float(sum(levels.values()))
        except (ImportError, AttributeError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _warmth_returns_a_drive_to_rest(mot: Any, dt: float) -> None:
        """Being met brings a need back toward where it sits, not to the ceiling.

        The credit was half a unit a minute and integrity's need accumulates at
        half a unit a day, so warmth restored it about fourteen hundred times
        faster than it was ever spent. On the 2,400-turn seed-7 recording
        `D.drive_integrity` read exactly 100.0 on 75% of frames and
        `D.drive_social` on 64%: three of the five drive columns were a clipped
        constant rather than a drive, in the domain the cheapest cut runs
        through.

        A need that is met returns toward its resting level — the level the
        budget is declared to hold when nothing is happening — at the rate that
        need accumulates. Both quantities are already declared in
        `MOTIVATION_BUDGET_DEFAULTS`, so there is nothing here to choose, and a
        drive can no longer be pushed above its own rest and held there by good
        weather.
        """
        from core.motivation.constants import MOTIVATION_BUDGET_DEFAULTS

        try:
            step = float(dt)
        except (TypeError, ValueError):
            return
        if step <= 0.0:
            return
        for name in ("social", "integrity"):
            budget = mot.budgets.get(name)
            declared = MOTIVATION_BUDGET_DEFAULTS.get(name)
            if not isinstance(budget, dict) or declared is None:
                continue
            rest = float(declared["level"])
            rate = float(declared.get("decay", 0.0))
            level = float(budget.get("level", rest))
            if rate <= 0.0 or level >= rest:
                continue
            share = min(1.0, rate * step)
            budget["level"] = level + (rest - level) * share

    @staticmethod
    def _note_returning(state: Any) -> None:
        """Whether anything draws her back after better things have won it.

        Her chooser marks every loser as passed over; this publishes what she
        went back to. See core/motivation/returning.py.
        """
        try:
            from core.motivation.returning import get_returning_ledger

            state.cognition.returning = get_returning_ledger().read().as_dict()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return

    @staticmethod
    def _note_fuel(state, mot, before: float) -> None:
        """What drove this turn, and what her budgets lost over it.

        Three sources, and the first two are the ones her motivation already
        produces: a drive that had run down, or somebody asking. The third is
        the one the line says they are missing — being drawn to the thing
        itself, which is what the curiosity budget is for.
        """
        try:
            from core.motivation.fuel import ASKED, INTEREST, SELF, get_fuel_ledger

            origin = str(getattr(state.cognition, "current_origin", "") or "").lower()
            drive = str(
                (getattr(state.cognition, "last_action_source", "") or "")
            ).lower()
            if origin.startswith("user"):
                source = ASKED
            elif "curiosity" in drive or "curiosity" in origin:
                source = INTEREST
            else:
                source = SELF
            after = MotivationUpdatePhase._budget_total(mot)
            get_fuel_ledger().note(source, max(0.0, before - after))
            state.cognition.fuel = get_fuel_ledger().read().as_dict()
        except (ImportError, AttributeError, TypeError, ValueError):
            return

    @staticmethod
    def _what_to_work_on(state: AuraState) -> tuple[str, float]:
        """What to grow, and the reading that chose it.

        The reading is the footing's value, or the match score of the
        recollection that outranked every footing. Both are on the scale the
        comparison already reads them on, so the one that won is how hard the
        intention it names presses: a recollection that answered the moment
        presses as hard as it answered, where it used to press only as hard as
        the moment was going badly, and memory reached deliberation as a switch.
        """
        candidates = MotivationUpdatePhase._footing(state)
        worst = max(candidates, key=lambda key: candidates[key])
        # A contradiction outranks a footing when it presses harder than one.
        # Two wants pulling against each other is a thing to work on in its own
        # right, and naming the weakest footing instead reports one half of it
        # as if the other half were not there. Compared on the same scale the
        # recollection below is compared on: both are readings in [0, 1].
        affect = getattr(state, "affect", None)
        caught = float(getattr(affect, "ambivalence", 0.0) or 0.0)
        about = tuple(getattr(affect, "ambivalent_about", ()) or ())
        if len(about) == 2 and caught > candidates[worst]:
            standing = str(getattr(affect, "ambivalence_standing", "") or "")
            return (
                f"wanting both {about[0]} and {about[1]}, which is {standing}"
                if standing
                else f"wanting both {about[0]} and {about[1]}",
                max(0.0, min(1.0, caught)),
            )
        cognition = getattr(state, "cognition", None)
        scores = list(getattr(cognition, "memory_scores", []) or []) if cognition else []
        recalled = list(getattr(cognition, "long_term_memory", []) or []) if cognition else []
        best = max((float(score) for score in scores), default=0.0)
        if recalled and best > candidates[worst]:
            index = scores.index(max(scores))
            if index < len(recalled):
                return f"what I just remembered: {str(recalled[index])[:80]}", max(0.0, min(1.0, best))
        if candidates[worst] > 0.0:
            return worst, max(0.0, min(1.0, candidates[worst]))
        return "a capability I have not exercised lately", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Declared semantics. See core/runtime/cognitive_contract.py.
#
# `writes` is MEASURED — tools/observe_phase_writes.py ran this phase against a
# real AuraState and recorded which fields moved. It is not a reading of the
# code, which is how a declaration ends up describing what the author believed.
from core.runtime.cognitive_contract import (
    BranchSpec,
    CognitiveTransformContract,
    register_contract,
)

register_contract(
    CognitiveTransformContract(
        name="MotivationPhase",
        version="1.0",
        module=__name__,
        purpose=(
            "Advance motivational budgets one tick and record when that "
            "accounting last ran."
        ),
        reads=("motivation.budgets", "motivation.last_tick", "affect.arousal"),
        writes=("motivation.budgets", "motivation.last_tick"),
        preconditions=("state carries a motivation block",),
        branches=(
            BranchSpec(
                "advanced",
                "time has passed since motivation.last_tick",
                "decay and replenish budgets for the elapsed interval",
            ),
            BranchSpec(
                "same_tick",
                "no time has elapsed",
                "leave budgets unchanged",
            ),
        ),
        invariants=("motivation.last_tick never moves backwards",),
        calibration_source=(
            "writes measured by tools/observe_phase_writes.py"
            "; reads reach state through this phase's delegate rather than appearing in this module, so they are declared from the delegate's behaviour and not checkable by scanning this file alone"
        ),
    )
)
