"""The signals a drive update is computed from.

Lifted whole out of `motivation_update`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .motivation_update import (
        AuraState,
    )


class _ReadsTheDriveSignals:
    """Lifted whole out of MotivationUpdatePhase; see motivation_update.py."""

    def _surprise_pressure(self, state: Any = None) -> float:
        """How urgently the world is asking to be acted on. 0.0 when unknown.

        The world model's surprise against how surprising its moments usually
        are, with arousal as the gain (core/affect/arousal_gain.py): a world
        going worse than its model expects presses harder on an aroused
        organism, and one going better than expected presses less. What the
        world model saw and how she feels meet here rather than being added
        somewhere downstream.

        Without a world model the free-energy engine's action urgency stands
        in, ungained, as it did before. Zero is the honest default when neither
        is there: an engine that is not there has not told us the world is
        calm, so the drives keep their ordinary rate rather than being told to
        hurry by an absence.
        """
        from .motivation_update import (
            get_runtime_service,
        )

        world = self._world_surprise_ratio()
        if world is not None:
            from core.affect.arousal_gain import gained

            affect = getattr(state, "affect", None)
            arousal = getattr(affect, "arousal", 0.0) if affect is not None else 0.0
            return gained(world, arousal, usual=0.5)
        try:
            # Through the registry: this is an observer reading a rate, and
            # `peek` is exactly right for it — a reading must not boot the
            # engine it is reading.
            engine = get_runtime_service("free_energy_engine", default=None)
            if engine is None:
                return 0.0
            return max(0.0, min(1.0, float(engine.get_action_urgency())))
        # not a failure: a reading that cannot be taken, or that is not a
        # number when it is, leaves this at the value below.
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _world_surprise_ratio() -> float | None:
        """The world model's surprise as a share of it plus its usual surprise, or None.

        None rather than zero when there is no reading, so the caller can tell
        a world behaving exactly as modelled from a world model that is not
        there. Read through the observer seam, like `_world_surprise`.
        """
        from .motivation_update import (
            get_runtime_service,
        )

        try:
            from core.consciousness.workspace_feed import surprise_ratio

            model = get_runtime_service("unified_world_model", default=None)
            surprise = model.surprise() if model is not None else None
            if surprise is None:
                return None
            return surprise_ratio(model, surprise)
        # not a failure: no world model means no surprise to ratio.
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def _substrate_dominance() -> float:
        """How settled the continuous substrate is, or zero when it cannot say."""
        from .motivation_update import (
            get_runtime_service,
        )

        try:
            from core.runtime.service_registry import get_runtime_service

            substrate = get_runtime_service("conscious_substrate", default=None)
            reading = substrate.get_substrate_affect() if substrate is not None else None
            if not isinstance(reading, dict):
                return 0.0
            return max(-1.0, min(1.0, float(reading.get("dominance", 0.0) or 0.0)))
        # not a failure: a reading that cannot be taken, or that is not a
        # number when it is, leaves this at the value below.
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _substrate_volatility() -> float:
        """How fast the continuous substrate is moving. 0.0 if it is not there."""
        from .motivation_update import (
            get_runtime_service,
        )

        try:
            from core.runtime.service_registry import get_runtime_service

            substrate = get_runtime_service("conscious_substrate", default=None)
            reading = substrate.get_state_summary_nowait() if substrate is not None else None
            if not isinstance(reading, dict) or reading.get("snapshot_stale"):
                return 0.0
            return max(0.0, min(1.0, float(reading.get("volatility", 0.0) or 0.0) / 100.0))
        # not a failure: a reading that cannot be taken, or that is not a
        # number when it is, leaves this at the value below.
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _novelty() -> float:
        """How unlike her ordinary life this moment is. 0.0 if unknown."""
        try:
            from core.ontogeny.lifetime import last_reading

            reading = last_reading()
            return max(0.0, min(1.0, float(getattr(reading, "novelty", 0.0) or 0.0)))
        # not a failure: a reading that cannot be taken, or that is not a
        # number when it is, leaves this at the value below.
        except (ImportError, AttributeError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _world_surprise() -> float:
        """How far the world just departed from the model of it. 0.0 if unknown."""
        from .motivation_update import (
            get_runtime_service,
        )

        try:
            # An observer: reading how far the world departed from the model
            # must not be what instantiates the model.
            model = get_runtime_service("unified_world_model", default=None)
            value = model.surprise() if model is not None else None
            return 0.0 if value is None else max(0.0, min(1.0, float(value)))
        # not a failure: a reading that cannot be taken, or that is not a
        # number when it is, leaves this at the value below.
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0

    @classmethod
    def _situational_pressure(cls, state: AuraState) -> float:
        """How badly the moment is going, in [0, 1]. Read, not chosen."""

        worst = max(cls._footing(state).values(), default=0.0)
        return max(0.0, min(1.0, worst))

