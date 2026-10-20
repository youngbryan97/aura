"""What one shard of the swarm does, and why shards are off.

Until 19 October 2026 a shard sent her model a script: "[SOVEREIGN SWARM SHARD]
... You are an autonomous cognitive shard ... DO NOT GUESS ... CRITICAL: You MUST
respond with a valid JSON object", with two tools described in prose and a JSON
protocol to call them. When the JSON did not parse, the shard wrote her answer
itself: "I am experiencing cognitive formatting degradation. Taking a defensive
posture."

Nothing read a shard's reply as hers. Its spawners read only whether it was
admitted. What it concluded went to two places, and each reworded it before
keeping it: the dialectical crucible, which wrapped it as "Provisional belief:
<reply> This remains revisable until future evidence confirms it." and committed
that to her belief engine; and the abstraction engine, which put it to her model
again inside a worded request and kept the answer as a first principle. On 19
October 266 of her 288 beliefs had come from the crucible, whose only caller was
the shard, and one of them was the fabricated line above, reinforced 38 times.
Every spawner's goal is an order as well ("Argue for the validity and necessity
of the concept.", "Identify root cause for X failure and suggest/trigger fix.",
"Critically audit this research insight"), so a shard would carry an
instruction to her model even with the script gone.

So shards are off (`AURA_SWARM_SHARDS`). On, a shard gives her model its goal and
context as runtime evidence and nothing else, offers no tools, logs her reply as
it came and keeps none of it. A reply that did not come is no reply.

Called only by the swarm. Every name taken from `agency_core` is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import json

from core.runtime.flags import FlagKind, declare

#: Off: nothing needs a shard's reply to be hers (the module docstring says what
#: read it, and what that did to her beliefs). On, a shard sends no instruction
#: and keeps nothing, so turning it on costs inference and changes nothing she
#: holds.
SHARDS_FLAG = declare(
    "AURA_SWARM_SHARDS",
    kind=FlagKind.BOOL,
    default=False,
    description="Let the sovereign swarm run shards: goal and context to her model as evidence, reply logged only",
    owner="core.agency.agency_shard_work",
)


def shards_are_on() -> bool:
    """Whether a shard sent to the swarm runs at all."""
    return bool(SHARDS_FLAG.value())


def shard_messages(goal: str, context: str) -> list[dict[str, str]]:
    """What a shard gives her model: its goal and context as one piece of runtime evidence."""
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    evidence = ("[GROUNDING EVIDENCE FOR THIS TURN]\n"
                + json.dumps({"goal": str(goal), "context": str(context)}, ensure_ascii=False)
                + "\n[END GROUNDING EVIDENCE FOR THIS TURN]")
    return [{"role": RUNTIME_EVIDENCE_ROLE, "content": evidence}]


class _RunsOneShard:
    """Lifted whole out of SovereignSwarm; see agency_core.py."""

    def _publish_shard_count(self) -> None:
        """Advertise the number of shards that actually exist right now."""
        from .agency_core import (
            _AGENCY_BOUNDARY_ERRORS,
            _record_agency_degradation,
            _schedule_agency_task,
            capture_and_log,
            get_registry,
        )

        try:
            if self._registry_shards_update_pending:
                return
            self._registry_shards_update_pending = True
            observed = len(self.active_shards)

            async def _run_shards_update() -> None:
                try:
                    await get_registry().update(active_shards=observed)
                finally:
                    self._registry_shards_update_pending = False

            registry_task = _schedule_agency_task(
                _run_shards_update(),
                name="agency.registry.active_shards",
                on_unscheduled=lambda: setattr(self, "_registry_shards_update_pending", False),
            )
            if registry_task is None:
                self._registry_shards_update_pending = False
        except _AGENCY_BOUNDARY_ERRORS as e:
            self._registry_shards_update_pending = False
            _record_agency_degradation(e, action="active shard count registry update skipped")
            capture_and_log(e, {"context": "AgencyCore._publish_shard_count"})

    async def _shard_wrapper(self, goal: str, context: str, shard_id: str = "unknown") -> None:
        """One shard: the goal and its context to her model as evidence, and her reply logged as it came."""
        from .agency_core import (
            _AGENCY_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_agency_degradation,
            capture_and_log,
            logger,
        )

        try:
            router = ServiceContainer.get("llm_router", default=None)
            if router is None:
                return
            async with self._inference_semaphore:
                result = await router.generate_with_metadata(
                    "", messages=shard_messages(goal, context),
                    origin="sovereign_swarm_shard", purpose="swarm_shard", is_background=True)
            reply = str((result or {}).get("text") or "").strip()
            if not reply:
                logger.info("Swarm: shard %s had no reply (%s).", shard_id, (result or {}).get("error") or "empty")
                return
            logger.info("🧠 Shard %s reply: %s", shard_id, reply[:200])
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                mycelium.pulse_hypha("collective", "distributed_agency", success=True)
        except _AGENCY_BOUNDARY_ERRORS as e:
            _record_agency_degradation(e, action=f"shard {shard_id} execution isolated")
            capture_and_log(e, {"module": "SovereignSwarm", "goal": goal})
