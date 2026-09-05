"""core/environment/embodied_simulator.py -- Persistent Embodied Simulator
========================================================================
Bridges Aura's cognitive architecture to a continuous physical simulator
loop. It maintains a persistent Scene Graph, interacts with the
Affordance Knowledge Base, and provides true physical grounding rather 
than just textual state representations.

This module actively updates physical object properties, distances,
and affordances, serving as the "raw physics" reality that MCTS
can plan against and SCM can perform do-calculus interventions on.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from core.brain.causal_world_model import CausalWorldModel
from core.perception.affordance_schema import AffordanceKnowledgeBase
from core.runtime.service_registry import get_runtime_service
from core.runtime.task_ownership import create_tracked_task, fire_and_forget

logger = logging.getLogger("Aura.EmbodiedSimulator")

@dataclass
class PhysicalEntity:
    """An object physically present in the embodied simulator."""
    entity_id: str
    class_name: str
    position: np.ndarray  # 3D coordinates [x, y, z]
    velocity: np.ndarray  = field(default_factory=lambda: np.zeros(3))
    state: dict[str, float] = field(default_factory=dict) # e.g. {"temperature": 0.5, "integrity": 1.0}
    last_updated: float = field(default_factory=time.time)

class SceneGraph:
    """Maintains the spatial and relational state of the immediate environment."""
    
    def __init__(self):
        self.entities: dict[str, PhysicalEntity] = {}
        self.agent_position = np.zeros(3)
        self.agent_velocity = np.zeros(3)

    def add_or_update_entity(self, entity_id: str, class_name: str, position: np.ndarray, state: dict[str, float] = None):
        if entity_id in self.entities:
            self.entities[entity_id].position = position
            if state:
                self.entities[entity_id].state.update(state)
            self.entities[entity_id].last_updated = time.time()
        else:
            self.entities[entity_id] = PhysicalEntity(
                entity_id=entity_id,
                class_name=class_name,
                position=position,
                state=state or {}
            )

    def get_nearby_entities(self, radius: float) -> list[PhysicalEntity]:
        nearby = []
        for ent in self.entities.values():
            dist = np.linalg.norm(ent.position - self.agent_position)
            if dist <= radius:
                nearby.append(ent)
        return nearby


class ContinuousSimulatorLoop:
    """The persistent physics/environment loop providing raw embodied grounding."""

    def __init__(self, affordance_kb: AffordanceKnowledgeBase, causal_model: CausalWorldModel):
        self.scene = SceneGraph()
        self.affordance_kb = affordance_kb
        self.causal_model = causal_model
        
        self.is_running = False
        self._loop_task: asyncio.Task | None = None
        self.tick_rate_hz = 10.0  # 10 updates per second

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._loop_task = create_tracked_task(
            self._physics_loop(),
            name="embodied_simulator.physics_loop",
        )
        logger.info("Embodied Simulator: Continuous physics loop STARTED.")

    async def stop(self):
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed %s in core.environment.embodied_simulator: %s", type(_exc).__name__, _exc)
        logger.info("Embodied Simulator: Continuous physics loop STOPPED.")

    async def _physics_loop(self):
        """Continuous physical update tick."""
        while self.is_running:
            start_time = time.time()
            
            # 1. Update simple kinematics
            self.scene.agent_position += self.scene.agent_velocity * (1.0 / self.tick_rate_hz)
            for ent in self.scene.entities.values():
                ent.position += ent.velocity * (1.0 / self.tick_rate_hz)
                
            # 2. Extract sensorimotor grounding vector (raw physics -> substrate)
            self._feed_sensorimotor_grounding()
            
            elapsed = time.time() - start_time
            sleep_time = max(0.01, (1.0 / self.tick_rate_hz) - elapsed)
            await asyncio.sleep(sleep_time)

    def _feed_sensorimotor_grounding(self):
        """Translates raw physics (scene graph) into neural substrate injections."""
        try:
            substrate = get_runtime_service("conscious_substrate", default=None)
            if substrate:
                # E.g. Arousal spikes if objects are moving fast near the agent
                nearby = self.scene.get_nearby_entities(radius=5.0)
                threat_level = 0.0
                for ent in nearby:
                    relative_vel = np.linalg.norm(ent.velocity - self.scene.agent_velocity)
                    if relative_vel > 2.0:
                        threat_level += 0.1
                        
                if threat_level > 0.0:
                    delta = np.zeros(64, dtype=np.float32)
                    delta[1] = min(0.5, threat_level) # Arousal
                    fire_and_forget(
                        substrate.inject_stimulus(delta, weight=0.2),
                        name="embodied_simulator.inject_stimulus",
                    )
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug("Embodied grounding injection failed: %s", e)

    def physical_intervention(self, entity_id: str, action: str, intensity: float) -> float:
        """
        Perform a true physical intervention (do-calculus physical execution).
        Executes the affordance, measures the physical result, and explicitly
        updates the Structural Causal Model with the discovery.
        """
        if entity_id not in self.scene.entities:
            return 0.0
            
        ent = self.scene.entities[entity_id]
        
        # Affordance priors modulate intervention strength instead of sitting
        # beside the simulator as narrative-only state.
        affs = self.affordance_kb.query(entities=[ent.class_name], action=action)
        confidence = max((aff.confidence for aff in affs), default=0.0)
        risk = max((aff.risk_level for aff in affs), default=0.0)
        bounded_intensity = max(0.0, min(1.0, float(intensity)))
        effective_intensity = bounded_intensity * (1.0 + (0.25 * confidence)) * (1.0 - (0.5 * risk))
        effective_intensity = max(0.0, min(1.25, effective_intensity))
        ent.state["last_affordance_confidence"] = confidence
        ent.state["last_affordance_risk"] = risk
        
        # The control outcome: what this measurement would have read had the
        # intervention not happened.
        #
        # The SCM refuses to mint a causal claim from a single level reading —
        # correctly, because one number is a correlation no matter how it was
        # produced. A treatment effect needs a control, and this simulator is
        # the one place that genuinely KNOWS its own counterfactual: it holds
        # the pre-intervention state and the physics that would have left it
        # alone. Discarding that and passing only the treated value meant a
        # real, executed intervention was recorded as a passive observation.
        if action == "push":
            control_outcome = 0.0                                  # no push, no force
        elif action == "heat":
            control_outcome = float(ent.state.get("temperature", 0.0))  # unheated reading
        else:
            control_outcome = 0.0

        # Execute "Physics" Simulation (Stubbed for actual environment hooks)
        # e.g., if action is 'push', alter velocity
        if action == "push":
            force = effective_intensity * 5.0
            ent.velocity[0] += force
            outcome_magnitude = force
        elif action == "heat":
            ent.state["temperature"] = ent.state.get("temperature", 0.0) + (effective_intensity * 10.0)
            outcome_magnitude = ent.state["temperature"]
        else:
            outcome_magnitude = effective_intensity
            
        # Explicit SCM Discovery via do-calculus
        # "I forced 'push' on 'block' and observed velocity increase"
        source_node = f"do_{action}_{ent.class_name}"
        target_node = f"{ent.class_name}_state_change"
        
        self.causal_model.discover_causality_via_intervention(
            source=source_node,
            target=target_node,
            source_val=effective_intensity,
            target_val_observed=outcome_magnitude,
            control_val=control_outcome,
            performed_by="embodied_simulator",
            environment=f"simulated_scene:{entity_id}",
        )
        
        logger.info("Embodied Intervention: do(%s on %s) -> SCM Updated.", action, ent.class_name)
        return outcome_magnitude
