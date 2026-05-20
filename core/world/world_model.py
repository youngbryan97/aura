"""core/world/world_model.py
=============================
Grounded Physics and Constraints Engine.

A deterministic, mathematical simulation of physical resource networks
representing real-world entities (vessels, ports, warehouses, routes)
with concrete physical limits, flow equations, and capacity constraints.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.PhysicsWorldModel")


@dataclass
class WorldEntity:
    """Represents a physical entity in the world network with hard constraints."""
    entity_id: str
    kind: str  # "node" (port, warehouse) or "edge" (vessel, route)
    capacity: float
    load: float
    flow_rate: float  # Current processing/travel speed
    max_flow_rate: float
    latency: float  # Current wait time or queue delay
    coordinates: Tuple[float, float] = (0.0, 0.0)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def enforce_constraints(self) -> None:
        """Enforces physical boundaries: non-negative inventory and capacity limits."""
        self.load = max(0.0, min(self.load, self.capacity))
        self.flow_rate = max(0.0, min(self.flow_rate, self.max_flow_rate))

        # Queue latency models exponential wait time if load exceeds 85% capacity
        utilization = self.load / max(1e-5, self.capacity)
        if utilization > 0.85:
            excess = utilization - 0.85
            self.latency = float(math.exp(excess * 5.0) - 1.0)
        else:
            self.latency = 0.0


class PhysicsWorldModel:
    """Grounded resource and constraints solver simulation."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.entities: Dict[str, WorldEntity] = {}
        self.sim_time: float = 0.0
        self._initialize_default_world()

    def _initialize_default_world(self) -> None:
        """Sets up a canonical supply chain resource network."""
        # Nodes
        self.add_entity(WorldEntity("Port_East", "node", capacity=1000.0, load=800.0, flow_rate=50.0, max_flow_rate=100.0, latency=0.0, coordinates=(35.6, 139.7)))
        self.add_entity(WorldEntity("Port_West", "node", capacity=1200.0, load=300.0, flow_rate=80.0, max_flow_rate=150.0, latency=0.0, coordinates=(37.7, -122.4)))
        self.add_entity(WorldEntity("Warehouse_Central", "node", capacity=5000.0, load=2000.0, flow_rate=200.0, max_flow_rate=400.0, latency=0.0, coordinates=(39.8, -98.5)))

        # Edges / Routes (flow corridors)
        self.add_entity(WorldEntity("Route_Pacific", "edge", capacity=500.0, load=0.0, flow_rate=20.0, max_flow_rate=30.0, latency=0.0, attributes={"source": "Port_East", "target": "Port_West", "distance_miles": 5000.0}))
        self.add_entity(WorldEntity("Vessel_Alpha", "edge", capacity=200.0, load=150.0, flow_rate=15.0, max_flow_rate=25.0, latency=0.0, coordinates=(36.0, 160.0), attributes={"route": "Route_Pacific", "heading": 90.0}))

    def add_entity(self, entity: WorldEntity) -> None:
        self.entities[entity.entity_id] = entity
        entity.enforce_constraints()

    def get_entity(self, entity_id: str) -> Optional[WorldEntity]:
        return self.entities.get(entity_id)

    def simulate(self, duration_s: float, actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Integrates physical states forward in time.

        Actions specify flow changes, routing changes, etc. E.g.
        actions = [{"type": "reroute", "entity_id": "Vessel_Alpha", "heading": 95.0, "speed": 18.0}]
        """
        actions = actions or []
        # 1. Apply Actions
        for action in actions:
            action_type = action.get("type")
            entity_id = action.get("entity_id")
            entity = self.get_entity(entity_id) if entity_id else None
            if not entity:
                continue

            if action_type == "reroute" and entity.kind == "edge":
                heading = action.get("heading")
                speed = action.get("speed")
                if heading is not None:
                    entity.attributes["heading"] = float(heading)
                if speed is not None:
                    entity.flow_rate = float(speed)
                    entity.enforce_constraints()
            elif action_type == "transfer" and entity.kind == "node":
                amount = float(action.get("amount", 0.0))
                target_id = action.get("target_id")
                target = self.get_entity(target_id) if target_id else None
                if target and entity.load >= amount:
                    # Move resource from entity to target
                    transfer_qty = min(amount, target.capacity - target.load)
                    entity.load -= transfer_qty
                    target.load += transfer_qty
                    entity.enforce_constraints()
                    target.enforce_constraints()

        # 2. Integrate Physics / Flow dynamics
        dt = min(duration_s, 3600.0)  # Integrate in max 1-hour ticks to preserve stability
        remaining = duration_s
        while remaining > 0:
            step = min(remaining, dt)
            self._physics_step(step)
            remaining -= step

        self.sim_time += duration_s

        return self.get_state_snapshot()

    def _physics_step(self, dt: float) -> None:
        """Calculates flow transfers and coordinates updates."""
        # Update Vessel Positions & Travel flow
        vessel = self.get_entity("Vessel_Alpha")
        if vessel and "heading" in vessel.attributes:
            speed_knots = vessel.flow_rate
            # 1 knot is approx 1.15 mph. In coordinate terms, rough Euler integration:
            dx = (speed_knots * 0.0001) * math.cos(math.radians(vessel.attributes["heading"])) * dt
            dy = (speed_knots * 0.0001) * math.sin(math.radians(vessel.attributes["heading"])) * dt
            lat, lon = vessel.coordinates
            vessel.coordinates = (lat + dy, lon + dx)
            vessel.enforce_constraints()

        # Node autonomous processing (decaying flow over time, shipping cargo arrival)
        port_east = self.get_entity("Port_East")
        port_west = self.get_entity("Port_West")
        if port_east and port_west:
            # Transfer cargo: Port East processes exports (load decreases), Port West processes imports
            processing_east = min(port_east.load, port_east.flow_rate * (dt / 3600.0))
            port_east.load -= processing_east
            port_east.enforce_constraints()

            # Central warehouse drains imports from Port West
            warehouse = self.get_entity("Warehouse_Central")
            if warehouse:
                intake = min(port_west.load, port_west.flow_rate * (dt / 3600.0))
                port_west.load -= intake
                warehouse.load += intake
                
                # Warehouse processes deliveries
                outflow = min(warehouse.load, warehouse.flow_rate * (dt / 3600.0))
                warehouse.load -= outflow

                port_west.enforce_constraints()
                warehouse.enforce_constraints()

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Returns the current state vectors for all entities."""
        return {
            "sim_time": self.sim_time,
            "entities": {
                eid: {
                    "kind": ent.kind,
                    "capacity": ent.capacity,
                    "load": ent.load,
                    "flow_rate": ent.flow_rate,
                    "latency": ent.latency,
                    "coordinates": ent.coordinates,
                    "attributes": ent.attributes.copy(),
                }
                for eid, ent in self.entities.items()
            }
        }


# Singleton Pattern
_instance: Optional[PhysicsWorldModel] = None


def get_physics_world_model() -> PhysicsWorldModel:
    global _instance
    if _instance is None:
        _instance = PhysicsWorldModel()
    return _instance
