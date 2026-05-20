import math

from core.world.world_model import PhysicsWorldModel, WorldEntity


def test_world_entity_constraints():
    entity = WorldEntity(
        entity_id="test_port",
        kind="port",
        capacity=100.0,
        load=80.0,
        flow_rate=10.0,
        max_flow_rate=20.0,
        latency=1.0,
        coordinates=(10.0, 20.0),
        attributes={},
    )
    assert entity.entity_id == "test_port"
    assert entity.kind == "port"
    assert entity.capacity == 100.0
    assert entity.load == 80.0
    assert entity.flow_rate == 10.0
    assert entity.max_flow_rate == 20.0
    assert entity.latency == 1.0


def test_world_model_rejects_non_finite_actions():
    model = PhysicsWorldModel()
    before = model.get_state_snapshot()

    after = model.simulate(
        duration_s=math.nan,
        actions=[
            {
                "type": "reroute",
                "entity_id": "Vessel_Alpha",
                "heading": math.nan,
                "speed": math.inf,
            },
            {
                "type": "transfer",
                "entity_id": "Port_East",
                "target_id": "Port_West",
                "amount": math.inf,
            },
        ],
    )

    assert after["sim_time"] == before["sim_time"]
    assert (
        after["entities"]["Vessel_Alpha"]["flow_rate"]
        == before["entities"]["Vessel_Alpha"]["flow_rate"]
    )


def test_world_model_bounds_malformed_entities_and_long_runs():
    entity = WorldEntity(
        entity_id="bad_coords",
        kind="node",
        capacity=math.inf,
        load=math.nan,
        flow_rate=math.inf,
        max_flow_rate=10.0,
        latency=math.nan,
        coordinates=(1.0,),
    )
    entity.enforce_constraints()

    assert entity.capacity == 0.0
    assert entity.load == 0.0
    assert entity.coordinates == (0.0, 0.0)

    model = PhysicsWorldModel()
    state = model.simulate(duration_s=10**12)
    assert state["sim_time"] == 24 * 3600


def test_physics_world_model_simulation():
    model = PhysicsWorldModel()

    # Add Port East, Port West, Vessel Alpha, Warehouse Central
    model.add_entity(
        WorldEntity(
            entity_id="Port_East",
            kind="node",
            capacity=1000.0,
            load=800.0,
            flow_rate=50.0,
            max_flow_rate=100.0,
            latency=2.0,
            coordinates=(0.0, 0.0),
            attributes={},
        )
    )
    model.add_entity(
        WorldEntity(
            entity_id="Port_West",
            kind="node",
            capacity=1200.0,
            load=400.0,
            flow_rate=30.0,
            max_flow_rate=120.0,
            coordinates=(100.0, 0.0),
            latency=1.0,
        )
    )
    model.add_entity(
        WorldEntity(
            entity_id="Vessel_Alpha",
            kind="edge",
            capacity=200.0,
            load=150.0,
            flow_rate=20.0,
            max_flow_rate=35.0,
            coordinates=(10.0, 0.0),
            latency=0.0,
            attributes={"heading": 90.0, "speed": 15.0},
        )
    )

    # Run simulation without actions
    state = model.simulate(duration_s=10.0)
    assert "entities" in state
    assert state["entities"]["Port_East"]["load"] < 800.0  # flow went out
    assert state["entities"]["Port_West"]["load"] < 400.0  # discharged cargo queue

    # Run simulation with actions (rerouting Vessel Alpha and transferring flow)
    actions = [
        {"type": "reroute", "entity_id": "Vessel_Alpha", "heading": 180.0, "speed": 25.0},
        {"type": "transfer", "entity_id": "Port_East", "target_id": "Port_West", "amount": 100.0},
    ]
    state_after = model.simulate(duration_s=10.0, actions=actions)
    assert state_after["entities"]["Port_West"]["load"] > 450.0  # received transfer
    assert state_after["entities"]["Vessel_Alpha"]["attributes"]["heading"] == 180.0
    assert state_after["entities"]["Vessel_Alpha"]["flow_rate"] == 25.0
