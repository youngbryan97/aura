import math

from core.actuators.actuator_registry import get_actuator_registry
from core.adaptation.immune_executor import ImmuneHeuristicExecutor
from core.sensors.sensor_registry import get_sensor_registry


def test_sensor_registry_sync_and_read():
    registry = get_sensor_registry()
    registry.sync_from_world_model()

    data = registry.read_all()
    assert "port_east_load" in data
    assert "port_west_load" in data
    assert "warehouse_load" in data
    assert "system_cpu_usage" in data

    reliability = registry.get_reliability_vector()
    assert reliability["port_east_load"] == 1.0
    assert reliability["system_cpu_usage"] == 1.0


def test_sensor_registry_rejects_non_finite_readings():
    registry = get_sensor_registry()
    before = registry.read_all()["port_east_load"]

    assert registry.record_reading("port_east_load", math.nan) is False
    assert registry.read_all()["port_east_load"] == before


def test_actuator_registry_actions():
    registry = get_actuator_registry()

    # Test valid RerouteVesselActuator call
    res = registry.execute_action(
        "reroute_vessel", {"vessel_id": "Vessel_Alpha", "heading": 120.0, "speed": 18.0}
    )
    assert res.success is True
    assert "Vessel_Alpha" in res.message

    # Test invalid actuator name
    res_invalid = registry.execute_action("invalid_actuator", {})
    assert res_invalid.success is False

    # Test invalid parameters (speed exceeding max)
    res_speed = registry.execute_action(
        "reroute_vessel", {"vessel_id": "Vessel_Alpha", "heading": 90.0, "speed": 100.0}
    )
    assert res_speed.success is False

    res_nan = registry.execute_action(
        "reroute_vessel", {"vessel_id": "Vessel_Alpha", "heading": math.nan, "speed": 10.0}
    )
    assert res_nan.success is False


def test_immune_executor_uses_safe_arithmetic_resolver():
    executor = ImmuneHeuristicExecutor()
    sensors = {"port_east_load": 800.0}

    resolved = executor.resolve_params({"amount": "$port_east_load * 0.25"}, sensors)
    assert resolved["amount"] == 200.0

    blocked = executor.resolve_params({"amount": "$port_east_load / 0"}, sensors)
    assert blocked["amount"] == "$port_east_load / 0"
