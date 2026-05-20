import pytest
from core.sensors.sensor_registry import get_sensor_registry, SensorRegistry
from core.actuators.actuator_registry import get_actuator_registry, ActuatorRegistry

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

def test_actuator_registry_actions():
    registry = get_actuator_registry()
    
    # Test valid RerouteVesselActuator call
    res = registry.execute_action("reroute_vessel", {"vessel_id": "Vessel_Alpha", "heading": 120.0, "speed": 18.0})
    assert res.success is True
    assert "Vessel_Alpha" in res.message
    
    # Test invalid actuator name
    res_invalid = registry.execute_action("invalid_actuator", {})
    assert res_invalid.success is False
    
    # Test invalid parameters (speed exceeding max)
    res_speed = registry.execute_action("reroute_vessel", {"vessel_id": "Vessel_Alpha", "heading": 90.0, "speed": 100.0})
    assert res_speed.success is False
