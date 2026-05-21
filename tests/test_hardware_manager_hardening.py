from __future__ import annotations

import asyncio

import pytest

from core.embodiment import hardware_manager as hardware_module
from core.embodiment.base_device import BaseHardwareDevice
from core.embodiment.hardware_manager import HardwareManager


class Device(BaseHardwareDevice):
    def __init__(self, device_id: str, *, fail_connect: bool = False, bad_dict: bool = False):
        super().__init__(device_id, f"Device {device_id}", "test")
        self.fail_connect = fail_connect
        self.bad_dict = bad_dict
        self.disconnected = False

    async def connect(self) -> bool:
        if self.fail_connect:
            self.connect_attempted = True
            raise RuntimeError("connection failed")
        self.is_connected = True
        return True

    async def disconnect(self) -> bool:
        self.disconnected = True
        self.is_connected = False
        return True

    async def get_status(self) -> dict[str, object]:
        return {"ok": self.is_connected}

    async def execute_command(self, command: str, **kwargs) -> dict[str, object]:
        return {"ok": True, "command": command, "kwargs": kwargs}

    def to_dict(self) -> dict[str, object]:
        if self.bad_dict:
            self.serialize_attempted = True
            raise RuntimeError("serialization failed")
        return super().to_dict()


def test_hardware_manager_records_connect_failure_and_reports_health(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    manager = HardwareManager()
    manager.register_device(Device("ok"))
    manager.register_device(Device("bad", fail_connect=True))
    monkeypatch.setattr(hardware_module, "record_degradation", record_degradation)

    asyncio.run(manager.start())
    health = manager.get_health()

    assert health["started"] is True
    assert health["registered_devices"] == 2
    assert health["connected_devices"] == 1
    assert health["status"] == "degraded"
    assert "bad" in health["connection_failures"]
    assert recorded
    assert recorded[0][0] == "hardware_manager"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True


def test_hardware_manager_list_devices_returns_fallback_metadata(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []
    manager = HardwareManager()
    manager.register_device(Device("broken", bad_dict=True))
    monkeypatch.setattr(
        hardware_module,
        "record_degradation",
        lambda module, exc, **kwargs: recorded.append((module, type(exc).__name__, kwargs)),
    )

    devices = manager.list_devices()

    assert devices[0]["device_id"] == "broken"
    assert devices[0]["serialization_error"] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True


def test_hardware_manager_rejects_empty_device_id():
    manager = HardwareManager()
    device = Device("valid")
    device.device_id = ""

    with pytest.raises(ValueError):
        manager.register_device(device)
