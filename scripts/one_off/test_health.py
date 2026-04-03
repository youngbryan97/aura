from core.health.boot_status import build_boot_health_snapshot
from types import SimpleNamespace
class DummyOrch:
    pass
orch = DummyOrch()
orch.status = SimpleNamespace(initialized=True, running=True, healthy=True, last_error="")
runtime = {"sha256": "abc", "signature": "def"}
lane = {"conversation_ready": False, "state": "cold"}
payload, status = build_boot_health_snapshot(orch, runtime, is_gui_proxy=False, conversation_lane=lane)
print(status)
