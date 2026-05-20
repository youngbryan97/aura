import asyncio
import json

from core.control import dynamic_router as router_module
from core.control.dynamic_router import DynamicRouter


def test_dynamic_router_quarantines_corrupt_history(tmp_path):
    router = DynamicRouter()
    router.db_path = tmp_path / "routing_history.json"
    router.db_path.write_text("{bad json", encoding="utf-8")

    router._load_history()

    assert router.performance_history == {}
    assert not router.db_path.exists()
    assert list(tmp_path.glob("routing_history.corrupt.*.json"))


def test_dynamic_router_migrates_and_sanitizes_legacy_history(tmp_path):
    router = DynamicRouter()
    router.db_path = tmp_path / "routing_history.json"
    router.db_path.write_text(
        json.dumps(
            {
                "Cortex": [
                    [True, "42", "fast_fact"],
                    {"success": "false", "tokens": "999999999", "task_type": "unknown"},
                    ["too-short"],
                ]
            }
        ),
        encoding="utf-8",
    )

    router._load_history()

    assert router.performance_history["Cortex"][0]["success"] is True
    assert router.performance_history["Cortex"][0]["tokens"] == 42
    assert router.performance_history["Cortex"][1]["success"] is False
    assert router.performance_history["Cortex"][1]["tokens"] == 1_000_000
    assert router.performance_history["Cortex"][1]["task_type"] == "autonomous_goal"


def test_dynamic_router_routes_tool_tasks_to_successful_tool_model(monkeypatch, tmp_path):
    router = DynamicRouter()
    router.db_path = tmp_path / "routing_history.json"
    router.performance_history = {
        "Weak": [
            {"success": False, "tokens": 10, "task_type": "tool_heavy", "recorded_at": 1.0}
            for _ in range(10)
        ],
        "ToolAgent": [
            {"success": True, "tokens": 10, "task_type": "tool_heavy", "recorded_at": 1.0}
            for _ in range(10)
        ],
    }

    class Backend:
        def get_tier_layout(self):
            return {"PRIMARY": ["Weak", "ToolAgent"], "SECONDARY": [], "TERTIARY": []}

        def is_unhealthy(self, _model):
            return False

    class Cel:
        def __init__(self):
            self.payloads = []

        def emit(self, payload):
            self.payloads.append(payload)

    cel = Cel()

    class StubContainer:
        @staticmethod
        def get(name, default=None):
            if name == "constitutive_expression_layer":
                return cel
            return default

    monkeypatch.setattr(router_module, "ServiceContainer", StubContainer)
    router.llm_router = Backend()

    decision = asyncio.run(
        router.route(
            "Search the web, open a browser, and save the result.",
            {"requires_tools": True},
        )
    )

    assert decision.model == "ToolAgent"
    assert decision.confidence == 1.0
    assert cel.payloads[-1]["task_type"] == "tool_heavy"
    assert cel.payloads[-1]["selected_model"] == "ToolAgent"


def test_dynamic_router_falls_back_to_cortex_after_backend_failure(monkeypatch, tmp_path):
    router = DynamicRouter()
    router.db_path = tmp_path / "routing_history.json"

    class BrokenBackend:
        def get_tier_layout(self):
            self.calls = getattr(self, "calls", 0) + 1
            raise RuntimeError("tier layout unavailable")

        @property
        def adapters(self):
            self.calls = getattr(self, "calls", 0) + 1
            raise RuntimeError("adapter inventory unavailable")

    class StubContainer:
        @staticmethod
        def get(_name, default=None):
            return default

    monkeypatch.setattr(router_module, "ServiceContainer", StubContainer)
    router.llm_router = BrokenBackend()

    decision = asyncio.run(router.route("What time is it?", {}))

    assert decision.model == "Cortex"
    assert 0.0 <= decision.confidence <= 1.0


def test_dynamic_router_record_outcome_sanitizes_and_persists(tmp_path):
    router = DynamicRouter()
    router.db_path = tmp_path / "routing_history.json"

    for _ in range(10):
        asyncio.run(router.record_outcome("Model\x00A", "yes", "not-int", "not-a-task"))

    assert router.db_path.exists()
    saved = json.loads(router.db_path.read_text(encoding="utf-8"))
    assert "ModelA" in saved
    assert saved["ModelA"][-1]["success"] is True
    assert saved["ModelA"][-1]["tokens"] == 0
    assert saved["ModelA"][-1]["task_type"] == "autonomous_goal"
