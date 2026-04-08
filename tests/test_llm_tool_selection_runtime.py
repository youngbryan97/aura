from types import SimpleNamespace

from core.capability_engine import CapabilityEngine


def test_select_tool_definitions_is_bounded_and_relevant():
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.SKILL_ALIASES = CapabilityEngine.SKILL_ALIASES
    engine.skills = {
        "web_search": SimpleNamespace(metabolic_cost=1),
        "clock": SimpleNamespace(metabolic_cost=1),
        "memory_ops": SimpleNamespace(metabolic_cost=1),
        "computer_use": SimpleNamespace(metabolic_cost=2),
        "os_manipulation": SimpleNamespace(metabolic_cost=2),
        "self_evolution": SimpleNamespace(metabolic_cost=3),
    }
    engine.detect_intent = lambda message: ["web_search", "clock", "memory_ops", "computer_use"]
    engine.get_tool_definitions = lambda: [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} description",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in engine.skills
    ]

    selected = CapabilityEngine.select_tool_definitions(
        engine,
        objective="Find the latest Bitcoin price and timestamp it for memory.",
        max_tools=3,
    )
    selected_names = [item["function"]["name"] for item in selected]

    assert len(selected_names) == 3
    assert "web_search" in selected_names
    assert "clock" in selected_names
    assert "memory_ops" in selected_names
    assert "self_evolution" not in selected_names
