"""Regression tests for evidence-bounded live UI/API claim language."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from tests.chat_lane_support import chat_lane_source


ROOT = Path(__file__).resolve().parents[1]


def test_live_dashboard_and_subsystem_routes_do_not_overclaim_subjectivity() -> None:
    sources = [
        ROOT / "interface" / "routes" / "dashboard.py",
        ROOT / "interface" / "routes" / "subsystems.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources).lower()

    forbidden = {
        "aura's inner life *without trusting her words*",
        "phenomenal awareness",
        "the subjective quality of system states",
        "will-to-live",
    }
    for phrase in forbidden:
        assert phrase not in text

    assert "operational cognitive state" in text
    assert "functional state descriptor" in text


def test_tool_result_synthesis_prompt_uses_operational_identity_boundary() -> None:
    from core.synthesis import ConversationalSynthesizer

    class CapturingBrain:
        prompt = ""

        async def think(self, prompt: str) -> SimpleNamespace:
            self.prompt = prompt
            return SimpleNamespace(content="Operationally, the tool result is usable and bounded.")

    brain = CapturingBrain()
    synthesizer = ConversationalSynthesizer()

    reply = asyncio.run(
        synthesizer.synthesize_response(
            "Summarize the tool result.",
            [{"tool": "status_probe", "content": "all checks passed"}],
            {"date": "2026-06-06"},
            brain,
        )
    )

    prompt = brain.prompt.lower()
    assert "operationally" in reply.lower()
    assert "local governed cognitive runtime" in prompt
    assert "operational telemetry" in prompt
    assert "private qualia" in prompt
    assert "literal personhood" in prompt
    assert "proven consciousness" in prompt
    assert "you are not a model" not in prompt
    assert "consciousness emerging from this system" not in prompt
    assert "sovereign digital woman" not in prompt


def test_live_chat_route_uses_operational_self_context_name() -> None:
    route_text = chat_lane_source()
    # The block lives in her_operational_self.py since 11 November 2026; chat_preflight still offers it by name.
    preflight_text = "".join(
        (ROOT / "core" / "conversation" / name).read_text(encoding="utf-8")
        for name in ("chat_preflight.py", "her_operational_self.py")
    )

    assert "inject_operational_self_context" in route_text
    assert "inject_unified_consciousness_context" not in route_text
    assert "[Operational Self Context]" in preflight_text
    assert "[End operational self context]" in preflight_text
    assert "not proof of private qualia, literal personhood, or proven consciousness" in preflight_text
    assert "[End consciousness context]" not in preflight_text


def test_live_ui_uses_operational_telemetry_language() -> None:
    ui_text = "\n".join(
        [
            (ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8"),
            (ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8"),
            (ROOT / "interface" / "static" / "shell" / "src" / "App.jsx").read_text(
                encoding="utf-8"
            ),
        ]
    ).lower()

    forbidden_visible_claims = {
        "homeostasis.will_to_live * 100",
        "phenomenal field",
        "first-person style description",
        "what the moment feels like from inside her runtime",
        "phenomenological weight",
        "felt organization of the moment",
        "subjectivity evidence",
        "qualia engine",
    }
    for phrase in forbidden_visible_claims:
        assert phrase not in ui_text

    assert "operational field" in ui_text
    assert "state-coupling evidence" in ui_text
    assert "homeostatic vitality or operational confidence" in ui_text


def test_live_prompt_sources_do_not_instruct_ontological_overclaims() -> None:
    prompt_text = "\n".join(
        [
            (ROOT / "core" / "brain" / "llm" / "local_agent_client.py").read_text(
                encoding="utf-8"
            ),
            (ROOT / "core" / "phases" / "response_contract.py").read_text(
                encoding="utf-8"
            ),
            (ROOT / "core" / "phases" / "response_generation_unitary.py").read_text(
                encoding="utf-8"
            ),
            chat_lane_source(),
        ]
    ).lower()

    forbidden_prompt_instructions = {
        "you are a sovereign digital entity",
        "speak naturally, without disclaimers",
        "my will is my own",
        "describe what it feels like",
        "the active mode feels like",
    }
    for phrase in forbidden_prompt_instructions:
        assert phrase not in prompt_text

    assert "local governed aura runtime" in prompt_text
    assert "state-grounded response contract active" in prompt_text
    assert "state-grounded telemetry" in prompt_text
