import asyncio
import json
from types import SimpleNamespace

from core.evolution import persona_evolver as persona_module
from core.evolution.persona_evolver import PersonaEvolver


class Emotion:
    def __init__(self, base_level=50.0, volatility=1.0):
        self.base_level = base_level
        self.volatility = volatility


class Personality:
    def __init__(self):
        self.traits = {"agreeableness": 0.5}
        self.emotions = {"frustration": Emotion()}
        self.interaction_memories = [
            {"message": f"helpful exchange {idx}", "sentiment": "positive"} for idx in range(12)
        ]
        self.reloads = 0

    def reload_persona(self):
        self.reloads += 1


class EvolutionEngine:
    def __init__(self, content):
        self.content = content

    async def think(self, *_args, **_kwargs):
        return SimpleNamespace(content=self.content)


def test_persona_evolver_clamps_and_atomically_persists_changes(monkeypatch, tmp_path):
    personality = Personality()
    monkeypatch.setattr(persona_module, "get_personality_engine", lambda: personality)
    monkeypatch.setattr(
        persona_module,
        "config",
        SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)),
    )
    engine = EvolutionEngine(
        'analysis {"traits":{"agreeableness": 99}, '
        '"emotions":{"frustration":{"base":999,"volatility":999}}}'
    )
    evolver = PersonaEvolver(SimpleNamespace(cognitive_engine=engine))

    asyncio.run(evolver.run_evolution_cycle(force=True))

    saved = json.loads((tmp_path / "evolved_persona.json").read_text(encoding="utf-8"))
    assert saved["traits"]["agreeableness"] == 0.55
    assert saved["emotions"]["frustration"]["base"] == 55.0
    assert saved["emotions"]["frustration"]["volatility"] == 6.0
    assert personality.interaction_memories == []
    assert personality.reloads == 1


def test_persona_evolver_leaves_persona_unchanged_on_invalid_response(monkeypatch, tmp_path):
    personality = Personality()
    monkeypatch.setattr(persona_module, "get_personality_engine", lambda: personality)
    monkeypatch.setattr(
        persona_module,
        "config",
        SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)),
    )
    evolver = PersonaEvolver(SimpleNamespace(cognitive_engine=EvolutionEngine("no json here")))

    asyncio.run(evolver.run_evolution_cycle(force=True))

    assert not (tmp_path / "evolved_persona.json").exists()
    assert len(personality.interaction_memories) == 12
    assert personality.reloads == 0


def test_persona_evolver_resets_corrupt_existing_persona_file(monkeypatch, tmp_path):
    personality = Personality()
    target = tmp_path / "evolved_persona.json"
    target.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(persona_module, "get_personality_engine", lambda: personality)
    monkeypatch.setattr(
        persona_module,
        "config",
        SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)),
    )
    evolver = PersonaEvolver(
        SimpleNamespace(cognitive_engine=EvolutionEngine('{"traits":{"agreeableness": 0.01}}'))
    )

    asyncio.run(evolver.run_evolution_cycle(force=True))

    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["traits"]["agreeableness"] == 0.51
    assert personality.reloads == 1
