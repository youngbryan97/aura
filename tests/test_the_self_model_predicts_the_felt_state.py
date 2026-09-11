"""A self-model that cannot be wrong is not a model.

The self-prediction loop predicts her valence, her dominant drive and what she
will be attending to, then grades itself against what happened. Its valence
error and its drive error were both exactly zero for the life of every process,
which is not a well-calibrated model; it is a model of a constant.

Two causes, one shape. The heartbeat read affect from the engine that feeds
`AffectUpdatePhase` rather than from `AuraState.affect`, which is what the
phase settles and what every other consumer of felt state reads — measured over
six turns of two conditions, the engine reported valence 0.0 every time while
the state's moved between 0.17 and 0.29. And the offline harness never
registered the vault it carries its state in under the name the tree reads it
by, so the fallback had nothing to fall back to: a dozen runtime paths take the
current state off `state_repository._current`, and every one of them saw None
while the phases were mutating a state.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_heartbeat_prefers_the_state_the_phases_settle():
    source = (ROOT / "core" / "consciousness" / "heartbeat.py").read_text()
    tree = ast.parse(source)
    gather = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "_felt_state_from_the_state" in (ast.get_source_segment(source, node) or "")
        and node.name != "_felt_state_from_the_state"
    )
    body = ast.get_source_segment(source, gather) or ""
    # The state is asked first; the engine is the fallback.
    assert body.index("_felt_state_from_the_state") < body.index("affect_engine")


def test_the_fallback_reports_whether_it_found_anything():
    """A fallback that cannot say it failed is a fallback that cannot be used."""
    from core.consciousness.heartbeat import CognitiveHeartbeat

    reading: dict = {}
    # Nothing is registered in this process, so the reader has nothing to read
    # and must say so rather than leaving the caller to guess.
    assert CognitiveHeartbeat._felt_state_from_the_state(reading) in (True, False)
    if not reading:
        assert CognitiveHeartbeat._felt_state_from_the_state(reading) is False


async def test_the_harness_publishes_its_state_where_the_tree_reads_it(tmp_path):
    from core.container import ServiceContainer
    from core.subject.driver import CONDITIONS, build_runtime, calibrate_clock, start_organism

    runtime = build_runtime(tmp_path / "runtime", seed=71)
    await start_organism(runtime, quiet=True)
    conditions = [c for c in CONDITIONS if c.name == "conversation"]
    await calibrate_clock(runtime, conditions, turns=1)
    try:
        await runtime.turn_once(conditions[0])
        repo = ServiceContainer.get("state_repository", default=None)
        assert repo is not None, "no state repository is registered"
        assert repo is runtime.kernel.vault, "a different repository won the name"
        assert getattr(repo, "_current", None) is runtime.state
    finally:
        if runtime.clock is not None:
            runtime.clock.uninstall()
