"""One substrate, whichever name a subsystem asks for it by.

The process held three. The consciousness system built one and published it as
`conscious_substrate` and `liquid_state`; the orchestrator built another in its
boot mixin and published that as `liquid_substrate` and `conscious_substrate`,
clobbering the first under the shared name; and the subject-core organism then
republished the consciousness system's under `liquid_substrate`. The result was
`conscious_substrate` resolving to the orchestrator's object and
`liquid_substrate` to the consciousness system's.

Only one of them is ever stepped. The other reports an infinitely old snapshot,
zero volatility and zero phi for the life of the process — so every consumer
that happened to ask by the losing name was reading a substrate with no
dynamics behind it: the attention gate, somatic qualia, temporal continuity,
the predictive hierarchy, the aesthetic engine, the philosophical stance, the
latent bridge, and the workspace's own bid for recurrent cognition.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Every name the tree reaches the substrate by.
NAMES = ("conscious_substrate", "liquid_substrate", "liquid_state", "liquid_neural_network")


def test_the_orchestrator_adopts_a_substrate_rather_than_building_a_second():
    """Read out of the source: the defect was an unconditional constructor."""
    source = (ROOT / "core/orchestrator/mixins/boot/boot_resilience.py").read_text()
    tree = ast.parse(source)
    built = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LiquidSubstrate"
    ]
    assert len(built) == 1, "one construction site"
    call = built[0]
    line = source.split("\n")[call.lineno - 1]
    holder = source.split("\n")[max(0, call.lineno - 4) : call.lineno]
    joined = "\n".join(holder) + "\n" + line
    assert "ServiceContainer.get" in joined, (
        "the substrate is constructed without first asking whether one is here"
    )


def test_the_consciousness_system_publishes_under_every_name():
    source = (ROOT / "core/consciousness/system.py").read_text()
    for name in NAMES:
        assert f'register_runtime_service("{name}"' in source, name


def test_the_workspace_bid_is_scale_free():
    """The substrate's claim on attention was its raw mean velocity.

    `get_state_summary_nowait` multiplies the mean velocity by a hundred to
    make it readable and the bid divided by a hundred, so the two cancelled and
    the bid was about two thousandths against a floor of five hundredths.
    Recurrent cognition could not reach attention by any route, on any turn.
    """
    from core.consciousness.workspace_feed import FLOOR, build_candidates
    from core.state.aura_state import AuraState

    del AuraState, build_candidates
    source = (ROOT / "core/consciousness/workspace_feed.py").read_text()
    block = source[source.index('source="substrate"') - 2000 : source.index('source="substrate"')]
    assert "global_energy" in block, "the bid is not measured against the substrate's own scale"
    assert FLOOR > 0.0
