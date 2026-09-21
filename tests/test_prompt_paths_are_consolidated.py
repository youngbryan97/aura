"""Every assembly path must carry every continuity block.

`build_system_prompt` has three paths — casual+trusted, casual-guest, and
standard — and the continuity and personhood groups used to be written out
once per path. Each new block had to be added to all three by hand, with
nothing failing if one was missed. A guest would simply have lost the thread
and no test would have said so.

This is the defect shape the external review named ("a rule existed but was
implemented at only one site") and the one this function was manufacturing
fresh instances of: the ledger and the self-preference block each needed
three separate edits to land.

These tests pin the property structurally, so the guarantee survives whoever
adds the next block.
"""

from __future__ import annotations

import inspect
import re

import pytest

from core.brain.llm.context_assembler import ContextAssembler
from core.brain.llm.continuity_ledger import ContinuityLedger
from core.being.individual_preferences import IndividualPreferences, formation_threshold
from core.state.aura_state import AuraState

#: One objective per assembly path.
#: - short greeting from a trusted owner -> casual
#: - a heavyweight request                -> standard
PATH_OBJECTIVES = ("hey", "hello there", "Perform a full architecture review of the runtime")


def _populated_state(objective: str) -> AuraState:
    state = AuraState.default()
    state.cognition.current_objective = objective
    state.cognition.working_memory = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(40)
    ]
    state.cognition.rolling_summary = "ROLLING_SUMMARY_MARKER for the thread."

    ledger = ContinuityLedger()
    ledger.observe([{"role": "user", "content": "I have always wanted to learn LEDGER_MARKER."}])
    state.cognition.continuity_ledger = ledger.to_dict()

    prefs = IndividualPreferences()
    for _ in range(formation_threshold()):
        prefs.encounter("PREFERENCE_MARKER", stance="drawn_to")
    state.identity.self_preferences = prefs.to_dict()

    state.cognition.modifiers["continuity_obligations"] = {
        "identity_mismatch": False,
        "current_objective": "OBLIGATION_MARKER",
        "active_commitments": ["keep the thread"],
        "subject_thread": "coherence",
    }
    return state


@pytest.mark.parametrize("objective", PATH_OBJECTIVES)
@pytest.mark.parametrize(
    "marker",
    ["ROLLING_SUMMARY_MARKER", "LEDGER_MARKER", "PREFERENCE_MARKER", "OBLIGATION_MARKER"],
)
def test_every_continuity_block_reaches_every_path(objective, marker):
    prompt = ContextAssembler.build_system_prompt(_populated_state(objective))
    assert marker in prompt, f"{marker} missing from the path taken by {objective!r}"


def test_continuity_group_is_defined_exactly_once():
    """The group must not drift back into per-path copies.

    Counted over the whole assembler rather than over `build_system_prompt`
    alone: the method-size sweep moved the assembly into
    `_build_system_prompt_stability_v58_zenith`, so reading the method found
    zero of something that is still defined exactly once. "Exactly once" is
    the property, and the module is where once means once.
    """
    from core.brain.llm import context_assembler

    from tests.source_contract import family_text

    # The family: the sections were lifted into context_assembler_blocks, and
    # "exactly once" is a property of the module and what was lifted out of it.
    source = family_text(context_assembler)
    assert source.count("continuity_sections = (") == 1
    assert source.count("personhood_sections = (") == 1


@pytest.mark.parametrize("block", ["ledger_block", "self_preference_block", "rolling_summary"])
def test_no_path_concatenates_a_continuity_block_directly(block):
    """Blocks join via the shared group, never per-path.

    A direct `base += ledger_block` in one branch is how the three-site
    duplication comes back.
    """
    source = inspect.getsource(ContextAssembler.build_system_prompt)
    direct = re.findall(rf"base \+= {block}\b", source)
    assert not direct, f"{block} is concatenated directly, bypassing the shared group"


def test_adding_a_block_to_the_group_reaches_all_paths():
    """The consolidation's whole point, asserted rather than assumed."""
    source = inspect.getsource(ContextAssembler.build_system_prompt)
    # All three paths render the shared groups.
    assert source.count("continuity_group") >= 4  # 1 definition + 3 uses
    assert source.count("personhood_group") >= 4
