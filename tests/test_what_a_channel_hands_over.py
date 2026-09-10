"""What each lesion channel hands over, and who could consume it."""
from __future__ import annotations

import ast

import pytest

from core.verify import influence_channels
from core.verify.what_a_channel_hands_over import (
    WHAT_A_NAME_MEANS,
    WHAT_A_PLAIN_CALL_CONSUMES,
    AHandover,
    how_the_handovers_read,
    what_a_direct_harness_can_move,
    what_each_channel_hands_over,
)


@pytest.fixture(scope="module", autouse=True)
def _the_owners_are_loaded():
    """The sites are read from source, but the channel ids come from imports."""
    import core.affect.affective_circumplex  # noqa: F401
    import core.being.affective_valence  # noqa: F401
    import core.brain.cognitive_engine  # noqa: F401
    import core.consciousness.qualia_synthesizer  # noqa: F401


def test_every_declared_channel_is_reported():
    reported = {h.channel for h in what_each_channel_hands_over()}
    assert reported == set(influence_channels.ALL_CHANNELS)


def test_the_circumplex_hands_over_sampler_parameters():
    """The largest direct actuation reaches temperature and a token budget."""
    handover = _one(influence_channels.AFFECT_CIRCUMPLEX_SAMPLING)
    assert "sampler" in handover.kinds
    assert handover.a_plain_call_can_move_it
    assert any("temperature" in d for d in handover.destinations)
    assert any("max_tokens" in d for d in handover.destinations)


def test_the_recurrent_channels_need_a_decoder_that_loops():
    """A single forward pass cannot consume a pass count or a steering gain."""
    for channel in (
        influence_channels.LIVE_MIND_RECURRENT_LOOPS,
        influence_channels.LIVE_MIND_STEERING_ALPHA,
    ):
        handover = _one(channel)
        assert handover.kinds == ("loop",)
        assert not handover.a_plain_call_can_move_it
        assert "loop" in handover.why_not


def test_the_context_block_is_prompt_text():
    handover = _one(influence_channels.LIVE_MIND_CONTEXT_BLOCK)
    assert "prompt" in handover.kinds
    assert handover.a_plain_call_can_move_it


def test_generation_controls_reach_the_router_keywords():
    """Its lesion is omission, so the destinations come from the guarded block."""
    handover = _one(influence_channels.LIVE_MIND_GENERATION_CONTROLS)
    assert "is_lesioned" in handover.bound_by
    assert "sampler" in handover.kinds
    assert any("top_p" in d for d in handover.destinations)


def test_a_class_bound_lesion_is_not_given_a_destination_by_hand():
    """@lesionable leaves nothing in the source to follow, and says so."""
    handover = _one(influence_channels.AFFECT_GENERATION_CONTROLS)
    assert handover.bound_by == ("lesionable",)
    assert handover.destinations == ()
    assert not handover.a_plain_call_can_move_it
    assert "no destination to follow" in handover.why_not


def test_qualia_richness_lands_downstream_of_any_generation():
    handover = _one(influence_channels.QUALIA_RICHNESS)
    assert not handover.a_plain_call_can_move_it
    assert handover.destinations, "the apply_channel site should still be found"


def test_more_than_one_channel_is_movable_and_the_count_is_reported():
    """The matched protocol moved one. This says how many it could move."""
    movable = what_a_direct_harness_can_move()
    assert len(movable) > 1
    assert influence_channels.AFFECT_CIRCUMPLEX_SAMPLING in movable
    assert how_the_handovers_read()["a_plain_call_can_move"] == len(movable)


def test_an_unmatched_destination_is_counted_rather_than_absorbed():
    """A name that matches no kind is reported, not quietly treated as one."""
    read = how_the_handovers_read()
    assert isinstance(read["unclassified_destinations"], list)
    made_up = AHandover(
        channel="x.y",
        bound_by=("apply_channel",),
        destinations=("some_new_consumer",),
        kinds=(),
    )
    assert made_up.unclassified == ("some_new_consumer",)
    assert not made_up.a_plain_call_can_move_it


def test_loop_is_checked_before_sampler():
    """`clean_user_surface_steering_alpha` must not fall into a sampler bucket."""
    order = [kind for kind, _ in WHAT_A_NAME_MEANS]
    assert order.index("loop") < order.index("sampler")


def test_a_plain_call_consumes_exactly_sampler_and_prompt():
    assert WHAT_A_PLAIN_CALL_CONSUMES == {"sampler", "prompt"}


def test_the_destinations_are_names_that_exist_in_their_source():
    """Derived, not authored: every destination is a real name in the tree."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    seen: set[str] = set()
    for relative in (
        "core/brain/cognitive_engine.py",
        "core/brain/inference_gate.py",
        "core/consciousness/qualia_synthesizer.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                seen.add(node.id)
            elif isinstance(node, ast.Attribute):
                seen.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                seen.add(node.value)
    for handover in what_each_channel_hands_over():
        for destination in handover.destinations:
            head = destination.split("[")[0].split(".")[-1]
            assert head in seen, f"{destination} is not a name in the source"


def _one(channel: str) -> AHandover:
    for handover in what_each_channel_hands_over():
        if handover.channel == channel:
            return handover
    raise AssertionError(f"{channel} was not reported")
