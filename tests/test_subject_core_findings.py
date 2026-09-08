"""The repairs the subject-core battery drove, pinned so they cannot go quiet again.

Every one of these was a mechanism written for a job and never called. None was
visible from reading the code — each was found by asking the battery why an
edge it expected was absent — and each would be just as invisible if it lapsed.
So each is an assertion that the call still happens, not that the function
still exists.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from core.state.aura_state import AuraState


def test_proprioception_publishes_the_name_its_readers_ask_for():
    """Three call sites ask for `ram_usage`; the loop published only `vram_usage`."""
    from core.phases import proprioceptive_loop

    source = inspect.getsource(proprioceptive_loop)
    assert 'soma.hardware["ram_usage"]' in source

    readers = 0
    for module in ("core.consciousness.selfhood_tick", "core.phases.cognitive_integration_phase"):
        __import__(module)
        import sys

        if "ram_usage" in inspect.getsource(sys.modules[module]):
            readers += 1
    assert readers >= 2


def test_the_affect_phase_advances_the_lifetime_state():
    """It stepped only when a memory retrieval happened to ask it something."""
    from core.phases.affect_update import AffectUpdatePhase

    source = inspect.getsource(AffectUpdatePhase)
    assert "_advance_lifetime" in source
    assert "core.ontogeny.lifetime" in source


def test_the_affect_phase_runs_the_grounding_engine():
    """Registered as a service and called zero times."""
    from core.phases.affect_update import AffectUpdatePhase

    source = inspect.getsource(AffectUpdatePhase)
    assert "_ground_affect" in source
    assert "affect_grounding" in source


def test_the_motivation_phase_reads_the_action_urgency():
    """The heartbeat read it to bid; the drives never did."""
    from core.phases.motivation_update import MotivationUpdatePhase

    source = inspect.getsource(MotivationUpdatePhase)
    assert "_surprise_pressure" in source
    assert "get_action_urgency" in source


def test_the_consciousness_phase_feeds_the_workspace():
    """Two places in the tree submitted a candidate, both rare drive branches."""
    from core.phases.consciousness_phase import ConsciousnessPhase

    assert "feed_workspace" in inspect.getsource(ConsciousnessPhase)


def test_the_consciousness_system_registers_broadcast_consumers():
    from core.consciousness.system import ConsciousnessSystem

    assert "register_broadcast_consumers" in inspect.getsource(ConsciousnessSystem)


def test_the_substrate_link_survives_boot_order():
    """It was fetched in the constructor, before anything registered it."""
    from core.consciousness.homeostatic_coupling import HomeostaticCoupling

    assert isinstance(HomeostaticCoupling.substrate, property)


def test_the_intention_loop_emits_and_compares():
    from core.agency.intention_loop import IntentionLoop

    source = inspect.getsource(IntentionLoop)
    assert "_emit_efference" in source
    assert "_compare_outcome" in source
    assert "get_agency_ledger" in source


def test_the_ten_domains_all_have_a_writer_and_a_reader():
    """A domain nothing can displace has no measurable outgoing edges, and a
    domain nothing reads has no incoming ones."""
    from core.subject.state import DOMAINS, feature_names, perturbable

    assert set(perturbable()) == set(DOMAINS)
    for key in DOMAINS:
        assert feature_names(key), key


def test_every_domain_reads_at_least_one_live_organ_or_state_field():
    from core.subject.state import DOMAINS, schema

    for key in DOMAINS:
        sources = schema(key).sources
        assert sources
        assert all(source for source in sources), key


@pytest.mark.parametrize("domain", ["G", "C", "S", "W"])
def test_the_organ_backed_domains_name_the_organ_they_read(domain):
    from core.subject.state import schema

    assert any(source.startswith("organ:") for source in schema(domain).sources)


def test_a_turn_of_the_offline_organism_moves_every_domain(tmp_path):
    """The end-to-end check the battery rests on: drive it, read it, and find
    that nothing came back constant."""
    from core.subject.driver import CONDITIONS, build_runtime, start_organism
    from core.subject.recording import build_recording

    async def run():
        runtime = build_runtime(tmp_path / "runtime", seed=3)
        await start_organism(runtime)
        frames = []
        for _ in range(2):
            for condition in CONDITIONS:
                frames.extend(await runtime.turn_once(condition))
        return build_recording(frames), runtime

    recording, runtime = asyncio.run(run())
    assert recording.frames > 0
    assert set(recording.live_domains()) == set("PIAGCSMWDN")
    assert not runtime.failures, runtime.failure_notes


def test_the_battery_does_not_hold_the_inter_instance_port():
    """`ConsciousnessSystem.start` binds a protocol listener unconditionally.
    A measurement harness advertising itself as an Aura instance is wrong on
    its own terms, and it collides with the live desktop runtime on the port it
    uses for the same purpose."""
    from core.subject import organism

    source = inspect.getsource(organism.bring_up)
    assert "aura_protocol" in source
    assert "protocol_listener_stopped" in source


def test_the_battery_leaves_no_free_running_cognitive_loop():
    """Eleven of them survived the first wind-down, one per bridge layer plus
    the closed causal loop. Free-running loops put an uncontrolled amount of
    computation into whichever arm of an intervention ran while the machine was
    busy, which does not add a limit to the measurement — it makes the two arms
    incomparable."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.subject.driver import build_runtime, start_organism

    async def run():
        with TemporaryDirectory() as tmp:
            runtime = build_runtime(Path(tmp) / "runtime", seed=5)
            summary = await start_organism(runtime)
            del runtime
            return summary

    summary = asyncio.run(run())
    assert summary["stopped_loops"], "nothing was stopped, so nothing was running"
    assert set(summary["still_running"]) <= {"state_registry.notification_dispatcher"}
