"""The fourteen have to be reachable, not merely present.

A module nobody can get to is a file with tests. This suite asks the four
questions that separate the two: does the container resolve it, does boot
declare its channels and register its invariants, does a live reading reach
the snapshot a conversation turn sees, and does the snapshot say anything
when something is wrong.

The last one matters most. Every organ here has a failure mode where it keeps
running and stops meaning anything — a signalling channel nobody can read, a
carer going short, a state that is no longer mostly her own — and the whole
point of wiring them to telemetry was that those states have somewhere to
appear.
"""

from __future__ import annotations

import pytest

#: Each organ's module-level singleton, and the reset its own module exposes.
#: Resetting the module is not enough on its own — the container hands out a
#: cached instance and would go on handing out the polluted one — so the
#: registration is dropped as well and rebuilt from scratch.
_SINGLETON_RESETS = (
    ("core.identity.constitutive_identity", "reset_constitutive_registry_for_test"),
    ("core.embodiment.expressive_dynamics", "reset_expressive_ledger_for_test"),
    ("core.ethics.care_allocation", "reset_care_allocator_for_test"),
    ("core.social.receptivity", "reset_receptivity_for_test"),
    ("core.social.conventions", "reset_convention_registry_for_test"),
    ("core.affect.dual_process_arbiter", "reset_dual_process_arbiter_for_test"),
    ("core.environment.prospect_refuge", "reset_position_registry_for_test"),
    ("core.learning.craft_practice", "reset_craft_practice_for_test"),
    ("core.creativity.novelty_value", "reset_novelty_valuer_for_test"),
    ("core.morality.reversible_alternative", "reset_reversibility_ledger_for_test"),
    ("core.social.costly_signaling", "reset_signal_channel_for_test"),
    ("core.social.reciprocity_engine", "reset_reciprocity_engine_for_test"),
    ("core.affect.empathic_coupling", "reset_empathic_field_for_test"),
    ("core.perception.aesthetic_response", "reset_aesthetic_observer_for_test"),
)


@pytest.fixture
def wired():
    """A container with everything registered and the group booted.

    Torn down as well as set up. These are process-wide singletons, so a test
    that drives one into a failure state leaves it there, and the next test to
    read the snapshot sees somebody else's carer going short. That is an
    order-dependence defect rather than a flaky test, and the runner reports
    the two differently.
    """
    import core.phenomena_wiring as wiring
    from core.container import get_container
    from core.fsw import phenomena_channels as channels
    from core.service_registration import register_all_services

    def fresh():
        # The telemetry dictionary keeps the last sample per channel, so a
        # test that drove a channel red leaves it red for whoever reads the
        # report next — the organ resets and the reading does not. Clearing
        # and re-declaring is what tests/test_claim_liveness.py does for the
        # same reason.
        from core.fsw.telemetry_dictionary import reset_telemetry_for_test

        reset_telemetry_for_test()
        channels.reset_for_test()
        container = get_container()
        for module_name, reset in _SINGLETON_RESETS:
            module = __import__(module_name, fromlist=[reset])
            getattr(module, reset)()
        for name in wiring.SERVICE_NAMES:
            container.unregister(name)
        register_all_services(container)
        wiring.reset_for_test()
        wiring.boot()

    fresh()
    yield wiring
    fresh()


def test_every_disposition_resolves_from_the_container(wired):
    from core.container import get_container

    container = get_container()
    for name in wired.SERVICE_NAMES:
        service = container.get(name)
        assert service is not None, f"{name} did not resolve"
        assert hasattr(service, "status") or hasattr(service, "names"), (
            f"{name} resolved to something with no readout"
        )
    assert len(wired.SERVICE_NAMES) == 14


def test_boot_declares_the_channels_and_registers_the_invariants(wired):
    from core.fsw.phenomena_channels import channel_names, event_names, reset_for_test

    reset_for_test()
    wired.reset_for_test()
    result = wired.boot()
    declared = set(result["telemetry"])
    assert declared >= set(channel_names()), "channels missing from the declaration"
    assert declared >= set(event_names()), "events missing from the declaration"
    assert len(result["invariants"]) == len(wired.INVARIANT_MODULES)


def test_no_channel_and_event_share_a_name():
    """Ids are a contract, and so is what a reader looks a name up by."""
    from core.fsw.phenomena_channels import channel_names, event_names

    both = channel_names() + event_names()
    assert len(both) == len(set(both)), "a channel and an event share a name"


def test_the_declared_invariants_run_and_pass(wired):
    from core.verify.invariants import verify

    report = verify("identity", "ethics", "social", "morality", "affect")
    assert report.checked >= 9
    mine = {
        "identity.constitution_is_one_way",
        "identity.coherence_carries_its_null",
        "ethics.care_floor_is_never_spent",
        "social.pooling_signal_yields_no_inference",
        "social.receptivity_prices_what_it_would_learn",
        "morality.reversibility_premium_is_bounded",
        "morality.foreclosure_is_never_free",
        "affect.arbitration_has_no_default_channel",
        "affect.empathy_keeps_a_return_path",
    }
    broken = {v.invariant for v in report.violations} & mine
    assert not broken, f"invariants failing: {sorted(broken)}"


def test_an_idle_system_trips_no_limits(wired):
    """A limit that fires on a healthy system trains its reader to ignore it."""
    from core.fsw.telemetry_dictionary import telemetry_report

    wired.sample()
    tripped = [
        row for row in (telemetry_report().get("violations") or [])
        if row.get("state") not in (None, "nominal", "stale")
        and str(row.get("channel", "")).split(".")[0] in {
            "identity", "expression", "care", "receptivity", "arbitration",
            "position", "craft", "creativity", "reversibility", "signalling",
            "reciprocity", "empathy", "aesthetic", "conventions",
        }
    ]
    assert not tripped, f"limits tripped with nothing happening: {tripped}"


def test_a_reading_reaches_the_live_mind_snapshot(wired):
    from core.runtime.live_mind_snapshot import _phenomena_snapshot

    section = _phenomena_snapshot()
    assert section["running"] == section["of"] == 14
    assert section["absent"] == []
    assert section["concerns"] == []


def _drive_into_quiet_failure() -> None:
    """Put every organ into the state it can be in while still running.

    Called by both tests below rather than left as a side effect of one of
    them. Depending on another test having run first is order dependence, and
    the fixture resets between tests precisely so it cannot happen by accident.
    """
    from core.container import get_container
    from core.social.costly_signaling import SignalSchedule
    from core.social.receptivity import Offer

    container = get_container()

    care = container.get("care_allocation")
    care.self_floor = 1.0
    care.set_need("someone", 20.0)
    for _ in range(3):
        care.allocate(10.0)
        care.begin_displacement(0.6)
        care.record_own_unmet(2.0)

    channel = container.get("signal_channel")
    channel.schedule = SignalSchedule(benefit=2.0, min_type=1.0, cost_slope=0.0)
    channel.receive(channel.send("her", 3.0))

    field = container.get("empathic_coupling")
    field.add_person("her", setpoint=0.0, anchor=0.2)
    for who, state in (("a", -3.0), ("b", -2.0), ("c", -4.0)):
        field.add_person(who, setpoint=state, anchor=2.0)
        field.couple("her", who, 1.0)

    container.get("constitutive_identity").get("maker").declare("a maker")

    receptivity = container.get("receptivity")
    for index in range(6):
        receptivity.receive(
            Offer(source="stranger", value=1.0, exposure=500.0, label=f"o{index}")
        )


def test_the_snapshot_names_each_state_it_exists_to_name(wired):
    _drive_into_quiet_failure()
    concerns = " | ".join(wired.snapshot()["concerns"])
    assert "going short" in concerns
    assert "read anything from" in concerns
    assert "not their own" in concerns
    assert "nothing accepted" in concerns
    assert "maker declared" in concerns


def test_those_states_also_reach_the_declared_channels(wired):
    """The concerns and the telemetry are separate paths and both must carry it."""
    from core.fsw import phenomena_channels as channels

    _drive_into_quiet_failure()
    written = wired.sample()
    assert written.get(channels.CHANNEL_CARE_DEPLETED) == pytest.approx(1.0)
    assert written.get(channels.CHANNEL_INFORMATIVE) == pytest.approx(0.0)
    assert written.get(channels.CHANNEL_AUTONOMY, 1.0) < 0.5
    assert written.get(channels.CHANNEL_UNSUPPORTED, 0.0) >= 1.0


def test_the_boot_activator_is_registered_with_the_foundations():
    from core.runtime import foundations

    assert "phenomena" in [name for name, _ in foundations._ACTIVATORS]


def test_a_missing_organ_is_reported_rather_than_raised(monkeypatch, wired):
    """The reason the snapshot reads through the container and not by import."""
    import core.phenomena_wiring as module

    monkeypatch.setattr(
        module, "_service",
        lambda name: None if name == "care_allocation" else module._container().get(name),
    )
    section = module.snapshot()
    assert "care_allocation" in [k for k, v in section["present"].items() if not v]
    assert section["running"] == 13
