"""A displacement that no consumer reads is not an intervention.

Every number the battery reports rests on one assumption: that displacing a
domain displaces the organism, not a field. The check that a writer moves its
own domain's reading lives next door, and it is not enough — it passes for a
field that is written, read back by the schema, and consulted by nothing. The
battery then measures a domain whose perturbation could not have propagated
and reports the absence of an edge as evidence about the organism.

Two defects this is written against were exactly that. `phi_estimate` is bumped
by the C writer and overwritten by executive closure from the closed loop later
in the same turn, so the displacement is gone before any consumer sees it; and
`cognition.phenomenal_state` carries a claim, not a number, so the valence and
arousal the writer reaches for are usually not there at all. Neither showed up
as a fault. Recurrent cognition simply had no outgoing edge.

So each case below names a consumer — a function the runtime calls to decide
something — and asserts that the displacement changes what that function
returns. Where a channel is dead the test says which one, rather than leaving
it to a campaign to report as a property of the subject.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.subject.state import perturb, perturb_organs
from core.ontogeny.state import OntogeneticState

DELTA = 0.15


# ── N: the developmental state ───────────────────────────────────────────


def test_displacing_the_reservoir_changes_what_novelty_reports() -> None:
    """Novelty is the reservoir's one live output. Every consumer reads it.

    The affect phase blends curiosity toward it, the workspace bids on it, the
    drive competition takes it as a signal and the self-model takes it as
    situation. If displacing the hidden state does not move it, N is a domain
    with thirteen columns and no way out.
    """
    # A life long enough that the distribution novelty is measured against
    # exists: below thirty observations the reservoir answers a flat one half,
    # which is honest and carries no signal.
    rng = np.random.default_rng(3)
    sham = OntogeneticState(input_width=8, units=64, seed=5)
    displaced = OntogeneticState(input_width=8, units=64, seed=5)
    for _ in range(120):
        row = rng.normal(size=8)
        sham.step(row)
        displaced.step(row)
    assert perturb(None, "N", DELTA, ontogeny=displaced) is True
    row = rng.normal(size=8)
    quiet = float(sham.step(row).novelty)
    moved = float(displaced.step(row).novelty)
    assert abs(moved - quiet) > 0.01, (
        f"the reservoir moved and novelty did not: {quiet:.4f} -> {moved:.4f}"
    )


def test_the_workspace_prices_the_ontogeny_bid_on_that_novelty(monkeypatch) -> None:
    """The bid is the developmental state's route into attention."""
    from core.consciousness.workspace_feed import build_candidates
    from core.ontogeny import lifetime
    from core.state.aura_state import AuraState

    state = AuraState.default()

    def at(novelty: float) -> float:
        reading = SimpleNamespace(
            novelty=novelty, displacement=0.1, relative_displacement=0.5
        )
        monkeypatch.setattr(lifetime, "last_reading", lambda: reading)
        bids = [
            bid for bid in build_candidates(state)
            if getattr(bid, "source", "") == "ontogeny"
        ]
        return max((float(bid.priority) for bid in bids), default=0.0)

    quiet, unprecedented = at(0.2), at(0.8)
    assert unprecedented > quiet, (
        f"a more unprecedented moment did not bid higher: {quiet} -> {unprecedented}"
    )


# ── C: recurrent cognition ───────────────────────────────────────────────


def test_displacing_the_substrate_changes_the_modifiers_cognition_runs_under() -> None:
    """The substrate's route out is the homeostatic coupling.

    It blends a third of the substrate into felt state and computes from it how
    hot and how deep the next thought may be. Those four numbers are read by
    the phases and are in the workspace domain's schema, so this is the edge
    from recurrent cognition to attention, measured at the place it lands.
    """
    from core.consciousness.homeostatic_coupling import HomeostaticCoupling

    coupling = HomeostaticCoupling(orchestrator=None)
    drives = {"energy": 0.8, "curiosity": 0.7, "persistence": 0.7}
    settled = {"valence": 0.2, "arousal": 0.5, "engagement": 0.5}
    for _ in range(200):
        coupling._compute_modifiers(drives, settled, 1.0)
    # A third of the displacement, which is the substrate's share of the blend.
    share = 0.3 * DELTA
    quiet = coupling._compute_modifiers(drives, settled, 1.0)
    moved = coupling._compute_modifiers(
        drives, {**settled, "valence": settled["valence"] + share}, 1.0
    )
    assert abs(moved.temperature_mod - quiet.temperature_mod) > 0.005, (
        "a displaced substrate did not reach the temperature the head runs at"
    )


def test_the_substrate_writer_moves_the_dimensions_the_readout_reports() -> None:
    """`get_substrate_affect` is what every consumer of the substrate calls."""
    from core.consciousness.liquid_substrate import LiquidSubstrate

    substrate = LiquidSubstrate()
    organs = SimpleNamespace(
        substrate=substrate, workspace=None, free_energy=None, self_model=None,
        world_model=None, ontogeny=None, agency=None, self_prediction=None,
        comparator=None, soma=None,
    )
    import numpy as np

    before_state = np.array(substrate.x, copy=True)
    before = dict(substrate.get_substrate_affect())
    assert asyncio.run(perturb_organs(organs, "C", DELTA, state=None)) is True
    after = dict(substrate.get_substrate_affect())
    moved = [
        key
        for key in ("valence", "arousal", "dominance", "energy")
        if abs(float(after[key]) - float(before[key])) > 1e-6
    ]
    assert len(moved) >= 3, f"only {moved} moved in the substrate's own readout"

    # And the recurrent state itself, not five readouts on top of it.
    #
    # The five named psychological dimensions are five of five hundred and
    # twelve, and `get_substrate_affect` takes x[0], x[1], x[2] and two means
    # over the whole vector. Writing only those moved recurrent cognition's own
    # reading to the clipping ceiling and moved the state the dynamics carry by
    # almost nothing — which is why C could be displaced hardest of the ten
    # domains and reach a tenth of a standard deviation anywhere else.
    spread = float(np.mean(np.abs(np.asarray(substrate.x) - before_state)))
    assert spread > DELTA * 0.5, (
        f"the whole recurrent state moved by {spread:.4f}; the intervention is "
        "still on the readouts rather than on the system"
    )
    assert abs(float(after["energy"]) - float(before["energy"])) > 1e-6, (
        "the energy readout is a mean over every unit and it did not move"
    )


def test_a_bumped_phi_estimate_is_not_a_displacement_of_anything() -> None:
    """The field the C writer also writes, and why it is not the channel.

    Executive closure assigns `phi_estimate` from the closed loop every turn it
    runs, so a value written before it is gone afterwards. This is here so that
    nobody reads the C writer and concludes the estimate carries the
    intervention: the substrate does, and this does not.
    """
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.phi = 0.42
    state.phi_estimate = 0.42 + DELTA
    # The one line in executive closure that decides the field's value.
    if state.phi:
        state.phi_estimate = float(state.phi)
    assert state.phi_estimate == pytest.approx(0.42), (
        "the closed loop no longer owns phi_estimate; the C writer may now use it"
    )


# ── D: deliberation ──────────────────────────────────────────────────────


def test_displacing_deliberation_changes_what_the_workspace_hears() -> None:
    """A probe intention has to reach the competition, priced on its urgency."""
    from core.consciousness.workspace_feed import _goal_priority
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [
        {"description": f"standing goal {index}", "goal": f"standing goal {index}",
         "status": "pending", "urgency": 0.5}
        for index in range(5)
    ]
    assert perturb(state, "D", DELTA, ontogeny=None) is True
    added = [
        goal for goal in state.cognition.active_goals
        if isinstance(goal, dict) and goal.get("id") == "subject_core_probe"
    ]
    assert added, "the deliberation writer left no intention behind"
    assert _goal_priority(added[0]) > 0.5, (
        "the probe intention states no more urgency than a standing one, so the "
        "competition cannot tell the arms apart"
    )


def test_a_displaced_drive_budget_changes_which_need_is_most_depleted() -> None:
    """What deliberation decides on is which need is worst, so the writer has
    to change that comparison rather than raise every budget equally."""
    from core.state.aura_state import AuraState

    state = AuraState.default()
    before = {name: dict(budget) for name, budget in state.motivation.budgets.items()}
    assert perturb(state, "D", DELTA, ontogeny=None) is True
    after = state.motivation.budgets
    gaps = {
        name: float(after[name]["level"]) - float(before[name]["level"])
        for name in before
    }
    # A budget already at capacity has no room and does not move, which is the
    # writer working: it fills a share of what is missing.
    room = {
        name: float(before[name]["capacity"]) - float(before[name]["level"])
        for name in before
    }
    movable = {name: gap for name, gap in gaps.items() if room[name] > 1e-9}
    assert movable, "no budget had room to be displaced"
    assert all(gap > 0.0 for gap in movable.values()), f"a budget did not move: {gaps}"
    assert max(movable.values()) > min(movable.values()) * 1.5, (
        "every budget moved by the same amount, so the most depleted drive is "
        f"still the most depleted and nothing dispatches differently: {gaps}"
    )


# ── the general rule ─────────────────────────────────────────────────────


def test_no_writer_reaches_into_another_domain() -> None:
    """An intervention that writes a downstream domain measures itself.

    Every writer is allowed to move its own domain and whatever follows from
    that through the organism. Writing a second domain's state directly is the
    one thing that would make an edge an artifact of the harness, so the source
    of each writer is read for an attribute path that belongs to another
    domain's schema.
    """
    import inspect
    import re

    from core.subject import state as subject_state

    #: Which domain owns each state path the schema reads, to two segments —
    #: `affect.valence` belongs to A, `cognition.active_goals` to D. Paths read
    #: by more than one domain belong to none of them for this purpose: a
    #: writer that moves a shared path is moving its own reading too.
    claims: dict[str, set[str]] = {}
    for domain in subject_state.DOMAINS:
        for path in subject_state.schema(domain).sources:
            if path.startswith("organ:"):
                continue
            key = ".".join(path.split("[")[0].split(".")[:2])
            claims.setdefault(key, set()).add(domain)
    owners = {key: next(iter(who)) for key, who in claims.items() if len(who) == 1}

    trespass: list[str] = []
    for domain in subject_state.DOMAINS:
        source = inspect.getsource(subject_state._WRITERS[domain])
        for target in re.findall(r'_bump\(\s*state,\s*"([^"]+)"', source):
            key = ".".join(target.split(".")[:2])
            owner = owners.get(key)
            if owner is not None and owner != domain:
                trespass.append(f"{domain}'s writer bumps {target}, owned by {owner}")
    assert not trespass, "; ".join(trespass)


# ── I: the body ──────────────────────────────────────────────────────────


def test_the_body_displacement_survives_the_host_freeze() -> None:
    """Two arms run seconds apart read different CPU and different thermals.

    Freezing the host is what makes the difference between them the
    intervention rather than the machine, and the driver puts the frozen
    readings back after every phase. So every hardware channel the body writer
    touches is overwritten within the same turn, and a displacement of
    interoception that lands only there is gone before anything reads it. What
    has to survive is the half of the body that is hers.
    """
    from core.state.aura_state import AuraState

    state = AuraState.default()
    frozen = dict(state.soma.hardware)
    assert perturb(state, "I", DELTA, ontogeny=None) is True
    exertion = float(getattr(state.soma, "exertion", 0.0))
    vitality = float(getattr(state, "vitality", 1.0))
    # What the driver does after each phase while a trial is in flight.
    state.soma.hardware.update(frozen)
    assert float(getattr(state.soma, "exertion", 0.0)) == pytest.approx(exertion)
    assert float(getattr(state, "vitality", 1.0)) == pytest.approx(vitality)
    assert exertion > 0.0, "the host freeze leaves the body with no channel to move"


# ── W: the world model ───────────────────────────────────────────────────


def test_displacing_the_world_model_changes_what_it_is_surprised_by() -> None:
    """A model of the world is displaced by being shown a world it is not in.

    `surprise` is what consumers read — the free-energy engine, the motivation
    phase's pressure, the workspace's own bid. If the observation goes in and
    the surprise does not move, the displacement was a call.
    """
    from core.state.aura_state import AuraState
    from core.world_model.unified_world_model import UnifiedWorldModel

    state = AuraState.default()
    quiet, moved = [], []
    for target, delta in ((quiet, 0.0), (moved, DELTA)):
        model = UnifiedWorldModel()
        organs = SimpleNamespace(
            world_model=model, substrate=None, workspace=None, free_energy=None,
            self_model=None, ontogeny=None, agency=None, self_prediction=None,
            comparator=None, soma=None,
        )
        for _ in range(12):
            asyncio.run(perturb_organs(organs, "W", delta, state=state))
        target.append(model.surprise())
    assert quiet[0] is not None and moved[0] is not None, "the model reported no surprise at all"
    assert abs(float(moved[0]) - float(quiet[0])) > 1e-6, (
        f"a displaced observation left the surprise where it was: {quiet[0]} -> {moved[0]}"
    )


# ── every writer, the two general rules ──────────────────────────────────


def test_every_displacement_moves_more_than_one_column() -> None:
    """One column that moves is a scalar with a name, not a displaced domain.

    A domain read through twenty features and displaced through one is being
    measured on nineteen constants and one dial, and an edge found from it is
    an edge from the dial.
    """
    from core.state.aura_state import AuraState
    from core.subject.state import DOMAINS, feature_names, read_core_state

    from core.consciousness.liquid_substrate import LiquidSubstrate
    from core.subject.state import Organs

    thin: list[str] = []
    for domain in DOMAINS:
        state = AuraState.default()
        reservoir = OntogeneticState(input_width=8, units=16, seed=0)
        # The whole intervention, both halves. Recurrent cognition's state is
        # the substrate: its columns in `AuraState` are a mode flag, a counter
        # and an estimate the closed loop owns, so measuring the state half
        # alone would report a domain displaced through one number.
        organs = Organs(substrate=LiquidSubstrate(), ontogeny=reservoir)
        before = read_core_state(state, organs=organs, ontogeny=reservoir).domain(domain)
        in_state = perturb(state, domain, DELTA, ontogeny=reservoir)
        in_organ = asyncio.run(perturb_organs(organs, domain, DELTA, state=state))
        assert in_state or in_organ, f"{domain} has no writer at all"
        after = read_core_state(state, organs=organs, ontogeny=reservoir).domain(domain)
        names = feature_names(domain)
        moved = [
            names[index]
            for index in range(len(before))
            if abs(float(after[index]) - float(before[index])) > 1e-9
        ]
        if len(moved) < 2:
            thin.append(f"{domain} moved {moved or 'nothing'}")
    assert not thin, "; ".join(thin)


def test_every_state_write_in_a_writer_is_clamped() -> None:
    """A displacement that leaves the range is a lesion wearing its name.

    Every writer's job is to move a field by a share of its own scale and stop
    at the bound, so that four times the campaign's displacement is still a
    state the organism could have been in. The bounds live at the call sites:
    `_bump` takes a low and a high, and a direct assignment is wrapped in
    min/max or a clip. An unbounded write is what turns an intervention into a
    lesion without anyone deciding to.
    """
    import inspect
    import re

    from core.subject import state as subject_state

    loose: list[str] = []
    for domain, writer in subject_state._WRITERS.items():
        source = inspect.getsource(writer)
        for call in re.findall(r"_bump\((.*?)\)\n", source, re.DOTALL):
            if call.count(",") < 3:
                loose.append(f"{domain}: _bump({' '.join(call.split())}) states no bound")
        for line in source.splitlines():
            body = line.strip()
            if not re.match(r"^(entry\[[^]]+\]|node\.\w+|\w+\.h)\s*=\s*", body):
                continue
            # Text has no range to leave. The workspace writer names what is
            # being attended to, and a name is not a quantity.
            if re.search(r"=\s*f?[\"']", body):
                continue
            if not any(token in body for token in ("min(", "max(", "np.clip", "clip(")):
                loose.append(f"{domain}: {body[:70]} is unbounded")
    assert not loose, "; ".join(loose)


def test_no_writer_consults_the_measurement() -> None:
    """A writer that can see the instrument can be written to satisfy it.

    Every displacement is a blind push on the organism: it takes a domain, a
    size and the state, and it must not read the recording, the edge test, the
    battery's thresholds or the reading functions. This is the difference
    between an intervention and an answer key.
    """
    import inspect

    from core.subject import state as subject_state

    forbidden = (
        "core.subject.causal", "core.subject.battery", "core.subject.recording",
        "core.subject.irreducibility", "read_core_state", "THRESHOLDS",
        "_read_", "EDGE_EFFECT",
    )
    guilty: list[str] = []
    for domain, writer in subject_state._WRITERS.items():
        source = inspect.getsource(writer)
        for name in forbidden:
            if name in source:
                guilty.append(f"{domain}'s writer mentions {name}")
    assert not guilty, "; ".join(guilty)


# ── G: the workspace ─────────────────────────────────────────────────────


def test_displacing_the_workspace_changes_which_bid_wins() -> None:
    """A displacement of attention is a change in what wins the competition.

    Writing `ignition_level` moved the number the schema reads and nothing
    downstream, because every consumer fires on a broadcast and a broadcast
    comes from a competition. And a candidate submitted under a probe's own
    name falls through every action table to whatever the sham did, so the
    route from attention through action to the world was closed to the one
    domain that should open it. The writer raises the runner-up under its own
    source until it wins.
    """
    from core.consciousness.global_workspace import (
        CognitiveCandidate,
        ContentType,
        GlobalWorkspace,
    )
    from core.subject.state import Organs

    async def run(delta: float) -> tuple[str, str]:
        workspace = GlobalWorkspace()
        for source, priority in (("perception", 0.70), ("memory", 0.55)):
            await workspace.submit(
                CognitiveCandidate(
                    content=f"{source} has something to say",
                    source=source,
                    priority=priority,
                    content_type=ContentType.PERCEPTUAL,
                )
            )
        organs = Organs(workspace=workspace)
        if delta:
            assert await perturb_organs(organs, "G", delta, state=None) is True
        winner = await workspace.run_competition()
        return (getattr(winner, "source", ""), getattr(winner, "content", ""))

    sham_source, _ = asyncio.run(run(0.0))
    moved_source, _ = asyncio.run(run(DELTA * 2.0))
    assert sham_source == "perception"
    assert moved_source == "memory", (
        f"displacing the workspace left {moved_source!r} winning, so both arms "
        "attend to the same thing and act the same way"
    )


# ── S: the self model ────────────────────────────────────────────────────


def test_displacing_the_self_model_changes_the_beliefs_it_holds() -> None:
    """The self model's beliefs are what the self-state reading is built from.

    The writer goes through `update_belief`, which is the organ's own governed
    path, so a write its authority would refuse is not made — reaching past it
    into the dict would measure a state the runtime can never reach.
    """
    from core.self_model import SelfModel
    from core.subject.state import Organs

    async def run() -> tuple[dict, dict]:
        model = SelfModel(id="a-displacement-probe")
        organs = Organs(self_model=model)
        before = dict(getattr(model, "beliefs", {}) or {})
        assert await perturb_organs(organs, "S", DELTA, state=None) is True
        return before, dict(getattr(model, "beliefs", {}) or {})

    before, after = asyncio.run(run())
    assert after != before, "the self model's beliefs did not move"
    assert "subject_core_probe" in after, (
        f"the displacement was accepted and left no belief: {sorted(after)[:8]}"
    )


def test_a_domain_faster_than_its_consumers_is_held_at_the_displacement() -> None:
    """`do(X)` holds X. A single push is a different intervention.

    It makes no difference to a domain whose state persists — a goal stays on
    the list, a budget stays where it was filled to — and all the difference to
    one whose own dynamics pull it back faster than its consumers look. The
    substrate integrates at twenty hertz with a tenth-of-a-second time
    constant, and the homeostatic coupling that carries it into how hot and how
    deep she may think runs on the heartbeat, once a second: a push delivered
    early in a turn had decayed by five e-foldings before anything downstream
    looked.
    """
    from core.subject.causal import SUSTAINED
    from core.subject.steppable import LAYERS
    from core.subject.driver import SECONDS_PER_TURN

    assert "C" in SUSTAINED, "recurrent cognition is the case this exists for"

    # And the rule that put it there, checked rather than remembered: the
    # substrate's own period is shorter than a turn.
    rates = {layer.name: layer.hz for layer in LAYERS}
    assert rates, "no layer declares a rate"
    fastest = max(rates.values())
    assert 1.0 / fastest < SECONDS_PER_TURN

    # Domains whose state persists are not held, because holding one would be
    # a clamp rather than a displacement.
    assert "D" not in SUSTAINED and "M" not in SUSTAINED and "S" not in SUSTAINED


def test_the_sustained_write_reaches_every_frame_after_the_injection() -> None:
    """The driver applies it after each frame from the injection point on."""
    import inspect

    from core.subject.driver import SubjectRuntime

    source = inspect.getsource(SubjectRuntime.turn_once)
    assert "sustain" in source
    assert "len(frames) - 1 > perturb_at" in source


def test_curiosity_is_read_from_both_channels_not_the_larger_one() -> None:
    """A max is not a derivation.

    `affect.curiosity` was `max(curiosity, anticipation or 0.5)`. Whichever
    channel is higher owns the readout outright and the other is invisible, and
    anticipation sits near a half — so the curiosity channel had to beat it
    before anything written there could be read at all. The developmental
    state's only route out of itself writes exactly that channel.
    """
    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    phase = AffectUpdatePhase(None)

    def curiosity_at(value: float) -> float:
        state = AuraState.default()
        affect = state.affect
        affect.emotions["anticipation"] = 0.5
        affect.emotions["curiosity"] = value
        phase._derive_metrics(affect)
        return float(affect.curiosity)

    low, high = curiosity_at(0.1), curiosity_at(0.4)
    assert high > low, (
        "the curiosity channel moved and the readout did not, so whatever "
        "writes it cannot be read"
    )


def test_displacing_the_body_reaches_what_prices_her_effort() -> None:
    """The host is held still, so the ledger is the body's only live channel.

    `_perturb_I` writes `soma.exertion`, and the proprioceptive loop derives
    that from the effort ledger at the top of every turn — so the write lasted
    until the next turn began. The one consumer that prices anything on how
    hard she has been working reads the ledger rather than the readout, so the
    displacement reached it never.
    """
    from core.soma.effort import (
        EffortLedger,
        get_effort_ledger,
        note_effort,
        reset_effort_for_test,
    )
    from core.subject.state import Organs

    reset_effort_for_test()
    for _ in range(29):
        note_effort("phases", 1.0)
    before = EffortLedger.exertion(dict(get_effort_ledger().peek()))

    assert asyncio.run(perturb_organs(Organs(), "I", DELTA, state=None)) is True
    after = EffortLedger.exertion(dict(get_effort_ledger().peek()))
    assert after > before, "the ledger the consumers read did not move"
    reset_effort_for_test()


def test_an_empty_ledger_is_not_displaced_into_existence() -> None:
    """A share of what she has spent. With nothing spent there is nothing to be
    more tired than, and inventing a number would be inventing a body."""
    from core.soma.effort import get_effort_ledger, reset_effort_for_test
    from core.subject.state import Organs

    reset_effort_for_test()
    asyncio.run(perturb_organs(Organs(), "I", DELTA, state=None))
    assert not dict(get_effort_ledger().peek())
