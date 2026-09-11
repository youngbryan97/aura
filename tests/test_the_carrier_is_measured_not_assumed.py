"""Subject Core v25: the parts that would report a carrier that is not there.

Five things could make this measurement say yes for the wrong reason, and each
has a test here.

The Fisher-Rao distance could be miscalibrated, so identical laws read as far
apart. The rank estimator could find structure in noise, or miss structure that
is there, and either way the grain it hands on is fiction. The cut enumeration
could quietly drop partitions, which turns a minimum over 511 into a minimum
over whichever ones were convenient. The exclusion rule could break a genuine
tie by sorting names, which manufactures a unique carrier out of a symmetry.
And the playback null — a cut arm that is the intact arm — could still score
damage, which is the movie objection landing on us rather than being answered.

The last test is about wording, and it is not cosmetic. A run reports
INTRINSIC_CARRIER_FOUND. Whether a carrier is a subject is a postulate nothing
here tests, and a report that used the other word would be claiming the thing
the whole design refuses to claim.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


# ── the metric ────────────────────────────────────────────────────────────


def test_identical_distributions_sit_at_zero() -> None:
    from core.subject.intrinsic_v25 import fisher_rao_categorical

    assert fisher_rao_categorical([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)


def test_orthogonal_distributions_sit_at_pi() -> None:
    """The normalisation that removes Cencov's remaining scale freedom."""
    from core.subject.intrinsic_v25 import fisher_rao_categorical

    assert fisher_rao_categorical([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.pi, abs=1e-9)


def test_the_metric_does_not_care_which_label_came_first() -> None:
    from core.subject.intrinsic_v25 import fisher_rao_categorical

    left, right = [0.2, 0.3, 0.5], [0.6, 0.1, 0.3]
    forward = fisher_rao_categorical(left, right)
    backward = fisher_rao_categorical(right, left)
    shuffled = fisher_rao_categorical(left[::-1], right[::-1])
    assert forward == pytest.approx(backward)
    assert forward == pytest.approx(shuffled)


def test_a_sample_estimator_separates_what_is_separate() -> None:
    from core.subject.intrinsic_v25 import crossfit_fisher_rao

    rng = np.random.default_rng(0)
    same_a = rng.normal(size=(200, 6))
    same_b = rng.normal(size=(200, 6))
    apart = rng.normal(loc=2.0, size=(200, 6))
    together = crossfit_fisher_rao(same_a, same_b, seed=0).distance_sq
    separate = crossfit_fisher_rao(same_a, apart, seed=0).distance_sq
    assert separate > together * 4.0, (
        "the estimator cannot tell two laws apart, so no cut result from it means anything"
    )


def test_the_sham_floor_is_subtracted() -> None:
    """Two untouched forks are not numerically identical, and the third arm is why."""
    from core.subject.intrinsic_v25 import intrinsic_rate_from_samples

    rng = np.random.default_rng(1)
    intact = rng.normal(size=(160, 5))
    sham_a = rng.normal(size=(160, 5))
    sham_b = rng.normal(size=(160, 5))
    estimate = intrinsic_rate_from_samples(
        intact, intact.copy(), sham_a, sham_b, tau_seconds=1.0, seed=0
    )
    assert estimate.excess_rate == pytest.approx(0.0, abs=1e-9)


def test_the_rate_falls_when_the_horizon_grows() -> None:
    """Dividing by tau is what stops a slower description from winning."""
    from core.subject.intrinsic_v25 import intrinsic_rate_from_samples

    rng = np.random.default_rng(2)
    intact = rng.normal(size=(160, 5))
    cut = rng.normal(loc=1.5, size=(160, 5))
    sham_a, sham_b = rng.normal(size=(160, 5)), rng.normal(size=(160, 5))
    fast = intrinsic_rate_from_samples(intact, cut, sham_a, sham_b, tau_seconds=1.0, seed=0)
    slow = intrinsic_rate_from_samples(intact, cut, sham_a, sham_b, tau_seconds=4.0, seed=0)
    assert slow.raw_rate < fast.raw_rate


# ── the grain ─────────────────────────────────────────────────────────────


def test_the_rank_estimator_finds_a_rank_that_is_there() -> None:
    from core.subject.intrinsic_v25 import fit_predictive_grain

    rng = np.random.default_rng(3)
    latent = rng.normal(size=(120, 3))
    loading = rng.normal(size=(3, 60))
    x = latent @ loading + 0.5 * rng.normal(size=(120, 60))
    assert fit_predictive_grain(x, null_draws=64, seed=1).rank == 3


def test_the_rank_estimator_finds_nothing_in_noise() -> None:
    """A grain fitted to noise is a grain that will pass anything."""
    from core.subject.intrinsic_v25 import fit_predictive_grain

    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 60))
    assert fit_predictive_grain(x, null_draws=64, seed=1).rank == 0


def test_the_frequency_bank_is_reproducible() -> None:
    """Preregistered means the same frequencies on the confirmatory run."""
    from core.subject.intrinsic_v25 import deterministic_frequency_bank

    first = deterministic_frequency_bank([8], frequencies_per_test=4, seed=99)
    second = deterministic_frequency_bank([8], frequencies_per_test=4, seed=99)
    assert np.allclose(first[0], second[0])


# ── the cuts ──────────────────────────────────────────────────────────────


def test_ten_domains_make_five_hundred_and_eleven_cuts() -> None:
    from core.subject.state import DOMAINS
    from core.subject.v25_runtime import bipartitions

    cuts = bipartitions(DOMAINS)
    assert len(cuts) == 511
    # Each cut exactly once: A|B and B|A are the same partition.
    seen = {frozenset([frozenset(left), frozenset(right)]) for left, right in cuts}
    assert len(seen) == 511
    for left, right in cuts:
        assert set(left) | set(right) == set(DOMAINS)
        assert not set(left) & set(right)


def test_no_cut_is_excluded() -> None:
    """The speed-up is sequential sampling, never a shorter list."""
    from core.subject.v25_cut import EXCLUDED

    assert not EXCLUDED


def test_one_undecided_cut_refuses_the_conjunction() -> None:
    """The claim is an intersection-union over cuts, so margin elsewhere buys nothing."""
    from core.subject.v25_cut import CutVerdict, SweepReport

    decided = CutVerdict(left=("A",), right=("B",), anchors_used=8, lower_bound=1.0, decided=True)
    open_one = CutVerdict(left=("B",), right=("A",), anchors_used=8, lower_bound=-0.1)
    report = SweepReport(tau_seconds=1.0, verdicts=[decided, open_one], undecided=[open_one.name])
    assert report.irreducible is False


# ── exclusion ─────────────────────────────────────────────────────────────


def test_components_are_found() -> None:
    from core.subject.v25_exclusion import strongly_connected_components

    nodes = list("ABCDE")
    edges = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "D")]
    assert strongly_connected_components(nodes, edges) == [("A", "B", "C"), ("D", "E")]


def test_crossing_spectra_are_incomparable() -> None:
    """No scale-free rule declares one of two crossing spectra globally larger."""
    from core.subject.v25_exclusion import Candidate, dominates, pareto_frontier

    left = {1.0: 2.0, 2.0: 1.0}
    right = {1.0: 1.0, 2.0: 2.0}
    assert dominates(left, right) is False
    assert dominates(right, left) is False

    a = Candidate(frozenset("AB"), True, True, left)
    b = Candidate(frozenset("BC"), True, True, right)
    frontier = pareto_frontier([a, b])
    assert len(frontier) == 2, "a genuine tie was broken; that manufactures a unique carrier"


def test_a_dominated_overlapping_candidate_leaves_the_frontier() -> None:
    from core.subject.v25_exclusion import Candidate, pareto_frontier

    weak = Candidate(frozenset("AB"), True, True, {1.0: 1.0, 2.0: 1.0})
    strong = Candidate(frozenset("BC"), True, True, {1.0: 2.0, 2.0: 2.0})
    assert [c.support for c in pareto_frontier([weak, strong])] == [frozenset("BC")]


def test_disjoint_candidates_do_not_exclude_each_other() -> None:
    """Two processes sharing no part are two subjects, not one and a loser."""
    from core.subject.v25_exclusion import Candidate, pareto_frontier

    small = Candidate(frozenset("AB"), True, True, {1.0: 0.5})
    large = Candidate(frozenset("CD"), True, True, {1.0: 9.0})
    assert len(pareto_frontier([small, large])) == 2


def test_closure_and_recurrence_are_gates_not_terms() -> None:
    """An open candidate with a huge rate is not a candidate at all."""
    from core.subject.v25_exclusion import Candidate, pareto_frontier

    open_one = Candidate(frozenset("AB"), closed=False, recurrent=True, spectrum={1.0: 99.0})
    feedforward = Candidate(frozenset("CD"), closed=True, recurrent=False, spectrum={1.0: 99.0})
    assert open_one.carrier is False
    assert feedforward.carrier is False
    assert pareto_frontier([open_one, feedforward]) == ()


def test_one_closed_component_over_everything_is_the_unique_carrier() -> None:
    from core.subject.v25_exclusion import select_carriers

    report = select_carriers(
        list("ABC"),
        [("A", "B"), ("B", "C"), ("C", "A")],
        {frozenset("ABC"): {1.0: 0.4}},
        {frozenset("ABC"): (True, 0.0)},
    )
    assert report.as_dict()["the_core_is_the_unique_domain_level_carrier"] is True
    assert report.status == "FOUND"


# ── the playback null ─────────────────────────────────────────────────────


def test_the_playback_null_collapses() -> None:
    """A cut arm that is the intact arm cannot have been damaged by the cut.

    This is the movie objection in one line. A measure that reports damage
    where the two arms carry identical states is measuring its own estimator.
    """
    from core.subject.intrinsic_v25 import intrinsic_rate_from_samples

    rng = np.random.default_rng(5)
    intact = rng.normal(size=(160, 6))
    sham_a, sham_b = rng.normal(size=(160, 6)), rng.normal(size=(160, 6))
    playback = intrinsic_rate_from_samples(
        intact, intact.copy(), sham_a, sham_b, tau_seconds=1.0, seed=0
    )
    assert playback.excess_rate == pytest.approx(0.0, abs=1e-9)


def test_duplicating_channels_does_not_raise_the_rate() -> None:
    """If copying a field made her more integrated, the measure is broken."""
    from core.subject.intrinsic_v25 import intrinsic_rate_from_samples

    rng = np.random.default_rng(6)
    intact = rng.normal(size=(200, 5))
    cut = rng.normal(loc=0.8, size=(200, 5))
    sham_a, sham_b = rng.normal(size=(200, 5)), rng.normal(size=(200, 5))
    plain = intrinsic_rate_from_samples(
        intact, cut, sham_a, sham_b, tau_seconds=1.0, seed=0
    ).excess_rate
    doubled = intrinsic_rate_from_samples(
        np.hstack([intact, intact]), np.hstack([cut, cut]),
        np.hstack([sham_a, sham_a]), np.hstack([sham_b, sham_b]),
        tau_seconds=1.0, seed=0,
    ).excess_rate
    assert doubled <= plain * 1.5


# ── provenance and wording ────────────────────────────────────────────────


def test_the_v25_campaign_is_not_the_batterys() -> None:
    """A different experiment is a different campaign, by construction."""
    from core.subject.provenance import campaign, campaign_v25

    battery = campaign(seed=7, rounds=3, trials=2, turns=1)
    v25 = campaign_v25(
        seed=7, rounds=3, anchors=3, history_turns=2, turns=1,
        cut_rounds=1, support=("P", "I"),
    )
    assert v25["fingerprint"] != battery["fingerprint"]
    assert v25["frozen"]["generation"] == "v25"


def test_moving_a_frozen_value_starts_a_different_campaign() -> None:
    from core.subject.provenance import campaign_v25

    base = dict(seed=7, rounds=3, anchors=3, history_turns=2, turns=1, cut_rounds=1,
                support=("P", "I"))
    first = campaign_v25(**base)["fingerprint"]
    second = campaign_v25(**{**base, "history_turns": 4})["fingerprint"]
    assert first != second


def test_the_verdict_is_never_the_word_conscious() -> None:
    """The report says what was measured. The bridge is a postulate, not a finding."""
    source = (REPO / "tools" / "run_subject_core_v25.py").read_text(encoding="utf-8")
    assert '"reported_as": "INTRINSIC_CARRIER_" + status' in source
    assert '"phenomenal_bridge": "UNVALIDATED"' in source
    for verdict in ('"CONSCIOUS"', "'CONSCIOUS'"):
        assert verdict not in source, "a v25 run must never report that word"


def test_the_bridge_is_named_as_a_postulate() -> None:
    source = (REPO / "tools" / "run_subject_core_v25.py").read_text(encoding="utf-8")
    assert "carrier-identity postulate" in source


# ── a screened sweep is a look, not a result ──────────────────────────────


def test_a_screened_sweep_never_reads_as_irreducible() -> None:
    """The score is the weakest cut, and a sample of cuts has not found it."""
    from core.subject.v25_cut import CutVerdict, SweepReport

    strong = [
        CutVerdict(left=("A",), right=("B",), anchors_used=8, lower_bound=1.0, decided=True)
        for _ in range(3)
    ]
    full = SweepReport(tau_seconds=1.0, verdicts=strong)
    screened = SweepReport(tau_seconds=1.0, verdicts=strong, screened=True, cuts_in_full=511)
    assert full.irreducible is True
    assert screened.irreducible is False


def test_a_screened_sweep_says_how_many_cuts_it_skipped() -> None:
    from core.subject.v25_cut import SweepReport

    report = SweepReport(tau_seconds=1.0, screened=True, cuts_in_full=511).as_dict()
    assert report["screened"] is True
    assert report["cuts_in_full"] == 511


def test_the_runner_refuses_a_screened_run() -> None:
    source = (REPO / "tools" / "run_subject_core_v25.py").read_text(encoding="utf-8")
    assert "only a sample of the bipartitions was scored" in source
