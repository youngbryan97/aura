"""ISC-v5: the two irreducibility lines, read by intervention.

ISC-v3 asks whether knowing the rest of the system improves the held-out
prediction of one side's next turn. That regression cannot see integration
that runs through what arrives: on input-driven pipelines integrated by
construction it scores the same as one with no coupling, because each turn's
fresh input fills the loss the cut is divided by. v5 asks the other way round.
From one snapshot, hold one side of a cut and let the other run, then the
reverse, compose the free halves, and compare with the untouched run. The
decision is `core.subject.v25_cut.decide_cut` at the preregistered schedule
below, and everything else in the conjunction is v3's.

The design is fixed here and in docs/ISC_V5_PREREGISTRATION.md, and the runner
reads it from here, so the two cannot drift apart:

    horizon     one turn, 33 frames at the experiment clock, decides; two turns
                are reported beside it and decide nothing
    looks       a cut is scored at its first 8, 16, 32, 64, 96 and 128 anchors
                and stops at the first look where its lower bound clears zero
    level       each look at alpha / 6, so a cut that costs nothing is decided
                at any look with probability under alpha
    draws       1000 bootstrap draws behind each lower bound

`partition_irreducibility` passes when every one of the 511 cuts is decided on
a sweep that is not a screen and not an unmerged shard: an intersection-union
test, so one undecided cut refuses the claim.

`partition_beats_nulls` passes when the playback control, which replays the
untouched run as the cut arm, is decided at no cut, and no null architecture
that passes every other v3 line is decided at all 511 under the same design.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "ALPHA",
    "ANCHORS",
    "DECIDING_LAG",
    "DRAWS",
    "LOOKS",
    "REPORTED_LAGS",
    "design",
    "lines",
    "nulls_passing_the_rest",
]

LOOKS: tuple[int, ...] = (8, 16, 32, 64, 96, 128)
ALPHA: float = 0.05
DRAWS: int = 1000
DECIDING_LAG: int = 33
REPORTED_LAGS: tuple[int, ...] = (66,)
ANCHORS: int = max(LOOKS)


def design() -> dict[str, Any]:
    """The schedule a v5 sweep runs, as the runner and the fingerprint read it."""
    return {
        "looks": list(LOOKS),
        "alpha": ALPHA,
        "alpha_per_look": ALPHA / len(LOOKS),
        "draws": DRAWS,
        "lags": [DECIDING_LAG, *REPORTED_LAGS],
        "deciding": [DECIDING_LAG],
        "anchors": ANCHORS,
    }


def nulls_passing_the_rest(table: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The null architectures that pass every v3 line but irreducibility, on one seed's table.

    These are the ones v5 has to sweep: a null already told apart from a
    subject by another line is not asked about again.
    """
    from core.subject.null_verdicts import REFERENCE, _passes_v3_all_but_irreducibility

    return sorted(
        name
        for name, row in table.items()
        if name != REFERENCE and row.get("kind") == "architecture" and _passes_v3_all_but_irreducibility(row)
    )


def _decided_everywhere(null: Mapping[str, Any]) -> bool:
    """Whether a null's sweep decided every cut, read off either report shape."""
    decided = null.get("cuts_decided", null.get("decided"))
    cuts = null.get("cuts_in_full", null.get("cuts"))
    return decided is not None and cuts is not None and int(decided) == int(cuts)


def _conforms(sweep: Mapping[str, Any]) -> list[str]:
    """Why a sweep report was not run to the v5 design, or nothing."""
    problems: list[str] = []
    if list(sweep.get("looks") or []) != list(LOOKS):
        problems.append(f"looks {sweep.get('looks')} are not the preregistered {list(LOOKS)}")
    if int(sweep.get("draws") or 0) != DRAWS:
        problems.append(f"{sweep.get('draws')} bootstrap draws, not {DRAWS}")
    if abs(float(sweep.get("alpha_per_look") or 0.0) - ALPHA / len(LOOKS)) > 1e-12:
        problems.append(f"each look read at {sweep.get('alpha_per_look')}, not {ALPHA / len(LOOKS)}")
    if not sweep.get("deciding", False):
        problems.append("the horizon read is not the deciding one")
    if sweep.get("screened"):
        problems.append("a screen of the cuts, which never decides the line")
    if sweep.get("shard"):
        problems.append(f"shard {sweep.get('shard')} on its own, not the merged sweep")
    return problems


def lines(
    sweep: Mapping[str, Any] | None,
    *,
    playback_decided: int | None,
    nulls_that_pass: Sequence[str],
    null_sweeps: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The two v5 lines, as plain records the battery turns into criteria.

    `sweep` is the merged `SweepReport.as_dict` at the deciding horizon.
    `playback_decided` is how many cuts the playback control decided.
    `nulls_that_pass` are the null architectures that pass every other v3
    line on this campaign, and `null_sweeps` their v5 sweeps by name.
    """
    if not sweep:
        blocked = "no v5 sweep was read"
        return [
            {"key": "partition_irreducibility", "passed": False, "value": None, "why": blocked},
            {"key": "partition_beats_nulls", "passed": False, "value": None, "why": blocked},
        ]
    problems = _conforms(sweep)
    cuts = int(sweep.get("cuts_tested") or 0)
    in_full = int(sweep.get("cuts_in_full") or 0)
    decided = int(sweep.get("cuts_decided") or 0)
    every = bool(cuts) and cuts == in_full and decided == cuts and not sweep.get("undecided")
    irreducible = every and not problems
    missing = [name for name in nulls_that_pass if name not in null_sweeps]
    decided_everywhere = sorted(
        name for name, null in null_sweeps.items() if name in nulls_that_pass and _decided_everywhere(null)
    )
    unconforming = sorted(
        name for name, null in null_sweeps.items() if name in nulls_that_pass and _conforms({**null, "deciding": True})
    )
    playback_clean = playback_decided == 0
    beats = irreducible and playback_clean and not missing and not decided_everywhere and not unconforming
    return [
        {
            "key": "partition_irreducibility",
            "passed": irreducible,
            "value": {"decided": decided, "cuts": cuts, "in_full": in_full},
            "why": "; ".join(problems) or (
                "every cut decided" if every else f"undecided: {list(sweep.get('undecided') or [])[:8]}"
            ),
            "weakest_cut": sweep.get("weakest_cut"),
            "weakest_lower_bound": sweep.get("weakest_lower_bound"),
            "anchors_spent": sweep.get("anchors_spent"),
        },
        {
            "key": "partition_beats_nulls",
            "passed": beats,
            "value": {
                "playback_decided": playback_decided,
                "nulls_that_pass": list(nulls_that_pass),
                "decided_at_every_cut": decided_everywhere,
                "unswept": missing,
                "not_run_to_the_design": unconforming,
            },
            "why": (
                "the line needs irreducibility first"
                if not irreducible
                else "the playback control was decided somewhere"
                if not playback_clean
                else f"no v5 sweep for {missing}"
                if missing
                else f"{unconforming} were not swept to the preregistered design"
                if unconforming
                else f"{decided_everywhere} decided at every cut"
                if decided_everywhere
                else "playback decided nowhere and no passing null decided everywhere"
            ),
        },
    ]
