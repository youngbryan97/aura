"""core/connectome/corroboration.py — two instruments on the same organism.

Operationally: takes the influence the connectome measured between phase
stations and the influence the subject battery measured between state domains,
maps the two onto each other by what they read, and reports where they agree
and where they do not.

Why this exists
---------------
Two things in this repository measure how one part of her moves another, and
until now neither read the other:

* The connectome records which CELLS fire during a turn, assigns them to seven
  phase stations, and scores every ordered pair against its own rotations.
  What it sees is the code running.
* The subject battery records ten state DOMAINS every tick, perturbs each one,
  and scores every ordered pair against its own null suite. What it sees is the
  state moving.

They share no code, no recording and no null. They are looking at the same
organism through different windows, and five of the things they name are the
same thing — her interoception is her interoception whether you read it off the
phases that compute it or the domain that holds it.

So they can check each other. Agreement between two instruments that share
nothing is worth more than either alone, and a disagreement says one of them is
measuring something other than what it is named after. Neither outcome is
available while they do not speak.

What this is not
----------------
Not a merge. The two keep their own numbers, their own nulls and their own
verdicts; nothing here averages them or lets one overrule the other. It reports
the overlap and leaves both instruments intact, because an instrument that has
been adjusted to agree with another one has stopped being a second opinion.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.Connectome.Corroboration")

__all__ = [
    "CORRESPONDENCE",
    "Agreement",
    "CorroborationReport",
    "corroborate",
    "read_domain_edges",
    "read_station_pairs",
]


#: Which phase station each state domain stands for, and which stand for none.
#:
#: Declared rather than inferred, for the reason `PHASE_STATIONS` is: this table
#: is the assumption the comparison rests on, so it belongs where a reader can
#: disagree with it. A domain mapped to "" is one the connectome has no station
#: for, and saying so is what keeps the overlap honest — scoring a pair the
#: other instrument never measured would be scoring a coincidence.
CORRESPONDENCE: dict[str, str] = {
    "I": "interoception",
    "A": "affect",
    "G": "workspace",
    "C": "higher_order",
    "S": "self_model",
    "D": "planning",
    # No station. The battery reads state and the connectome reads phases, and
    # these four are read by one and not the other.
    "P": "",  # perception: the connectome folds sensing into interoception
    "M": "",  # active memory: storage, deliberately outside the ring
    "W": "",  # world model: no phase station stands for it
    "N": "",  # developmental state: a slower clock than a turn
}

#: The connectome's seventh station. The battery has no domain for it, because
#: it measures state and an action is an output.
STATION_WITHOUT_A_DOMAIN = "action"

#: What would refuse the comparison, written before it was run. Two instruments
#: with nothing in common will agree on some pairs by chance; agreeing on fewer
#: than they would by coin-flip is the outcome that says one of them is not
#: measuring what it is named after.
FALSIFIED_BELOW = 0.5


@dataclass(frozen=True, slots=True)
class Agreement:
    """One ordered pair, as each instrument saw it."""

    station_link: str
    domain_link: str
    connectome_carries: bool
    connectome_gain: float
    battery_carries: bool
    battery_effect: float
    battery_q: float

    @property
    def agree(self) -> bool:
        return self.connectome_carries == self.battery_carries

    def as_json(self) -> dict[str, Any]:
        return {
            "stations": self.station_link,
            "domains": self.domain_link,
            "connectome_carries": self.connectome_carries,
            "connectome_gain": round(self.connectome_gain, 5),
            "battery_carries": self.battery_carries,
            "battery_effect": round(self.battery_effect, 5),
            "battery_q": self.battery_q,
            "agree": self.agree,
        }


@dataclass(slots=True)
class CorroborationReport:
    """What the two instruments said about the pairs they both measured."""

    pairs: list[Agreement] = field(default_factory=list)
    unmatched_stations: tuple[str, ...] = ()
    unmatched_domains: tuple[str, ...] = ()

    @property
    def agreed(self) -> int:
        return sum(1 for pair in self.pairs if pair.agree)

    @property
    def rate(self) -> float:
        return self.agreed / len(self.pairs) if self.pairs else 0.0

    @property
    def holds(self) -> bool:
        return bool(self.pairs) and self.rate >= FALSIFIED_BELOW

    def as_json(self) -> dict[str, Any]:
        both = [pair for pair in self.pairs if pair.connectome_carries and pair.battery_carries]
        neither = [
            pair
            for pair in self.pairs
            if not pair.connectome_carries and not pair.battery_carries
        ]
        return {
            "compared": len(self.pairs),
            "agreed": self.agreed,
            "rate": round(self.rate, 4),
            "both_say_influence": len(both),
            "neither_says_influence": len(neither),
            "disagreed": [pair.as_json() for pair in self.pairs if not pair.agree],
            "holds": self.holds,
            "unmatched_stations": list(self.unmatched_stations),
            "unmatched_domains": list(self.unmatched_domains),
            "verdict": (
                f"{self.agreed} of {len(self.pairs)} ordered pairs are seen the same way "
                f"by two instruments that share no code, no recording and no null"
            ),
        }


def read_station_pairs(path: str | Path) -> dict[tuple[str, str], tuple[bool, float]]:
    """The connectome's ordered station pairs, by (from, to)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = (
        payload
        if isinstance(payload, list)
        else payload.get("rows") or payload.get("pairs") or []
    )
    found: dict[tuple[str, str], tuple[bool, float]] = {}
    for row in rows:
        link = str(row.get("link") or "")
        if "->" not in link:
            continue
        left, right = (part.strip() for part in link.split("->", 1))
        found[(left, right)] = (bool(row.get("carries")), float(row.get("gain") or 0.0))
    return found


def read_domain_edges(
    path: str | Path, *, q_ceiling: float = 0.05
) -> dict[tuple[str, str], tuple[bool, float, float]]:
    """The battery's ordered domain edges, by (from, to).

    An edge counts as carrying when it survived the battery's own multiple-
    comparison correction. The threshold is the battery's, not one invented
    here: reading its numbers under a different rule would make the comparison
    about the rule.
    """
    found: dict[tuple[str, str], tuple[bool, float, float]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                effect = float(row["effect"])
                q = float(row["q"])
            except (KeyError, TypeError, ValueError):
                continue
            found[(row["source"].strip(), row["target"].strip())] = (
                q <= q_ceiling,
                effect,
                q,
            )
    return found


def corroborate(
    station_pairs: Mapping[tuple[str, str], tuple[bool, float]],
    domain_edges: Mapping[tuple[str, str], tuple[bool, float, float]],
    *,
    correspondence: Mapping[str, str] | None = None,
) -> CorroborationReport:
    """Compare the two instruments on every pair they both measured."""
    table = dict(correspondence or CORRESPONDENCE)
    to_station = {domain: station for domain, station in table.items() if station}
    report = CorroborationReport()

    for (source, target), (battery_carries, effect, q) in sorted(domain_edges.items()):
        left, right = to_station.get(source), to_station.get(target)
        if not left or not right or left == right:
            continue
        seen = station_pairs.get((left, right))
        if seen is None:
            continue
        report.pairs.append(
            Agreement(
                station_link=f"{left} -> {right}",
                domain_link=f"{source} -> {target}",
                connectome_carries=seen[0],
                connectome_gain=seen[1],
                battery_carries=battery_carries,
                battery_effect=effect,
                battery_q=q,
            )
        )

    matched = {station for station in to_station.values()}
    report.unmatched_stations = tuple(
        sorted({left for left, _ in station_pairs} - matched)
    )
    report.unmatched_domains = tuple(
        sorted(domain for domain, station in table.items() if not station)
    )
    return report


def corroborate_from_disk(
    station_pairs_path: str | Path,
    domain_edges_path: str | Path,
    *,
    correspondence: Mapping[str, str] | None = None,
) -> CorroborationReport:
    """Read both instruments' recorded output and compare them."""
    return corroborate(
        read_station_pairs(station_pairs_path),
        read_domain_edges(domain_edges_path),
        correspondence=correspondence,
    )
