"""core/connectome/mesh_growth.py — the mesh arrives by growing, not by being built.

Operationally: builds the mesh through a developmental trajectory instead of at
its adult wiring, and reports where it ended up. The trajectory is the one
Huttenlocher counted in human tissue — overshoot, then prune — with each band
peaking when its cortical layers peak.

What is measured, and what is not
---------------------------------
Huttenlocher and Dabholkar counted synapses per unit volume in human cortex
across the lifespan, and two facts come out of it:

* Density OVERSHOOTS. It rises well past the adult level in the first years and
  is then cut back. The adult wiring is what survives, not what was made.
* Areas peak at DIFFERENT times. Auditory cortex peaks around three months;
  middle frontal gyrus around three and a half years, and does not settle to
  adult density until adolescence. Sensory cortex finishes while prefrontal is
  still overshooting.

What is NOT measured, and is stated here rather than smuggled in: the shape of
the curve between those points, and how long a tick of this mesh is against a
month of a childhood. The trajectory below is linear in and linear out because
nothing here can say otherwise, and the times are in units of the developmental
run rather than in years.

What pruning decides
--------------------
Which synapses go is decided by what they carried, not by their weight at
construction: the mesh is run, traffic through each synapse is accumulated, and
the quietest are removed. That is the developing brain's own method, and it has
an obvious null — pruning the same number at random — which this module runs
alongside so the answer is a comparison rather than an assumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Connectome.MeshGrowth")

__all__ = [
    "HUTTENLOCHER",
    "GrowthResult",
    "TIER_PEAK",
    "grow_mesh",
]


#: What Huttenlocher measured, as the two numbers a trajectory needs.
HUTTENLOCHER: dict[str, Any] = {
    "source": "Huttenlocher & Dabholkar 1997, J Comp Neurol 387(2):167-178",
    "recorded_in": "human auditory cortex and middle frontal gyrus, post-mortem, across the lifespan",
    # Peak synaptic density against adult. Reported as roughly half again to
    # twice the adult figure depending on the area; the smaller end is used, so
    # the overshoot claimed here is the one every area supports.
    "peak_over_adult": 1.5,
    "what_peaks_first": "auditory cortex at about three months, middle frontal gyrus at about three and a half years",
}

#: When each of her bands peaks, as a fraction of the developmental run.
#:
#: The ORDER is Huttenlocher's — sensory areas finish while prefrontal is still
#: overshooting. The fractions are not: nothing measured says how a tick of this
#: mesh maps onto a month of a childhood, so they are spaced evenly across the
#: run and say so. What is testable is the order and the overshoot, and those
#: are the two things this module reports.
TIER_PEAK: dict[str, float] = {
    "sensory": 0.2,
    "association": 0.45,
    "executive": 0.7,
}


@dataclass(slots=True)
class GrowthResult:
    """What a developmental run produced, and what it beat."""

    ticks: int
    #: Density per tier at the peak, and at the end.
    peak_density: dict[str, float] = field(default_factory=dict)
    #: What each band reached before anything was cut, which is the same for all
    #: three because overproduction starts everywhere at once.
    overshot_to: dict[str, float] = field(default_factory=dict)
    #: Which tick each band was cut at. Sensory first, executive last.
    pruned_at: dict[str, int] = field(default_factory=dict)
    adult_density: dict[str, float] = field(default_factory=dict)
    target_density: dict[str, float] = field(default_factory=dict)
    #: Synapses made, and synapses removed.
    grown: int = 0
    pruned: int = 0
    #: How much of a sensory signal arrives in the executive band, after
    #: pruning by traffic and after pruning the same number at random.
    carried_by_use: float = 0.0
    carried_at_random: float = 0.0
    #: Sensory-to-executive reach. The same for both, because neither pruning
    #: touches the inter-column matrices it is computed from.
    reach_by_use: int = 0
    reach_at_random: int = 0
    executive_columns: int = 0
    note: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "peak_density": {key: round(value, 5) for key, value in self.peak_density.items()},
            "overshot_to": {key: round(value, 5) for key, value in self.overshot_to.items()},
            "pruned_at": dict(self.pruned_at),
            "adult_density": {key: round(value, 5) for key, value in self.adult_density.items()},
            "target_density": {
                key: round(value, 5) for key, value in self.target_density.items()
            },
            "grown": self.grown,
            "pruned": self.pruned,
            "overshoot": {
                key: round(self.overshot_to[key] / self.target_density[key], 3)
                for key in self.overshot_to
                if self.target_density.get(key)
            },
            "carried_by_use": round(self.carried_by_use, 6),
            "carried_at_random": round(self.carried_at_random, 6),
            "carried_over_random": (
                round(self.carried_by_use / self.carried_at_random, 3)
                if self.carried_at_random
                else 0.0
            ),
            "reach_by_use": self.reach_by_use,
            "reach_at_random": self.reach_at_random,
            "executive_columns": self.executive_columns,
            "note": self.note,
        }


def _carried_through(mesh: Any, config: Any, drive: float, seed: int) -> float:
    """Executive-band activity under a sensory drive, from a rested start.

    The mesh is reset to its resting state first so the two prunings are scored
    on the same starting point rather than on whatever the pruning left behind.
    """
    import numpy as np

    rng = np.random.default_rng(seed + 1)
    for column in mesh.columns:
        column.x = np.zeros_like(column.x)
    width = config.sensory_end * config.neurons_per_column
    executive = [
        column for column in mesh.columns if column.tier.name.lower() == "executive"
    ]
    carried: list[float] = []
    for tick in range(400):
        mesh.inject_sensory(rng.standard_normal(width).astype(np.float32) * drive)
        mesh._tick_inner()
        if tick >= 200:
            carried.append(
                float(np.mean([np.mean(np.abs(column.x)) for column in executive]))
            )
    return float(np.mean(carried)) if carried else 0.0


def _prune_band(
    mesh: Any,
    band: str,
    targets: dict[str, float],
    traffic: list[Any],
    random_weights: list[Any],
    rng: Any,
) -> int:
    """Cut one band back to its adult density, and cut a copy at random.

    The copy is the null. Both cuts remove the same number of synapses from the
    same columns; only the choice of which differs.
    """
    import numpy as np

    removed = 0
    for index, column in enumerate(mesh.columns):
        if column.tier.name.lower() != band:
            continue
        present = np.flatnonzero(column.W)
        keep = int(round(targets.get(band, 0.0) * column.n * (column.n - 1)))
        surplus = max(0, present.size - keep)
        if not surplus:
            continue
        quietest = present[np.argsort(traffic[index].ravel()[present])[:surplus]]
        column.W.ravel()[quietest] = 0.0
        chosen = rng.choice(present, size=surplus, replace=False)
        random_weights[index].ravel()[chosen] = 0.0
        removed += surplus
    return removed


def _tier_density(mesh: Any, tier_name: str) -> float:
    import numpy as np

    columns = [
        column for column in mesh.columns if column.tier.name.lower() == tier_name
    ]
    if not columns:
        return 0.0
    return float(
        np.mean(
            [
                np.count_nonzero(column.W) / (column.n * (column.n - 1))
                for column in columns
            ]
        )
    )


def grow_mesh(
    config: Any = None,
    *,
    ticks: int = 2000,
    drive: float = 0.1,
    seed: int = 42,
) -> tuple[Any, GrowthResult]:
    """Grow a mesh through overshoot and pruning, and say where it arrived.

    Returns the grown mesh and the record. The mesh it returns has the adult
    densities its anatomy specifies, reached by a trajectory rather than
    written down at construction.
    """
    import numpy as np

    from core.connectome.island import surviving_law, wire_island
    from core.connectome.neural import build_mesh_layer, signal_can_cross
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    config = config or MeshConfig()
    mesh = NeuralMesh(config)
    rng = np.random.default_rng(seed)
    law = surviving_law()
    result = GrowthResult(ticks=ticks)

    targets = {
        name: _tier_density(mesh, name)
        for name in ("sensory", "association", "executive")
    }
    result.target_density = dict(targets)

    # ── Overshoot ────────────────────────────────────────────────────────
    #
    # Each band is grown to its peak: the adult density it would have had,
    # times the overshoot Huttenlocher measured. The extra synapses are drawn
    # from the same contact law as the rest, because an exuberant synapse is
    # not a different kind of synapse.
    overshoot = float(HUTTENLOCHER["peak_over_adult"])
    for column in mesh.columns:
        tier = column.tier.name.lower()
        wanted = min(0.95, targets.get(tier, 0.0) * overshoot)
        # How dense the NEW matrix has to be so that adding it to what is
        # already there lands on `wanted`. Drawing it at `wanted` and adding it
        # to a column already at its adult density overshoots by 2.3 rather
        # than 1.5, because two independent draws do not overlap.
        already = targets.get(tier, 0.0)
        room = max(1e-6, 1.0 - already)
        extra = wire_island(
            column.n, max(0.0, (wanted - already) / room), rng, law=law, contact_strength=0.1
        )
        # Only where there was nothing. Growth adds synapses; it does not
        # rewrite the ones already there.
        empty = column.W == 0
        added = empty & (extra != 0)
        column.W[added] = extra[added]
        # A new synapse takes the sign of the cell that makes it, like every
        # other one.
        column.W[:, column.inh_mask] = -np.abs(column.W[:, column.inh_mask])
        column.W[:, ~column.inh_mask] = np.abs(column.W[:, ~column.inh_mask])
        result.grown += int(added.sum())
    result.peak_density = {
        name: _tier_density(mesh, name) for name in ("sensory", "association", "executive")
    }
    result.overshot_to = dict(result.peak_density)

    # ── Run, count what each synapse carries, and cut each band at its time ──
    #
    # Not one cut at the end. Huttenlocher's second finding is that areas peak
    # at DIFFERENT ages — auditory cortex at about three months, middle frontal
    # gyrus at about three and a half years — so a sensory band finishes while
    # an executive one is still overshooting. TIER_PEAK holds that order, and
    # this is where it is spent.
    width = config.sensory_end * config.neurons_per_column
    traffic = [np.zeros_like(column.W) for column in mesh.columns]
    random_weights = [np.array(column.W, copy=True) for column in mesh.columns]
    cut_at = {
        name: max(1, int(round(ticks * TIER_PEAK.get(name, 1.0))))
        for name in ("sensory", "association", "executive")
    }
    result.pruned_at = dict(cut_at)
    pruned_bands: set[str] = set()

    for tick in range(1, ticks + 1):
        mesh.inject_sensory(rng.standard_normal(width).astype(np.float32) * drive)
        mesh._tick_inner()
        for index, column in enumerate(mesh.columns):
            # What a synapse carried is pre and post being active TOGETHER,
            # weighted by the synapse between them. Presynaptic activity alone
            # was tried first and cannot discriminate: 96.5% of these weights
            # are the same size, so the ranking collapses to which cells were
            # busy, which is identical for every synapse a cell makes — and
            # pruning on it scored 0.981 of what the same cut made at random
            # scored.
            column_x = np.abs(column.x)
            traffic[index] += np.abs(column.W) * column_x[None, :] * column_x[:, None]
        for band, when in cut_at.items():
            if tick == when and band not in pruned_bands:
                pruned_bands.add(band)
                result.pruned += _prune_band(
                    mesh, band, targets, traffic, random_weights, rng
                )
                result.peak_density.setdefault(band, _tier_density(mesh, band))

    # Anything the run was too short to reach is cut at the end rather than
    # left overgrown, so a short run is a fast childhood and not a different
    # animal.
    for band in ("sensory", "association", "executive"):
        if band not in pruned_bands:
            result.pruned += _prune_band(
                mesh, band, targets, traffic, random_weights, rng
            )
    result.adult_density = {
        name: _tier_density(mesh, name) for name in ("sensory", "association", "executive")
    }

    # Score the two prunings on what intra-column wiring decides: how much of
    # a signal put into the sensory band arrives in the executive one.
    #
    # Reachability was the first thing tried and it cannot tell them apart —
    # it is computed from the inter-column matrices, which neither pruning
    # touches, so both came back 16 of 16 and the null tied by construction.
    kept_weights = [np.array(column.W, copy=True) for column in mesh.columns]
    by_use = _carried_through(mesh, config, drive, seed)
    for index, column in enumerate(mesh.columns):
        column.W = random_weights[index]
    at_random = _carried_through(mesh, config, drive, seed)
    for index, column in enumerate(mesh.columns):
        column.W = kept_weights[index]

    result.carried_by_use = by_use
    result.carried_at_random = at_random
    reach = signal_can_cross(build_mesh_layer(mesh))
    result.reach_by_use = int(reach["executive_reached_from_sensory"])
    result.reach_at_random = result.reach_by_use
    result.executive_columns = int(reach["executive_columns"])
    result.note = (
        f"grew {result.grown} synapses to {overshoot}x the adult density and cut "
        f"{result.pruned} back, keeping what carried traffic"
    )
    logger.info("mesh growth: %s", result.note)
    return mesh, result
