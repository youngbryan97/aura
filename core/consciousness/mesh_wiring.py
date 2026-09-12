"""core/consciousness/mesh_wiring.py — how the mesh is joined up.

Operationally: builds the long-range matrices the mesh integrates, and
guarantees the two things a graph has to have before anything can be measured
on it — that every column can send and receive, and that a signal put into the
sensory band can arrive in the executive one.

Lifted out of `neural_mesh.py`, which had grown to hold two different subjects:
how the mesh is wired and how it runs. They change for different reasons and
are read by different questions.

**The receiver is the ROW.** `recurrent = W @ x` means
`recurrent[i] = sum_j W[i, j] * x[j]`, so W[i, j] is the weight FROM j INTO i.
The tick is the authority on that, because it is what actually moves activity.
Every builder here writes `weights[target, source]`; three of them wrote the
other way round until 2026-09-09, and so did the reachability reader, so each
agreed with itself and none agreed with the tick.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.Consciousness.MeshWiring")

__all__ = [
    "CorticalTier",
    "MeshWiring",
]


class CorticalTier(Enum):
    """Hierarchical tier of a cortical column.

    Here rather than in `neural_mesh` because every builder below is written in
    terms of it, and a mesh that runs has to import its wiring anyway.
    """

    SENSORY = auto()       # columns 0-15   — close to embodiment/interoception
    ASSOCIATION = auto()   # columns 16-47  — cross-modal integration
    EXECUTIVE = auto()     # columns 48-63  — executive control / self-model


class MeshWiring:
    """The wiring half of `NeuralMesh`, mixed into it.

    A mixin rather than a helper object because every one of these reads the
    mesh's own configuration, its per-structure generators and its tier map,
    and threading five of those through five signatures would say less about
    the code than `self` already does.
    """

    def _build_inter_column_weights(self) -> np.ndarray:
        """Build sparse, distance-weighted inter-column connectivity.

        Connectivity probability decays exponentially with column index distance.
        Feedforward (sensory→assoc→exec) is stronger than feedback.
        """
        n = self.cfg.columns
        weights = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist = abs(i - j)
                prob = self.cfg.inter_column_density * np.exp(-dist * self.cfg.inter_column_distance_decay)
                if self._hubs[i] and self._hubs[j]:
                    # Two hubs, and the distance term goes with them. Cortex's
                    # club spans the brain — its members are wired to each
                    # other whether they are neighbours or opposite poles —
                    # which is the whole reason it shortens long paths. Keeping
                    # the decay here left hub pairs at a probability of about
                    # one in a hundred and produced no hubs at all: degree
                    # topped out at 8 either way.
                    prob = min(1.0, self.cfg.inter_column_density * self.cfg.hub_coupling)
                if self._rng_inter.random() < prob:
                    strength = self._rng_inter.standard_normal() * 0.05
                    # Feedforward bias: sensory→assoc→exec gets 1.5× strength
                    tier_i = self._tier_for(i)
                    tier_j = self._tier_for(j)
                    if (tier_i == CorticalTier.SENSORY and tier_j == CorticalTier.ASSOCIATION) or \
                       (tier_i == CorticalTier.ASSOCIATION and tier_j == CorticalTier.EXECUTIVE):
                        strength *= 1.5
                    # i is the source and j the target here, so the receiver is
                    # the row. See RECEIVER_IS_THE_ROW.
                    weights[j, i] = strength
        return weights

    def _build_feedforward_weights(self) -> np.ndarray:
        """The bottom-up pathway, built the way the top-down one is.

        The mesh had an explicit feedback matrix and no explicit feedforward
        one. What carried signal upward was ``_build_inter_column_weights``,
        which is local wiring — its probability decays with the distance between
        column indices — and a tier boundary is exactly where that distance is
        large. So sensory injection could not reach the executive projection,
        and the two ends of the mesh that the rest of the system actually
        touches were in different connected components on every seed tried.

        Distance is left out here on purpose. A projection between cortical
        areas is an axon bundle, not a local connection, and its existence does
        not fall off with how far apart the areas are.
        """
        n = self.cfg.columns
        weights = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            tier_i = self._tier_for(i)
            for j in range(n):
                if i == j:
                    continue
                tier_j = self._tier_for(j)
                forward = (
                    tier_i == CorticalTier.SENSORY and tier_j == CorticalTier.ASSOCIATION
                ) or (
                    tier_i == CorticalTier.ASSOCIATION and tier_j == CorticalTier.EXECUTIVE
                )
                direct = (
                    tier_i == CorticalTier.SENSORY and tier_j == CorticalTier.EXECUTIVE
                )
                if forward:
                    probability = self.cfg.feedforward_density
                    scale = self.cfg.feedforward_strength
                elif direct:
                    probability = self.cfg.feedforward_direct_density
                    scale = self.cfg.feedforward_strength
                else:
                    continue
                if self._rng_feedforward.random() < probability:
                    # i drives j, and the receiver is the row.
                    weights[j, i] = self._rng_feedforward.standard_normal() * scale
        return weights

    def _connect_every_column(self) -> int:
        """Give every column at least one way in and one way out.

        Measured on seed 42: 22 of 64 columns had no outgoing inter-column edge
        and 21 had none incoming, four had neither, and a signal injected into
        the sensory tier reached 11 of 16 executive columns. A column with no
        edges is not a quiet column, it is tissue the rest of the mesh cannot
        use, and there is no such thing in cortex.

        The cause is the sampling rather than the densities. Each pair is a
        coin flip at a probability that decays with distance, and over 63
        chances a column can lose all of them; the expected number of columns
        that do is not zero and never was. What cortex says is that every
        column has connections, which is a statement about the conditional
        distribution: given that a column IS connected, where does it connect?

        So a column with nothing gets one edge drawn from that conditional —
        the same distance-decayed weights it was already being sampled under,
        renormalised over the partners it could have had. No new constant, and
        the shape of who connects to whom is unchanged.

        Returns how many edges it had to add.
        """
        added = 0
        n = self.cfg.columns
        if n < 2:
            return 0
        indices = np.arange(n)
        # The same distance decay the matrix was sampled under.
        distance = np.abs(indices[:, None] - indices[None, :]).astype(np.float64)
        affinity = np.exp(-distance * self.cfg.inter_column_distance_decay)
        np.fill_diagonal(affinity, 0.0)

        def _draw(row: np.ndarray) -> int:
            total = row.sum()
            if total <= 0:
                return int(self._rng_inter.integers(0, n))
            return int(self._rng_inter.choice(n, p=row / total))

        # A column's OUTPUT is its column of the matrix and its INPUT is its
        # row, because the receiver is the row. Read the other way round this
        # guaranteed the two things it was not checking.
        present = np.abs(self._inter_W) > 0
        for column in range(n):
            if not present[:, column].any():
                target = _draw(affinity[column].copy())
                self._inter_W[target, column] = (
                    self._rng_inter.standard_normal() * 0.05
                )
                present[target, column] = True
                added += 1
            if not present[column].any():
                source = _draw(affinity[:, column].copy())
                self._inter_W[column, source] = (
                    self._rng_inter.standard_normal() * 0.05
                )
                present[column, source] = True
                added += 1
        # And every executive column has to be REACHABLE from the sensory
        # band, which having an in-edge does not guarantee. The two ends of the
        # mesh the rest of the system touches are `inject_sensory` and
        # `get_executive_projection`; an executive column no sensory signal can
        # arrive at is tissue with nothing to do.
        #
        # Found when the rich club changed the graph: on one seed a signal
        # reached 15 of 16, with every column holding edges at both ends. Degree
        # is not reachability.
        added += self._connect_what_the_readers_cannot_reach(affinity)

        if added:
            logger.debug(
                "NeuralMesh connected %d column ends the unconditional draw left empty",
                added,
            )
        return added

    def _connect_what_the_readers_cannot_reach(self, affinity: np.ndarray) -> int:
        """Give every executive column a route from the sensory band."""
        from collections import deque

        n = self.cfg.columns
        added = 0
        for _ in range(n):
            # Walking forward means following a column's OUTPUT, which is its
            # column of the matrix.
            present = np.abs(self._inter_W) > 0
            seen = set(range(min(self.cfg.sensory_end, n)))
            queue = deque(seen)
            while queue:
                for target in np.flatnonzero(present[:, queue.popleft()]):
                    if int(target) not in seen:
                        seen.add(int(target))
                        queue.append(int(target))
            stranded = [
                column
                for column in range(self.cfg.association_end, n)
                if column not in seen
            ]
            if not stranded:
                break
            column = stranded[0]
            # From something the signal already reaches, chosen under the same
            # distance affinity the rest of the graph was drawn under.
            row = affinity[:, column].copy()
            row[[index for index in range(n) if index not in seen]] = 0.0
            total = row.sum()
            source = (
                int(self._rng_inter.choice(n, p=row / total))
                if total > 0
                else int(self._rng_inter.integers(0, max(1, self.cfg.sensory_end)))
            )
            # From a column the signal already reaches, into this one.
            self._inter_W[column, source] = self._rng_inter.standard_normal() * 0.05
            added += 1
        return added

    def _build_feedback_weights(self):
        """Build the explicit top-down (exec→sensory) feedback pathway.

        Lamme's RPT: consciousness arises specifically from recurrent feedback
        from higher cortical areas back to lower sensory areas. This creates
        an architecturally distinct pathway from feedforward processing.

        The feedback matrix connects executive columns (48-63) back to sensory
        columns (0-15) via association columns (16-47) as relay. The strength
        is configurable and the pathway can be ablated for adversarial testing.
        """
        n = self.cfg.columns
        weights = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            tier_i = self._tier_for(i)
            for j in range(n):
                tier_j = self._tier_for(j)
                # Executive → Association (feedback)
                if tier_i == CorticalTier.EXECUTIVE and tier_j == CorticalTier.ASSOCIATION:
                    dist = abs(i - j)
                    prob = 0.08 * np.exp(-dist * 0.1)
                    if self._rng_feedback.random() < prob:
                        # i sends to j, and the receiver is the row.
                        weights[j, i] = self._rng_feedback.standard_normal() * 0.04
                # Association → Sensory (feedback)
                elif tier_i == CorticalTier.ASSOCIATION and tier_j == CorticalTier.SENSORY:
                    dist = abs(i - j)
                    prob = 0.06 * np.exp(-dist * 0.1)
                    if self._rng_feedback.random() < prob:
                        # i sends to j, and the receiver is the row.
                        weights[j, i] = self._rng_feedback.standard_normal() * 0.03
                # Direct executive → Sensory (long-range feedback, sparser)
                elif tier_i == CorticalTier.EXECUTIVE and tier_j == CorticalTier.SENSORY:
                    dist = abs(i - j)
                    prob = 0.03 * np.exp(-dist * 0.05)
                    if self._rng_feedback.random() < prob:
                        # i sends to j, and the receiver is the row.
                        weights[j, i] = self._rng_feedback.standard_normal() * 0.02

        self._feedback_W = weights.astype(np.float32)


def _from_human_connectome(name: str, fallback: float) -> float:
    """One number from the human connectome reference, or the fallback."""
    try:
        from core.connectome.rich_club import HUMAN_RICH_CLUB

        return float(HUMAN_RICH_CLUB[name])
    except (ImportError, KeyError, TypeError, ValueError):
        return float(fallback)


def _is_human_island(index: int, cfg: Any) -> bool:
    """Whether this column takes its local wiring from H01. -1 means all of them."""
    declared = int(getattr(cfg, "human_island_columns", 0) or 0)
    if declared < 0:
        return True
    return index < declared
