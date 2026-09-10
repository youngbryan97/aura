"""core/connectome/reading.py — the reading hierarchy, run on her own machinery.

Turker, Fumagalli, Kuhnke and Hartwigsen pooled 163 fMRI experiments, 3,031
subjects and 5,444 activation peaks into one question: what does a brain recruit
when it reads, and how does that change with what is being read and what is
being done with it. Four of their findings are shaped like something this
package can ask of Aura, because a coordinate-based meta-analysis over
conditions and a recording over conditions are the same instrument pointed at
different tissue.

    1. A common core across every level of reading, plus machinery specific to
       each level — letters, words, sentences, text — and the specific part is
       exclusively left-hemispheric, which is to say: lateralised, not spread.
    2. The level-specific parts differ in SIZE, not only in place. Letters give
       one focal cluster; words give a distributed network; text gives one small
       frontal cluster again. The middle of the hierarchy recruits the most.
    3. A dual-route dissociation. Reading a familiar word engages different
       frontal tissue from decoding an unfamiliar letter string, which is the
       lexical and non-lexical routes given anatomy.
    4. The TASK moves the network more than the stimulus does. Identical words,
       read silently versus judged for lexicality, produce clearly distinct
       profiles — decision machinery for the judgement, a reading profile for
       the reading.

None of that is imported as a claim about Aura. It is imported as four
predictions, each of which her own recording can refuse. What this module does
is drive her over the same contrasts and measure which of her cells fire, so the
answer is about her rather than about the analogy.

The honest limits, stated once. A parcel of cortex is not a Python module, an
ALE score is not a call count, and nothing here recovers the coordinates. What
transfers is the SHAPE of the comparison — a common core, level-specific
machinery, two routes, and a task effect larger than the stimulus effect — and
each of those is a number this instrument can produce for her.

Source: Turker, S., Fumagalli, B., Kuhnke, P., Hartwigsen, G. (2025). The
'reading' brain: Meta-analytic insight into functional activation during reading
in adults. Neuroscience and Biobehavioral Reviews 173, 106166.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Connectome.Reading")

__all__ = [
    "HUMAN_READING",
    "LEVELS",
    "ReadingProfile",
    "ReadingReport",
    "compare_levels",
    "profile_condition",
    "core_and_specific",
    "dual_route",
    "task_over_stimulus",
]

#: The levels of the hierarchy, coarsest input first. Named as the paper names
#: them so a reader can put the two side by side.
LEVELS: tuple[str, ...] = ("letter", "word", "sentence", "text")


@dataclass(frozen=True, slots=True)
class HumanFinding:
    """One thing the meta-analysis measured, and what would refuse it here."""

    name: str
    in_humans: str
    predicted_here: str
    falsifier: str


#: What the paper found, and what each finding predicts about her, written down
#: before the recording is made. A prediction noticed in a result is not the
#: same kind of evidence as one written first, and the only way to turn the
#: first into the second is to write it down and run it again.
HUMAN_READING: tuple[HumanFinding, ...] = (
    HumanFinding(
        name="common_core",
        in_humans=(
            "Every reading task recruits one shared peri-Sylvian network — left "
            "inferior frontal, ventral occipito-temporal and temporo-parietal "
            "cortex — whatever is being read."
        ),
        predicted_here=(
            "A set of cells fires under every level of input, and it is a large "
            "fraction of what fires at any single level."
        ),
        falsifier=(
            "The levels share almost no cells, so there is no core and each level "
            "runs its own machinery."
        ),
    ),
    HumanFinding(
        name="level_specificity",
        in_humans=(
            "Each level also recruits machinery no other level does, and that "
            "specific part is left-lateralised rather than spread over both "
            "hemispheres."
        ),
        predicted_here=(
            "Each level fires cells no other level fires, and those cells are "
            "concentrated in few regions rather than spread evenly."
        ),
        falsifier=(
            "No level has cells of its own, or the level-specific cells are spread "
            "across regions no more concentrated than the core is."
        ),
    ),
    HumanFinding(
        name="the_middle_recruits_most",
        in_humans=(
            "Letters give one focal occipital cluster and text one small frontal "
            "cluster, while words and sentences give distributed networks. The "
            "middle of the hierarchy is where the most machinery is specific."
        ),
        predicted_here=(
            "Specific machinery is largest for the middle levels and smaller at "
            "both ends."
        ),
        falsifier=(
            "Specific machinery grows monotonically with level, or shrinks "
            "monotonically, rather than peaking in the middle."
        ),
    ),
    HumanFinding(
        name="dual_route",
        in_humans=(
            "A familiar word engages pars orbitalis and the pars opercularis/MFG "
            "intersection; an unfamiliar letter string engages ventral pars "
            "opercularis toward premotor cortex. Known and unknown take "
            "different routes."
        ),
        predicted_here=(
            "A name she already holds and a well-formed name she has never seen "
            "fire measurably different cells, not merely more of the same ones."
        ),
        falsifier=(
            "Known and unknown strings fire the same cells in the same "
            "proportions; there is one route."
        ),
    ),
    HumanFinding(
        name="task_beats_stimulus",
        in_humans=(
            "Identical words read silently versus judged for lexicality produce "
            "clearly distinct networks — decision machinery for the judgement, a "
            "reading profile for the reading."
        ),
        predicted_here=(
            "Changing what she is asked to DO with identical text changes which "
            "cells fire more than changing the text at a fixed task does."
        ),
        falsifier=(
            "The task makes less difference than the stimulus, so what she runs is "
            "driven by the input rather than by the question."
        ),
    ),
)


@dataclass(slots=True)
class ReadingProfile:
    """Which cells fired under one condition, and where they live."""

    condition: str
    cells: frozenset[str] = field(default_factory=frozenset)
    by_region: dict[str, int] = field(default_factory=dict)
    frames: int = 0

    @property
    def size(self) -> int:
        return len(self.cells)

    def concentration(self) -> float:
        """How much of this profile sits in its largest region.

        The paper's lateralisation claim is about concentration, not about
        sides: a specific process lives in a few places rather than everywhere.
        A profile spread evenly over its regions scores near its own share; one
        that piles into a single region scores near one.
        """
        total = sum(self.by_region.values())
        if total <= 0:
            return 0.0
        return max(self.by_region.values()) / total

    def as_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "cells": self.size,
            "frames": self.frames,
            "regions": len(self.by_region),
            "concentration": round(self.concentration(), 4),
            "largest_regions": [
                {"region": name, "cells": count}
                for name, count in sorted(
                    self.by_region.items(), key=lambda item: -item[1]
                )[:6]
            ],
        }


def profile_condition(
    trace: Any,
    snapshot: Any,
    condition: str,
    *,
    min_frames: int = 20,
    frames_cap: int = 0,
    seed: int = 0,
) -> ReadingProfile:
    """The cells that fired in one condition, grouped by the region they live in.

    ``frames_cap`` equalises the conditions before they are compared. A
    condition recorded for twice as long fires more cells for that reason
    alone, and "text recruits the most machinery" would then be a statement
    about how many frames text got. Frames are sampled rather than truncated,
    so a condition is not judged by its first half.
    """
    import numpy as np

    matrix = trace.matrix()
    rows = [i for i, name in enumerate(trace.conditions) if name == condition]
    if len(rows) < min_frames:
        return ReadingProfile(condition=condition, frames=len(rows))
    if frames_cap and len(rows) > frames_cap:
        picked = np.random.default_rng(seed).choice(
            len(rows), size=frames_cap, replace=False
        )
        rows = [rows[int(index)] for index in sorted(picked)]
    activity = np.asarray(matrix[rows], dtype=np.float64)
    fired = activity.sum(axis=0) > 0
    cells = {uid for uid, on in zip(trace.uids, fired, strict=False) if on}
    by_region: dict[str, int] = {}
    for uid in cells:
        unit = snapshot.units.get(uid)
        if unit is None:
            continue
        by_region[unit.region] = by_region.get(unit.region, 0) + 1
    return ReadingProfile(
        condition=condition,
        cells=frozenset(cells),
        by_region=by_region,
        frames=len(rows),
    )


@dataclass(slots=True)
class ReadingReport:
    """What her own recording says about each of the five predictions."""

    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    findings: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "profiles": self.profiles,
            "findings": self.findings,
            "skipped": self.skipped,
            "confirmed": sorted(
                name for name, row in self.findings.items() if row.get("holds")
            ),
            "refused": sorted(
                name
                for name, row in self.findings.items()
                if row.get("holds") is False
            ),
        }


def core_and_specific(
    profiles: Mapping[str, ReadingProfile], levels: Sequence[str] = LEVELS
) -> dict[str, Any]:
    """The cells every level fires, and the cells only one level fires."""
    present = [name for name in levels if name in profiles and profiles[name].size]
    if len(present) < 2:
        return {"levels": present, "skipped": "fewer than two levels fired"}
    sets = {name: profiles[name].cells for name in present}
    core = frozenset.intersection(*sets.values())
    specific = {
        name: cells - frozenset.union(*[sets[other] for other in present if other != name])
        for name, cells in sets.items()
    }
    union = frozenset.union(*sets.values())
    return {
        "levels": present,
        "core": len(core),
        "union": len(union),
        "core_share_of_union": round(len(core) / len(union), 4) if union else 0.0,
        "core_share_of_each": {
            name: round(len(core) / len(cells), 4) if cells else 0.0
            for name, cells in sets.items()
        },
        "specific": {name: len(cells) for name, cells in specific.items()},
        "specific_share": {
            name: round(len(specific[name]) / len(cells), 4) if cells else 0.0
            for name, cells in sets.items()
        },
    }


def compare_levels(
    profiles: Mapping[str, ReadingProfile], levels: Sequence[str] = LEVELS
) -> dict[str, Any]:
    """Where in the hierarchy the level-specific machinery is largest."""
    shape = core_and_specific(profiles, levels)
    specific = shape.get("specific") or {}
    present = [name for name in levels if name in specific]
    if len(present) < 3:
        return {"skipped": "fewer than three levels fired; a peak needs a middle"}
    sizes = [specific[name] for name in present]
    peak = present[sizes.index(max(sizes))]
    ends = {present[0], present[-1]}
    return {
        "order": present,
        "specific": {name: specific[name] for name in present},
        "peak": peak,
        "peaks_in_the_middle": peak not in ends,
        "monotonic": sizes == sorted(sizes) or sizes == sorted(sizes, reverse=True),
    }


def _overlap(left: ReadingProfile, right: ReadingProfile) -> float:
    """Jaccard: how much two conditions fired the same cells."""
    if not left.cells and not right.cells:
        return 1.0
    union = left.cells | right.cells
    if not union:
        return 1.0
    return len(left.cells & right.cells) / len(union)


def dual_route(known: ReadingProfile, unknown: ReadingProfile) -> dict[str, Any]:
    """Do a name she holds and a name she has never seen take different routes?

    Overlap alone cannot answer it: two conditions of different sizes overlap
    less whatever else is true. So the number that decides is how much of each
    side is its OWN — cells the other condition did not fire — against how much
    they share.
    """
    if not known.cells or not unknown.cells:
        return {"skipped": "one of the two conditions fired nothing"}
    shared = known.cells & unknown.cells
    only_known = known.cells - unknown.cells
    only_unknown = unknown.cells - known.cells
    return {
        "known_cells": known.size,
        "unknown_cells": unknown.size,
        "shared": len(shared),
        "only_known": len(only_known),
        "only_unknown": len(only_unknown),
        "overlap": round(_overlap(known, unknown), 4),
        "own_share_known": round(len(only_known) / known.size, 4),
        "own_share_unknown": round(len(only_unknown) / unknown.size, 4),
        "two_routes": bool(only_known) and bool(only_unknown),
    }


def task_over_stimulus(
    profiles: Mapping[str, ReadingProfile],
    *,
    same_task_pairs: Sequence[tuple[str, str]],
    same_stimulus_pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Does changing the question move the network more than changing the text?

    Both sides are measured the same way — one minus the overlap between a pair
    of conditions — so the comparison is between two distances of the same kind
    rather than between a distance and a count.
    """

    def spread(pairs: Sequence[tuple[str, str]]) -> list[float]:
        found = []
        for left, right in pairs:
            if left in profiles and right in profiles:
                if profiles[left].size and profiles[right].size:
                    found.append(1.0 - _overlap(profiles[left], profiles[right]))
        return found

    stimulus = spread(same_task_pairs)
    task = spread(same_stimulus_pairs)
    if not stimulus or not task:
        return {"skipped": "one side of the comparison has no measurable pair"}
    stimulus_mean = statistics.fmean(stimulus)
    task_mean = statistics.fmean(task)
    return {
        "stimulus_pairs": len(stimulus),
        "task_pairs": len(task),
        "distance_when_the_text_changes": round(stimulus_mean, 4),
        "distance_when_the_question_changes": round(task_mean, 4),
        "task_beats_stimulus": task_mean > stimulus_mean,
        "ratio": round(task_mean / stimulus_mean, 4) if stimulus_mean else 0.0,
    }
