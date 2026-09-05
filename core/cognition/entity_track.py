"""core/cognition/entity_track.py — the same thing, still there when it is hidden.

A world model that cannot say "that is the same window I saw before it went
behind the other one" is not modelling a world; it is modelling a sequence of
unrelated pictures. Aura has real protections against this inside screen
pursuit — page identity, window lattice, task-region checks — and each is a
local guard against a specific way she lost the thread. None of them produces
a durable identity that memory and the world model can both refer to.

An :class:`EntityTrack` is that identity. It survives three things a frame-by-
frame matcher does not:

* **Occlusion.** A track that stops being observed is ``OCCLUDED``, not gone.
  It stays alive on a budget of missed observations proportional to how well
  established it was, so a thing seen sixty times survives a longer absence
  than a thing seen twice. Only after that budget does it become ``LOST``.
* **Ambiguity.** Two candidate matches do not silently pick the nearer one.
  When the margin between them is thin the update is refused and the track is
  marked ``AMBIGUOUS``, because a confident wrong association is worse than a
  gap — it writes a false history that later inference treats as observed.
* **Splits and merges.** Two things that turn out to be one, or one that turns
  out to be two, keep their lineage, so an episodic memory pointing at the old
  track can still be resolved.

The semantic layer stays separate on purpose. ``hypotheses`` holds what the
track might BE, with weights; ``support`` holds how much was actually seen.
Collapsing them is how "the thing at (400, 300)" becomes "the Save button"
with the confidence of a geometric match.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

__all__ = [
    "TrackState",
    "Observation",
    "EntityTrack",
    "TrackStore",
    "get_track_store",
    "reset_track_store_for_test",
]


class TrackState(StrEnum):
    VISIBLE = "visible"
    #: Not observed this frame, still within its persistence budget.
    OCCLUDED = "occluded"
    #: Two candidates matched too closely to choose between.
    AMBIGUOUS = "ambiguous"
    #: Out of budget. Kept for lineage; never matched again.
    LOST = "lost"
    #: Folded into another track.
    MERGED = "merged"


@dataclass(frozen=True, slots=True)
class Observation:
    """One sighting: where it was, what it looked like, when."""

    at: float
    geometry: tuple[float, ...] = ()
    features: tuple[float, ...] = ()
    label: str = ""
    frame: str = ""

    def distance(self, other: "Observation") -> float:
        """Geometry distance in whatever units the caller uses, plus feature distance.

        Returns ``inf`` when the two cannot be compared at all, so an
        incomparable pair can never look like a close match.
        """
        parts: list[float] = []
        if self.geometry and other.geometry and len(self.geometry) == len(other.geometry):
            parts.append(math.dist(self.geometry, other.geometry))
        if self.features and other.features and len(self.features) == len(other.features):
            norm = math.sqrt(sum(f * f for f in self.features)) * math.sqrt(
                sum(f * f for f in other.features)
            )
            cosine = (
                sum(a * b for a, b in zip(self.features, other.features, strict=True)) / norm
                if norm > 0
                else 0.0
            )
            parts.append(1.0 - cosine)
        if not parts:
            return math.inf
        return sum(parts) / len(parts)


#: Missed observations a track survives per sighting, and the ceiling on that
#: budget. A thing seen once should not be assumed present for a minute.
_PERSISTENCE_PER_SIGHTING = 2
_MAX_PERSISTENCE = 60

#: A second candidate closer than this multiple of the best makes the match
#: ambiguous. 1.25 means "the runner-up was within 25 percent" — close enough
#: that picking the winner is a coin flip dressed as a measurement.
_AMBIGUITY_RATIO = 1.25

#: Two candidates both within this distance are in the same place, whatever
#: the ratio between them says. Without this floor a best distance of 1e-9 and
#: a runner-up of 1e-8 read as a clean win at a ratio of ten, and a best of
#: exactly zero divided the decision by nothing at all.
_SAME_PLACE = 1e-6


@dataclass
class EntityTrack:
    """One thing, tracked across time, with its identity's history."""

    track_id: str
    state: TrackState = TrackState.VISIBLE
    support: int = 0
    misses: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_observation: Observation | None = None
    #: What this might be, weighted. Never collapsed into an identity.
    hypotheses: dict[str, float] = field(default_factory=dict)
    #: Handles from core.cognition.concept_handle, when the track has been
    #: recognised as a known concept.
    concepts: tuple[str, ...] = ()
    lineage: tuple[str, ...] = ()
    #: Geometry change per unit time, from the last two sightings. A thing
    #: that kept moving while it was out of sight comes back where it was
    #: GOING, and matching against where it was last seen breaks the track of
    #: anything that moves — which is every case this module is for.
    velocity: tuple[float, ...] = ()

    @property
    def persistence_budget(self) -> int:
        return min(_MAX_PERSISTENCE, _PERSISTENCE_PER_SIGHTING * max(1, self.support))

    @property
    def alive(self) -> bool:
        return self.state in (TrackState.VISIBLE, TrackState.OCCLUDED, TrackState.AMBIGUOUS)

    def observe(self, observation: Observation) -> None:
        previous = self.last_observation
        elapsed = observation.at - self.last_seen
        if (
            previous is not None
            and previous.geometry
            and len(previous.geometry) == len(observation.geometry)
            and elapsed > 0
        ):
            self.velocity = tuple(
                (new - old) / elapsed
                for new, old in zip(observation.geometry, previous.geometry, strict=True)
            )
        self.support += 1
        self.misses = 0
        self.last_seen = observation.at
        self.last_observation = observation
        self.state = TrackState.VISIBLE

    def predicted_at(self, at: float) -> Observation | None:
        """Where this would be at ``at``, carrying its last measured motion.

        None before there is anything to carry — one sighting is a position
        and not a trajectory, and extrapolating from it invents the motion.
        """
        if self.last_observation is None:
            return None
        if not self.velocity or not self.last_observation.geometry:
            return self.last_observation
        elapsed = at - self.last_seen
        if elapsed <= 0:
            return self.last_observation
        return replace(
            self.last_observation,
            at=at,
            geometry=tuple(
                position + speed * elapsed
                for position, speed in zip(
                    self.last_observation.geometry, self.velocity, strict=True
                )
            ),
        )

    def miss(self) -> None:
        """A frame in which this track was not seen."""
        if not self.alive:
            return
        self.misses += 1
        self.state = (
            TrackState.LOST if self.misses > self.persistence_budget else TrackState.OCCLUDED
        )

    def suggest(self, hypothesis: str, weight: float) -> None:
        self.hypotheses[hypothesis] = max(0.0, min(1.0, float(weight)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "velocity": list(self.velocity),
            "state": self.state.value,
            "support": self.support,
            "misses": self.misses,
            "persistence_budget": self.persistence_budget,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "hypotheses": dict(self.hypotheses),
            "concepts": list(self.concepts),
            "lineage": list(self.lineage),
        }


class TrackStore:
    """Association across frames, refusing the associations it cannot justify."""

    def __init__(self, *, match_threshold: float = 0.35, max_tracks: int = 4096) -> None:
        self._lock = checked_lock("core.cognition.entity_track.TrackStore", reentrant=True)
        self._tracks: dict[str, EntityTrack] = {}
        self._counter = 0
        self._match_threshold = float(match_threshold)
        self._max_tracks = int(max_tracks)
        self._refused_ambiguous = 0

    def update(self, observations: Sequence[Observation]) -> list[EntityTrack]:
        """Fold a frame of observations into the tracks, one association each.

        Returns the tracks touched. Every live track not matched takes a miss,
        which is what turns "not seen" into occlusion rather than absence.
        """
        with self._lock:
            live = [t for t in self._tracks.values() if t.alive]
            matched: dict[str, EntityTrack] = {}
            touched: list[EntityTrack] = []
            for observation in observations:
                track = self._associate_locked(observation, live, matched)
                matched[track.track_id] = track
                touched.append(track)
            for track in live:
                if track.track_id not in matched:
                    track.miss()
            self._evict_locked()
            return touched

    def _associate_locked(
        self,
        observation: Observation,
        live: list[EntityTrack],
        already: dict[str, EntityTrack],
    ) -> EntityTrack:
        # Against where each track is EXPECTED to be, not where it was last
        # seen. The two are the same for a thing that has not moved and for a
        # track with one sighting, and they diverge exactly when it matters:
        # a thing that crossed an occluder reappears a whole occlusion's worth
        # of travel from its last sighting.
        candidates = []
        for track in live:
            if track.track_id in already or track.last_observation is None:
                continue
            expected = track.predicted_at(observation.at) or track.last_observation
            candidates.append((expected.distance(observation), track))
        scored = sorted(candidates, key=lambda pair: pair[0])
        if not scored or scored[0][0] > self._match_threshold:
            return self._new_track_locked(observation)
        best_distance, best = scored[0]
        if len(scored) > 1:
            runner_up = scored[1][0]
            # A near-zero best distance is the LEAST ambiguous case there is,
            # not the most. Reading it as ambiguous refused every association
            # the moment prediction made matches near-exact, and every track
            # was rebuilt from scratch each frame — the tracker got worse the
            # better its evidence became.
            close = runner_up <= max(best_distance * _AMBIGUITY_RATIO, _SAME_PLACE)
            if close:
                # Two things this could be. Starting a new track is honest;
                # picking one writes a history that was never observed.
                self._refused_ambiguous += 1
                best.state = TrackState.AMBIGUOUS
                return self._new_track_locked(observation)
        best.observe(observation)
        return best

    def _new_track_locked(self, observation: Observation) -> EntityTrack:
        self._counter += 1
        track = EntityTrack(track_id=f"t{self._counter}", first_seen=observation.at)
        track.observe(observation)
        self._tracks[track.track_id] = track
        return track

    def merge(self, keep_id: str, absorb_id: str) -> EntityTrack:
        """Two tracks turned out to be one thing."""
        with self._lock:
            keep, absorb = self._tracks[keep_id], self._tracks[absorb_id]
            keep.support += absorb.support
            keep.first_seen = min(keep.first_seen, absorb.first_seen)
            keep.lineage = (*keep.lineage, absorb_id)
            for name, weight in absorb.hypotheses.items():
                keep.hypotheses[name] = max(keep.hypotheses.get(name, 0.0), weight)
            absorb.state = TrackState.MERGED
            absorb.lineage = (*absorb.lineage, keep_id)
            return keep

    def split(self, track_id: str, observation: Observation) -> EntityTrack:
        """One track turned out to be two things."""
        with self._lock:
            parent = self._tracks[track_id]
            child = self._new_track_locked(observation)
            child.lineage = (*child.lineage, track_id)
            parent.lineage = (*parent.lineage, child.track_id)
            return child

    def resolve(self, track_id: str) -> EntityTrack | None:
        """Follow lineage so a reference to a merged track still lands."""
        with self._lock:
            seen: set[str] = set()
            current = self._tracks.get(track_id)
            while current is not None and current.state is TrackState.MERGED:
                if current.track_id in seen:
                    return current
                seen.add(current.track_id)
                nxt = next((self._tracks.get(x) for x in reversed(current.lineage)), None)
                if nxt is None:
                    return current
                current = nxt
            return current

    def _evict_locked(self) -> None:
        if len(self._tracks) <= self._max_tracks:
            return
        dead = sorted(
            (t for t in self._tracks.values() if not t.alive), key=lambda t: t.last_seen
        )
        for track in dead[: len(self._tracks) - self._max_tracks]:
            del self._tracks[track.track_id]

    def report(self) -> dict[str, Any]:
        with self._lock:
            by_state: dict[str, int] = {}
            for track in self._tracks.values():
                by_state[track.state.value] = by_state.get(track.state.value, 0) + 1
            survived = [t for t in self._tracks.values() if t.support > 1 and t.misses > 0]
            return {
                "tracks": len(self._tracks),
                "by_state": by_state,
                "ambiguous_associations_refused": self._refused_ambiguous,
                "survived_an_occlusion": len(survived),
            }

    def tracks(self) -> list[EntityTrack]:
        with self._lock:
            return list(self._tracks.values())


_store_lock = checked_lock("core.cognition.entity_track.singleton")
_store: TrackStore | None = None


def get_track_store() -> TrackStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = TrackStore()
        return _store


def reset_track_store_for_test() -> TrackStore:
    global _store
    with _store_lock:
        _store = TrackStore()
        return _store
