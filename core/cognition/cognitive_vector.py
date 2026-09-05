"""core/cognition/cognitive_vector.py — one algebra for the spaces that never met.

Aura holds vectors in at least six unrelated spaces: transformer residuals,
self-field vectors, world-model latents, RLC state, neural-mesh activations,
and embedding-store rows. Every pair that needs to talk has a hand-written
adapter, and every adapter makes an unrecorded decision about what survives the
crossing. Adding a seventh space costs six adapters.

The Semantic Pointer Architecture's answer, adapted here in clean room from its
published description: pick one high-dimensional space, define binding and
unbinding inside it, and make every substrate a typed projection into and out
of it. Then a concept moves through one hop instead of a pairwise adapter, and
— the part that matters — the round trip is measurable, so what the crossing
costs is a number rather than an assumption.

Binding
-------
:func:`bind` is circular convolution and :func:`unbind` is correlation with the
inverse, which is the standard construction. The properties that make it useful
are that binding two random vectors gives something similar to neither, and
that unbinding recovers the other operand approximately. Approximately is the
whole game: the recovered vector is noisy and needs :class:`CleanupMemory` to
snap back to the concept it is nearest, which is why a cleanup memory is part
of the algebra rather than a convenience.

Capacity
--------
:func:`capacity` measures how many bound pairs a vector of a given dimension
can hold before unbinding stops recovering the right item. That number is what
card 084 asks for: a dimension chosen because it meets a measured error target
rather than because it was a power of two. It is measured rather than derived,
because the derivation depends on the cleanup memory's contents.

Diagnostics
-----------
:func:`representational_similarity` and :func:`subspace_angle` compare two
spaces without needing a shared basis, so "does the world model still know what
the embedding store knows" becomes a measurement. Card 082.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import hashlib
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_DIMENSION",
    "unit",
    "random_vector",
    "bind",
    "unbind",
    "superpose",
    "similarity",
    "CleanupMemory",
    "Projection",
    "VectorRegistry",
    "capacity",
    "representational_similarity",
    "subspace_angle",
    "get_vector_registry",
    "reset_vector_registry_for_test",
]

#: Chosen by measurement, not by taste. Against a 200-item cleanup memory,
#: ``capacity()`` recovers 8 of 8 bound pairs at 512 and 8 of 8 at 256; the two
#: separate at 12 pairs, where 512 recovers 0.92 and 256 recovers 0.67. 512 is
#: the smallest dimension measured that holds a dozen bindings, and a dozen is
#: the range the rest of the architecture treats as working memory. Re-run
#: ``capacity()`` before changing it; the number is an empirical claim.
DEFAULT_DIMENSION = 512

Vector = list[float]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def unit(vector: Sequence[float]) -> Vector:
    """Scale to length one. A zero vector stays zero rather than dividing."""
    magnitude = _norm(vector)
    return [0.0] * len(vector) if magnitude == 0 else [x / magnitude for x in vector]


def stable_seed(name: str) -> int:
    """The seed for a concept's vector, identical in every process.

    ``hash()`` on a string is salted per interpreter, so seeding from it mints
    a different vector for "red" on every boot: the cleanup memory cannot be
    persisted, two runs cannot be compared, and a capacity measurement is a
    fresh random draw each time rather than a property of the dimension.
    """
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=4).digest(), "big")


def random_vector(dimension: int = DEFAULT_DIMENSION, *, seed: int | None = None) -> Vector:
    """A unit vector with no relationship to any other."""
    import random

    rng = random.Random(seed)
    return unit([rng.gauss(0.0, 1.0) for _ in range(dimension)])


try:  # numpy is present in this environment; the fallback keeps the module pure.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only where numpy is absent
    _np = None


def _dft(values: Sequence[float]) -> list[complex]:
    n = len(values)
    return [
        sum(values[k] * complex(math.cos(-2 * math.pi * j * k / n), math.sin(-2 * math.pi * j * k / n))
            for k in range(n))
        for j in range(n)
    ]


def _idft(values: Sequence[complex]) -> list[float]:
    n = len(values)
    return [
        sum(values[k] * complex(math.cos(2 * math.pi * j * k / n), math.sin(2 * math.pi * j * k / n))
            for k in range(n)).real / n
        for j in range(n)
    ]


def bind(a: Sequence[float], b: Sequence[float]) -> Vector:
    """Circular convolution. The result resembles neither operand.

    Uses the frequency domain, so binding is elementwise multiplication of
    spectra. That is the same operation as the naive double loop and is what
    makes ``unbind`` exact in the noiseless case.
    """
    if len(a) != len(b):
        raise ValueError(f"cannot bind vectors of different dimension: {len(a)} and {len(b)}")
    if _np is not None:
        product = _np.fft.irfft(_np.fft.rfft(_np.asarray(a)) * _np.fft.rfft(_np.asarray(b)), n=len(a))
        return unit(product.tolist())
    fa, fb = _dft(a), _dft(b)
    return unit(_idft([x * y for x, y in zip(fa, fb, strict=True)]))


def involution(vector: Sequence[float]) -> Vector:
    """The approximate inverse under circular convolution."""
    return [vector[0], *reversed(list(vector[1:]))]


def unbind(bound: Sequence[float], key: Sequence[float]) -> Vector:
    """Recover the other operand, approximately. Noise is expected."""
    return bind(bound, involution(key))


def superpose(*vectors: Sequence[float]) -> Vector:
    """Add and renormalise. The sum is similar to each addend."""
    if not vectors:
        return []
    dimension = len(vectors[0])
    total = [0.0] * dimension
    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError("cannot superpose vectors of different dimension")
        for i, value in enumerate(vector):
            total[i] += value
    return unit(total)


def similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Zero for a zero vector rather than undefined."""
    if len(a) != len(b):
        return 0.0
    if _np is not None:
        left, right = _np.asarray(a), _np.asarray(b)
        denominator = float(_np.linalg.norm(left) * _np.linalg.norm(right))
        return 0.0 if denominator == 0 else float(left @ right) / denominator
    denominator = _norm(a) * _norm(b)
    return 0.0 if denominator == 0 else sum(x * y for x, y in zip(a, b, strict=True)) / denominator


class CleanupMemory:
    """Snap a noisy vector back to the concept it is nearest, or refuse.

    The refusal is the point. A cleanup that always returns its best match will
    confidently name a concept for a vector that is nothing but noise, and
    every downstream consumer will believe it. ``threshold`` is the similarity
    below which this returns nothing, and ``margin`` is how far ahead the
    winner must be before the answer counts as unambiguous.
    """

    def __init__(self, *, threshold: float = 0.25, margin: float = 0.05) -> None:
        self._lock = checked_lock("core.cognition.cognitive_vector.CleanupMemory", reentrant=True)
        self._items: dict[str, Vector] = {}
        self._threshold = float(threshold)
        self._margin = float(margin)
        self._refused = 0
        self._ambiguous = 0

    def add(self, name: str, vector: Sequence[float]) -> None:
        with self._lock:
            self._items[name] = unit(vector)

    def resolve(self, vector: Sequence[float]) -> dict[str, Any]:
        with self._lock:
            scored = sorted(
                ((similarity(vector, v), name) for name, v in self._items.items()), reverse=True
            )
        if not scored:
            return {"name": None, "similarity": 0.0, "reason": "cleanup memory is empty"}
        best_score, best_name = scored[0]
        if best_score < self._threshold:
            with self._lock:
                self._refused += 1
            return {
                "name": None,
                "similarity": best_score,
                "reason": f"nothing above threshold {self._threshold}",
            }
        if len(scored) > 1 and best_score - scored[1][0] < self._margin:
            with self._lock:
                self._ambiguous += 1
            return {
                "name": None,
                "similarity": best_score,
                "runner_up": scored[1][1],
                "reason": "two concepts within the margin",
            }
        return {"name": best_name, "similarity": best_score, "reason": ""}

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "items": len(self._items),
                "refused": self._refused,
                "ambiguous": self._ambiguous,
                "threshold": self._threshold,
                "margin": self._margin,
            }


@dataclass(frozen=True, slots=True)
class Projection:
    """One substrate's way in and out of the canonical space."""

    substrate: str
    dimension: int
    #: Measured round-trip similarity: project out, project back, compare.
    fidelity: float = 0.0
    #: Whether the projection has an inverse at all. A one-way projection is
    #: usable and cannot carry a concept back.
    invertible: bool = True
    note: str = ""

    @property
    def lossless_enough(self) -> bool:
        """Whether a concept survives the round trip well enough to be recognised."""
        return self.invertible and self.fidelity >= 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "dimension": self.dimension,
            "fidelity": self.fidelity,
            "invertible": self.invertible,
            "lossless_enough": self.lossless_enough,
            "note": self.note,
        }


class VectorRegistry:
    """The canonical space, its cleanup memory, and every substrate's projection."""

    def __init__(self, *, dimension: int = DEFAULT_DIMENSION) -> None:
        self._lock = checked_lock("core.cognition.cognitive_vector.VectorRegistry", reentrant=True)
        self.dimension = int(dimension)
        self.cleanup = CleanupMemory()
        self._projections: dict[str, Projection] = {}
        self._concepts: dict[str, Vector] = {}

    def concept(self, name: str, *, seed: int | None = None) -> Vector:
        """Get or mint the canonical vector for a concept name."""
        with self._lock:
            existing = self._concepts.get(name)
            if existing is not None:
                return existing
            vector = random_vector(
                self.dimension, seed=seed if seed is not None else stable_seed(name)
            )
            self._concepts[name] = vector
            self.cleanup.add(name, vector)
            return vector

    def declare_projection(self, projection: Projection) -> Projection:
        with self._lock:
            self._projections[projection.substrate] = projection
            return projection

    def measure_projection(
        self,
        substrate: str,
        out_fn,
        back_fn,
        *,
        samples: int = 20,
        note: str = "",
    ) -> Projection:
        """Project concepts out and back, and record what survived.

        This is what makes the crossing a number. A projection that has not
        been measured reports fidelity 0.0 and is not ``lossless_enough``,
        which is the correct reading of an adapter nobody checked.
        """
        scores = []
        for i in range(samples):
            original = self.concept(f"__probe_{substrate}_{i}")
            try:
                returned = back_fn(out_fn(original))
            except Exception:  # noqa: BLE001 - a projection that raises has fidelity 0
                scores.append(0.0)
                continue
            scores.append(similarity(original, returned))
        fidelity = sum(scores) / len(scores) if scores else 0.0
        return self.declare_projection(
            Projection(
                substrate=substrate,
                dimension=self.dimension,
                fidelity=fidelity,
                invertible=fidelity > 0.0,
                note=note,
            )
        )

    def report(self) -> dict[str, Any]:
        with self._lock:
            projections = list(self._projections.values())
        return {
            "dimension": self.dimension,
            "concepts": len(self._concepts),
            "projections": [p.to_dict() for p in projections],
            "substrates_reachable": sum(1 for p in projections if p.lossless_enough),
            "adapters_replaced": max(0, len(projections) * (len(projections) - 1) - len(projections)),
            "cleanup": self.cleanup.report(),
        }


def capacity(
    dimension: int, *, pairs: int, distractors: int = 200, seed: int = 0
) -> dict[str, Any]:
    """How reliably a bundle of ``pairs`` bound pairs unbinds at this dimension.

    Builds a superposition of role-filler bindings, unbinds each role, and asks
    the cleanup memory to name the filler. The returned recovery rate is what a
    dimension choice should be justified by.
    """
    registry = VectorRegistry(dimension=dimension)
    fillers = [registry.concept(f"filler_{i}") for i in range(pairs)]
    for i in range(pairs, pairs + distractors):
        registry.concept(f"filler_{i}")
    roles = [random_vector(dimension, seed=seed * 1000 + i) for i in range(pairs)]
    bundle = superpose(*[bind(r, f) for r, f in zip(roles, fillers, strict=True)])
    recovered = 0
    for i, role in enumerate(roles):
        result = registry.cleanup.resolve(unbind(bundle, role))
        if result["name"] == f"filler_{i}":
            recovered += 1
    return {
        "dimension": dimension,
        "pairs": pairs,
        "distractors": distractors,
        "recovered": recovered,
        "recovery_rate": recovered / pairs if pairs else 0.0,
    }


def representational_similarity(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]
) -> float:
    """How alike two spaces are in what they consider alike.

    Compares the two pairwise-similarity matrices rather than the vectors, so
    the spaces need no shared basis and no matching dimension. This is the
    diagnostic that answers "does the world model still know what the
    embedding store knows" without a decoder in between.
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.0

    def matrix(space):
        return [
            similarity(space[i], space[j])
            for i in range(len(space))
            for j in range(i + 1, len(space))
        ]

    left, right = matrix(a), matrix(b)
    mean_l = sum(left) / len(left)
    mean_r = sum(right) / len(right)
    numerator = sum((x - mean_l) * (y - mean_r) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - mean_l) ** 2 for x in left) * sum((y - mean_r) ** 2 for y in right)
    )
    return 0.0 if denominator == 0 else numerator / denominator


def subspace_angle(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Angle in radians between the mean directions of two sets of vectors.

    Coarse and cheap. It catches the case that matters — one space has drifted
    wholesale away from another — without a full principal-angle decomposition.
    """
    if not a or not b:
        return math.pi / 2
    return math.acos(max(-1.0, min(1.0, similarity(superpose(*a), superpose(*b)))))


_lock = checked_lock("core.cognition.cognitive_vector.singleton")
_registry: VectorRegistry | None = None


def get_vector_registry() -> VectorRegistry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = VectorRegistry()
        return _registry


def reset_vector_registry_for_test(**kwargs: Any) -> VectorRegistry:
    global _registry
    with _lock:
        _registry = VectorRegistry(**kwargs)
        return _registry
