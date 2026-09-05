"""core/interiority/faculty.py — the contract each of the forty-three keeps.

A faculty is not a function that returns a mood. It is an object that
declares, in data, what would show it is wrong. Four declarations are
required and the registry refuses a faculty missing any of them:

``requires``
    Appraisal checks that must be present. A faculty whose inputs are
    absent :meth:`declines` — it returns an activation of zero with a
    stated reason — rather than substituting a default. Substituting a
    default is how six of six reviewed prototypes turn "nothing is
    known" into a confident number.

``counterfactuals``
    Interventions and their expected direction, as data. Set
    ``self_agency`` to zero and guilt must collapse. Make the loss
    reversible and grief must fall. These are Pearl's do() on the
    appraisal frame, and ``tests/interiority/test_counterfactuals.py``
    runs every one that every faculty declares — so the causal claim is
    checked by construction rather than by a test somebody remembered
    to write. A faculty declaring none does not load.

``null``
    The neutral frame under which the activation must be zero. This is
    the check the reviewed work most often fails: Grok's hatred ledger
    charges a tax of 0.04 to an agent with no hatred and no investment,
    because a default policy term leaks into the sum. A mechanism that
    fires on nothing cannot be evidence of anything.

``falsifier``
    One sentence naming an observation that would show the mechanism is
    not doing what it claims. Not a limitation; a refutation.

The base class enforces two things the subclass cannot route around.
Intensity is clamped into [0, 1] and then capped by the provenance
ceiling of the checks the faculty actually read, so a faculty running on
assumed inputs cannot report the confidence of a measurement. And the
receipt records which checks were read, at what provenance, with what
ceiling — so a surprising activation can be traced to the evidence that
produced it rather than to the author's intent.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from enum import StrEnum
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.interiority.appraisal import ALL_CHECKS, AppraisalFrame
from core.interiority.effects import Effects
from core.interiority.evidence import Reading, absent, ceiling_for, weakest
from core.interiority.ledger import RelationalLedger
from core.interiority.other_minds import OtherEstimate
from core.interiority.params import Param, registry as param_registry

logger = logging.getLogger("Aura.Interiority.Faculty")


class Direction(StrEnum):
    """What an intervention is expected to do to the activation."""

    #: Falls to zero. The strongest claim: this cause is necessary.
    COLLAPSES = "collapses"
    #: Falls, without necessarily reaching zero.
    DECREASES = "decreases"
    INCREASES = "increases"
    #: Does not move. Used to pin what a faculty must *not* depend on —
    #: the audience-invariance checks are all of this kind.
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class Counterfactual:
    """One do() on the appraisal frame, and the direction it must produce."""

    name: str
    #: check name -> forced value, or None to make the check absent.
    do: Mapping[str, float | None]
    expect: Direction
    because: str
    #: Ledger facts to WITHHOLD when this intervention runs. A faculty that
    #: reads the world rather than the frame — mourning reads a registered
    #: loss, sympathy reads a custody's vulnerability — is untouched by a
    #: do() on the frame, and the run returns the baseline to the last
    #: decimal while reporting a wrong direction. Naming the fact here is how
    #: the intervention reaches what the faculty actually reads.
    withhold: tuple[str, ...] = ()
    #: Readings to force on the other-agent estimate, for the faculties whose
    #: variable lives there rather than in the frame or the world.
    do_other: Mapping[str, float | None] = MappingProxyType({})
    #: Interior readings to force, for the faculties that read their own
    #: body rather than the frame, the world, or another agent.
    do_interior: Mapping[str, Any] = MappingProxyType({})
    #: Build one world fact DIFFERENTLY rather than withholding it. Mourning
    #: reads a loss's irreversibility off the ledger, so "recoverable" is not
    #: a frame value and not an absent fact — it is the same fact with a
    #: different number in it.
    do_world: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        unknown = set(self.do) - set(ALL_CHECKS)
        if unknown:
            raise ValueError(
                f"counterfactual {self.name!r} intervenes on unknown checks "
                f"{sorted(unknown)}"
            )
        if not self.because.strip():
            raise ValueError(
                f"counterfactual {self.name!r} has no stated reason. A test that "
                "does not say why it must hold cannot tell a fix from a defeat"
            )


@dataclass(frozen=True)
class NullSpec:
    """The neutral frame under which the faculty must not fire."""

    #: check -> the value that means "nothing here". Checks not named are
    #: made absent, which is the stronger neutral.
    values: Mapping[str, float] = field(default_factory=dict)
    #: Activation allowed under the null. Zero unless the faculty has a
    #: stated reason for a floor, which must be in the string.
    tolerance: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.tolerance > 0.0 and not self.reason.strip():
            raise ValueError(
                "a non-zero null tolerance needs a stated reason, or it is a "
                "leak with permission"
            )


@dataclass(frozen=True)
class Activation:
    """What a faculty produced, and what it rests on."""

    faculty: str
    intensity: float
    #: The action readiness this state is. Empty when the faculty is not
    #: about readiness (the two cellular ones, and the ledger auditors).
    tendency: str = ""
    effects: Effects = field(default_factory=Effects)
    #: Non-empty when the faculty refused, with the reason.
    declined: str = ""
    receipt: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        return self.intensity > 0.0 and not self.declined

    def to_dict(self) -> dict[str, Any]:
        return {
            "faculty": self.faculty,
            "intensity": self.intensity,
            "tendency": self.tendency,
            "declined": self.declined,
            "effects": self.effects.to_dict(),
            "receipt": dict(self.receipt),
        }


@dataclass
class FacultyContext:
    """Everything a faculty may read. There is nothing else."""

    frame: AppraisalFrame
    ledger: RelationalLedger
    other: OtherEstimate | None = None
    #: Aura's own interior readings — load, latency, error rate, free
    #: energy. Named channels only; see core/interiority/interoception.py.
    interior: Mapping[str, float] = field(default_factory=dict)
    now: float = 0.0
    #: The gain bank and the transmission medium. The two substrate
    #: reporters have these as their subject, so they must arrive through
    #: the context rather than through a module global: a faculty that
    #: reaches for a singleton cannot be measured against a world, and the
    #: ablation harness measured both of them as changing nothing because
    #: it was priming a bank they never read.
    bank: Any | None = None
    cleft: Any | None = None

    def receptors(self) -> Any:
        if self.bank is not None:
            return self.bank
        from core.interiority.receptors import get_receptor_bank

        return get_receptor_bank()

    def medium(self) -> Any:
        if self.cleft is not None:
            return self.cleft
        from core.interiority.cleft import get_cleft

        return get_cleft()

    def check(self, name: str) -> Reading:
        return self.frame[name]

    def v(self, name: str) -> float:
        return self.frame[name].value

    def interior_value(self, name: str, default: float = 0.0) -> float:
        value = self.interior.get(name, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default


class Faculty(ABC):
    """One capacity, with the evidence it needs and the tests that refute it."""

    #: Stable id, ``fNN_snake_name``. Never reused, never renamed.
    id: str = ""
    #: Position in the list as it was given. Used for ordering only.
    number: int = 0
    #: The capacity, in the words it was asked for. Verbatim, so a reader
    #: can check that the mechanism answers the question that was asked
    #: rather than a nearby one that was easier.
    question: str = ""
    #: One line: what this mechanism claims to be.
    mechanism: str = ""
    #: Appraisal checks that must be present or the faculty declines.
    requires: tuple[str, ...] = ()
    #: Checks the faculty reads when they are there and does without.
    optional: tuple[str, ...] = ()
    counterfactuals: tuple[Counterfactual, ...] = ()
    null: NullSpec = NullSpec()
    #: Appraisal values under which this faculty should fire, overriding the
    #: proving harness's defaults. A faculty that cannot produce an
    #: activation in the world it declares is a mechanism that cannot fire,
    #: and tests/interiority/test_faculties_can_fire.py refuses it.
    activation: Mapping[str, float] = MappingProxyType({})
    #: Interior readings the harness must supply for it to reach its own
    #: mechanism. Merged over the defaults.
    activation_interior: Mapping[str, Any] = MappingProxyType({})
    #: What the ledger must contain. Empty means everything the harness
    #: builds; naming subsets keeps worlds that contradict each other apart,
    #: such as a live bond and a registered loss for the same person.
    activation_world: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._validate_declaration()

    # ── declaration checks ────────────────────────────────────────────
    def _validate_declaration(self) -> None:
        cls = type(self).__name__
        if not self.id:
            raise ValueError(f"{cls}: a faculty needs a stable id")
        if not self.question.strip():
            raise ValueError(
                f"{self.id}: the question is empty. The capacity has to be "
                "written down in the words it was asked for, or nobody can "
                "check that this answers it"
            )
        if not self.mechanism.strip():
            raise ValueError(f"{self.id}: mechanism is empty")
        unknown = set(self.requires) | set(self.optional) | set(self.activation)
        unknown -= set(ALL_CHECKS)
        if unknown:
            raise ValueError(f"{self.id}: unknown appraisal checks {sorted(unknown)}")
        if not self.counterfactuals:
            raise ValueError(
                f"{self.id}: no counterfactuals declared. A mechanism with no "
                "stated intervention is not a causal claim, and a test suite "
                "that only checks it runs will pass on a constant"
            )
        if not self.falsifier().strip():
            raise ValueError(f"{self.id}: no falsifier declared")

    # ── what the subclass writes ──────────────────────────────────────
    @abstractmethod
    def compute(self, ctx: FacultyContext) -> Activation:
        """Produce the activation. Called only when ``requires`` are present."""

    @abstractmethod
    def falsifier(self) -> str:
        """One observation that would show this mechanism is not what it claims."""

    def params(self) -> tuple[Param, ...]:
        """Parameters this faculty declared, from the global registry."""
        return param_registry().for_owner(self.owner())

    def owner(self) -> str:
        module = type(self).__module__.replace(".", "/")
        return f"{module}.py"

    # ── the call the runtime makes ────────────────────────────────────
    def evaluate(self, ctx: FacultyContext) -> Activation:
        """Run the faculty with the guards the subclass cannot route around."""
        missing = [name for name in self.requires if not ctx.frame[name].present]
        if missing:
            return Activation(
                faculty=self.id,
                declined=(
                    f"required appraisal checks are absent: {', '.join(missing)}. "
                    "Declining rather than assuming: a default here would be "
                    "reported as a reading"
                ),
                intensity=0.0,
                receipt={"missing": missing},
            )

        try:
            raw = self.compute(ctx)
        except Exception as exc:  # noqa: BLE001 — a faculty that raises is a defect
            from core.runtime.errors import record_degradation

            record_degradation(
                f"interiority.{self.id}", exc, action="faculty returned no activation"
            )
            return Activation(
                faculty=self.id,
                declined=f"{type(exc).__name__}: {exc}",
                intensity=0.0,
            )

        read = tuple(self.requires) + tuple(
            n for n in self.optional if ctx.frame[n].present
        )
        ceiling = ceiling_for(ctx.frame[n] for n in read) if read else 0.0
        clamped = 0.0 if not math.isfinite(raw.intensity) else raw.intensity
        clamped = 0.0 if clamped < 0.0 else 1.0 if clamped > 1.0 else clamped
        capped = min(clamped, ceiling)
        scale = 0.0 if clamped <= 0.0 else capped / clamped

        receipt = {
            "checks_read": list(read),
            "provenance": weakest(ctx.frame[n] for n in read).label if read else "absent",
            "ceiling": ceiling,
            "raw_intensity": clamped,
            "confidence": ctx.frame.confidence(*read) if read else 0.0,
            **dict(raw.receipt),
        }
        return replace(
            raw,
            faculty=self.id,
            intensity=capped,
            effects=raw.effects.scaled(scale) if scale != 1.0 else raw.effects,
            receipt=receipt,
        )

    # ── description ───────────────────────────────────────────────────
    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "number": self.number,
            "question": self.question,
            "mechanism": self.mechanism,
            "requires": list(self.requires),
            "optional": list(self.optional),
            "activation": dict(self.activation),
            "activation_world": list(self.activation_world),
            "falsifier": self.falsifier(),
            "counterfactuals": [
                {
                    "name": c.name,
                    "do": dict(c.do),
                    "expect": str(c.expect),
                    "because": c.because,
                }
                for c in self.counterfactuals
            ],
            "params": [p.to_dict() for p in self.params()],
        }


class FacultyRegistry:
    """The forty-three, in the order they were asked for."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.interiority.faculty.FacultyRegistry", reentrant=True)
        self._faculties: dict[str, Faculty] = {}

    def register(self, faculty: Faculty) -> Faculty:
        with self._lock:
            existing = self._faculties.get(faculty.id)
            if existing is not None and type(existing) is not type(faculty):
                raise ValueError(
                    f"{faculty.id} is already registered by {type(existing).__name__}. "
                    "Faculty ids are a contract"
                )
            self._faculties[faculty.id] = faculty
            return faculty

    def get(self, faculty_id: str) -> Faculty | None:
        with self._lock:
            return self._faculties.get(faculty_id)

    def all(self) -> tuple[Faculty, ...]:
        with self._lock:
            return tuple(
                sorted(self._faculties.values(), key=lambda f: (f.number, f.id))
            )

    def ids(self) -> tuple[str, ...]:
        return tuple(f.id for f in self.all())

    def __len__(self) -> int:
        with self._lock:
            return len(self._faculties)

    def clear_for_test(self) -> None:
        with self._lock:
            self._faculties.clear()


_REGISTRY = FacultyRegistry()


def registry() -> FacultyRegistry:
    return _REGISTRY


def register(faculty_cls: type[Faculty]) -> type[Faculty]:
    """Class decorator: instantiate and register at import."""
    _REGISTRY.register(faculty_cls())
    return faculty_cls


def intervene(frame: AppraisalFrame, do: Mapping[str, float | None]) -> AppraisalFrame:
    """Pearl's do() on an appraisal frame.

    Forcing a check breaks whatever produced it, which is the point: the
    question is not "what does the ledger say" but "what would this
    faculty do if this one variable were otherwise". The forced reading
    keeps the provenance of the one it replaced, so an intervention does
    not accidentally lift a faculty's ceiling and make the counterfactual
    pass for the wrong reason.
    """
    checks = dict(frame.checks)
    for name, value in do.items():
        original = checks[name]
        if value is None:
            checks[name] = absent(source=f"do({name}=absent)")
        else:
            checks[name] = Reading(
                float(value),
                original.provenance if original.present else Reading(
                    0.0, original.provenance
                ).provenance,
                original.confidence if original.present else 1.0,
                source=f"do({name}={value})",
            )
            if not original.present:
                # An intervention that supplies a value where there was
                # none must not pretend it was measured.
                from core.interiority.evidence import Provenance

                checks[name] = Reading(
                    float(value), Provenance.INFERRED, 1.0, source=f"do({name}={value})"
                )
    return AppraisalFrame(
        event=frame.event, checks=checks, ledger_revision=frame.ledger_revision
    )


__all__ = [
    "Activation",
    "Counterfactual",
    "Direction",
    "Faculty",
    "FacultyContext",
    "FacultyRegistry",
    "NullSpec",
    "intervene",
    "register",
    "registry",
]
