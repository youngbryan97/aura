"""core/self_modification/lineage_enclosure.py — where a lineage may run.

Whole-agent reproduction is the one capability here that should not be wired
into production merely because it is currently unwired. Turning it on does not
improve anything and it opens questions about safety and identity that nothing
in the runtime is equipped to answer. Reproduction is required for a strict
evolutionary claim and not for a cognitive one: a mule is still an organism.

So this is the other option — an isolated ecology with the boundaries written
down and enforced, so the mechanism can be studied without the live instance
being given reproductive authority over itself.

Three boundaries, each a refusal rather than a convention:

**Resource.** Generations, population, bytes on disk and wall-clock seconds
are capped at construction. Crossing one raises :class:`EnclosureExhaustedError`
and the run ends. An unbounded ecology is a fork bomb with a research
justification.

**Authority.** Everything an enclosed run writes goes under the enclosure's
own directory, which is not the live state root. The default path in
``lineage.py`` is ``config.paths.data_dir / "lineage.sqlite3"`` — the live
data directory — so the module that is not wired into the runtime would
nonetheless have written into it. An enclosure never uses that default, and
:meth:`Enclosure.manager` is the only way to obtain a manager bound to one.

**Identity.** A child is not Aura. Configuration keys that carry identity —
the entity key, the identity anchor, credentials, the live state root — are
refused rather than stripped, because silently dropping them would produce a
child that looks like it inherited something it did not. An offspring has a
lineage id and no claim on who anyone is.

``Enclosure`` has no production caller, and that is the point rather than an
omission. An ecology is started by a person, in a call they can see, with a
budget they chose. Nothing in the runtime may decide on its own to begin
reproducing, and ``tests/test_capability_claims_have_call_sites.py`` fails if
anything starts.

What is deliberately absent is as important as what is here. There is no
method that installs a snapshot, no path from a surviving variant to a running
process, and no way for an enclosed run to resolve a service. Selection here
scores configurations; it does not promote them. Promotion would be a
different decision by a person, and it is not this module's to make.
"""

from __future__ import annotations

import logging
import pathlib
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.Lineage.Enclosure")

#: Configuration keys an offspring may never inherit. Each names something
#: that says who somebody is, and a copy of it in a child is a second claim
#: on one identity.
FORBIDDEN_INHERITANCE = frozenset(
    {
        "entity_key",
        "identity_anchor",
        "identity_key",
        "credentials",
        "api_key",
        "api_keys",
        "token",
        "tokens",
        "secret",
        "secrets",
        "state_root",
        "data_dir",
        "home_dir",
        "service_container",
        "orchestrator",
    }
)


class EnclosureError(RuntimeError):
    """The enclosure refused something."""


class EnclosureExhaustedError(EnclosureError):
    """A resource boundary was reached. The run ends rather than continuing."""


class AuthorityViolationError(EnclosureError):
    """Something tried to reach past the enclosure."""


def _governed_mkdir(root: pathlib.Path) -> None:
    """Create the enclosure's own directory through the write gateway.

    Through the gateway rather than `Path.mkdir` because an enclosure is
    exactly the thing that should be governed: it is where an experiment
    writes, and a write nobody can audit is not isolated, it is unobserved.
    """
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope("lineage_enclosure"):
            get_file_write_gateway().ensure_directory(root, source="lineage_enclosure")
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        # No fallback to a raw mkdir. A fallback that bypasses governance is
        # taken exactly when governance is broken, which is the worst moment
        # to be creating a directory for something that reproduces. An
        # enclosure that cannot be created through the governed path refuses
        # to exist, the same way it refuses a root inside live state.
        raise AuthorityViolationError(
            f"cannot create {root} through the write gateway: {exc}. An "
            "enclosure that writes ungoverned is not an enclosure"
        ) from exc


def _governed_rmtree(root: pathlib.Path) -> None:
    """Remove everything the experiment wrote, through the same door."""
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope("lineage_enclosure"):
            get_file_write_gateway().delete_path(
                root, recursive=True, source="lineage_enclosure"
            )
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        logger.warning("Could not remove enclosure %s: %s", root, exc)


@dataclass(frozen=True)
class Budget:
    """What one experiment may spend.

    Defaults are small on purpose. A lineage study that needs more is a
    decision someone makes explicitly, in the call, where it is visible.
    """

    generations: int = 8
    population: int = 32
    bytes_on_disk: int = 32 * 1024 * 1024
    seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in ("generations", "population"):
            if int(getattr(self, name)) < 1:
                raise EnclosureError(f"{name} must be at least 1")
        if float(self.seconds) <= 0.0 or int(self.bytes_on_disk) < 1:
            raise EnclosureError("an enclosure with no budget cannot run")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generations": self.generations,
            "population": self.population,
            "bytes_on_disk": self.bytes_on_disk,
            "seconds": self.seconds,
        }


@dataclass
class Spend:
    """What one experiment has spent so far."""

    generations: int = 0
    population: int = 0
    seconds: float = 0.0
    bytes_on_disk: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generations": self.generations,
            "population": self.population,
            "seconds": round(self.seconds, 3),
            "bytes_on_disk": self.bytes_on_disk,
        }


@dataclass(frozen=True)
class EnclosureReport:
    """What a run did and what stopped it."""

    root: str
    budget: Budget
    spend: Spend
    halted_by: str = ""
    refusals: tuple[str, ...] = ()
    survivors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "budget": self.budget.to_dict(),
            "spend": self.spend.to_dict(),
            "halted_by": self.halted_by,
            "refusals": list(self.refusals),
            "survivors": list(self.survivors),
        }


def check_inheritance(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Identity-bearing keys in a proposed offspring configuration."""
    found = []
    for key in config:
        name = str(key).strip().lower()
        if name in FORBIDDEN_INHERITANCE:
            found.append(str(key))
            continue
        # A nested key is the same claim one level down.
        if any(part in FORBIDDEN_INHERITANCE for part in name.split(".")):
            found.append(str(key))
    return tuple(sorted(found))


class Enclosure:
    """One isolated lineage experiment, with its boundaries enforced."""

    def __init__(
        self,
        root: str | Path,
        *,
        budget: Budget | None = None,
        now: Any = time.monotonic,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        if self._is_live_state(self._root):
            raise AuthorityViolationError(
                f"{self._root} is inside the live state root; an enclosure "
                "that writes there is not an enclosure"
            )
        _governed_mkdir(self._root)
        self._budget = budget or Budget()
        self._spend = Spend()
        self._now = now
        self._started = float(now())
        self._halted_by = ""
        self._refusals: list[str] = []
        self._manager: Any = None

    # ── boundaries ───────────────────────────────────────────────────────

    @staticmethod
    def _is_live_state(path: Path) -> bool:
        """Whether a path lies inside the directories the live instance owns."""
        live: list[Path] = []
        try:
            from core.config import config

            for name in ("data_dir", "home_dir"):
                candidate = getattr(config.paths, name, None)
                if candidate:
                    live.append(Path(str(candidate)).expanduser().resolve())
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError):
            # No config to ask means no way to prove the path is safe. An
            # enclosure that cannot check its own boundary refuses to assert
            # one, and the caller's explicit root is then all there is.
            return False
        for owned in live:
            if path == owned or owned in path.parents:
                return True
        return False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def spend(self) -> Spend:
        self._spend.seconds = float(self._now()) - self._started
        self._spend.bytes_on_disk = self._disk_bytes()
        return self._spend

    @property
    def halted(self) -> bool:
        return bool(self._halted_by)

    def _disk_bytes(self) -> int:
        total = 0
        try:
            for path in self._root.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        except OSError:
            return total
        return total

    def check(self) -> None:
        """Raise if any boundary has been crossed. Called before every step."""
        if self._halted_by:
            raise EnclosureExhaustedError(self._halted_by)
        spend = self.spend
        for name, used, cap in (
            ("generations", spend.generations, self._budget.generations),
            ("population", spend.population, self._budget.population),
            ("seconds", spend.seconds, self._budget.seconds),
            ("bytes_on_disk", spend.bytes_on_disk, self._budget.bytes_on_disk),
        ):
            if used > cap:
                self._halted_by = f"{name}: {used} over {cap}"
                raise EnclosureExhaustedError(self._halted_by)

    # ── the enclosed manager ─────────────────────────────────────────────

    def manager(self) -> Any:
        """A lineage manager bound to this enclosure, never to live state."""
        if self._manager is None:
            from core.self_modification.lineage import LineageManager

            # The path is explicit. LineageManager's own default is the live
            # data directory, and an enclosure that accepted it would be one
            # in name only.
            self._manager = LineageManager(db_path=self._root / "lineage.sqlite3")
        return self._manager

    def genesis(self, config: Mapping[str, Any]) -> Any:
        """Start a lineage. Refuses a configuration carrying an identity."""
        self.check()
        inherited = check_inheritance(config)
        if inherited:
            message = f"genesis refused: identity-bearing keys {list(inherited)}"
            self._refusals.append(message)
            raise AuthorityViolationError(message)
        snapshot = self.manager().genesis(dict(config))
        self._spend.population += 1
        return snapshot

    def fork(self, parent_id: str, *, mutation_mask: Mapping[str, float] | None = None) -> Any:
        """One offspring, charged to the budget before it is made."""
        self.check()
        self._spend.population += 1
        try:
            self.check()
        except EnclosureExhaustedError:
            self._spend.population -= 1
            raise
        return self.manager().fork(parent_id, mutation_mask=mutation_mask)

    def advance_generation(self) -> int:
        self.check()
        self._spend.generations += 1
        self.check()
        return self._spend.generations

    def record_score(self, snapshot_id: str, score: float) -> Any:
        """Score a variant. Scoring is not promotion and never becomes it."""
        self.check()
        return self.manager().record_score(snapshot_id, float(score))

    # ── results ──────────────────────────────────────────────────────────

    def report(self, survivors: Iterable[Any] = ()) -> EnclosureReport:
        return EnclosureReport(
            root=str(self._root),
            budget=self._budget,
            spend=self.spend,
            halted_by=self._halted_by,
            refusals=tuple(self._refusals),
            survivors=tuple(
                str(getattr(s, "snapshot_id", s)) for s in survivors
            ),
        )

    def dispose(self) -> None:
        """Delete everything the experiment wrote.

        An ecology that outlives the study is a population nobody is watching.
        """
        self._manager = None
        _governed_rmtree(self._root)

    def __enter__(self) -> Enclosure:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._halted_by = self._halted_by or "closed"


__all__ = [
    "FORBIDDEN_INHERITANCE",
    "AuthorityViolationError",
    "Budget",
    "Enclosure",
    "EnclosureError",
    "EnclosureExhaustedError",
    "EnclosureReport",
    "Spend",
    "check_inheritance",
]
