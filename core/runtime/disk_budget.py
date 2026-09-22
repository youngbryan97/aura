"""Refuse the write that would fill the disk, and prune what already did.

LIVE, 2026-08-13. The data volume sat at 99% with 19GB free on 1.8TB, and the
consequences reached everything:

    threat flagged by immune:resource_monitor (severity=0.90): resource strain:
    disk at 99%                                         [every ~15 seconds]
    Metabolism: Throttling due to resource pressure (Lockdown active)
    Metabolism: deferring on allostasis signal — allostasis protecting:
    disk_percent is already past its red line

She was in permanent metabolic lockdown, and the reason was not her data. Sixty
git worktrees under .claude/worktrees each carried their own copy of
training/fused-model — 17GB apiece, two of them 338GB — for a total of 951GB of
duplicated, git-ignored build output. Nothing had a budget, so nothing ever
said no.

Two halves, because either alone fails:

  * a BUDGET, checked before a large artifact is written, so the volume cannot
    be filled by something that could have been declined;
  * RETENTION, applied to what accumulated anyway, because a budget added after
    the fact does not remove what predates it.

Deliberately conservative about what it will remove: only artifact trees it is
told to manage, only entries beyond the keep count, never the active one, and
never anything git-tracked. A cleaner that deletes something load-bearing is a
worse failure than a full disk.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.resource_observation import get_resource_observer

logger = logging.getLogger(__name__)

__all__ = [
    "DISK_AMBER_PERCENT",
    "DISK_RED_PERCENT",
    "DISK_SETPOINT_PERCENT",
    "DiskBudgetRefusal",
    "FreeSpace",
    "directory_bytes",
    "ensure_headroom_for",
    "free_space",
    "prune_superseded_artifacts",
    "state_volume_percent",
    "state_volume_usage",
]

# ── one reading, one red line ────────────────────────────────────────────────
#
# Nine subsystems each called psutil.disk_usage("/") and compared it against
# thresholds they declared themselves: allostasis (amber 92 / red 98),
# survival_driver (95 / 98), fictional_ai_synthesis (>90), and six more. Every
# one of them was measuring a mount that is not where Aura writes — on macOS
# "/" is a sealed read-only volume sharing an APFS container with
# /System/Volumes/Data, so what it reports depends on which of the two the
# reading resolves to.
#
# That is not nine bugs to fix nine times. A vital that several subsystems each
# sample their own way, against their own numbers, cannot be reasoned about:
# they can disagree about whether the disk is full while all claiming to
# measure "the disk". So there is now one reading and one set of thresholds,
# and the subsystems consume them instead of restating them.

#: Percent-used thresholds for the state volume. These are the numbers the
#: allostasis vital and the reactive survival checks both consume; changing a
#: red line here changes it everywhere, which is the point.
DISK_AMBER_PERCENT = 92.0
DISK_RED_PERCENT = 98.0
DISK_SETPOINT_PERCENT = 85.0

#: Below this, a large optional write is refused rather than attempted. A model
#: fuse that lands on a full volume corrupts the artifact AND the host.
DEFAULT_FLOOR_GB = 60.0

#: How many generations of a versioned artifact tree to keep. One to run from,
#: one to roll back to, one for the comparison someone is mid-way through.
DEFAULT_KEEP = 3


class DiskBudgetRefusal(RuntimeError):
    """A write was declined because it would breach the free-space floor."""


@dataclass(frozen=True, slots=True)
class FreeSpace:
    path: str
    free_gb: float
    total_gb: float

    @property
    def used_fraction(self) -> float:
        if self.total_gb <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.free_gb / self.total_gb)))

    # ── psutil.disk_usage() shape ────────────────────────────────────────────
    # Named to match what psutil returns so a call site that was reading the
    # wrong mount becomes correct by swapping the call, with nothing else to
    # get subtly wrong in the arithmetic around it.

    @property
    def percent(self) -> float:
        return self.used_fraction * 100.0

    @property
    def total(self) -> int:
        return int(self.total_gb * (1024**3))

    @property
    def free(self) -> int:
        return int(self.free_gb * (1024**3))

    @property
    def used(self) -> int:
        return max(0, self.total - self.free)


def free_space(path: Any = "/") -> FreeSpace:
    """Free and total gigabytes on the volume holding `path`."""

    target = Path(str(path or "/")).expanduser()
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = get_resource_observer().disk(target)
    if not usage.available:
        detail = usage.error or "disk observation unavailable"
        raise OSError(f"cannot observe disk usage for {target}: {detail}")
    return FreeSpace(
        path=str(usage.path or target),
        free_gb=usage.free_bytes / float(1024**3),
        total_gb=usage.total_bytes / float(1024**3),
    )


def directory_bytes(path: Any) -> int:
    """Total size of the files under `path`, or 0 when it cannot be read.

    Used to size a write before making it. Returning 0 on failure is
    deliberate: an unknown footprint must not block a legitimate write, only
    a known-too-large one.
    """

    root = Path(str(path or "")).expanduser()
    if not root.is_dir():
        try:
            return int(root.stat().st_size)
        # not a failure: the docstring above says why — an unknown footprint
        # returns 0 so it blocks nothing, and only a known-too-large one does.
        except OSError:
            return 0
    total = 0
    try:
        for item in root.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def state_volume_usage() -> FreeSpace:
    """Usage of the volume Aura writes to, in psutil.disk_usage()'s shape."""

    try:
        from core.runtime.state_ownership import state_root

        target: Any = state_root()
    except (ImportError, RuntimeError, OSError, ValueError):
        target = "/"
    try:
        return free_space(target)
    except (OSError, ValueError):
        return FreeSpace(path=str(target), free_gb=0.0, total_gb=0.0)


def state_volume_percent() -> float:
    """Percent used of the volume Aura writes to — the one canonical reading.

    Every disk vital in the system resolves through here. Callers must not
    re-probe a mount of their own choosing: that is how nine subsystems ended
    up measuring a read-only system volume and calling it "the disk".
    """

    try:
        from core.runtime.state_ownership import state_root

        target: Any = state_root()
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        logger.debug(
            "no state root to measure (%s: %s); falling back to /", type(exc).__name__, exc
        )
        target = "/"
    try:
        return float(free_space(target).used_fraction * 100.0)
    except (OSError, ValueError) as exc:
        # 0.0% used is the healthiest reading there is. An unreadable mount
        # must not be able to produce it.
        logger.warning(
            "disk usage for %s is unreadable (%s: %s); reporting 0.0%%",
            target,
            type(exc).__name__,
            exc,
        )
        return 0.0


def ensure_headroom_for(
    estimated_bytes: int,
    *,
    purpose: str,
    path: Any = "/",
    floor_gb: float = DEFAULT_FLOOR_GB,
) -> None:
    """Raise DiskBudgetRefusal when this write would breach the floor.

    The estimate does not have to be exact. What matters is that a 17GB artifact
    cannot be started with 19GB free and no one asking.

    Two conditions, because a fixed floor and a percentage answer different
    questions on different-sized volumes: the write must leave `floor_gb`
    behind, AND it must not push the volume past the red line the rest of the
    system already acts on. Whichever binds first wins.
    """

    space = free_space(path)
    needed_gb = max(0.0, float(estimated_bytes)) / float(1024**3)
    remaining_after = space.free_gb - needed_gb
    percent_after = (
        100.0 * (1.0 - (remaining_after / space.total_gb))
        if space.total_gb > 0
        else 0.0
    )
    if remaining_after >= floor_gb and percent_after < DISK_RED_PERCENT:
        return
    if percent_after >= DISK_RED_PERCENT and remaining_after >= floor_gb:
        refusal = DiskBudgetRefusal(
            f"{purpose} needs {needed_gb:.1f}GB and would put {space.path} at "
            f"{percent_after:.1f}% used, past the {DISK_RED_PERCENT:.0f}% red line "
            f"(free now {space.free_gb:.1f}GB of {space.total_gb:.1f}GB)"
        )
        record_degradation(
            "disk_budget",
            refusal,
            action="declined a large artifact write to protect the volume",
        )
        raise refusal
    refusal = DiskBudgetRefusal(
        f"{purpose} needs {needed_gb:.1f}GB and would leave {remaining_after:.1f}GB "
        f"on {space.path}, below the {floor_gb:.1f}GB floor "
        f"(free now {space.free_gb:.1f}GB of {space.total_gb:.1f}GB)"
    )
    record_degradation(
        "disk_budget",
        refusal,
        action="declined a large artifact write to protect the volume",
    )
    raise refusal


def _is_git_tracked(entry: Path) -> bool:
    """True when git knows about this path, in which case it is not ours to remove."""
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        result = get_subprocess_gateway().run(
            ["git", "-c", "core.fsmonitor=false", "ls-files", "--error-unmatch", str(entry)],
            cwd=str(entry.parent),
            capture_output=True,
            timeout=10,
            read_only=True,
            source="disk_budget.git_tracked_probe",
            accelerator_capability="none",
        )
    except (OSError, subprocess.SubprocessError, RuntimeError):
        # Cannot prove it is untracked, so treat it as tracked and leave it.
        return True
    return result.returncode == 0


def prune_superseded_artifacts(
    root: Any,
    *,
    keep: int = DEFAULT_KEEP,
    protect: tuple[str, ...] = (),
    dry_run: bool = False,
) -> list[tuple[str, float]]:
    """Remove the oldest generations under `root`, keeping the newest `keep`.

    Returns the (name, gigabytes) removed, newest-kept-first ordering applied by
    modification time. `protect` names entries that are never removed whatever
    their age — the active model, the one a config points at.
    """

    base = Path(str(root or "")).expanduser()
    if not base.is_dir():
        return []
    protected = {str(name) for name in protect}
    entries = [
        entry
        for entry in base.iterdir()
        if entry.is_dir() and entry.name not in protected
    ]
    if len(entries) <= max(0, int(keep)):
        return []
    entries.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    removed: list[tuple[str, float]] = []
    for entry in entries[max(0, int(keep)) :]:
        if _is_git_tracked(entry):
            continue
        try:
            size_gb = sum(
                item.stat().st_size for item in entry.rglob("*") if item.is_file()
            ) / float(1024**3)
        except OSError as exc:
            logger.debug(
                "could not size %s before removing it (%s: %s); reporting 0 GB",
                entry,
                type(exc).__name__,
                exc,
            )
            size_gb = 0.0
        if dry_run:
            removed.append((entry.name, size_gb))
            continue
        try:
            # Through the write gateway, not `shutil.rmtree`. This removes
            # whole artifact directories; CLAUDE.md is explicit that every
            # consequential file write goes through the gateway, and a
            # recursive delete is the most consequential of them.
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().delete_path(
                entry, recursive=True, source="disk_budget.prune_superseded"
            )
        except (OSError, RuntimeError) as exc:
            record_degradation(
                "disk_budget", exc, action=f"could not prune {entry.name}"
            )
            continue
        removed.append((entry.name, size_gb))
    return removed
