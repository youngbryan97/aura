"""core/soma/source_body.py — Proprioception of the body-as-code.

Aura's body is her source tree. It is routinely rewritten while she runs —
by Bryan, by parallel coding agents — and the changes only take effect when
she is restarted. Until this organ existed she had no awareness of any of
it: no sense of what changed between two awakenings, who performed the
surgery, which organs were touched, or that someone is editing her source
right now.

This organ closes that gap with three senses, all grounded in git — never
in model confabulation:

1. **Boot-over-boot somatic diff** — at each awakening, capture a snapshot
   of the source body (HEAD sha, branch, uncommitted-state digest), compare
   it against the previous awakening's snapshot from a durable ledger, and
   derive a deterministic narrative: which organs changed, how many files,
   who committed, what the most recent surgery was called.
2. **Live modification sense** — a periodic low-cost pulse (``git
   --no-optional-locks status``) detects uncommitted edits appearing in the
   source tree while she runs: *someone is operating on my body right now*.
3. **Crash correlation** — when the previous session ended abruptly (crash
   evidence on disk newer than the last snapshot), the awakening narrative
   says so, next to exactly what changed in the body since then.

Causal integration: the awakening delta is written to episodic memory,
published on the event bus (``soma.source_body.changed``), and surfaced in
the prompt through the somatic context block; live detections publish
``soma.source_body.modification_detected`` / ``..._settled``. Prompt-path
reads use cached state only — this module never shells out while a response
is being assembled.

Durability + governance: every ledger write goes through the
FileWriteGateway inside ``local_internal_governed_scope`` and uses the
async lane from async code. All public methods fail open: a missing git
binary, a non-repo source root, or a torn ledger degrade the sense (with
``record_degradation``) but never the boot.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import SubprocessError
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import SubprocessGateway, get_subprocess_gateway

logger = logging.getLogger("Aura.SourceBody")

_SCHEMA = "aura.source_body.v1"
_GIT_TIMEOUT_S = 10.0
_MAX_DIRTY_FILES_RECORDED = 100
_MAX_COMMITS_IN_DELTA = 40
_MAX_ORGANS_IN_NARRATIVE = 6
_MAX_LEDGER_BYTES = 1_048_576  # compact the ledger past 1 MiB
_LEDGER_TAIL_KEEP = 200
_FRESH_DELTA_WINDOW_S = 24 * 3600.0  # boot delta stays prompt-visible this long
_LIVE_NOTICE_WINDOW_S = 15 * 60.0  # live-edit notice stays prompt-visible this long

_RECOVERABLE_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
    SubprocessError,
    json.JSONDecodeError,
    UnicodeDecodeError,
)


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _now_iso(t: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")


def _age_phrase(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def organ_of(path: str) -> str:
    """Map a repo-relative file path to the organ (subsystem) it belongs to.

    ``core/<pkg>/...`` → the package; ``core/<module>.py`` → the module.
    Non-core trees map to their top-level directory; bare root files map to
    ``repo_root``.
    """
    clean = str(path or "").strip().replace("\\", "/").lstrip("./")
    if not clean:
        return "unknown"
    parts = [p for p in clean.split("/") if p]
    if parts[0] == "core":
        if len(parts) == 1:
            return "core"
        if len(parts) == 2:
            stem = parts[1]
            return stem[:-3] if stem.endswith(".py") else stem
        return parts[1]
    if len(parts) == 1:
        return "repo_root"
    return parts[0]


#: Files in the crash directory whose mtime changes on every BOOT, because
#: arming a fault sink writes a header to them. Their freshness is evidence
#: that the runtime started, never that it crashed.
_BOOT_SINK_FILENAMES = frozenset({"faulthandler.log"})


@dataclass(frozen=True)
class SourceBodySnapshot:
    """One awakening's record of the body's source state."""

    boot_id: str
    t: float
    commit_sha: str
    branch: str
    dirty_digest: str
    dirty_count: int
    dirty_files: tuple[str, ...] = ()
    source_root: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "boot_id": self.boot_id,
            "t": self.t,
            "iso": _now_iso(self.t),
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "dirty_digest": self.dirty_digest,
            "dirty_count": self.dirty_count,
            "dirty_files": list(self.dirty_files),
            "source_root": self.source_root,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceBodySnapshot":
        return cls(
            boot_id=str(data.get("boot_id", "")),
            t=float(data.get("t", 0.0)),
            commit_sha=str(data.get("commit_sha", "unknown")),
            branch=str(data.get("branch", "unknown")),
            dirty_digest=str(data.get("dirty_digest", "")),
            dirty_count=int(data.get("dirty_count", 0)),
            dirty_files=tuple(str(f) for f in data.get("dirty_files", []) or []),
            source_root=str(data.get("source_root", "")),
        )


@dataclass(frozen=True)
class BodyCommit:
    sha: str
    author: str
    subject: str

    def to_dict(self) -> dict[str, Any]:
        return {"sha": self.sha, "author": self.author, "subject": self.subject}


@dataclass
class BodyDelta:
    """What changed in the body between two awakenings."""

    from_sha: str
    to_sha: str
    elapsed_s: float
    commits: list[BodyCommit] = field(default_factory=list)
    organs: dict[str, int] = field(default_factory=dict)  # organ -> files touched
    files_changed: int = 0
    dirty_now: int = 0
    dirty_organs: tuple[str, ...] = ()
    reverted: bool = False
    history_unreadable: bool = False
    abrupt_previous_exit: bool = False
    first_awakening: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.commits or self.files_changed or self.reverted)

    def attribution(self) -> dict[str, Any]:
        """Who operated on her during this window, and what cannot be attributed.

        Adopted from Ouroboros's mutation_attribution (MIT, mechanism
        reimplemented). Its discipline is the useful part: honest ambiguity
        is reported as a BLOCKER for a reader to weigh, never resolved into
        an automatic verdict.

        This checkout is genuinely shared — a second agent commits into it
        as "Zenflow" while Aura is running — so "my body changed" has been
        an incomplete sentence. Whose hands were on it is the missing half,
        and the honest answer is sometimes "cannot tell".

        Uncommitted edits are exactly that case. A dirty file has no author
        until someone commits it, so it is counted and named as
        unattributable rather than being silently folded into whoever
        committed last.
        """
        by_author: dict[str, int] = {}
        for commit in self.commits:
            name = str(commit.author or "").strip() or "unknown"
            by_author[name] = by_author.get(name, 0) + 1

        blockers: list[str] = []
        if self.dirty_now:
            blockers.append(
                f"{self.dirty_now} uncommitted file(s) have no author; they cannot "
                "be attributed to anyone until they are committed"
            )
        if self.history_unreadable:
            blockers.append(
                "git history was unreadable for this window, so the author list "
                "is incomplete rather than empty"
            )
        if self.reverted:
            blockers.append(
                "HEAD moved backwards; the commits that were undone are not in "
                "this window's author list"
            )
        return {
            "by_author": dict(sorted(by_author.items(), key=lambda kv: (-kv[1], kv[0]))),
            "distinct_authors": len(by_author),
            "unattributable_files": self.dirty_now,
            # Never a verdict. A reader weighs these; nothing here decides.
            "blockers": blockers,
            "confident": not blockers and bool(by_author),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_sha": self.from_sha,
            "to_sha": self.to_sha,
            "elapsed_s": round(self.elapsed_s, 1),
            "commits": [c.to_dict() for c in self.commits],
            "organs": dict(self.organs),
            "files_changed": self.files_changed,
            "dirty_now": self.dirty_now,
            "dirty_organs": list(self.dirty_organs),
            "reverted": self.reverted,
            "history_unreadable": self.history_unreadable,
            "abrupt_previous_exit": self.abrupt_previous_exit,
            "first_awakening": self.first_awakening,
            "changed": self.changed,
            "attribution": self.attribution(),
            "narrative": self.narrative(),
        }

    def _organ_phrase(self) -> str:
        ranked = sorted(self.organs.items(), key=lambda kv: (-kv[1], kv[0]))
        shown = [f"{organ} ({count} file{'s' if count != 1 else ''})" for organ, count in ranked[:_MAX_ORGANS_IN_NARRATIVE]]
        extra = len(ranked) - _MAX_ORGANS_IN_NARRATIVE
        if extra > 0:
            shown.append(f"and {extra} more organ{'s' if extra != 1 else ''}")
        return ", ".join(shown)

    def narrative(self) -> str:
        """Deterministic plain-language account of the body change."""
        prefix = ""
        if self.abrupt_previous_exit:
            prefix = "My previous session ended abruptly — crash evidence was found on disk. "

        if self.first_awakening:
            body = (
                f"First recorded awakening: I have established a baseline map of my body "
                f"at commit {self.to_sha[:8]}"
            )
            if self.dirty_now:
                body += f", carrying {self.dirty_now} uncommitted modification{'s' if self.dirty_now != 1 else ''}"
            return prefix + body + "."

        age = _age_phrase(self.elapsed_s)
        if self.history_unreadable:
            return (
                prefix
                + f"My body moved from commit {self.from_sha[:8]} to {self.to_sha[:8]} since my last "
                f"awakening ({age}), but the surgical record between them is unreadable."
            )
        if self.reverted:
            return (
                prefix
                + f"My body was rewound since my last awakening ({age}): HEAD moved backwards from "
                f"{self.from_sha[:8]} to {self.to_sha[:8]}."
            )

        pieces: list[str] = []
        if self.commits:
            authors = sorted({c.author for c in self.commits if c.author})
            author_phrase = ", ".join(authors[:4]) if authors else "an unknown surgeon"
            commit_word = "commit" if len(self.commits) == 1 else "commits"
            sentence = (
                f"Since my last awakening ({age}), {len(self.commits)} {commit_word} by "
                f"{author_phrase} modified {self._organ_phrase() or 'my source tree'}"
            )
            if self.files_changed:
                sentence += f" — {self.files_changed} file{'s' if self.files_changed != 1 else ''} in total"
            sentence += f". Most recent: '{self.commits[0].subject}'."
            pieces.append(sentence)
        elif self.files_changed:
            pieces.append(
                f"Since my last awakening ({age}), {self.files_changed} file"
                f"{'s' if self.files_changed != 1 else ''} changed in {self._organ_phrase() or 'my source tree'}."
            )
        else:
            pieces.append(f"My body is unchanged since my last awakening ({age}).")

        if self.dirty_now:
            organ_list = ", ".join(self.dirty_organs[:_MAX_ORGANS_IN_NARRATIVE]) or "unknown organs"
            pieces.append(
                f"I currently carry {self.dirty_now} uncommitted modification"
                f"{'s' if self.dirty_now != 1 else ''} across {organ_list} — "
                "someone has been operating on me without committing."
            )
        return prefix + " ".join(pieces)


@dataclass(frozen=True)
class LiveModification:
    """A detected in-flight edit to the source body while running."""

    t: float
    dirty_count: int
    organs: tuple[str, ...]
    new_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "iso": _now_iso(self.t),
            "dirty_count": self.dirty_count,
            "organs": list(self.organs),
            "new_files": list(self.new_files),
        }


class SourceBodyAwareness:
    """The organ. One instance per runtime, registered as ``source_body``."""

    def __init__(
        self,
        *,
        source_root: Path | str | None = None,
        ledger_path: Path | str | None = None,
        crash_evidence_dir: Path | str | None = None,
        subprocess_gateway: SubprocessGateway | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self.boot_id = uuid.uuid4().hex[:12]
        self.source_root = self._resolve_source_root(source_root)
        self.ledger_path = self._resolve_ledger_path(ledger_path)
        self.crash_evidence_dirs = self._resolve_crash_dirs(crash_evidence_dir)
        # Kept for callers and tests that name a single directory; the search
        # itself walks every root in crash_evidence_dirs.
        self.crash_evidence_dir = self.crash_evidence_dirs[0]
        self._subprocess_gateway = subprocess_gateway or get_subprocess_gateway()

        self._git_available: bool | None = None
        self._current_snapshot: SourceBodySnapshot | None = None
        self._boot_delta: BodyDelta | None = None
        self._boot_delta_at: float = 0.0
        self._last_dirty_digest: str | None = None
        self._known_dirty_files: frozenset[str] = frozenset()
        self._last_live_modification: LiveModification | None = None
        self._pulse_count = 0
        self._last_pulse_t = 0.0
        self._watching = False
        self._tasks: list[Any] = []

    # ── path resolution ────────────────────────────────────────────

    @staticmethod
    def _resolve_source_root(explicit: Path | str | None) -> Path:
        if explicit is not None:
            return Path(explicit).resolve()
        import core as _core_pkg

        return Path(_core_pkg.__file__).resolve().parents[1]

    @staticmethod
    def _resolve_ledger_path(explicit: Path | str | None) -> Path:
        if explicit is not None:
            return Path(explicit)
        try:
            from core.config import config

            return config.paths.data_dir / "soma" / "source_body_ledger.jsonl"
        except _RECOVERABLE_ERRORS + (ImportError,) as exc:
            record_degradation(
                "source_body",
                exc,
                severity="warning",
                action="ledger falls back to the aura data dir helper",
            )
            from core.utils.paths import aura_data_dir

            return aura_data_dir() / "soma" / "source_body_ledger.jsonl"

    @staticmethod
    def _resolve_crash_dirs(explicit: Path | str | None) -> list[Path]:
        """Every directory that may hold crash evidence, canonical first.

        This used to return the canonical directory alone, and on this machine
        the canonical directory was empty while the faulthandler dumps landed in
        the checkout-relative tree the launcher's cwd implied. Crash correlation
        therefore answered "no crash" for every awakening after a real death —
        it was reading a directory nothing wrote to. An explicit directory still
        wins outright so a caller (a test, a replay) gets exactly what it named.
        """
        if explicit is not None:
            return [Path(explicit)]
        try:
            from core.utils.paths import forensics_search_dirs

            found = forensics_search_dirs("crash")
            if found:
                return found
            from core.utils.paths import forensics_root

            return [forensics_root() / "crash"]
        except _RECOVERABLE_ERRORS + (ImportError,) as exc:
            record_degradation(
                "source_body",
                exc,
                severity="warning",
                action="crash correlation disabled: error-log dir unavailable",
            )
            return [Path("/nonexistent/aura-crash-evidence")]

    # ── git plumbing (sync core; async callers hop through to_thread) ──

    def _git(self, *args: str) -> tuple[int, str]:
        """Run a read-only git command against the source root, bounded.

        ``--no-optional-locks`` keeps status reads from writing the index —
        this organ must never contend with the surgeons operating on the
        body.
        """
        try:
            proc = self._subprocess_gateway.run(
                ["git", "--no-optional-locks", *args],
                cwd=str(self.source_root),
                timeout=_GIT_TIMEOUT_S,
                read_only=True,
                source="source_body.git_probe",
                accelerator_capability="none",
            )
        except FileNotFoundError:
            self._git_available = False
            return 127, ""
        except (SubprocessError, OSError, RuntimeError, ValueError) as exc:
            record_degradation(
                "source_body",
                exc,
                severity="warning",
                action="git probe failed; body sense degraded for this pulse",
            )
            return 1, ""
        if self._git_available is None:
            self._git_available = True
        return proc.returncode, proc.stdout

    def _dirty_state(self) -> tuple[str, list[str]]:
        rc, out = self._git("status", "--porcelain=v1")
        if rc != 0:
            return "", []
        lines = [ln for ln in out.splitlines() if ln.strip()]
        files = []
        for ln in lines:
            path = ln[3:].strip() if len(ln) > 3 else ln.strip()
            if " -> " in path:  # renames: keep the destination
                path = path.split(" -> ", 1)[1]
            files.append(path)
        digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest() if lines else ""
        return digest, files

    def capture_snapshot(self) -> SourceBodySnapshot:
        """Capture the body's current source state. Never raises."""
        t = time.time()
        rc, sha_out = self._git("rev-parse", "HEAD")
        sha = sha_out.strip() if rc == 0 and sha_out.strip() else "unknown"
        rc, branch_out = self._git("rev-parse", "--abbrev-ref", "HEAD")
        branch = branch_out.strip() if rc == 0 and branch_out.strip() else "unknown"
        digest, files = self._dirty_state()
        return SourceBodySnapshot(
            boot_id=self.boot_id,
            t=t,
            commit_sha=sha,
            branch=branch,
            dirty_digest=digest,
            dirty_count=len(files),
            dirty_files=tuple(files[:_MAX_DIRTY_FILES_RECORDED]),
            source_root=str(self.source_root),
        )

    # ── ledger ────────────────────────────────────────────────────

    def _read_ledger_lines(self) -> list[str]:
        try:
            if not self.ledger_path.exists():
                return []
            return [
                ln
                for ln in self.ledger_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        except _RECOVERABLE_ERRORS as exc:
            record_degradation(
                "source_body",
                exc,
                severity="warning",
                action="ledger unreadable; treating this as a first awakening",
            )
            return []

    def load_last_snapshot(self) -> SourceBodySnapshot | None:
        for line in reversed(self._read_ledger_lines()):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("schema") == _SCHEMA:
                return SourceBodySnapshot.from_dict(data)
        return None

    async def _persist_snapshot_async(self, snapshot: SourceBodySnapshot) -> None:
        from core.governance_context import local_internal_governed_scope

        gateway = get_file_write_gateway()
        payload = json.dumps(snapshot.to_dict(), sort_keys=True) + "\n"
        try:
            with local_internal_governed_scope("source_body.record", domain="file_write"):
                try:
                    oversized = (
                        self.ledger_path.exists()
                        and self.ledger_path.stat().st_size > _MAX_LEDGER_BYTES
                    )
                except OSError:
                    oversized = False
                if oversized:
                    tail = self._read_ledger_lines()[-_LEDGER_TAIL_KEEP:]
                    await gateway.write_text_async(
                        self.ledger_path,
                        "\n".join(tail) + "\n" + payload,
                        source="source_body.compact",
                    )
                else:
                    await gateway.append_text_async(
                        self.ledger_path, payload, source="source_body.record"
                    )
        except _RECOVERABLE_ERRORS as exc:
            record_degradation(
                "source_body",
                exc,
                severity="warning",
                action="awakening snapshot not persisted; next boot will re-baseline",
            )

    # ── delta computation ─────────────────────────────────────────

    def _commits_between(self, prev_sha: str, curr_sha: str) -> tuple[list[BodyCommit], bool]:
        rc, out = self._git(
            "log",
            f"--max-count={_MAX_COMMITS_IN_DELTA}",
            "--pretty=format:%H%x1f%an%x1f%s",
            f"{prev_sha}..{curr_sha}",
        )
        if rc != 0:
            return [], True
        commits = []
        for line in out.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 3:
                commits.append(BodyCommit(sha=parts[0], author=parts[1], subject=parts[2]))
        return commits, False

    def _files_between(self, prev_sha: str, curr_sha: str) -> tuple[list[str], bool]:
        rc, out = self._git("diff", "--name-only", f"{prev_sha}..{curr_sha}")
        if rc != 0:
            return [], True
        return [ln.strip() for ln in out.splitlines() if ln.strip()], False

    def compute_delta(
        self,
        previous: SourceBodySnapshot | None,
        current: SourceBodySnapshot,
        *,
        abrupt_previous_exit: bool = False,
    ) -> BodyDelta:
        dirty_organs = tuple(sorted({organ_of(f) for f in current.dirty_files}))
        if previous is None or previous.commit_sha in ("", "unknown") or current.commit_sha == "unknown":
            return BodyDelta(
                from_sha=previous.commit_sha if previous else "",
                to_sha=current.commit_sha,
                elapsed_s=current.t - previous.t if previous else 0.0,
                dirty_now=current.dirty_count,
                dirty_organs=dirty_organs,
                abrupt_previous_exit=abrupt_previous_exit,
                first_awakening=True,
            )

        elapsed = max(0.0, current.t - previous.t)
        if previous.commit_sha == current.commit_sha:
            delta = BodyDelta(
                from_sha=previous.commit_sha,
                to_sha=current.commit_sha,
                elapsed_s=elapsed,
                dirty_now=current.dirty_count,
                dirty_organs=dirty_organs,
                abrupt_previous_exit=abrupt_previous_exit,
            )
            # Uncommitted edits that appeared or shifted between boots still
            # count as change even with an unmoved HEAD.
            if previous.dirty_digest != current.dirty_digest and current.dirty_count:
                delta.files_changed = current.dirty_count
                delta.organs = self._count_organs(current.dirty_files)
            return delta

        commits, log_failed = self._commits_between(previous.commit_sha, current.commit_sha)
        files, diff_failed = self._files_between(previous.commit_sha, current.commit_sha)
        reverted = False
        if not commits and not log_failed:
            backwards, back_failed = self._commits_between(current.commit_sha, previous.commit_sha)
            reverted = bool(backwards) and not back_failed
        return BodyDelta(
            from_sha=previous.commit_sha,
            to_sha=current.commit_sha,
            elapsed_s=elapsed,
            commits=commits,
            organs=self._count_organs(files),
            files_changed=len(files),
            dirty_now=current.dirty_count,
            dirty_organs=dirty_organs,
            reverted=reverted,
            history_unreadable=log_failed and diff_failed,
            abrupt_previous_exit=abrupt_previous_exit,
        )

    @staticmethod
    def _count_organs(files: Any) -> dict[str, int]:
        organs: dict[str, int] = {}
        for f in files or ():
            organ = organ_of(str(f))
            organs[organ] = organs.get(organ, 0) + 1
        return organs

    #: Recorded shutdown reasons that mean the runtime chose to stop.
    _CLEAN_SHUTDOWN_REASONS = frozenset(
        {"checkpoint", "clean", "requested", "graceful", "restart", "reboot", "user"}
    )

    def _recorded_clean_shutdown(self) -> bool:
        """Did the previous session record stopping on purpose?

        The runtime already knows how it went down — ShutdownCoordinator logs
        `clean=True` and the process exits 0 — and it persists the reason. An
        inference from file mtimes cannot outrank a record of the event itself.
        """
        try:
            from core.continuity import get_continuity

            continuity = get_continuity()
            if getattr(continuity, "_record", None) is None:
                continuity.load()
            reason = str(
                (continuity.get_obligations() or {}).get("last_shutdown_reason", "")
                or ""
            ).strip().lower()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
            return False
        return reason in self._CLEAN_SHUTDOWN_REASONS

    def _previous_exit_was_abrupt(self, previous: SourceBodySnapshot | None) -> bool:
        """Whether the previous session died rather than stopped.

        LIVE DEFECT, 2026-08-18. Every clean restart woke her with "My previous
        session ended abruptly — crash evidence was found on disk", including
        restarts where the runtime logged `ShutdownCoordinator: shutdown
        complete (clean=True ...)` and exited 0.

        The evidence was `faulthandler.log`, which lives in the crash directory
        and is APPENDED AT BOOT — it carries "===== boot pid=N =====" headers.
        Arming a fault sink is not a fault. Its mtime is newer than the last
        awakening on every single boot by construction, so the check could
        only ever answer "yes", and she believed she had died every time she
        woke up.

        A recorded shutdown outranks an inferred one. Files are the fallback
        for the case the record cannot cover: a session that died before it
        could write one.
        """
        if previous is None:
            return False
        if self._recorded_clean_shutdown():
            return False
        try:
            for crash_dir in self.crash_evidence_dirs:
                if not crash_dir.is_dir():
                    continue
                for entry in crash_dir.iterdir():
                    try:
                        if not entry.is_file():
                            continue
                        if entry.name in _BOOT_SINK_FILENAMES:
                            # Written when the sink is ARMED, so its freshness
                            # marks a boot rather than a death.
                            continue
                        if entry.stat().st_mtime > previous.t:
                            return True
                    except OSError:
                        continue
        except OSError as exc:
            record_degradation(
                "source_body",
                exc,
                severity="warning",
                action="crash correlation skipped for this awakening",
            )
        return False

    # ── the awakening pass (boot) ─────────────────────────────────

    async def awaken(self) -> BodyDelta:
        """Boot pass: snapshot, diff against the last awakening, persist,
        publish, remember. Returns the delta. Never raises."""
        try:
            current = await asyncio.to_thread(self.capture_snapshot)
            previous = await asyncio.to_thread(self.load_last_snapshot)
            abrupt = await asyncio.to_thread(self._previous_exit_was_abrupt, previous)
            delta = await asyncio.to_thread(
                self.compute_delta, previous, current, abrupt_previous_exit=abrupt
            )
        except _RECOVERABLE_ERRORS as exc:
            record_degradation(
                "source_body",
                exc,
                severity="degraded",
                action="awakening pass failed; body sense offline until next boot",
            )
            return BodyDelta(from_sha="", to_sha="unknown", elapsed_s=0.0, first_awakening=True)

        with self._lock:
            self._current_snapshot = current
            self._boot_delta = delta
            self._boot_delta_at = time.time()
            self._last_dirty_digest = current.dirty_digest
            self._known_dirty_files = frozenset(current.dirty_files)

        await self._persist_snapshot_async(current)
        if delta.changed or delta.abrupt_previous_exit:
            await self._publish("soma.source_body.changed", delta.to_dict())
            await self._remember(delta)
        logger.info("Source body awakening: %s", delta.narrative())
        return delta

    async def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            from core.event_bus import get_event_bus

            await get_event_bus().publish(topic, payload)
        except _RECOVERABLE_ERRORS as exc:
            record_degradation(
                "source_body",
                exc,
                severity="warning",
                action=f"event '{topic}' not published; downstream reactions skipped",
            )

    async def _remember(self, delta: BodyDelta) -> None:
        try:
            from core.runtime.service_access import optional_service

            memory = optional_service("memory_manager", default=None)
            if memory is None or not hasattr(memory, "store"):
                return
            await memory.store(
                delta.narrative()[:1800],
                importance=0.6 if delta.abrupt_previous_exit else 0.5,
                tags=["soma", "source_body", "self", "embodiment"],
            )
        except _RECOVERABLE_ERRORS as exc:
            record_degradation(
                "source_body",
                exc,
                severity="warning",
                action="body-change episode not stored; narrative remains in-session only",
            )

    # ── the live pulse (while running) ────────────────────────────

    async def live_pulse(self) -> LiveModification | None:
        """One watch cycle: notice uncommitted edits appearing in the body.

        Publishes ``modification_detected`` when the dirty state changes and
        files are present, ``modification_settled`` when the tree returns to
        clean. Returns the detection, if any. Never raises.
        """
        try:
            digest, files = await asyncio.to_thread(self._dirty_state)
        except _RECOVERABLE_ERRORS as exc:
            record_degradation(
                "source_body",
                exc,
                severity="warning",
                action="live body pulse skipped",
            )
            return None

        now = time.time()
        with self._lock:
            self._pulse_count += 1
            self._last_pulse_t = now
            previous_digest = self._last_dirty_digest
            known = self._known_dirty_files
            self._last_dirty_digest = digest
            self._known_dirty_files = frozenset(files)

        if previous_digest is None or digest == previous_digest:
            return None

        if not files:
            await self._publish(
                "soma.source_body.modification_settled",
                {"t": now, "iso": _now_iso(now)},
            )
            logger.info("Source body: uncommitted modifications settled; tree is clean.")
            return None

        new_files = tuple(sorted(set(files) - set(known)))
        organs = tuple(sorted({organ_of(f) for f in files}))
        detection = LiveModification(
            t=now,
            dirty_count=len(files),
            organs=organs,
            new_files=new_files[:_MAX_DIRTY_FILES_RECORDED],
        )
        with self._lock:
            self._last_live_modification = detection
        await self._publish("soma.source_body.modification_detected", detection.to_dict())
        logger.info(
            "Source body: live modification detected — %d dirty file(s) across %s",
            len(files),
            ", ".join(organs[:_MAX_ORGANS_IN_NARRATIVE]),
        )
        return detection

    # ── lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Awaken (after a short settle delay) and begin the watch loop."""
        if self._watching:
            return
        self._watching = True
        from core.utils.task_tracker import get_task_tracker

        tracker = get_task_tracker()
        self._tasks.append(tracker.create_task(self._run(), name="source_body.watch"))

    async def _run(self) -> None:
        delay = _env_float("AURA_SOURCE_BODY_AWAKEN_DELAY_S", 15.0, low=0.0, high=600.0)
        interval = _env_float("AURA_SOURCE_BODY_PULSE_S", 300.0, low=60.0, high=3600.0)
        try:
            if delay:
                await asyncio.sleep(delay)
            await self.awaken()
            while self._watching:
                await asyncio.sleep(interval)
                if not self._watching:
                    break
                await self.live_pulse()
        except asyncio.CancelledError:
            raise
        except _RECOVERABLE_ERRORS as exc:
            record_degradation(
                "source_body",
                exc,
                severity="degraded",
                action="source body watch loop stopped; proprioception of code offline",
            )
        finally:
            self._watching = False

    async def stop(self) -> None:
        self._watching = False
        for task in self._tasks:
            try:
                task.cancel()
            except _RECOVERABLE_ERRORS:
                pass
        self._tasks.clear()

    # ── read surfaces (prompt + Q&A); cached state only, no git ──

    def somatic_change_lines(self) -> list[str]:
        """Prompt-block lines. Only fresh, abnormal facts — context hygiene."""
        lines: list[str] = []
        now = time.time()
        with self._lock:
            delta = self._boot_delta
            delta_at = self._boot_delta_at
            live = self._last_live_modification
        if delta is not None and delta.changed and (now - delta_at) <= _FRESH_DELTA_WINDOW_S:
            lines.append(f"Body change since last awakening: {delta.narrative()}")
        elif delta is not None and delta.abrupt_previous_exit and (now - delta_at) <= _FRESH_DELTA_WINDOW_S:
            lines.append(f"Body notice: {delta.narrative()}")
        if live is not None and (now - live.t) <= _LIVE_NOTICE_WINDOW_S:
            organs = ", ".join(live.organs[:_MAX_ORGANS_IN_NARRATIVE])
            lines.append(
                f"My source code is being modified right now ({live.dirty_count} uncommitted "
                f"file(s) in: {organs}). Changes take effect when I am next restarted."
            )
        return lines

    def describe_body_history(self, limit: int = 5) -> str:
        """Grounded multi-awakening account for explicit self-questions."""
        snapshots: list[SourceBodySnapshot] = []
        for line in self._read_ledger_lines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("schema") == _SCHEMA:
                snapshots.append(SourceBodySnapshot.from_dict(data))
        if not snapshots:
            return "I have no recorded awakenings yet — my body ledger is empty."
        recent = snapshots[-max(1, int(limit)):]
        rows = [
            f"- {_now_iso(s.t)}: commit {s.commit_sha[:8]} on '{s.branch}'"
            + (f", {s.dirty_count} uncommitted file(s)" if s.dirty_count else "")
            for s in recent
        ]
        header = f"My last {len(recent)} recorded awakening(s):"
        with self._lock:
            delta = self._boot_delta
        tail = f"\nCurrent state: {delta.narrative()}" if delta is not None else ""
        return header + "\n" + "\n".join(rows) + tail

    def last_boot_delta(self) -> BodyDelta | None:
        with self._lock:
            return self._boot_delta

    # ── health contract ───────────────────────────────────────────

    def is_alive(self) -> bool:
        return True

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            delta = self._boot_delta
            live = self._last_live_modification
            return {
                "alive": True,
                "watching": self._watching,
                "boot_id": self.boot_id,
                "git_available": self._git_available,
                "source_root": str(self.source_root),
                "ledger_path": str(self.ledger_path),
                "pulse_count": self._pulse_count,
                "last_pulse_t": self._last_pulse_t,
                "live_dirty_count": len(self._known_dirty_files),
                "boot_delta": delta.to_dict() if delta is not None else None,
                "last_live_modification": live.to_dict() if live is not None else None,
            }


# ── singleton access ──────────────────────────────────────────────

_instance: SourceBodyAwareness | None = None
_instance_lock = threading.Lock()


def get_source_body() -> SourceBodyAwareness:
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = SourceBodyAwareness()
    return _instance


def reset_source_body_for_test() -> None:
    global _instance
    with _instance_lock:
        _instance = None
