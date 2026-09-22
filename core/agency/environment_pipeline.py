"""Local environmental agency — a concrete, governed background pipeline (#36).

Aura's autonomous agency cycle (``agency_facade.run_cycle`` /
``autonomous_initiative_loop``) decides *what* to do. This module is one concrete
thing she can do to her *local environment* without a model in the loop: survey a
workspace directory and write a plain-language digest of it. It is deliberately:

  - **non-destructive** — a read-only survey plus a single digest file it owns
    (``<workspace>/.aura/workspace_digest.md``); it never edits or deletes
    anything else;
  - **bounded** — caps the number of entries walked and skips heavy/noise dirs
    (``.git``, ``node_modules``, ``__pycache__``, ``.venv`` …) so it can't run away;
  - **governed** — the write goes through ``local_internal_governed_scope``;
  - **safe-mode aware** — when safe mode is on it surveys but does not write;
  - **provable** — it returns a structured receipt and appends a run-ledger line,
    so a proof harness (``tools/environmental_agency_proof.py``) can verify a real
    on-disk effect, and the run shows up in the activity view.

The work here is deterministic and offline-testable; wiring it as an
autonomously-*chosen* initiative rides the existing agency loop.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.runtime_settings import get_runtime_setting
from core.runtime.state_ownership import state_root

logger = logging.getLogger(__name__)

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
              ".pytest_cache", ".ruff_cache", "dist", "build", ".aura"}
_MAX_ENTRIES = 20_000          # hard ceiling so a survey can never run away
_DIGEST_RELPATH = Path(".aura") / "workspace_digest.md"
_RUN_LEDGER = state_root() / "data" / "environmental_agency" / "runs.jsonl"


@dataclass
class WorkspaceDigestReceipt:
    """Provable record of one environmental-agency run."""
    workspace: str
    surveyed_at: str
    total_files: int
    total_dirs: int
    total_bytes: int
    by_extension: dict[str, int]
    largest_file: str
    recent_files: list[str]
    digest_path: str
    wrote_digest: bool
    success: bool
    note: str = ""
    truncated: bool = False
    action_kind: str = "environmental.workspace_digest"
    origin_drive: str = "order"
    duration_ms: int = 0
    receipt_id: str = field(default_factory=lambda: f"envrun-{int(time.time()*1000)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


def survey_workspace(workspace: Path, *, max_entries: int = _MAX_ENTRIES) -> dict[str, Any]:
    """Read-only survey of ``workspace``. Pure (no writes), bounded, deterministic."""
    root = Path(workspace)
    total_files = total_dirs = total_bytes = 0
    by_ext: Counter[str] = Counter()
    largest = ("", 0)
    recent: list[tuple[float, str]] = []
    seen = 0
    truncated = False

    for path in root.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        seen += 1
        if seen > max_entries:
            truncated = True
            break
        try:
            if path.is_dir():
                total_dirs += 1
                continue
            st = path.stat()
        except OSError:
            continue
        total_files += 1
        total_bytes += st.st_size
        ext = path.suffix.lower() or "(no ext)"
        by_ext[ext] += 1
        if st.st_size > largest[1]:
            largest = (str(path.relative_to(root)), st.st_size)
        recent.append((st.st_mtime, str(path.relative_to(root))))

    recent.sort(reverse=True)
    return {
        "total_files": total_files,
        "total_dirs": total_dirs,
        "total_bytes": total_bytes,
        "by_extension": dict(by_ext.most_common(12)),
        "largest_file": largest[0],
        "largest_bytes": largest[1],
        "recent_files": [r[1] for r in recent[:8]],
        "truncated": truncated,
    }


def _compose_digest(workspace: Path, s: dict[str, Any], when: str) -> str:
    lines = [
        f"# Workspace digest — {workspace.name}",
        "",
        f"_Surveyed by Aura on {when}. Read-only; nothing was changed._",
        "",
        f"- **{s['total_files']:,} files** across **{s['total_dirs']:,} folders**, "
        f"**{_human_bytes(s['total_bytes'])}** total.",
    ]
    if s["largest_file"]:
        lines.append(f"- Largest file: `{s['largest_file']}` ({_human_bytes(s['largest_bytes'])}).")
    if s["by_extension"]:
        kinds = ", ".join(f"{ext} ×{n}" for ext, n in s["by_extension"].items())
        lines.append(f"- File kinds: {kinds}.")
    if s["recent_files"]:
        lines.append("- Most recently changed:")
        lines += [f"  - `{r}`" for r in s["recent_files"]]
    if s["truncated"]:
        lines.append(f"\n_(Survey stopped at the {_MAX_ENTRIES:,}-entry ceiling.)_")
    lines.append("")
    return "\n".join(lines)


def _append_run_ledger(receipt: WorkspaceDigestReceipt, ledger: Path) -> None:
    try:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        from core.governance_context import local_internal_governed_scope
        with local_internal_governed_scope("environment_pipeline.run_ledger", domain="file_write"):
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(receipt.to_dict(), default=str) + "\n")
    except OSError as exc:
        logger.debug(
            "%s unavailable (%s: %s); the run ledger line was not appended; the digest and the returned receipt are the proof",
            "local_internal_governed_scope",
            type(exc).__name__,
            exc,
        )


def run_workspace_digest(
    workspace: str | Path,
    *,
    safe_mode: bool | None = None,
    ledger_path: Path | None = None,
) -> WorkspaceDigestReceipt:
    """Survey ``workspace`` and write a plain-language digest (governed, safe-mode aware).

    Returns a :class:`WorkspaceDigestReceipt`. When safe mode is active the survey
    still runs but the digest is *not* written (``wrote_digest=False``), so the
    pipeline honors the user's safe-mode brake.
    """
    started = time.monotonic()
    root = Path(workspace)
    when = datetime.now(tz=UTC).isoformat(timespec="seconds")
    if safe_mode is None:
        safe_mode = bool(get_runtime_setting("safety.safe_mode", False))

    if not root.is_dir():
        return WorkspaceDigestReceipt(
            workspace=str(root), surveyed_at=when, total_files=0, total_dirs=0,
            total_bytes=0, by_extension={}, largest_file="", recent_files=[],
            digest_path="", wrote_digest=False, success=False,
            note="workspace is not a directory",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    s = survey_workspace(root)
    digest_path = root / _DIGEST_RELPATH
    receipt = WorkspaceDigestReceipt(
        workspace=str(root), surveyed_at=when,
        total_files=s["total_files"], total_dirs=s["total_dirs"], total_bytes=s["total_bytes"],
        by_extension=s["by_extension"], largest_file=s["largest_file"],
        recent_files=s["recent_files"], digest_path=str(digest_path),
        wrote_digest=False, success=True, truncated=s["truncated"],
    )

    if safe_mode:
        receipt.note = "safe mode is on — surveyed only, did not write the digest"
    else:
        try:
            from core.governance_context import local_internal_governed_scope
            with local_internal_governed_scope("environment_pipeline.workspace_digest", domain="file_write"):
                atomic_write_text(digest_path, _compose_digest(root, s, when))
            receipt.wrote_digest = True
            receipt.note = "wrote a fresh workspace digest"
        except OSError as exc:
            receipt.success = False
            receipt.note = f"survey ok but digest write failed: {exc}"

    receipt.duration_ms = int((time.monotonic() - started) * 1000)
    _append_run_ledger(receipt, ledger_path or _RUN_LEDGER)
    return receipt
