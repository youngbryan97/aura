#!/usr/bin/env python3
"""tools/release_train.py — boring update, boring rollback.

The missing half of the release story. Manifest and provenance existed, but
updating the live checkout was `git pull` and hope: a dirty tree made the
pull fail silently mid-script (observed July 8 — the instance rebooted onto
the OLD tip while the operator believed it updated), and there was no recorded
way back. This tool makes both directions boring:

  update    fetch → record the rollback point → labeled-autostash any WIP →
            ff-only pull → compile sanity → history entry. Refuses loudly
            instead of half-succeeding; restores the stash if the pull fails.
  rollback  reset --hard to the last recorded good point (always an ancestor
            of origin/main, so the NEXT update still fast-forwards cleanly).
  status    where we are, what's dirty, what the history says.

Every action lands in artifacts/release/history.jsonl (machine-local ledger).
`--relaunch` hands off to launch_aura.sh --reboot after a successful move.

Usage:
  python tools/release_train.py status
  python tools/release_train.py update [--relaunch] [--smoke]
  python tools/release_train.py rollback [--to COMMIT] [--relaunch]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HISTORY_PATH = REPO_ROOT / "artifacts" / "release" / "history.jsonl"


def _default_runner(command: tuple[str, ...], timeout_s: float) -> Any:
    from core.tasks.managed_command import run_project_command

    return run_project_command(command, timeout_s=timeout_s)


class ReleaseTrain:
    """Update/rollback engine; command execution injected for tests."""

    def __init__(
        self,
        root: Path = REPO_ROOT,
        *,
        runner: Callable[[tuple[str, ...], float], Any] | None = None,
        history_path: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self._run = runner or _default_runner
        self.history_path = Path(history_path or HISTORY_PATH)

    # ── git plumbing ─────────────────────────────────────────────────────────

    def _git(self, *args: str, timeout_s: float = 120.0) -> Any:
        return self._run(("git", "-c", "core.fsmonitor=false", "-C", str(self.root), *args), timeout_s)

    def head(self) -> str:
        result = self._git("rev-parse", "HEAD")
        return result.stdout.strip() if getattr(result, "ok", False) else ""

    def dirty_files(self) -> list[str]:
        result = self._git("status", "--porcelain")
        if not getattr(result, "ok", False):
            return []
        return [line for line in result.stdout.splitlines() if line.strip()]

    def _is_ancestor(self, commit: str, of: str) -> bool:
        result = self._git("merge-base", "--is-ancestor", commit, of)
        return getattr(result, "returncode", 1) == 0

    # ── history ledger ───────────────────────────────────────────────────────

    def _record(self, entry: dict[str, Any]) -> None:
        entry = {"at": time.time(), "at_human": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    def history(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        entries = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries[-limit:]

    def last_rollback_point(self) -> str:
        for entry in reversed(self.history(limit=100)):
            if entry.get("action") == "update" and entry.get("ok") and entry.get("from"):
                return str(entry["from"])
        return ""

    # ── the stash, addressed by what it is, never by where it sits ──────────
    #
    # This checkout is shared with other sessions, and each of them may push
    # or pop the stash stack at any moment. `git stash pop` takes whatever is
    # on top — which after a parallel session's push is that session's work —
    # so an entry is found by its label, applied by its SHA, and dropped by
    # re-finding the label. And a successful update restores the WIP it set
    # aside: "autostashed" that stays stashed is uncommitted work made to
    # disappear.

    def _stash_sha(self, label: str) -> str:
        listed = self._git("stash", "list", "--format=%H %gs")
        for line in (listed.stdout or "").splitlines():
            sha, _space, subject = line.partition(" ")
            if label and label in subject:
                return sha
        return ""

    def _stash_ref(self, label: str) -> str:
        listed = self._git("stash", "list", "--format=%gd %gs")
        for line in (listed.stdout or "").splitlines():
            ref, _space, subject = line.partition(" ")
            if label and label in subject:
                return ref
        return ""

    def _restore_stash(self, label: str) -> bool:
        sha = self._stash_sha(label)
        if not sha:
            return False
        applied = self._git("stash", "apply", sha)
        if not getattr(applied, "ok", False):
            return False
        ref = self._stash_ref(label)
        if ref:
            self._git("stash", "drop", ref)
        return True

    # ── the boring paths ─────────────────────────────────────────────────────

    def update(self, *, smoke: bool = False) -> dict[str, Any]:
        fetch = self._git("fetch", "origin", "main", timeout_s=180.0)
        if not getattr(fetch, "ok", False):
            return {"ok": False, "step": "fetch", "error": (fetch.stderr or "fetch failed")[-400:]}

        current = self.head()
        target_result = self._git("rev-parse", "origin/main")
        target = target_result.stdout.strip() if getattr(target_result, "ok", False) else ""
        if not current or not target:
            return {"ok": False, "step": "resolve", "error": "could not resolve HEAD or origin/main"}
        if current == target:
            return {"ok": True, "step": "noop", "from": current, "to": target,
                    "detail": "already at origin/main"}

        stash_label = ""
        if self.dirty_files():
            stash_label = (
                f"release-train: WIP autostashed {time.strftime('%H:%M')} "
                f"before update {current[:8]}→{target[:8]}"
            )
            stashed = self._git("stash", "push", "-m", stash_label)
            if not getattr(stashed, "ok", False):
                return {"ok": False, "step": "stash",
                        "error": (stashed.stderr or "stash failed")[-400:]}

        pulled = self._git("pull", "--ff-only", "origin", "main", timeout_s=300.0)
        if not getattr(pulled, "ok", False):
            outcome: dict[str, Any] = {
                "ok": False, "step": "pull",
                "error": (pulled.stderr or pulled.stdout or "pull failed")[-400:],
            }
            if stash_label:
                outcome["stash_restored"] = self._restore_stash(stash_label)
                if not outcome["stash_restored"]:
                    outcome["stash_label"] = stash_label
            self._record({"action": "update", **outcome, "from": current, "to": target})
            return outcome

        compile_result = self._run(
            (sys.executable, "-m", "compileall", "-q", str(self.root / "core")), 300.0
        )
        compile_ok = bool(getattr(compile_result, "ok", False))

        smoke_ok: bool | None = None
        if smoke:
            smoke_result = self._run(
                (sys.executable, "-m", "pytest", "tests/test_response_contract.py", "-q"),
                600.0,
            )
            smoke_ok = bool(getattr(smoke_result, "ok", False))

        stash_restored: bool | None = None
        if stash_label:
            stash_restored = self._restore_stash(stash_label)
        outcome = {
            "ok": compile_ok and (smoke_ok is not False),
            "step": "complete", "from": current, "to": target,
            "compile_ok": compile_ok, "smoke_ok": smoke_ok,
            "stash_label": stash_label if not stash_restored else "",
            "stash_restored": stash_restored,
        }
        self._record({"action": "update", **outcome})
        return outcome

    def rollback(self, *, to: str = "") -> dict[str, Any]:
        target = to or self.last_rollback_point()
        if not target:
            return {"ok": False, "step": "resolve",
                    "error": "no rollback point recorded and none given (--to COMMIT)"}
        current = self.head()
        if current == target:
            return {"ok": True, "step": "noop", "from": current, "to": target,
                    "detail": "already at rollback point"}

        # Guard the invariant that keeps update boring FOREVER after: we only
        # ever roll back to an ancestor of origin/main, so the next update is
        # still a fast-forward. Anything else needs a human and a real merge.
        self._git("fetch", "origin", "main", timeout_s=180.0)
        if not self._is_ancestor(target, "origin/main"):
            return {"ok": False, "step": "guard",
                    "error": f"{target[:12]} is not an ancestor of origin/main; refusing "
                             "a rollback that would break future fast-forward updates"}

        stash_label = ""
        if self.dirty_files():
            stash_label = (
                f"release-train: WIP autostashed {time.strftime('%H:%M')} "
                f"before rollback {current[:8]}→{target[:8]}"
            )
            stashed = self._git("stash", "push", "-m", stash_label)
            if not getattr(stashed, "ok", False):
                return {"ok": False, "step": "stash",
                        "error": (stashed.stderr or "stash failed")[-400:]}

        reset = self._git("reset", "--hard", target)
        outcome = {
            "ok": bool(getattr(reset, "ok", False)),
            "step": "complete" if getattr(reset, "ok", False) else "reset",
            "from": current, "to": target, "stash_label": stash_label,
        }
        if not outcome["ok"]:
            outcome["error"] = (reset.stderr or "reset failed")[-400:]
        self._record({"action": "rollback", **outcome})
        return outcome

    def status(self) -> dict[str, Any]:
        head = self.head()
        self._git("fetch", "origin", "main", timeout_s=180.0)
        target_result = self._git("rev-parse", "origin/main")
        origin = target_result.stdout.strip() if getattr(target_result, "ok", False) else ""
        return {
            "head": head,
            "origin_main": origin,
            "up_to_date": bool(head and head == origin),
            "dirty_files": len(self.dirty_files()),
            "rollback_point": self.last_rollback_point(),
            "history_tail": self.history(limit=5),
        }


def _relaunch(root: Path) -> None:
    # The launcher self-detaches (nohup inside), so the managed runner returns
    # as soon as the boot handoff completes — no raw subprocess needed.
    print("[release-train] relaunching via launch_aura.sh --reboot ...")
    result = _default_runner((str(root / "launch_aura.sh"), "--reboot"), 300.0)
    if not getattr(result, "ok", False):
        print(f"[release-train] relaunch reported failure: {(result.stderr or '')[-300:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "update", "rollback"])
    parser.add_argument("--to", default="", help="rollback target commit (default: last update's 'from')")
    parser.add_argument("--smoke", action="store_true", help="run a smoke test after update")
    parser.add_argument("--relaunch", action="store_true", help="reboot the live instance on success")
    args = parser.parse_args()

    train = ReleaseTrain()
    if args.command == "status":
        print(json.dumps(train.status(), indent=2, sort_keys=True))
        return 0
    if args.command == "update":
        outcome = train.update(smoke=args.smoke)
    else:
        outcome = train.rollback(to=args.to)

    print(json.dumps(outcome, indent=2, sort_keys=True))
    if outcome.get("ok") and args.relaunch and outcome.get("step") != "noop":
        _relaunch(train.root)
    return 0 if outcome.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
