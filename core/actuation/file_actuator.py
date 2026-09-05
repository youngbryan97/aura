"""core/actuation/file_actuator.py — File and Repository Actuator.

This is an actuation BOUNDARY: whatever it forwards becomes a real filesystem
write or a real git operation. It therefore validates and classifies here
rather than trusting the caller's arguments downstream.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from core.actuation.world_actuator import get_world_actuator
from core.runtime.state_ownership import state_root

#: Hard ceiling on a single actuated write. A boundary with no byte limit is
#: a disk-fill primitive (CP126 1fa38892).
MAX_WRITE_BYTES = 8 * 1024 * 1024

#: Repository actions that can destroy or publish work. CP126 c0ecefb0: only
#: publish_code and push were classified, so reset/force/branch-deletion/
#: remote and credential changes were actuated as ORDINARY risk — and an
#: unrecognized action name was treated as benign, which is the wrong default
#: for a boundary that runs git.
HIGH_RISK_REPO_ACTIONS = frozenset({
    "publish_code",
    "push",
    "force_push",
    "reset",
    "hard_reset",
    "revert",
    "rebase",
    "clean",
    "delete_branch",
    "branch_delete",
    "prune",
    "gc",
    "tag_delete",
    "remote_add",
    "remote_remove",
    "remote_set_url",
    "set_credentials",
    "config_set",
    "submodule_update",
    "checkout_force",
    "stash_drop",
    "filter_branch",
})

#: Actions this boundary considers read-only/benign. Anything not listed is
#: treated as HIGH RISK — unknown means unclassified, not safe.
LOW_RISK_REPO_ACTIONS = frozenset({
    "status",
    "diff",
    "log",
    "show",
    "branch_list",
    "fetch",
    "add",
    "commit",
    "checkout",
    "stash",
    "pull",
})


class FileActuationError(ValueError):
    """A file/repository actuation request was refused at the boundary."""


def _workspace_roots() -> list[Path]:
    """Directories this actuator may write inside.

    CP126 1fa38892: the request accepted an unrestricted path with no
    workspace capability. Roots come from the environment so tests and the
    live runtime can each declare their own, and the process CWD is never an
    implicit root.
    """
    raw = os.environ.get("AURA_FILE_ACTUATOR_ROOTS", "")
    roots: list[Path] = []
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        try:
            roots.append(Path(entry).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            continue
    if not roots:
        # Default capability: the Aura data/workspace tree and the system
        # temp dir (where tests and sandboxes legitimately write).
        import tempfile

        roots = [
            state_root().resolve(strict=False),
            Path(tempfile.gettempdir()).resolve(strict=False),
        ]
    return roots


def resolve_write_target(path: str) -> Path:
    """Resolve a write target inside an allowed root, or refuse.

    Symlinks are resolved BEFORE the containment check, so a link inside the
    workspace cannot redirect a write outside it.
    """
    raw = str(path or "").strip()
    if not raw:
        raise FileActuationError("write path is empty")
    try:
        target = Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FileActuationError(f"write path is unresolvable: {exc}") from exc
    for root in _workspace_roots():
        try:
            target.relative_to(root)
        except ValueError:
            continue
        return target
    raise FileActuationError(
        f"write path is outside every allowed workspace root: {target}"
    )


def _current_sha256(target: Path) -> str:
    try:
        if not target.is_file():
            return ""
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError:
        return ""


class FileActuator:
    """Wrapper for file writes and repository git operations."""

    @classmethod
    async def write_file(
        cls,
        path: str,
        content: str,
        source: str = "file_actuator",
        *,
        expected_sha256: str | None = None,
        allow_replace: bool = True,
    ) -> dict[str, Any]:
        """Actuate one bounded, contained file write.

        ``expected_sha256`` is a compare-and-swap precondition (CP126
        174481d8): the write proceeds only if the file's current contents hash
        to this value — pass ``""`` to require that the file does NOT exist.
        The pre-write hash is returned as rollback data so a caller can undo
        the change without re-reading a file another writer may have changed.
        """
        target = resolve_write_target(path)
        payload = content if isinstance(content, str) else str(content)
        encoded = payload.encode("utf-8", errors="surrogateescape")
        if len(encoded) > MAX_WRITE_BYTES:
            raise FileActuationError(
                f"write exceeds the {MAX_WRITE_BYTES} byte boundary limit "
                f"({len(encoded)} bytes)"
            )

        existed = target.is_file()
        previous_sha256 = _current_sha256(target)
        if not allow_replace and existed:
            raise FileActuationError(f"refusing to replace an existing file: {target}")
        if expected_sha256 is not None and previous_sha256 != expected_sha256:
            # Concurrent change: overwriting it would silently discard
            # somebody else's write.
            raise FileActuationError(
                "compare-and-swap failed: file changed since it was read "
                f"(expected {expected_sha256[:12] or '<absent>'}, "
                f"found {previous_sha256[:12] or '<absent>'})"
            )

        result = await get_world_actuator().actuate(
            category="local_files",
            action_name="write_file",
            params={
                "path": str(target),
                "text": payload,
                "bytes": len(encoded),
                "expected_sha256": expected_sha256,
                "previous_sha256": previous_sha256,
                "existed": existed,
                "new_sha256": hashlib.sha256(encoded).hexdigest(),
            },
            source=source,
        )
        if isinstance(result, dict):
            # Rollback data travels WITH the receipt: reversing the write must
            # not depend on re-reading a file that may have changed again.
            result.setdefault("previous_sha256", previous_sha256)
            result.setdefault("existed_before", existed)
        return result

    @classmethod
    def classify_repo_action(cls, action: str) -> bool:
        """True when this repository action must be treated as high risk."""
        name = str(action or "").strip().lower()
        if not name:
            return True
        if name in HIGH_RISK_REPO_ACTIONS:
            return True
        if name in LOW_RISK_REPO_ACTIONS:
            return False
        # Unknown action names are UNCLASSIFIED, and an unclassified git
        # operation is not evidence of safety.
        return True

    @classmethod
    async def modify_repo(
        cls,
        repo_path: str,
        action: str,
        params: dict[str, Any],
        source: str = "file_actuator",
    ) -> dict[str, Any]:
        """Actuate a repository operation with the risk of the action ACTUALLY run.

        CP126 b7894f2c: ``act_params`` spread the caller's params AFTER the
        action key, so ``params={"action": "push"}`` replaced the operation
        while ``high_risk_flag`` was still computed from the original benign
        argument — risk classification and the executed action could disagree.
        The action and repo path are now applied LAST and the flag is derived
        from the same value that is sent.
        """
        resolved_action = str(action or "").strip().lower()
        if not resolved_action:
            raise FileActuationError("repository action is empty")
        repo_target = resolve_write_target(repo_path)
        if not repo_target.is_dir():
            raise FileActuationError(f"repository path is not a directory: {repo_target}")
        # CP126 e0f2716e (partial): the path is at least CONTAINED and real.
        # Binding it to a registered repository identity (remote, branch,
        # ownership, clean tree) needs a repository registry this boundary
        # does not have — the containment check is the honest subset.
        if not (repo_target / ".git").exists():
            raise FileActuationError(f"not a git repository: {repo_target}")

        caller_params = dict(params or {})
        caller_params.pop("action", None)
        caller_params.pop("repo_path", None)
        act_params = {
            **caller_params,
            "repo_path": str(repo_target),
            "action": resolved_action,
        }
        return await get_world_actuator().actuate(
            category="code_repos",
            action_name="modify_repo",
            params=act_params,
            source=source,
            high_risk_flag=cls.classify_repo_action(resolved_action),
        )
