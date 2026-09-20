from __future__ import annotations

import subprocess
from pathlib import Path

from tools.closeout.audit_checkpoint_hygiene import audit


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    return result.stdout.strip()


def _checkpoint_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Checkpoint Test")
    _git(root, "config", "user.email", "checkpoint@example.invalid")
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "initial")
    _git(root, "remote", "add", "origin", root.as_uri())
    head = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/main", head)
    _git(root, "config", "branch.main.remote", "origin")
    _git(root, "config", "branch.main.merge", "refs/heads/main")
    return root


def test_clean_checkpoint_is_exactly_pushed_main(tmp_path: Path) -> None:
    report = audit(_checkpoint_repo(tmp_path))

    assert report["passed"] is True
    assert report["clean"] is True
    assert report["head_is_pushed_main"] is True
    assert report["upstream"] == "origin/main"
    assert report["issues"] == []


def test_dirty_tracked_change_is_an_unfinished_obligation(tmp_path: Path) -> None:
    root = _checkpoint_repo(tmp_path)
    (root / "tracked.txt").write_text("changed\n", encoding="utf-8")

    report = audit(root)

    assert report["passed"] is False
    assert any("tracked or untracked obligation" in issue for issue in report["issues"])


def test_untracked_file_is_an_unfinished_obligation(tmp_path: Path) -> None:
    root = _checkpoint_repo(tmp_path)
    (root / "untracked.txt").write_text("unfinished\n", encoding="utf-8")

    report = audit(root)

    assert report["passed"] is False
    assert any("tracked or untracked obligation" in issue for issue in report["issues"])


def test_local_only_commit_is_not_a_pushed_checkpoint(tmp_path: Path) -> None:
    root = _checkpoint_repo(tmp_path)
    (root / "tracked.txt").write_text("two\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "local only")

    report = audit(root)

    assert report["passed"] is False
    assert report["head_is_pushed_main"] is False
    assert any("ahead/behind=1\t0" in issue for issue in report["issues"])


def test_non_main_upstream_is_rejected(tmp_path: Path) -> None:
    root = _checkpoint_repo(tmp_path)
    head = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/review", head)
    _git(root, "config", "branch.main.merge", "refs/heads/review")

    report = audit(root)

    assert report["passed"] is False
    assert "checkpoint upstream is 'origin/review', expected 'origin/main'" in report["issues"]


def test_in_progress_git_operation_is_rejected(tmp_path: Path) -> None:
    root = _checkpoint_repo(tmp_path)
    git_dir = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-dir"))
    (git_dir / "MERGE_HEAD").write_text(_git(root, "rev-parse", "HEAD") + "\n")

    report = audit(root)

    assert report["passed"] is False
    assert report["active_git_markers"] == ["MERGE_HEAD"]
    assert any("git operation or lock is still active" in issue for issue in report["issues"])
