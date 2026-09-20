"""An artifact must say whether the code that produced it is obtainable.

"The public repository is not the deployed code" cannot be answered by
asserting otherwise. `tools/reqproof/capture.py` has refused to capture from
an unpushed or dirty tree for a long time, and that refusal was reachable by
exactly one evidence producer. Everything else in `artifacts/` asserted
results about Aura while saying nothing about whether the code behind them had
ever left this machine.

The property these tests hold hardest: `unknown` must never read as
`published`. A provenance check that could not run is not a provenance check
that passed, and a missing git binary silently stamping artifacts as
reproducible is the exact failure this module exists to prevent.
"""

from __future__ import annotations

import subprocess

import pytest

from core.evaluation.source_attestation import SCHEMA, SourceAttestation, attest


def _attestation(**overrides) -> SourceAttestation:
    base = {
        "verdict": "published",
        "head_sha": "a" * 40,
        "upstream_ref": "origin/main",
        "upstream_sha": "a" * 40,
        "is_clean": True,
        "is_pushed": True,
        "unpushed_commits": 0,
        "dirty_files": (),
        "dirty_count": 0,
        "detail": "",
    }
    base.update(overrides)
    return SourceAttestation(**base)


def test_only_published_is_reproducible() -> None:
    assert _attestation(verdict="published").reproducible_from_public_source is True
    assert _attestation(verdict="divergent").reproducible_from_public_source is False
    assert _attestation(verdict="unknown").reproducible_from_public_source is False


def test_unknown_does_not_read_as_published() -> None:
    """The failure that would quietly undo the whole module."""
    payload = _attestation(verdict="unknown", is_clean=False, is_pushed=False).to_dict()

    assert payload["verdict"] == "unknown"
    assert payload["reproducible_from_public_source"] is False


def test_attest_on_a_real_repo_returns_a_usable_verdict() -> None:
    result = attest()

    assert result.verdict in {"published", "divergent", "unknown"}
    assert result.to_dict()["schema"] == SCHEMA
    # Whatever the verdict, it must justify itself: a bare "divergent" with no
    # detail leaves a reader unable to act on it.
    if result.verdict != "published":
        assert result.detail, "a non-published verdict must name what differs"


def test_dirty_tree_is_never_published(tmp_path) -> None:
    """A clean-tree claim from a dirty tree is the whole criticism, in one field."""
    if not _git_available():
        pytest.skip("git unavailable")

    _run(tmp_path, "init", "-q")
    _run(tmp_path, "config", "user.email", "t@example.com")
    _run(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _run(tmp_path, "add", "-A")
    _run(tmp_path, "commit", "-qm", "one")

    # No upstream at all, and now a dirty file on top.
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    result = attest(tmp_path)

    assert result.verdict == "divergent"
    assert result.is_clean is False
    assert result.dirty_count >= 1
    assert result.reproducible_from_public_source is False


def test_missing_upstream_is_divergent_not_published(tmp_path) -> None:
    """No upstream means a reader has nowhere to get the code from."""
    if not _git_available():
        pytest.skip("git unavailable")

    _run(tmp_path, "init", "-q")
    _run(tmp_path, "config", "user.email", "t@example.com")
    _run(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _run(tmp_path, "add", "-A")
    _run(tmp_path, "commit", "-qm", "one")

    result = attest(tmp_path)

    assert result.verdict == "divergent"
    assert result.is_clean is True, "tree is clean; the divergence is the missing upstream"
    assert "upstream" in result.detail.lower()


def test_non_repo_directory_is_unknown(tmp_path) -> None:
    """Not a repo is not a clean repo."""
    result = attest(tmp_path / "nowhere")

    assert result.verdict == "unknown"
    assert result.reproducible_from_public_source is False


def test_attest_never_raises(tmp_path) -> None:
    """A provenance statement that can crash its own run gets deleted from the run.

    Then artifacts carry no provenance at all, which is the state this module
    exists to end.
    """
    for target in (tmp_path, tmp_path / "missing", "/", ""):
        assert attest(target).verdict in {"published", "divergent", "unknown"}


def test_dirty_file_list_is_bounded() -> None:
    """An attestation is a statement, not a diff."""
    result = attest()

    assert len(result.dirty_files) <= 20


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _run(cwd, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *args], cwd=str(cwd), capture_output=True, timeout=30, check=False
    )
