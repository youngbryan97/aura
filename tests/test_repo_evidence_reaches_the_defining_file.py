"""Repo evidence must be able to reach every file, and open the right one.

Found 2026-08-18. `_inprocess_search` — the path taken on any machine without
ripgrep, which includes this one — walked the tree and stopped after 4,000
files. The tree holds 6,299, and core/runtime/subprocess_gateway.py sits at
position 4,152. Asked where SubprocessGateway lives, the honest answer was
incidental mentions in aura_main.py and interface/server.py, because the file
named after the symbol was never opened. A third of the repository was
unreachable as evidence, decided by directory-walk order, for any question.

The scan cap was not the defect; spending it in walk order was. Listing paths
is cheap and reading them is not, so the order is chosen first and the cap
bounds reads. Pruning skipped directories during the walk rather than after it
took the listing from 1,907ms to 91ms over the same 6,299 files.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest

from core.brain.evidence_provider import (
    _SKIP_DIRS,
    EvidenceProvider,
    _filename_candidates,
    _salient_terms,
    snake_case,
)


@pytest.fixture
def provider() -> EvidenceProvider:
    return EvidenceProvider(memory_facade=None)


def _order(provider: EvidenceProvider, objective: str):
    terms = _salient_terms(objective)
    # The production computation, not a re-derivation of it.
    snake = {t: snake_case(t) for t in terms}
    return provider._search_order(terms, snake, _filename_candidates(objective))


def test_the_defining_file_is_read_before_the_budget_runs_out(provider) -> None:
    """It sat 152 files past the cap, so it was never opened at all."""
    order = _order(provider, "explain how SubprocessGateway routes effects")
    names = [p.name for p in order[:200]]

    assert "subprocess_gateway.py" in names


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("explain how SubprocessGateway routes effects", "subprocess_gateway.py"),
        ("how does the file write gateway work", "file_write_gateway.py"),
        ("how does the health contract work", "health_contract.py"),
        ("how does lockdep detect deadlocks", "lockdep.py"),
    ],
)
def test_the_named_file_is_first_in_the_read_order(
    provider, objective: str, expected: str
) -> None:
    order = _order(provider, objective)

    assert order, f"no files ordered for {objective!r}"
    assert order[0].name == expected, f"{objective!r} -> {order[0]}"


def test_a_filename_can_be_rebuilt_from_the_words_asked(provider) -> None:
    """"file" is dropped as a common term, so the name needs the raw words."""
    candidates = _filename_candidates("how does the file write gateway work")

    assert "file_write_gateway" in candidates


def test_the_answer_is_the_file_named_after_the_subject(provider) -> None:
    spans = asyncio.run(
        provider.gather(
            "explain how SubprocessGateway routes through effect governance",
            task_type="repo_audit",
            limit=6,
        )
    )
    repo = [s for s in spans if s.source == "repo"]

    assert repo, "no repo evidence at all"
    assert "subprocess_gateway" in repo[0].ref.lower(), [s.ref for s in repo]


def test_self_modification_snapshots_are_not_citable() -> None:
    """.aura_architect holds candidate/ and original/ copies of real modules.

    Citing one cites a version of herself that was considered and rejected, at
    a line number that looks authentic.
    """
    assert ".aura_architect" in _SKIP_DIRS


def test_skipped_directories_are_never_listed(provider) -> None:
    """Judged RELATIVE to the root being walked.

    The provider prunes directory NAMES while walking, so the question is
    whether the walk descended into one. Testing the absolute path asks a
    different question, and it answers wrong whenever the checkout itself
    lives under one of those names — a worktree under .claude/worktrees, which
    is where CONTRIBUTING tells an agent to work. Every one of 6,762 results
    then looked like an offender.
    """
    order = _order(provider, "explain how SubprocessGateway routes effects")
    root = pathlib.Path(provider._root).resolve()

    offenders = [
        p
        for p in order
        if set(pathlib.Path(p).resolve().relative_to(root).parts) & _SKIP_DIRS
    ]

    assert not offenders, f"listed {len(offenders)} paths under skipped dirs"
