"""The bytes are on this disk, so the evidence is not on the network.

LIVE, 2026-08-27: "ok this is driving me nuts. /private/tmp/claude-501/…/
invoice-tools — clean run, nothing raises, but invoice two comes out holding
invoice one's lines. what's the actual cause, and what do I change?"

The turn ran a web search for that whole sentence, path included, and came back
with GitHub issues about disk usage under /private/tmp/claude-501. The search
then discarded all five results as irrelevant — correctly — having already
spent the turn's tool budget on a question whose evidence was sitting in a
directory the person had named.

The guard for this existed and said so: "A file on this disk is not a live-
search question." It recognised a named FILE. "Debug the project at <path>"
names a folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interface.routes.chat import _should_collect_desktop_required_search_evidence


def test_a_named_directory_settles_it(tmp_path: Path) -> None:
    (tmp_path / "lib.py").write_text("def f():\n    return 1\n")
    asked = (
        f"ok this is driving me nuts. {tmp_path} — clean run, nothing raises, but "
        "invoice two comes out holding invoice one's lines. what's the actual "
        "cause, and what do I change?"
    )
    should, _query, _contract = _should_collect_desktop_required_search_evidence(asked)
    assert should is False, "a directory on this disk sent the turn to a web search"


def test_a_named_file_still_settles_it(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("the first rule\n")
    should, _query, _contract = _should_collect_desktop_required_search_evidence(
        f"read {target} and tell me the first rule it states"
    )
    assert should is False


def test_a_path_that_does_not_exist_does_not_settle_it() -> None:
    """Only bytes that are actually here make the evidence local."""
    should, _query, _contract = _should_collect_desktop_required_search_evidence(
        "who founded Hugging Face and what do they sell? /nope/not/here — give me links"
    )
    assert should is True


@pytest.mark.parametrize(
    "asked",
    [
        "who founded Hugging Face, and what do they actually sell? give me links",
        "what can you tell me about Anthropic the company",
    ],
)
def test_a_real_web_question_still_searches(asked: str) -> None:
    should, query, _contract = _should_collect_desktop_required_search_evidence(asked)
    assert should is True
    assert query
