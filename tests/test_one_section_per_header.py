"""Two subsystems, one kind of block, and the prompt carried both.

Measured live on 2026-08-28: a 46,996-character system message contained
"## GOAL EXECUTION STATE" twice, at 1,966 and 1,078 characters. Neither copy was
wrong — they were assembled from different objectives by different callers — and
the second cost a thousand characters of a prompt the resident model reads at
about twelve tokens a second.
"""

from __future__ import annotations


def _assembled(blocks: list[str]) -> list[str]:
    """The dedup rule, exercised the way the assembler applies it."""

    seen: set[str] = set()
    kept: list[str] = []
    for block in blocks:
        header = str(block or "").lstrip().split("\n", 1)[0].strip()
        if header and header in seen:
            continue
        if header:
            seen.add(header)
        kept.append(block)
    return kept


def test_a_repeated_header_is_dropped() -> None:
    blocks = [
        "## GOAL EXECUTION STATE\nImmediate execution: finish the ledger.",
        "## LIVE MIND CONTEXT\nSomething else entirely.",
        "## GOAL EXECUTION STATE\nA second, differently filtered copy.",
    ]
    kept = _assembled(blocks)
    assert len(kept) == 2
    assert kept[0].startswith("## GOAL EXECUTION STATE")
    assert "second, differently filtered" not in "\n".join(kept)


def test_the_first_wins() -> None:
    """Priority blocks are assembled first and chosen for this objective."""

    blocks = [
        "## GOAL EXECUTION STATE\nchosen for this objective",
        "## GOAL EXECUTION STATE\nthe generic one",
    ]
    assert _assembled(blocks) == [blocks[0]]


def test_different_headers_all_survive() -> None:
    blocks = [
        "## GOAL EXECUTION STATE\na",
        "## LIVE MIND CONTEXT\nb",
        "[WORLD MODEL BELIEFS]\nc",
    ]
    assert len(_assembled(blocks)) == 3


def test_a_block_with_no_header_is_never_deduplicated() -> None:
    """Two headerless blocks are two different things, not one repeated."""

    blocks = ["just some text", "some other text", ""]
    assert len(_assembled(blocks)) == 3


def test_the_assembler_really_applies_it() -> None:
    from pathlib import Path

    body = Path("core/runtime/conversation_support.py").read_text()
    assert "Dropped a repeated context section" in body
    assert "seen: set[str] = set()" in body
