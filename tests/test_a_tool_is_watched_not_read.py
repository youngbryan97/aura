"""A tool of this machine's own is read from its manual and run on inputs drawn from it, in a sandbox that refuses every
file write and the network; what it printed becomes the checks of her version. Its code is never read, and a tool
that runs other code or removes files is never run."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from core.rebuilding.what_it_was_seen_to_do import (
    Way,
    as_calls,
    seen,
    the_original_here,
    ways_to_try,
)

pytestmark = [pytest.mark.unit, pytest.mark.skipif(sys.platform != "darwin", reason="the macOS sandbox")]


def test_only_this_systems_own_tools_are_originals_and_none_that_run_code():
    assert the_original_here("wc") == "/usr/bin/wc" and the_original_here("Wc (Unix)") == "/usr/bin/wc"
    for name in ("python3", "rm -rf", "bash", "sudo", "curl", "Microsoft Paint"):
        assert the_original_here(name) is None, name
    with pytest.raises(ValueError):
        seen("/bin/rm", [Way(args=["-rf", "x"])])


def test_what_the_tool_prints_is_seen_and_becomes_a_check():
    runs = seen("/usr/bin/wc", [Way(args=["-l"], stdin="a\nb\nc\n"), Way(args=["-w"], stdin="one two three\n")])
    assert [r.out.split() for r in runs] == [["3"], ["3"]] and all(r.code == 0 for r in runs)
    calls = as_calls(runs, "behaves as the original does")
    assert calls[0].function == "main" and calls[0].args == [["-l"], "a\nb\nc\n"] and calls[0].value == runs[0].out


def test_a_tool_cannot_write_a_file_in_the_sandbox(tmp_path):
    target = tmp_path / "written.txt"
    runs = seen("/usr/bin/tee", [Way(args=[str(target)], stdin="x")])
    assert not target.exists() and runs[0].code != 0


def test_ways_that_name_a_file_are_never_tried():
    async def ask(prompt, schema, _tokens):
        assert "Never name a file" in prompt and "word, line" in prompt
        return schema(ways=[Way(args=["-l"], stdin="a\n"), Way(args=["/etc/passwd"]), Way(args=["../x"]),
                            Way(args=["~/notes"])])

    ways = asyncio.run(ways_to_try("wc", "wc – word, line, character, and byte count", ask))
    assert [w.args for w in ways] == [["-l"]]


def test_a_rebuilt_tool_is_checked_by_what_the_original_printed():
    from core.rebuilding.rebuilding_a_program import AS_THE_ORIGINAL, _as_seen
    from core.rebuilding.what_a_program_does import Feature, Genome, Source

    async def ask(prompt, schema, _tokens):
        return schema(ways=[Way(args=["-c"], stdin="hello\n"), Way(args=["-l"], stdin="1\n2\n")])

    genome = Genome(name="Counter", kind="code", features=[Feature(name="count lines")])
    manual = [Source("wc manual", "wc – word, line, character, and byte count", "its manual page on this machine")]
    said: list[str] = []

    async def tell(line: str) -> None:
        said.append(line)

    calls = asyncio.run(_as_seen("wc", genome, manual, ask, tell))
    assert [c.value.split() for c in calls] == [["6"], ["2"]]
    assert genome.features[-1].name == AS_THE_ORIGINAL and "refuses every file write" in said[0]
    assert Path("/usr/bin/wc").exists()
