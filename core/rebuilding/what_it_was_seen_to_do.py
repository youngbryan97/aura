"""What a program on this machine was seen to do: its manual read, and it run on inputs drawn from that, never its code.

A clean-room rebuild is about the code, not about using the program: people who rebuild a tool run the original
and write down what it does (differential testing, McKeeman 1998). Where the program asked for is a tool of this
machine's own system (``wc``, ``sort``, ``cal``), its manual page is the deepest account of it there is, and running it
is the oracle: each run's output becomes a check her version must pass, ``main(args, stdin)`` giving what the
original printed.

The original is run only from the system's own folders, under the macOS sandbox with every file write and the
network refused, for a few seconds at most; what is tried is proposed by her model from the manual and never names
a file. Nothing here knows any tool.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("Rebuilding.Seen")

__all__ = ["Seen", "Way", "as_calls", "its_manual", "seen", "the_original_here", "ways_to_try"]

#: Where a tool may be run from: the system's own, never one a person or a program put there.
SYSTEM = ("/bin", "/usr/bin", "/sbin", "/usr/sbin")
#: Tools that run other code, act as someone else or reach other machines: never an original to run, sandbox or not.
_NOT_TO_RUN = re.compile(r"^(?:[a-z]*sh|python[\d.]*|perl[\d.]*|ruby|php|osascript|swift|node|tclsh[\d.]*|lua|awk|"
                         r"env|xargs|nohup|exec|sudo|su|login|doas|launchctl|open|ssh|scp|sftp|curl|wget|nc|ftp|telnet|"
                         r"rsync|osacompile|security|defaults|kill(?:all)?|shutdown|reboot|halt|dd|rm|mv|chmod|chown)$")
#: How long one run may take, and how much of its output is kept, in seconds and characters.
RUN_S = 5.0
OUTPUT_CHARS = 4000
MANUAL_CHARS = 24_000
MOST_WAYS = 24


class Way(BaseModel):
    args: list[str] = Field(default_factory=list, description="the arguments, none of them a file")
    stdin: str = Field(default="", max_length=2000, description="what is given it to read")
    shows: str = Field(default="", max_length=200, description="which part of the manual it tries")


class _Ways(BaseModel):
    ways: list[Way] = Field(default_factory=list)


@dataclass(frozen=True)
class Seen:
    way: Way
    out: str
    code: int


def the_original_here(program: str) -> str | None:
    """The system tool a program's name names, where this machine has one in its own folders."""
    names = dict.fromkeys(n for n in (program.strip(), re.sub(r"\s*\(.*\)$", "", program).strip(),
                                      program.strip().split()[0] if program.strip() else "") if n)
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9][\w.+-]{0,40}", name):
            continue
        found = shutil.which(name.lower(), path=os.pathsep.join(SYSTEM))
        if (found and Path(found).resolve().parent.as_posix() in {str(Path(d).resolve()) for d in SYSTEM}
                and not _NOT_TO_RUN.match(Path(found).name)):
            return found
    return None


def its_manual(command: str) -> str:
    """The tool's manual page as plain text, "" where it has none."""
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    name = Path(command).name
    try:
        shown = get_subprocess_gateway().run(["man", name], env={**os.environ, "MANPAGER": "cat", "MANWIDTH": "100"},
                                             timeout=15.0, source="rebuilding.its_manual", stdin_devnull=True,
                                             accelerator_capability="none")
    except (OSError, TimeoutError) as why:
        logger.info("rebuilding: no manual read for %s: %s", name, why)
        return ""
    text = re.sub(r".\x08", "", str(shown.stdout or ""))  # the overstrikes a terminal makes bold
    return text[:MANUAL_CHARS] if shown.returncode == 0 else ""


async def ways_to_try(program: str, manual: str, ask: Any, *, at_most: int = MOST_WAYS) -> list[Way]:
    """What her model proposes to run the tool with, to see each thing its manual says it does."""
    prompt = (f"The command-line tool {program}. Its manual:\n{manual[:MANUAL_CHARS]}\n\n"
              f"Give at most {at_most} ways to run it that together show what it does: each the arguments, and the "
              "text given it to read on standard input.")
    said = await ask(prompt, _Ways, 2048)
    ways = []
    for way in getattr(said, "ways", []) or []:
        # What it reads is the text given: a way that names a file or a path is not run.
        if any("/" in a or a.startswith("~") or ".." in a for a in way.args):
            continue
        ways.append(way)
    return ways[:at_most]


def _profile(command: str) -> str:
    return "\n".join([
        "(version 1)", "(deny default)",
        f'(allow process-exec (literal "{command}"))', "(allow process-fork)", "(allow sysctl-read)",
        "(allow mach-lookup)", "(allow signal (target self))",
        '(allow file-read* (subpath "/System") (subpath "/usr") (subpath "/bin") (subpath "/sbin") (subpath "/Library")'
        ' (subpath "/private/var/db") (subpath "/dev") (literal "/") (literal "/private") (literal "/private/etc/localtime"))',
        '(allow file-write* (literal "/dev/null"))',
        "(deny network*)",
    ])


def seen(command: str, ways: list[Way]) -> list[Seen]:
    """Each way run under the sandbox: what it printed and how it exited; a run past ``RUN_S`` is not seen."""
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    if (Path(command).resolve().parent.as_posix() not in {str(Path(d).resolve()) for d in SYSTEM}
            or _NOT_TO_RUN.match(Path(command).name)):
        raise ValueError(f"{command} is not a tool of this system's own that may be run")
    found: list[Seen] = []
    with tempfile.TemporaryDirectory(prefix="aura-seen-") as scratch:
        profile = Path(scratch) / "profile.sb"
        profile.write_text(_profile(command))
        for way in ways:
            try:
                run = get_subprocess_gateway().run(["sandbox-exec", "-f", str(profile), command, *way.args], input=way.stdin,
                                                   timeout=RUN_S, cwd=scratch, env={"PATH": "/usr/bin:/bin", "LANG": "C"},
                                                   source="rebuilding.seen", accelerator_capability="none")
            except TimeoutError:
                continue
            found.append(Seen(way, str(run.stdout or "")[:OUTPUT_CHARS], int(run.returncode)))
    return found


def as_calls(seen_runs: list[Seen], feature: str = "what the original was seen to do") -> list[Any]:
    """Each run as a check her version must pass: ``main(args, stdin)`` gives what the original printed."""
    from core.rebuilding.programs_in_code import CodeCall

    return [CodeCall(feature=feature, function="main", args=[run.way.args, run.way.stdin], expect="equals", value=run.out)
            for run in seen_runs if run.code == 0]
