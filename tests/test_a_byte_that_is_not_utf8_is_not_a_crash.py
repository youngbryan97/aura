"""A tool's output with a byte that is not UTF-8 reaches its caller as text.

LIVE 15 November 2026: her find_in_source skill crashed twice in two minutes, "'utf-8' codec can't decode byte 0x92
in position 81584585": a cp1252 apostrophe in a tracked text file her git grep matched. The gateway's async path
already replaced such a byte; the path a read-only run takes decoded strictly.
"""
from __future__ import annotations

import sys

from core.runtime.subprocess_gateway import get_subprocess_gateway

_PRINTS_A_CP1252_APOSTROPHE = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'it\\x92s here\\n')"]


def test_a_read_only_run_reads_a_stray_byte_as_a_replacement():
    done = get_subprocess_gateway().run(_PRINTS_A_CP1252_APOSTROPHE, read_only=True, source="tests.not_utf8",
                                        timeout=30.0, accelerator_capability="none")
    assert done.returncode == 0
    assert done.stdout == "it�s here\n"


def test_bytes_are_still_bytes_for_a_caller_that_asks_for_them():
    done = get_subprocess_gateway().run(_PRINTS_A_CP1252_APOSTROPHE, read_only=True, source="tests.not_utf8",
                                        timeout=30.0, text=False, accelerator_capability="none")
    assert done.stdout == b"it\x92s here\n"
