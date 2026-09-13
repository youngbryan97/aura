"""Where a soak writes, and how it opens its receipts.

Three soak harnesses had the same two lines each — resolve an output directory,
open a receipts file and stream JSON lines into it. Three copies of a file
mutation is three call sites somebody has to govern for one behaviour, and the
effect-ownership ledger counts them that way: six entries of migration debt for
what is one place a soak writes.

Here once, so the ledger carries one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import IO


def resolve_output_dir(raw_path: str) -> Path:
    """The directory a soak writes into, made if it is not there."""
    out_dir = Path(raw_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def append_line(path: Path, line: str) -> None:
    """One line onto the end of a file a soak is streaming into."""
    with open_receipts(Path(path), mode="a") as handle:
        handle.write(line)


def append_row(path: Path, row: list[object]) -> None:
    """One CSV row onto the end of a resource log."""
    import csv

    with open_receipts(Path(path), mode="a") as handle:
        csv.writer(handle).writerow(row)


@contextlib.contextmanager
def open_receipts(path: Path, *, mode: str = "w") -> Iterator[IO[str]]:
    """The receipts file, opened for streaming and closed afterwards.

    A soak writes a line per tick and may run for hours, so the file is held
    open rather than reopened; the context manager is what guarantees the last
    line is on disk when the run ends, however it ends.
    """
    handle = Path(path).open(mode, encoding="utf-8")
    try:
        yield handle
    finally:
        handle.close()
