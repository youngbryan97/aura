"""Atomic text-file writes.

Write-then-rename in the same directory so readers never observe a partial
file. fsync before rename so the rename is durable across power loss / lid
close. Used by config persistence, journals, and the container seal.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, os.PathLike[str], Path]


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    data = text.encode(encoding)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, p)
