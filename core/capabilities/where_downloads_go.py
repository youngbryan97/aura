"""Where a file her browser downloads is kept: her own folder, or the one a task names.

A page that saves or exports a file hands it to the browser as a download.
Nothing kept them: Playwright removes a download's file when the browser
closes, so a document a program exported was gone by the time anybody looked.
Every download is now saved, under its own name, to her downloads folder, or
to the folder the task in hand names ("export it to my Desktop") for as long
as that task runs.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger("PhantomBrowser.Downloads")

__all__ = ["downloads_go_to", "keep_downloads", "kept_downloads"]

#: The folders a task has named, innermost last.
_NAMED: list[Path] = []

#: Every file kept this run, newest last.
_KEPT: list[Path] = []


def _default() -> Path:
    from core.runtime.state_ownership import state_root

    return state_root() / "downloads"


@contextlib.contextmanager
def downloads_go_to(folder: Path) -> Iterator[Path]:
    """While this is open, what her browser downloads is kept in ``folder``."""
    _NAMED.append(Path(folder))
    try:
        yield Path(folder)
    finally:
        _NAMED.remove(Path(folder))


def kept_downloads(since: int = 0) -> list[Path]:
    """The files kept from ``since`` (an index into everything kept this run) on."""
    return list(_KEPT[since:])


def keep_downloads(context: Any) -> None:
    """Keep every download from pages of this browser context."""
    context.on("download", lambda download: asyncio.ensure_future(_keep(download)))


async def _keep(download: Any) -> None:
    folder = _NAMED[-1] if _NAMED else _default()
    name = re.sub(r"[/\\:\x00]", "_", str(download.suggested_filename or "download")).strip(". ") or "download"
    try:
        await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
        target = await asyncio.to_thread(_free_name, folder / name)
        await download.save_as(str(target))
    except Exception as why:  # noqa: BLE001 - a download that could not be kept is said, not raised into the page
        logger.warning("a download (%s) could not be kept in %s: %s", name, folder, why)
        return
    _KEPT.append(target)
    logger.info("📥 kept %s in %s", target.name, folder)


def _free_name(wanted: Path) -> Path:
    """``wanted``, or the same name with a number, so nothing already there is written over."""
    if not wanted.exists():
        return wanted
    for n in range(2, 1000):
        other = wanted.with_name(f"{wanted.stem} ({n}){wanted.suffix}")
        if not other.exists():
            return other
    return wanted.with_name(f"{wanted.stem} (new){wanted.suffix}")

