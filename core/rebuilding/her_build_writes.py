"""Her build's writes, through the file gateway in a governed scope of their own (Master Todo Q06).

The pages she builds, the records beside them, their folders and her scratch files are hers to keep, in her state
(core/runtime/her_writes.py).
"""

from __future__ import annotations

from pathlib import Path

from core.runtime.her_writes import governed

__all__ = ["a_folder_for", "governed"]


def a_folder_for(path: Path, *, source: str = "rebuilding_a_program") -> None:
    """Make the folder ``path`` goes in when it is not there. A folder that is there is left as it is: the gateway
    makes a folder private, and a page may be written beside other files, in a folder that is not hers alone."""
    if not path.parent.is_dir():
        governed("ensure_directory", path.parent, source=source)
