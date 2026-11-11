"""Her build's writes, through the file gateway in a governed scope of their own (Master Todo Q06).

The pages she builds, the records beside them, their folders and her scratch files are hers to keep, in her state.
Each gateway call is written out by name, so the effect inventory (tools/lint_governance.py) reads it as the
gateway's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def governed(how: str, path: str | Path, *payload: Any, source: str = "rebuilding_a_program") -> Any:
    """One write of her build's: ``write_text`` (text), ``append_text`` (text), ``ensure_directory``,
    ``delete_file`` or ``copy_path`` (destination)."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope(f"rebuilding.{how}", domain="file_write", receipt_prefix="rebuilding",
                                       constraints={"artifact": "aura.rebuilt_program.v1", "operation": how}):
        if how == "write_text":
            return get_file_write_gateway().write_text(path, *payload, source=source)
        if how == "append_text":
            return get_file_write_gateway().append_text(path, *payload, source=source)
        if how == "ensure_directory":
            return get_file_write_gateway().ensure_directory(path, source=source)
        if how == "delete_file":
            return get_file_write_gateway().delete_file(path, source=source)
        if how == "copy_path":
            return get_file_write_gateway().copy_path(path, *payload, source=source)
    raise ValueError(f"no governed write named {how!r}")


def a_folder_for(path: Path, *, source: str = "rebuilding_a_program") -> None:
    """Make the folder ``path`` goes in when it is not there. A folder that is there is left as it is: the gateway
    makes a folder private, and a page may be written beside other files, in a folder that is not hers alone."""
    if not path.parent.is_dir():
        governed("ensure_directory", path.parent, source=source)
