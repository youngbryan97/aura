"""What she keeps, written through the file gateway in a governed scope of its own.

Each gateway call is written out by name, so the effect inventory (tools/lint_governance.py) reads it as the gateway's. What
keeps a thing of hers writes it here, and adds no way of writing of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["governed"]


def governed(how: str, path: str | Path, *payload: Any, source: str = "rebuilding_a_program") -> Any:
    """One write of hers: ``write_text`` (text), ``write_bytes`` (bytes), ``append_text`` (text), ``ensure_directory``,
    ``delete_file``, ``copy_path`` or ``move_path`` (destination)."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope(f"rebuilding.{how}", domain="file_write", receipt_prefix="rebuilding",
                                       constraints={"artifact": "aura.rebuilt_program.v1", "operation": how}):
        if how in {"write_text", "write_bytes"}:
            held = str(payload[0]).encode("utf-8") if how == "write_text" else payload[0]
            return get_file_write_gateway().write_bytes(path, held, source=source)
        if how == "append_text":
            return get_file_write_gateway().append_text(path, *payload, source=source)
        if how == "ensure_directory":
            return get_file_write_gateway().ensure_directory(path, source=source)
        if how == "delete_file":
            return get_file_write_gateway().delete_file(path, source=source)
        if how == "copy_path":
            return get_file_write_gateway().copy_path(path, *payload, source=source)
        if how == "move_path":
            return get_file_write_gateway().move_path(path, *payload, source=source)
    raise ValueError(f"no governed write named {how!r}")
