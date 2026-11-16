"""Private state writes with authority supplied by the caller."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from core.runtime.file_write_gateway import get_file_write_gateway


async def write_private_state_async(
    path: Path,
    payload: Any,
    *,
    source: str,
    scope_factory: Callable[[], AbstractContextManager[Any]],
    schema_version: int | None = None,
    schema_name: str | None = None,
    append: bool = False,
) -> None:
    """Create the private parent, then write text, a log entry or an envelope using scopes supplied by the caller."""
    if append and schema_version is not None:
        raise ValueError("a JSON envelope cannot be appended")
    if schema_version is None and not isinstance(payload, str):
        raise TypeError("a plain state payload must be text")
    target = Path(path)
    gateway = get_file_write_gateway()
    with scope_factory():
        await gateway.ensure_directory_async(target.parent, source=source)
    with scope_factory():
        if append:
            await gateway.append_text_async(target, payload, source=source)
        elif schema_version is None:
            await gateway.write_text_async(target, payload, source=source)
        else:
            await gateway.write_json_async(
                target, payload, schema_version=schema_version, schema_name=schema_name, source=source,
            )
