"""Persist an external repair and independently witness its file effect."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path


async def save_repair(path: Path, original: str, repaired: str) -> str:
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.conversation.surface_disposition import record_tool_receipt

    original_bytes, repaired_bytes = original.encode("utf-8"), repaired.encode("utf-8")
    if await asyncio.to_thread(path.read_bytes) != original_bytes:
        raise ValueError("the program changed during verification; refusing to overwrite newer work")
    digest = hashlib.sha256(repaired_bytes).hexdigest()
    backup = path.with_name(path.name + ".before-repair")
    if await asyncio.to_thread(backup.exists) and await asyncio.to_thread(backup.read_bytes) != original_bytes:
        backup = path.with_name(path.name + ".before-repair-" + hashlib.sha256(original_bytes).hexdigest()[:12])
    gateway = get_file_write_gateway()
    await gateway.write_text_async(backup, original, source="saving_a_verified_repair")
    if await asyncio.to_thread(backup.read_bytes) != original_bytes:
        raise RuntimeError("the repair backup did not match the original")
    if await asyncio.to_thread(path.read_bytes) != original_bytes:
        raise ValueError("the program changed before the repair was saved")
    await gateway.write_text_async(path, repaired, source="saving_a_verified_repair")
    observed = await asyncio.to_thread(path.read_bytes) == repaired_bytes
    record_tool_receipt("artifact_repair", ok=observed, action="write", object_ref=str(path),
                        effect_observed=observed, verification="file readback",
                        evidence=f"sha256={digest}; backup={backup}",
                        observed_content="The repaired source was read back from the named file." if observed else "File readback differed.")
    if not observed:
        raise RuntimeError("the saved repair did not match its verified source")
    return str(backup)
