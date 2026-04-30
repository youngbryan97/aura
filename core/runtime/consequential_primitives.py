"""Receipt-enforced consequential primitives.

These are the canonical side-effect surfaces used by self-repair and autonomy
code.  Direct callers without an active Will/governance receipt fail closed via
`effect_sink`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.effect_boundary import effect_sink


@effect_sink("primitive.file_write", allowed_domains=("state_mutation", "file_write"))
def guarded_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_text(Path(path), text, encoding=encoding)


@effect_sink("primitive.shell_exec", allowed_domains=("tool_execution",))
def guarded_shell_exec(argv: list[str], *, cwd: str | Path | None = None, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, timeout=timeout, capture_output=True, text=True, check=False)


@effect_sink("primitive.memory_write", allowed_domains=("memory_write",))
def guarded_memory_write(writer: Any, payload: Mapping[str, Any]) -> Any:
    write = getattr(writer, "write", None) or getattr(writer, "add", None) or getattr(writer, "upsert", None)
    if not callable(write):
        raise TypeError("writer must expose write/add/upsert")
    return write(dict(payload))


@effect_sink("primitive.code_mutation", allowed_domains=("state_mutation", "file_write"))
def guarded_code_mutation(path: str | Path, source: str) -> None:
    compile(source, str(path), "exec")
    atomic_write_text(Path(path), source, encoding="utf-8")


@effect_sink("primitive.scar_formation", allowed_domains=("memory_write", "state_mutation"))
def guarded_scar_formation(system: Any, **kwargs: Any) -> Any:
    return system.form_scar(**kwargs)


@effect_sink("primitive.lora_training", allowed_domains=("state_mutation", "tool_execution"))
def guarded_lora_training(trainer: Any, *args: Any, **kwargs: Any) -> Any:
    train = getattr(trainer, "train", None) or getattr(trainer, "run", None)
    if not callable(train):
        raise TypeError("trainer must expose train/run")
    return train(*args, **kwargs)


@effect_sink("primitive.network_call", allowed_domains=("tool_execution",))
def guarded_network_call(client: Any, *args: Any, **kwargs: Any) -> Any:
    request = getattr(client, "request", None) or getattr(client, "get", None)
    if not callable(request):
        raise TypeError("client must expose request/get")
    return request(*args, **kwargs)


@effect_sink("primitive.hot_reload", allowed_domains=("state_mutation",))
def guarded_hot_reload(reloader: Any, module_name: str) -> Any:
    reload = getattr(reloader, "reload", None) or getattr(reloader, "hot_reload", None)
    if not callable(reload):
        raise TypeError("reloader must expose reload/hot_reload")
    return reload(module_name)


__all__ = [
    "guarded_write_text",
    "guarded_shell_exec",
    "guarded_memory_write",
    "guarded_code_mutation",
    "guarded_scar_formation",
    "guarded_lora_training",
    "guarded_network_call",
    "guarded_hot_reload",
]
