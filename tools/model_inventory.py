#!/usr/bin/env python3
"""Every model on this machine, and every model the runtime names, side by side.

Q01 asks for the inventory and for the check behind it: that each live role
resolves to an artifact that is actually here, with the identity, tokenizer,
geometry, context window and declared allocation its consumers read; and that
whatever sits on disk unnamed by any code is listed with its size, because a
15GB checkpoint nothing can reach is not "cache", it is the disk.

    python tools/model_inventory.py            # the table
    python tools/model_inventory.py --json     # the record
    python tools/model_inventory.py --check    # exit 1 on a live role that does not resolve
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS_DIR = Path(os.environ.get("AURA_MODELS_DIR", "~/.aura/models")).expanduser()
HF_HUB = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()

#: Where code names a model. Anything else importing a name is a doc.
_CODE_ROOTS = ("core", "interface", "aura_main.py", "config")


def _du_bytes(path: Path, *, follow: bool = False) -> int:
    """Bytes actually held.

    A hub cache directory holds the bytes once, in blobs/, and every snapshot
    is symlinks into it — counting through the links reported every checkpoint
    twice. A resolved snapshot is the other way round: its files ARE the links,
    each to a distinct blob, so a role's size follows them.
    """
    total = 0
    if path.is_file():
        return (path.stat() if follow else path.lstat()).st_size
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                entry = Path(root) / name
                total += (entry.stat() if follow else entry.lstat()).st_size
            except OSError:
                continue
    return total


def _hf_snapshot(repo_id: str) -> Path | None:
    """The resolved snapshot directory for a hub repo, if it is cached."""
    folder = HF_HUB / ("models--" + repo_id.replace("/", "--"))
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _config(path: Path) -> dict[str, Any]:
    for name in ("config.json",):
        candidate = path / name
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
    return {}


def _geometry(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    out: dict[str, Any] = {}
    for key in ("architectures", "model_type", "num_hidden_layers", "hidden_size",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
                "sliding_window", "vocab_size"):
        if key in text:
            out[key] = text[key]
        elif key in config:
            out[key] = config[key]
    quant = config.get("quantization") or {}
    if isinstance(quant, dict) and quant.get("bits"):
        out["quantization_bits"] = quant.get("bits")
    return out


def _tokenizer_identity(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    template = path / "chat_template.jinja"
    if template.is_file():
        out["chat_template_sha256"] = hashlib.sha256(template.read_bytes()).hexdigest()[:16]
    tok = path / "tokenizer.json"
    if tok.is_file():
        out["tokenizer_sha256"] = hashlib.sha256(tok.read_bytes()).hexdigest()[:16]
    cfg = path / "tokenizer_config.json"
    if cfg.is_file():
        try:
            payload = json.loads(cfg.read_text(encoding="utf-8"))
            out["tokenizer_class"] = payload.get("tokenizer_class")
            if "chat_template" in payload and "chat_template_sha256" not in out:
                out["chat_template_sha256"] = hashlib.sha256(
                    str(payload["chat_template"]).encode()
                ).hexdigest()[:16]
        except (OSError, ValueError):
            pass
    return out


def _manifest_mentions(needle: str) -> int:
    """The fused-model manifest names its base; that is a live dependency."""
    manifest = REPO_ROOT / "training" / "fused-model" / "active.json"
    try:
        return int(needle in manifest.read_text(encoding="utf-8"))
    except OSError:
        return 0


def _code_mentions(needle: str) -> int:
    """How many production files name this model. `grep -rl`, bounded to code."""
    try:
        done = subprocess.run(
            ["grep", "-rlF", needle, *_CODE_ROOTS],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    return len([line for line in done.stdout.splitlines() if not line.endswith(".pyc")])


def live_roles() -> list[dict[str, Any]]:
    """The roles the runtime serves, resolved the way the runtime resolves them."""
    from core.brain.llm import model_registry as registry

    roles: list[dict[str, Any]] = []

    def add(
        role: str, name: str, path: str, *, consumer: str, optional: bool = False,
        language_model: bool = True,
    ) -> None:
        resolved = Path(str(path)).expanduser()
        if not resolved.exists() and re.fullmatch(r"[\w.-]+/[\w.-]+", str(path)):
            snapshot = _hf_snapshot(str(path))
            resolved = snapshot if snapshot is not None else resolved
        exists = resolved.exists()
        row: dict[str, Any] = {
            "role": role,
            "name": name,
            "path": str(resolved),
            "exists": exists,
            "optional": optional,
            "consumer": consumer,
        }
        if exists and resolved.is_dir():
            config = _config(resolved)
            row["size_gb"] = round(_du_bytes(resolved, follow=True) / 1e9, 2)
            row["geometry"] = _geometry(config)
            row["tokenizer"] = _tokenizer_identity(resolved)
        if exists and resolved.is_dir() and language_model:
            # Only a language model has a context window or an MLX worker
            # footprint. Asking an ASR checkpoint for one records a live
            # degradation about a context budget nothing will ever size.
            try:
                evidence = registry.get_context_window_evidence(str(resolved))
                row["context_window"] = {
                    "tokens": int(evidence.tokens),
                    "source": str(getattr(evidence.source, "value", evidence.source)),
                    "detail": str(evidence.detail),
                }
            except Exception as exc:  # noqa: BLE001 — an inventory reports, it does not stop
                row["context_window"] = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                from core.brain.llm.mlx_client import _declared_mlx_worker_footprint_gb

                row["declared_footprint_gb"] = round(_declared_mlx_worker_footprint_gb(str(resolved)), 2)
            except Exception as exc:  # noqa: BLE001
                row["declared_footprint_gb"] = f"{type(exc).__name__}: {exc}"
        roles.append(row)

    add("cortex", registry.CORTEX_LOGICAL_NAME, registry.get_model_path(registry.CORTEX_LOGICAL_NAME),
        consumer="core/brain/llm/mlx_client.py (resident worker)")
    add("brainstem", registry.BRAINSTEM_MODEL, registry.get_model_path(registry.BRAINSTEM_MODEL),
        consumer="core/brain/llm/mlx_client.py (fallback lane)")
    fallback = registry._FLAG_FALLBACK_MODEL.value()
    add("reflex", str(fallback), registry.get_model_path(str(fallback)),
        consumer="core/brain/llm/mlx_client.py (reflex lane)")
    deep = registry._FLAG_DEEP_MODEL.value()
    if deep:
        add("deep", str(deep), registry.get_model_path(str(deep)), consumer="deep solver lane", optional=True)
    else:
        roles.append({"role": "deep", "name": "", "path": "", "exists": False, "optional": True,
                      "consumer": "deep solver lane", "note": "no AURA_DEEP_MODEL configured"})

    # Perception and voice name hub repos directly.
    try:
        from core.memory.vector_memory_engine import EmbeddingEngine

        add("embedding", EmbeddingEngine.PREFERRED_MODEL, EmbeddingEngine.PREFERRED_MODEL,
            consumer="core/memory/vector_memory_engine.py (in-process, torch)", language_model=False)
        try:
            from core.memory import embedding_model

            roles[-1]["declared_footprint_gb"] = float(embedding_model.FOOTPRINT_GB)
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
    except Exception as exc:  # noqa: BLE001
        roles.append({"role": "embedding", "name": "", "path": "", "exists": False,
                      "optional": False, "consumer": "vector memory", "note": repr(exc)})
    try:
        from core.brain.llm import mlx_vision_client

        vision = getattr(mlx_vision_client, "VISION_MODEL", None) or getattr(
            mlx_vision_client, "DEFAULT_VISION_MODEL", None
        )
        if vision:
            add("vision", str(vision), str(vision), consumer="core/brain/llm/mlx_vision_client.py")
    except Exception as exc:  # noqa: BLE001
        roles.append({"role": "vision", "name": "", "path": "", "exists": False,
                      "optional": True, "consumer": "vision worker", "note": repr(exc)})
    for env, default, role in (
        ("AURA_VOICE_ASR_PARTIAL", "mlx-community/parakeet-tdt-0.6b-v3", "asr_partial"),
        ("AURA_VOICE_ASR_FINAL", "mlx-community/parakeet-tdt-0.6b-v3", "asr_final"),
    ):
        repo = os.environ.get(env, default)
        add(role, repo, repo, consumer="core/voice/duplex/config.py", optional=True, language_model=False)
    return roles


def on_disk() -> list[dict[str, Any]]:
    """Everything under the models directory and the hub cache, named or not."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if MODELS_DIR.is_dir():
        for child in sorted(MODELS_DIR.iterdir()):
            if child.name.startswith("."):
                continue
            mentions = (
                _code_mentions(child.name)
                + _code_mentions(re.sub(r"-[0-9a-f]{12,}$", "", child.name))
                + _manifest_mentions(child.name)
            )
            rows.append({"location": "models", "name": child.name, "path": str(child),
                         "size_gb": round(_du_bytes(child) / 1e9, 2), "code_mentions": mentions})
            seen.add(child.name)
    if HF_HUB.is_dir():
        for child in sorted(HF_HUB.glob("models--*")):
            repo = child.name[len("models--"):].replace("--", "/")
            short = repo.split("/")[-1]
            mentions = _code_mentions(repo) + _code_mentions(short)
            rows.append({"location": "hub", "name": repo, "path": str(child),
                         "size_gb": round(_du_bytes(child) / 1e9, 2), "code_mentions": mentions})
    return rows


def inventory() -> dict[str, Any]:
    roles = live_roles()
    disk = on_disk()
    named = {Path(r["path"]).resolve() for r in roles if r.get("exists")}
    unreferenced = [
        r for r in disk
        if r["code_mentions"] == 0 and not any(str(n).startswith(r["path"]) for n in named)
    ]
    missing = [r for r in roles if not r.get("exists") and not r.get("optional")]
    return {
        "schema": "aura.model_inventory.v1",
        "models_dir": str(MODELS_DIR),
        "hub_cache": str(HF_HUB),
        "roles": roles,
        "on_disk": disk,
        "unreferenced_on_disk": unreferenced,
        "unreferenced_gb": round(sum(r["size_gb"] for r in unreferenced), 1),
        "live_roles_missing": missing,
    }


def _table(record: dict[str, Any]) -> str:
    lines = ["ROLE          MODEL                                          ON DISK  SIZE     CTX      DECLARED  TOKENIZER"]
    for r in record["roles"]:
        ctx = (r.get("context_window") or {}).get("tokens", "")
        lines.append(
            f"{r['role']:<13} {str(r['name'])[:46]:<46} {'yes' if r.get('exists') else ('opt' if r.get('optional') else 'NO '):<8} "
            f"{str(r.get('size_gb', '')):<8} {str(ctx):<8} {str(r.get('declared_footprint_gb', '')):<9} "
            f"{(r.get('tokenizer') or {}).get('chat_template_sha256', '')}"
        )
    lines.append("")
    lines.append(f"on disk: {len(record['on_disk'])} entries; unreferenced by any code: "
                 f"{len(record['unreferenced_on_disk'])} ({record['unreferenced_gb']}GB)")
    for r in record["unreferenced_on_disk"]:
        lines.append(f"  {r['size_gb']:>6}GB  {r['location']:<6} {r['name']}")
    if record["live_roles_missing"]:
        lines.append("")
        lines.append("LIVE ROLES THAT DO NOT RESOLVE:")
        for r in record["live_roles_missing"]:
            lines.append(f"  {r['role']}: {r['name']} -> {r['path']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true", help="exit 1 if a live role does not resolve")
    args = parser.parse_args()
    record = inventory()
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True, default=str))
    else:
        print(_table(record))
    if args.check and record["live_roles_missing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
