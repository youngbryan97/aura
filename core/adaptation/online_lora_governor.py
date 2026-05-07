"""Governed online LoRA updates from self-reflection cycles.

This is intentionally conservative: a Will-approved reflection can enqueue a
small adapter update, but the governor refuses to start while another mlx-lm
LoRA process is active. The current full training run therefore remains safe.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import psutil

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation


@dataclass
class OnlineLoRAReceipt:
    requested_at: float
    status: str
    reflection_hash: str
    will_receipt_id: str = ""
    reason: str = ""
    dataset_path: str = ""
    optimizer_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.blake2b(str(text or "").encode("utf-8"), digest_size=12).hexdigest()


class OnlineLoRAGovernor:
    """Owns the reflection -> dataset -> governed LoRA update path."""

    def __init__(
        self,
        *,
        receipt_path: str | Path | None = None,
        process_iter: Callable[..., Any] | None = None,
    ) -> None:
        self.receipt_path = Path(
            receipt_path or Path.home() / ".aura" / "data" / "runtime" / "online_lora_updates.jsonl"
        )
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self._process_iter = process_iter or psutil.process_iter
        self._lock = asyncio.Lock()
        self.last_receipt: Optional[OnlineLoRAReceipt] = None

    @staticmethod
    def enabled() -> bool:
        return os.getenv("AURA_ONLINE_LORA", "1").strip().lower() not in {"0", "false", "off", "no"}

    def active_lora_processes(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for proc in self._process_iter(["pid", "cmdline", "name"]):
            try:
                info = getattr(proc, "info", {}) or {}
                cmdline = info.get("cmdline") or []
                joined = " ".join(str(part) for part in cmdline).lower()
                if "mlx_lm" in joined and "lora" in joined:
                    found.append({"pid": info.get("pid"), "cmdline": cmdline})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception as exc:
                record_degradation("online_lora_governor", exc)
        return found

    def _record(self, receipt: OnlineLoRAReceipt) -> OnlineLoRAReceipt:
        self.last_receipt = receipt
        with open(self.receipt_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt.to_dict(), sort_keys=True, default=str) + "\n")
        return receipt

    async def maybe_update_from_reflection(
        self,
        reflection: str,
        *,
        conversation_context: str = "",
        will_receipt_id: str = "",
        force: bool = False,
    ) -> OnlineLoRAReceipt:
        """Capture a reflection and, when allowed, run a tiny LoRA update."""
        async with self._lock:
            if not self.enabled() and not force:
                return self._record(
                    OnlineLoRAReceipt(
                        requested_at=time.time(),
                        status="disabled",
                        reflection_hash=_hash_text(reflection),
                        reason="AURA_ONLINE_LORA disabled",
                    )
                )

            running = self.active_lora_processes()
            if running and not force:
                return self._record(
                    OnlineLoRAReceipt(
                        requested_at=time.time(),
                        status="blocked_existing_training",
                        reflection_hash=_hash_text(reflection),
                        reason=f"active mlx_lm lora process pid={running[0].get('pid')}",
                    )
                )

            decision = self._decide(reflection, will_receipt_id=will_receipt_id)
            if not decision.get("approved"):
                return self._record(
                    OnlineLoRAReceipt(
                        requested_at=time.time(),
                        status="will_blocked",
                        reflection_hash=_hash_text(reflection),
                        will_receipt_id=str(decision.get("receipt_id", "")),
                        reason=str(decision.get("reason", "Will did not approve")),
                    )
                )

            dataset_path = await self._capture_training_example(reflection, conversation_context)
            optimizer_result = await self._run_optimizer(dataset_path)
            status = "updated" if optimizer_result.get("ok") else "optimizer_failed"
            return self._record(
                OnlineLoRAReceipt(
                    requested_at=time.time(),
                    status=status,
                    reflection_hash=_hash_text(reflection),
                    will_receipt_id=str(decision.get("receipt_id", "")),
                    reason=str(optimizer_result.get("error") or optimizer_result.get("message") or ""),
                    dataset_path=str(dataset_path),
                    optimizer_result=optimizer_result,
                )
            )

    def _decide(self, reflection: str, *, will_receipt_id: str = "") -> dict[str, Any]:
        if will_receipt_id:
            return {"approved": True, "receipt_id": will_receipt_id, "reason": "upstream Will-approved reflection"}
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=f"online_lora_update:{_hash_text(reflection)}",
                source="online_lora_governor",
                domain=ActionDomain.STATE_MUTATION,
                priority=0.45,
                context={"operation": "small_lora_adapter_update", "reflection": reflection[:240]},
            )
            return {
                "approved": decision.is_approved(),
                "receipt_id": decision.receipt_id,
                "reason": decision.reason,
            }
        except Exception as exc:
            record_degradation("online_lora_governor", exc)
            return {"approved": False, "receipt_id": "", "reason": f"will_unavailable:{type(exc).__name__}"}

    async def _capture_training_example(self, reflection: str, conversation_context: str) -> Path:
        from core.adaptation.finetune_pipe import get_finetune_pipe

        pipe = get_finetune_pipe()
        await pipe.register_success(
            task_description="Will-approved self-reflection",
            context=conversation_context[:800],
            reasoning="Self-reflection accepted as a plasticity signal.",
            final_action=reflection[:1200],
            quality_score=0.72,
        )
        await pipe.flush()
        return pipe.dataset_path

    async def _run_optimizer(self, dataset_path: Path) -> dict[str, Any]:
        try:
            from core.adaptation.self_optimizer import SelfOptimizer, get_self_optimizer

            optimizer = get_self_optimizer()
            if isinstance(optimizer, SelfOptimizer):
                optimizer.dataset_path = Path(dataset_path)
            result = await optimizer.optimize(iters=int(os.getenv("AURA_ONLINE_LORA_ITERS", "20")), batch_size=1)
            return dict(result or {})
        except Exception as exc:
            record_degradation("online_lora_governor", exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def write_status(self, path: str | Path) -> dict[str, Any]:
        payload = {
            "enabled": self.enabled(),
            "active_lora_processes": self.active_lora_processes(),
            "last_receipt": self.last_receipt.to_dict() if self.last_receipt else None,
            "receipt_path": str(self.receipt_path),
        }
        atomic_write_text(Path(path), json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return payload


_instance: Optional[OnlineLoRAGovernor] = None


def get_online_lora_governor() -> OnlineLoRAGovernor:
    global _instance
    if _instance is None:
        _instance = OnlineLoRAGovernor()
    return _instance


__all__ = [
    "OnlineLoRAGovernor",
    "OnlineLoRAReceipt",
    "get_online_lora_governor",
]
