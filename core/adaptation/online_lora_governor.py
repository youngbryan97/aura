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
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.resource_observation import ResourceObserver, get_resource_observer
from core.runtime.state_ownership import state_root


@dataclass
class OnlineLoRAReceipt:
    requested_at: float
    status: str
    reflection_hash: str
    will_receipt_id: str = ""
    reason: str = ""
    dataset_path: str = ""
    optimizer_result: dict[str, Any] = field(default_factory=dict)
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""
    process_observation_available: bool = False

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
        observer: ResourceObserver | None = None,
    ) -> None:
        self.receipt_path = Path(
            receipt_path or state_root() / "data" / "runtime" / "online_lora_updates.jsonl"
        )
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self._observer = observer
        self._lock = asyncio.Lock()
        self.last_receipt: OnlineLoRAReceipt | None = None

    @staticmethod
    def enabled() -> bool:
        return os.getenv("AURA_ONLINE_LORA", "1").strip().lower() not in {"0", "false", "off", "no"}

    def active_lora_processes(self) -> list[dict[str, Any]]:
        observer = self._observer or get_resource_observer()
        table = observer.process_table()
        if not table.available:
            return [
                {
                    "pid": None,
                    "cmdline": [],
                    "observation_error": table.error or "process_table_unavailable",
                }
            ]
        found: list[dict[str, Any]] = []
        current_pid = os.getpid()
        for process in table.processes:
            if process.pid == current_pid:
                continue
            joined = " ".join(process.cmdline).lower()
            matched = self._training_signature(joined)
            if matched:
                found.append({
                    "pid": process.pid,
                    "cmdline": list(process.cmdline),
                    "matched": matched,
                })
        return found

    #: Command-line signatures of processes that are training or fusing model
    #: weights. Matching only "mlx_lm ... lora" made the recurrence-native 32B
    #: trainer and every other training entrypoint INVISIBLE to the exclusion
    #: check, so this governor would happily start a competing LoRA update
    #: beside a running trainer while reporting that it had checked.
    #: Ordered MOST SPECIFIC FIRST, so the reported label names the actual
    #: entrypoint rather than whichever generic marker happened to appear in
    #: the command line. Detection is unaffected by order; the label is not.
    _TRAINING_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("mlx_lm.lora", ("mlx_lm", "lora")),
        ("mlx_lm.fuse", ("mlx_lm", "fuse")),
        ("self_optimizer", ("self_optimizer",)),
        ("resident_trainer", ("resident", "train")),
        ("recurrence_trainer", ("recurrence", "train")),
        ("grpo_trainer", ("grpo",)),
        # Launchers before the generic script/verb markers they wrap.
        ("torch_run", ("torchrun",)),
        ("accelerate_launch", ("accelerate", "launch")),
        ("distillation", ("distill",)),
        ("finetune", ("finetune",)),
        ("fine_tune", ("fine_tune",)),
        ("train_script", ("train.py",)),
    )

    @classmethod
    def _training_signature(cls, joined_cmdline: str) -> str:
        """Name the training signature this command line matches, if any."""
        for label, needles in cls._TRAINING_SIGNATURES:
            if all(needle in joined_cmdline for needle in needles):
                return label
        return ""

    def _record(self, receipt: OnlineLoRAReceipt) -> OnlineLoRAReceipt:
        provenance = (self._observer or get_resource_observer()).provenance
        receipt.observation_source = provenance.source.value
        receipt.observation_scenario_id = provenance.scenario_id
        receipt.process_observation_available = not receipt.reason.startswith(
            "process_table_unavailable"
        )
        self.last_receipt = receipt
        get_file_write_gateway().append_text(
            self.receipt_path,
            json.dumps(receipt.to_dict(), sort_keys=True, default=str) + "\n",
            source="adaptation.online_lora.receipt",
        )
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

            # A command-line census enriches every host process and is allowed
            # only on a worker thread. On large developer workstations it can
            # otherwise stall the owner loop for multiple seconds.
            # `force` may override the POLICY gate above (an operator choosing
            # to run while the feature flag is off). It must not override this
            # one: another trainer holding the model is a physical collision,
            # not a preference, and two concurrent LoRA updates corrupt the
            # adapter they share. There is no emergency authority that makes
            # concurrent training safe, so force does not grant one.
            running = await asyncio.to_thread(self.active_lora_processes)
            if running:
                observation_error = str(running[0].get("observation_error") or "")
                return self._record(
                    OnlineLoRAReceipt(
                        requested_at=time.time(),
                        status="blocked_existing_training",
                        reflection_hash=_hash_text(reflection),
                        reason=(
                            f"process_table_unavailable:{observation_error}"
                            if observation_error
                            else (
                                f"active training process "
                                f"({running[0].get('matched') or 'unknown'}) "
                                f"pid={running[0].get('pid')}"
                                + (" [force does not override a training collision]"
                                   if force else "")
                            )
                        ),
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
            
            # Delegate to the already-booted learning owner. This path is
            # collect-only; it must not instantiate a second learner or report
            # a weight update before the scheduler validates and promotes one.
            try:
                from core.container import ServiceContainer

                learner = ServiceContainer.get("continuous_learner", default=None)
                if learner is None:
                    learner = ServiceContainer.get("live_learner", default=None)
                if learner is None:
                    optimizer_result = {
                        "ok": False,
                        "message": "queued_collect_only: no canonical learning owner registered",
                    }
                elif hasattr(learner, "record_turn"):
                    learner.record_turn(
                        system_prompt="You are Aura.",
                        user_input=conversation_context[-500:] if conversation_context else "Self-reflection trigger.",
                        response=reflection,
                        # NOT explicit_positive. This is Aura's own reflection
                        # with no user feedback, no outcome evidence, and no
                        # factual verification behind it. Labelling it explicit
                        # positive feedback made the model train on its own
                        # output as though a human had endorsed it — the
                        # textbook self-training loop, where whatever the model
                        # already tends to say becomes what it is reinforced to
                        # say. It is captured as unlabelled material for the
                        # scheduler's own validation to grade.
                        explicit_positive=False,
                        emotional_context={"arousal": 0.5, "valence": 0.5},
                    )
                    optimizer_result = {
                        "ok": True,
                        "message": "queued_for_scheduler_validation",
                        # Stated on the receipt (which this module owns) so
                        # nothing downstream can mistake self-authored text for
                        # externally-corroborated training signal.
                        "signal_source": "self_reflection",
                        "externally_validated": False,
                    }
                else:
                    optimizer_result = {
                        "ok": False,
                        "message": "queued_collect_only: learning owner lacks record_turn",
                    }
            except (ImportError, AttributeError, RuntimeError) as _e:
                record_degradation("online_lora_governor", _e)
                optimizer_result = {"ok": False, "message": f"queued_collect_only: {type(_e).__name__}"}

            status = "queued_for_validation" if optimizer_result.get("ok") else "queued_collect_only"
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

    @staticmethod
    def _verify_will_receipt(receipt_id: str) -> bool | None:
        """True/False when the Will can adjudicate the id, None when it cannot."""
        try:
            from core.will import get_will

            will = get_will()
            lookup = getattr(will, "get_receipt", None) or getattr(
                will, "lookup_receipt", None
            )
            if not callable(lookup):
                return None
            receipt = lookup(str(receipt_id))
            if receipt is None:
                return False
            approved = getattr(receipt, "is_approved", None)
            if callable(approved):
                return bool(approved())
            if isinstance(receipt, dict):
                return bool(receipt.get("approved"))
            return None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            # not a failure: a receipt that cannot be read does not say
            # approved, and the branch above returns the same None for one
            # that says nothing.
            return None

    def _decide(self, reflection: str, *, will_receipt_id: str = "") -> dict[str, Any]:
        # A caller-supplied receipt id used to be accepted as approval on the
        # sole basis of being non-empty: any string at all authorised a weight
        # mutation. The id is now checked against the Will that supposedly
        # issued it, and only a receipt the Will confirms as approved counts.
        # Full binding (signature, action digest, scope, expiry, replay) needs
        # the signing boundary this module does not own; what is closed here is
        # the case where NOTHING was verified.
        if will_receipt_id:
            verified = self._verify_will_receipt(will_receipt_id)
            if verified is True:
                return {
                    "approved": True,
                    "receipt_id": will_receipt_id,
                    "reason": "Will-confirmed upstream approval",
                }
            if verified is False:
                return {
                    "approved": False,
                    "receipt_id": will_receipt_id,
                    "reason": "supplied Will receipt is not an approval",
                }
            # Verification unavailable: fall through to asking the Will
            # directly rather than trusting the unverifiable id.
            record_degradation(
                "online_lora_governor",
                RuntimeError("could not verify supplied Will receipt"),
                severity="warning",
                action="re-requested a decision from the Will instead of trusting the id",
            )
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
        except (ImportError, AttributeError, RuntimeError) as exc:
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
        except (ImportError, AttributeError, RuntimeError) as exc:
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


_instance: OnlineLoRAGovernor | None = None


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
