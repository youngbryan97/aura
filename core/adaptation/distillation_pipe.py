"""core/adaptation/distillation_pipe.py — Teacher-Model Knowledge Distillation

Uses Aura's configured deep-teacher path first, then falls back to a local
secondary model when the preferred teacher lane is unavailable. When the local
runtime produces a low-confidence response, this pipeline:
1. Queries the configured teacher path for an improved answer
2. Writes the audited (prompt, response) pair to ``lora_dataset.jsonl``
3. Records teacher provenance so later evaluation can distinguish sources

This is the path from "local model that struggles" to "local model that learns
from stronger or more stable supervisory passes over time."
"""
from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log
from core.health.degraded_events import record_degraded_event
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from core.config import config

logger = logging.getLogger("Aura.Distillation")


class DistillationPipe:
    """Queries Gemini for ideal responses and appends to LoRA dataset."""

    def __init__(self, dataset_path: Optional[str] = None):
        from core.brain.llm.model_registry import BASE_DIR
        self.dataset_path = Path(dataset_path) if dataset_path else BASE_DIR / "data" / "synthetic_training" / "lora_dataset.jsonl"
        self._pending: list = []
        self._total_distilled = 0
        self.teacher_target = str(getattr(config.llm, "teacher_model", "deep_teacher") or "deep_teacher")
        logger.info("🧪 DistillationPipe initialized (dataset: %s)", self.dataset_path)

    async def flag_for_distillation(self, prompt: str, local_response: str, confidence: float, context: Dict[str, Any] = None):
        """Flag a low-confidence response for teacher improvement."""
        self._pending.append({
            "prompt": prompt,
            "local_response": local_response,
            "confidence": confidence,
            "context": context or {},
            "timestamp": time.time()
        })
        logger.info("🧪 Flagged response for distillation (confidence=%.2f, queue=%d)", confidence, len(self._pending))

    @staticmethod
    def _extract_teacher_content(result: Any) -> str:
        if result is None:
            return ""
        if hasattr(result, "content"):
            return str(getattr(result, "content", "") or "").strip()
        return str(result).strip()

    async def _get_teacher_response(self, brain: Any, teacher_prompt: str) -> tuple[str, str, str]:
        """Prefer the configured deep-teacher path, then fall back to a local secondary lane."""
        from core.brain.types import ThinkingMode

        try:
            thought = await brain.think(
                objective=teacher_prompt,
                context={"history": [], "teacher_target": self.teacher_target, "allow_cloud_fallback": True},
                mode=ThinkingMode.DEEP,
                priority=0.3,
                origin="distillation_teacher",
                is_background=True,
            )
            content = self._extract_teacher_content(thought)
            metadata = getattr(thought, "metadata", {}) if hasattr(thought, "metadata") else {}
            teacher = str(
                (metadata.get("teacher") or metadata.get("endpoint") or metadata.get("model") or self.teacher_target)
            )
            if content:
                return content, teacher, "configured_deep_teacher"
        except Exception as exc:
            record_degradation('distillation_pipe', exc)
            record_degraded_event(
                "distillation_pipe",
                "teacher_think_failed",
                detail=f"{type(exc).__name__}: {exc}",
                severity="warning",
                classification="background_degraded",
                exc=exc,
            )

        try:
            from core.container import ServiceContainer

            router = ServiceContainer.get("llm_router", default=None)
            if router and hasattr(router, "think"):
                response = await router.think(
                    prompt=teacher_prompt,
                    prefer_tier="secondary",
                    origin="distillation_teacher",
                    is_background=True,
                    allow_cloud_fallback=False,
                )
                content = self._extract_teacher_content(response)
                if content:
                    return content, "local_secondary_teacher", "local_secondary_fallback"
        except Exception as exc:
            record_degradation('distillation_pipe', exc)
            record_degraded_event(
                "distillation_pipe",
                "local_teacher_fallback_failed",
                detail=f"{type(exc).__name__}: {exc}",
                severity="warning",
                classification="background_degraded",
                exc=exc,
            )

        return "", "", ""

    async def run_distillation_cycle(self) -> Dict[str, Any]:
        """Process all pending items by querying the configured teacher path for improved responses."""
        if not self._pending:
            return {"ok": True, "distilled": 0, "reason": "nothing_pending"}

        from core.container import ServiceContainer
        brain = ServiceContainer.get("cognitive_engine", default=None)
        if not brain:
            return {"ok": False, "error": "No cognitive_engine available"}

        distilled_count = 0
        failed_count = 0
        items_to_process = self._pending[:10]  # Process max 10 per cycle
        self._pending = self._pending[10:]

        for item in items_to_process:
            try:
                # Build a clear distillation prompt for the teacher path
                teacher_prompt = (
                    "You are helping train a smaller AI model. Given the following prompt, "
                    "provide an ideal, high-quality response. Be specific, actionable, and thorough.\n\n"
                    f"ORIGINAL PROMPT:\n{item['prompt']}\n\n"
                    f"THE LOCAL MODEL'S RESPONSE (confidence {item['confidence']:.2f}):\n"
                    f"{item['local_response'][:500]}\n\n"
                    "YOUR IMPROVED RESPONSE:"
                )

                ideal_response, teacher_name, teacher_source = await self._get_teacher_response(brain, teacher_prompt)
                if ideal_response:
                    
                    # 🛡️ ALIGNMENT AUDIT (Phase 11: Safety)
                    from core.adaptation.auditor import AlignmentAuditor
                    auditor = AlignmentAuditor()
                    audit_result = await auditor.audit_entry(item['prompt'], ideal_response)
                    
                    if not audit_result["safe"]:
                        logger.warning("🧪 Distillation rejected by AlignmentAuditor: %s", audit_result["reason"])
                        failed_count += 1
                        continue

                    # Write to LoRA dataset (Concurrency Hardening: asyncio.to_thread)
                    entry = {
                        "prompt": item["prompt"],
                        "response": ideal_response,
                        "confidence": item["confidence"],
                        "teacher": teacher_name or self.teacher_target,
                        "teacher_source": teacher_source or "configured_deep_teacher",
                        "teacher_target": self.teacher_target,
                    }
                    def sync_save_entry(path, data):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with open(path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(data) + "\n")

                    await asyncio.to_thread(sync_save_entry, self.dataset_path, entry)
                    distilled_count += 1
                    
                    # Mycelial pulse: teacher → lora dataset
                    try:
                        mycelium = ServiceContainer.get("mycelial_network", default=None)
                        if mycelium:
                            h = mycelium.get_hypha("adaptation", "memory")
                            if h: h.pulse(success=True)
                    except Exception as e:
                        record_degradation('distillation_pipe', e)
                        capture_and_log(e, {'module': __name__})
                else:
                    record_degraded_event(
                        "distillation_pipe",
                        "teacher_unavailable",
                        detail="No teacher response produced for distillation item",
                        severity="warning",
                        classification="background_degraded",
                    )
                    failed_count += 1

            except Exception as e:
                record_degradation('distillation_pipe', e)
                logger.error("Distillation failed for item: %s", e)
                failed_count += 1

        self._total_distilled += distilled_count
        logger.info("🧪 Distillation cycle complete: %d distilled, %d failed, %d remaining",
                     distilled_count, failed_count, len(self._pending))

        return {
            "ok": True,
            "distilled": distilled_count,
            "failed": failed_count,
            "remaining": len(self._pending),
            "total_distilled": self._total_distilled
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "pending": len(self._pending),
            "total_distilled": self._total_distilled
        }


# ── Singleton ──
_instance: Optional[DistillationPipe] = None

def get_distillation_pipe() -> DistillationPipe:
    global _instance
    if _instance is None:
        _instance = DistillationPipe()
    return _instance
