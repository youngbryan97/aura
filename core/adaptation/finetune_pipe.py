"""core/adaptation/finetune_pipe.py - LoRA Synthetic Data Pipe

This module hooks into the event bus to listen for successful task executions,
extracts the reasoning trajectory, and writes it to a JSONL dataset in Alpaca/ShareGPT format.
This allows Aura to generate her own active-learning data for future fine-tuning.
"""
from core.runtime.numeric_guards import is_finite_number, unit_float

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
import os
import json
import logging
import asyncio
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger("Aura.FinetunePipe")

class FinetunePipe:
    """Listens for successful events and writes them to a JSONL dataset."""

    def __init__(self, data_dir: str = "data/synthetic_training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.data_dir / "lora_dataset.jsonl"
        self._batch: List[Dict[str, Any]] = []
        self._flush_lock = asyncio.Lock()
        # AUDIT-FIX: each entry carries a quality_score for priority rotation
        
    def _format_text_prompt(self, instruction: str, input_context: str, output: str) -> Dict[str, str]:
        """Formats the trace into the standard 'text' field format for MLX-LM."""
        prompt = f"User: {instruction}\n"
        if input_context:
            prompt += f"Context: {input_context}\n"
        prompt += f"Aura: {output}"
        return {"text": prompt}

    @staticmethod
    def _compute_quality_score(reasoning: str, final_action: str) -> float:
        """Heuristic quality score [0, 1] for a training sample.

        AUDIT-FIX: Rotation uses quality scores, not chronological order.
        Higher = better sample to retain.
        """
        score = 0.5
        # Length heuristic: longer reasoning = richer supervision signal
        reasoning_len = len(reasoning.split())
        score += min(0.2, reasoning_len / 500.0)
        # Action specificity: JSON/code responses are more learnable
        if any(tok in final_action for tok in ["{", "```", "def ", "class ", "import "]):
            score += 0.15
        # Penalize very short outputs
        if len(final_action.strip()) < 20:
            score -= 0.2
        return max(0.0, min(1.0, score))

    async def register_success(
        self,
        task_description: str,
        context: str,
        reasoning: str,
        final_action: str,
        quality_score: float = -1.0,
        metadata: Dict[str, Any] | None = None,
    ):
        """Records a successful trace for future fine-tuning."""
        try:
            full_response = f"<thought>\n{reasoning}\n</thought>\n<action>\n{final_action}\n</action>"
            formatted_entry = self._format_text_prompt(
                instruction=task_description,
                input_context=context,
                output=full_response
            )
            # Attach quality score for priority rotation
            # CP126 (high): "Caller quality accepts non-finite and
            # out-of-range values." `quality_score < 0` is False for NaN, so
            # a NaN skipped the heuristic fallback AND became the stored
            # score — and rotation sorts on it, where NaN compares False
            # against everything and pins a sample at one end of the
            # priority order permanently.
            if not is_finite_number(quality_score) or quality_score < 0:
                quality_score = self._compute_quality_score(reasoning, final_action)
            quality_score = unit_float(quality_score, default=0.0)
            formatted_entry["_quality"] = round(quality_score, 4)
            if metadata:
                formatted_entry["_meta"] = metadata

            self._batch.append(formatted_entry)
            logger.info("FinetunePipe: Captured trace for '%s' (quality=%.2f).", task_description[:30], quality_score)
            
            # Flush to disk if batch size > 5
            if len(self._batch) >= 5:
                await self.flush()
                
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('finetune_pipe', e)
            logger.error("Failed to register success trace: %s", e)

    async def flush(self):
        """Writes the current batch to the JSONL dataset file asynchronously."""
        if not self._batch:
            return
            
        async with self._flush_lock:
            try:
                # Offload synchronous file append to a background thread
                def _write_batch(batch, path):
                    from core.governance_context import local_internal_governed_scope

                    payload = "".join(json.dumps(entry) + "\n" for entry in batch)
                    # Governed scope, established INSIDE the worker thread:
                    # without it the live runtime refused every flush as a
                    # governance violation ("append_text:adaptation.
                    # finetune_pipe.dataset called outside governed context",
                    # ×12 in the Jul 18 live log) and the learning traces
                    # were silently dropped.
                    with local_internal_governed_scope(
                        "adaptation.finetune_pipe.dataset"
                    ):
                        get_file_write_gateway().append_text(
                            path,
                            payload,
                            source="adaptation.finetune_pipe.dataset",
                        )
                            
                await asyncio.to_thread(_write_batch, self._batch.copy(), self.dataset_path)
                
                logger.info("FinetunePipe: Flushed %d traces to %s", len(self._batch), self.dataset_path)
                self._batch.clear()
                
                # AUDIT-FIX: Quality-score-based rotation — keep highest-quality 1000 samples,
                # not the most recent 1000 (chronological rotation discards good old samples).
                def _rotate_dataset(path):
                    from core.governance_context import local_internal_governed_scope

                    lines = path.read_text(encoding="utf-8").splitlines()
                    if len(lines) <= 1000:
                        return
                    logger.info("FinetunePipe: Dataset exceeds 1000 samples. Quality-rotating...")
                    # Parse entries and sort by quality score descending
                    entries = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            q = obj.pop("_quality", 0.5)
                            entries.append((q, json.dumps(obj)))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            entries.append((0.5, line))
                    entries.sort(key=lambda x: x[0], reverse=True)
                    with local_internal_governed_scope(
                        "adaptation.finetune_pipe.rotate_dataset"
                    ):
                        get_file_write_gateway().write_text(
                            path,
                            "".join(line + "\n" for _, line in entries[:1000]),
                            source="adaptation.finetune_pipe.rotate_dataset",
                        )
                            
                try:
                    await asyncio.to_thread(_rotate_dataset, self.dataset_path)
                except (RuntimeError, AttributeError, TypeError, ValueError) as rotate_err:
                    record_degradation('finetune_pipe', rotate_err)
                    logger.debug("Dataset rotation skipped: %s", rotate_err)
                    
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                record_degradation('finetune_pipe', e)
                logger.error("Failed to flush traces to disk: %s", e)

# Singleton instance
_pipe_instance = None

def get_finetune_pipe() -> FinetunePipe:
    global _pipe_instance
    if _pipe_instance is None:
        # Base directory resolution
        base = Path(__file__).resolve().parent.parent.parent
        _pipe_instance = FinetunePipe(data_dir=str(base / "data" / "synthetic_training"))
    return _pipe_instance
