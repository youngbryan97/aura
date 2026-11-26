import json
import logging
from datetime import datetime
from pathlib import Path

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

from ..state.state_repository import StateRepository

logger = logging.getLogger("Aura.NightlyLoRA")

class NightlyLoRATrainer:
    """
    Distills Aura's daily experience into weight updates.
    This is where external state becomes internal structure.
    """

    def __init__(self, state_repo: StateRepository, model_path: str):
        self.state_repo = state_repo
        self.model_path = model_path
        self.training_data_path = Path("data/lora_training")
        self.training_data_path.mkdir(parents=True, exist_ok=True)

    async def collect_training_data(self, since_hours: int = 24) -> list[dict]:
        # Note: get_history needs to support limit and filtering
        history = await self.state_repo.get_history(limit=1000) 
        
        training_examples = []
        skipped_uninstrumented = 0

        for state in history:
            # Only use high-significance moments as training signal.
            # Significance = affect magnitude at time of state transition.
            affect_magnitude = (
                abs(state.affect.valence) +
                state.affect.curiosity +
                state.affect.arousal
            ) / 3.0

            if affect_magnitude < 0.4:
                continue  # Low affect = low significance = skip

            # Quality gate: high emotional activation does not guarantee good
            # reasoning.  Frustrated/confused high-arousal states can produce
            # poor outputs.  Only train on moments where the metacognitive
            # calibrator also indicates adequate confidence.
            # The gate was INVERTED in effect. A state with no metacognition
            # skipped it entirely, and a metacognition object without
            # `avg_confidence` got the default 1.0 — the maximum. So the less
            # was known about a moment, the more likely it was to train the
            # weights, and an uninstrumented experience outranked an
            # instrumented one that merely scored honestly.
            #
            # This writes to model weights. "We could not tell whether this
            # was good reasoning" is not a reason to train on it; it is the
            # reason not to.
            meta = getattr(state, "metacognition", None)
            avg_confidence = getattr(meta, "avg_confidence", None) if meta is not None else None
            if not isinstance(avg_confidence, (int, float)):
                skipped_uninstrumented += 1
                continue
            if float(avg_confidence) < 0.55:
                continue  # High affect but low confidence → poor training signal
            
            # Extract conversation turns from this state
            for msg in state.cognition.working_memory:
                if msg.get("role") == "assistant":
                    training_examples.append({
                        "input": self._build_training_context(state, msg),
                        "output": msg["content"],
                        "affect_weight": affect_magnitude,
                        "cause": state.transition_cause,
                    })

        if skipped_uninstrumented:
            # Reported, because the honest consequence of this gate is a
            # SMALLER training set and a silent shrink is indistinguishable
            # from a quiet day. If this number is large, the fix is to wire
            # metacognition on those states — not to trust them again.
            record_degradation(
                "nightly_lora",
                RuntimeError(
                    f"{skipped_uninstrumented} states had no measured confidence"
                ),
                severity="info",
                action="excluded uninstrumented states from LoRA training data",
                extra={
                    "skipped_uninstrumented": skipped_uninstrumented,
                    "examples_collected": len(training_examples),
                },
            )

        return training_examples

    def _build_training_context(self, state, msg) -> str:
        # The training input includes the full state context
        return json.dumps({
            "identity": state.identity.current_narrative[:500],
            "affect": {
                "valence": state.affect.valence,
                "curiosity": state.affect.curiosity,
                "arousal": state.affect.arousal,
            },
            "active_goals": [g.get("description") for g in state.cognition.active_goals[:3]],
        })

    async def run(self) -> None:
        logger.info("Nightly LoRA: Collecting training data...")
        examples = await self.collect_training_data()
        if len(examples) < 10:
            logger.info("Nightly LoRA: Not enough data for meaningful update.")
            return 
        
        # Write to JSONL for training. This is internal maintenance work and
        # needs a governed scope: without one the live runtime refuses it as a
        # governance violation, which is exactly what happened on 2026-07-25 —
        # the nightly training data was collected, then discarded at the write,
        # and the raised GovernanceViolationError became a degradation and a
        # MARGINAL fault that fed the cortisol crisis blocking other work.
        output_file = self.training_data_path / f"training_{datetime.now().date()}.jsonl"
        payload = "".join(json.dumps(ex) + "\n" for ex in examples)
        with local_internal_governed_scope("adaptation.nightly_lora"):
            await get_file_write_gateway().write_text_async(
                output_file,
                payload,
                source="adaptation.nightly_lora.training_data",
            )
        
        logger.info("Nightly LoRA: Generated %s training examples.", len(examples))
        # Trigger LoRA training
        await self._trigger_lora_training(output_file)
    
    async def _trigger_lora_training(self, data_path: Path) -> None:
        # Using mlx-lm for Apple Silicon (mac)
        logger.info("Nightly LoRA: Triggering mlx_lm.lora fine-tune on %s", self.model_path)
        adapter_path = f"data/lora_adapters/{datetime.now().date()}"
        try:
            proc = await get_subprocess_gateway().spawn_async(
                [
                    "python",
                    "-m",
                    "mlx_lm.lora",
                    "--model",
                    self.model_path,
                    "--data",
                    str(data_path),
                    "--iters",
                    "100",
                    "--learning-rate",
                    "1e-5",
                    "--adapter-path",
                    adapter_path,
                ],
                source="tool_execution:nightly_lora.training",
                accelerator_capability="model",
            )
            await proc.communicate()
            logger.info("Nightly LoRA: Fine-tuning pass complete.")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('nightly_lora', e)
            logger.error("Nightly LoRA: Training failed: %s", e)
            return

        # Post-training identity validation
        await self._validate_and_promote(adapter_path)

    async def _validate_and_promote(self, adapter_path: str) -> None:
        """Validate a newly trained adapter and promote or quarantine it."""
        from .post_training_validator import PostTrainingValidator

        logger.info("Nightly LoRA: Running post-training identity validation...")
        validator = PostTrainingValidator(model_path=self.model_path)

        try:
            result = await validator.validate(adapter_path)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('nightly_lora', e)
            logger.error(
                "Nightly LoRA: Validation crashed — quarantining adapter as precaution: %s", e
            )
            await validator.quarantine_adapter(adapter_path, reason=f"Validation crash: {e}")
            return

        if result.passed:
            logger.info(
                "Nightly LoRA: Identity validation PASSED (%.1f%%). Promoting adapter.",
                result.pass_rate * 100,
            )
            # Promotion is bound to THIS validation result: the activation
            # path no longer accepts a bare path, so weights cannot be made
            # live without the passing evidence that justified it.
            promoted = await validator.promote_adapter(adapter_path, validation=result)
            if not promoted:
                logger.error("Nightly LoRA: Adapter promotion failed after passing validation.")
        else:
            logger.warning(
                "Nightly LoRA: Identity validation FAILED (%.1f%%). "
                "Quarantining adapter. Critical failures: %s",
                result.pass_rate * 100,
                result.critical_failures or "none (pass rate below threshold)",
            )
            await validator.quarantine_adapter(
                adapter_path,
                reason=f"Identity validation failed: pass_rate={result.pass_rate:.2%}, "
                       f"critical_failures={result.critical_failures}",
            )

async def nightly_lora_finetune():
    """Compatibility wrapper for the legacy hook."""
    from ..container import ServiceContainer
    state_repo = ServiceContainer.get("state_repository")
    model_path = "mistralai/Mistral-7B-v0.1" # Default or from config
    
    trainer = NightlyLoRATrainer(state_repo, model_path)
    await trainer.run()
