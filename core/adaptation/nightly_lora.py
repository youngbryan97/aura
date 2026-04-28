from core.runtime.errors import record_degradation
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
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
            meta = getattr(state, "metacognition", None)
            if meta is not None:
                avg_confidence = getattr(meta, "avg_confidence", 1.0)
                if avg_confidence < 0.55:
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
        logger.info("🌙 Nightly LoRA: Collecting training data...")
        examples = await self.collect_training_data()
        if len(examples) < 10:
            logger.info("🌙 Nightly LoRA: Not enough data for meaningful update.")
            return 
        
        # Write to JSONL for training
        output_file = self.training_data_path / f"training_{datetime.now().date()}.jsonl"
        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        
        logger.info(f"🌙 Nightly LoRA: Generated {len(examples)} training examples.")
        # Trigger LoRA training
        await self._trigger_lora_training(output_file)
    
    async def _trigger_lora_training(self, data_path: Path) -> None:
        # Using mlx-lm for Apple Silicon (mac)
        logger.info(f"🌙 Nightly LoRA: Triggering mlx_lm.lora fine-tune on {self.model_path}")
        adapter_path = f"data/lora_adapters/{datetime.now().date()}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "mlx_lm.lora",
                "--model", self.model_path,
                "--data", str(data_path),
                "--iters", "100",
                "--learning-rate", "1e-5",
                "--adapter-path", adapter_path,
            )
            await proc.communicate()
            logger.info("✅ Nightly LoRA: Fine-tuning pass complete.")
        except Exception as e:
            record_degradation('nightly_lora', e)
            logger.error(f"❌ Nightly LoRA: Training failed: {e}")
            return

        # Post-training identity validation
        await self._validate_and_promote(adapter_path)

    async def _validate_and_promote(self, adapter_path: str) -> None:
        """Validate a newly trained adapter and promote or quarantine it."""
        from .post_training_validator import PostTrainingValidator

        logger.info("🔍 Nightly LoRA: Running post-training identity validation...")
        validator = PostTrainingValidator(model_path=self.model_path)

        try:
            result = await validator.validate(adapter_path)
        except Exception as e:
            record_degradation('nightly_lora', e)
            logger.error(
                "❌ Nightly LoRA: Validation crashed — quarantining adapter as precaution: %s", e
            )
            await validator.quarantine_adapter(adapter_path, reason=f"Validation crash: {e}")
            return

        if result.passed:
            logger.info(
                "✅ Nightly LoRA: Identity validation PASSED (%.1f%%). Promoting adapter.",
                result.pass_rate * 100,
            )
            promoted = await validator.promote_adapter(adapter_path)
            if not promoted:
                logger.error("❌ Nightly LoRA: Adapter promotion failed after passing validation.")
        else:
            logger.warning(
                "🚫 Nightly LoRA: Identity validation FAILED (%.1f%%). "
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
