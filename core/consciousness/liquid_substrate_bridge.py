import hashlib
import logging
import numpy as np
from typing import TYPE_CHECKING, Dict, Any, Optional

if TYPE_CHECKING:
    from core.orchestrator import RobustOrchestrator

logger = logging.getLogger("LiquidSubstrate.Bridge")

def encode_message_to_stimulus(text: str, neuron_count: int = 512) -> np.ndarray:
    """
    Convert a text message to a stimulus vector for the CTRNN.
    Uses character-frequency histogram + structural features projected to neuron_count dims.
    """
    hist = np.zeros(256, dtype=np.float32)
    for ch in text[:512]:
        hist[ord(ch) & 0xFF] += 1.0
    total = hist.sum() or 1.0
    hist /= total

    length_norm = min(1.0, len(text) / 500.0)
    punct_density = sum(1 for c in text if c in ".,!?;:") / max(1, len(text))
    upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, len(text))
    features = np.array([length_norm, punct_density, upper_ratio, digit_ratio], dtype=np.float32)

    raw = np.concatenate([hist, features])

    rng = np.random.RandomState(neuron_count)
    proj = rng.randn(neuron_count, 260).astype(np.float32) * (1.0 / np.sqrt(260))
    stimulus = np.tanh(proj @ raw)
    return stimulus

def bridge_to_orchestrator(orchestrator: "RobustOrchestrator"):
    """Wire LiquidSubstrate into the orchestrator lifecycle."""
    from core.container import ServiceContainer
    substrate = ServiceContainer.get("liquid_substrate", default=None)

    if substrate is None:
        logger.warning("LiquidSubstrate not in ServiceContainer — bridge skipped.")
        return

    if hasattr(substrate, 'running') and not substrate.running:
        substrate.start()
        logger.info("LiquidSubstrate started by bridge.")

    neuron_count = getattr(substrate.config, 'neuron_count', 512)

    _original_enqueue = orchestrator.enqueue_message
    def enqueue_with_stimulus(message: str):
        _original_enqueue(message)
        try:
            stimulus = encode_message_to_stimulus(message, neuron_count)
            substrate.inject_stimulus(stimulus, weight=0.5)
            logger.debug("Stimulus injected for message (%d chars)", len(message))
        except Exception as e:
            logger.warning("Stimulus injection failed: %s", e)
    orchestrator.enqueue_message = enqueue_with_stimulus

    def get_substrate_affect() -> Dict[str, float]:
        try:
            # Fix: get_state_summary is async — use sync get_substrate_affect instead
            summary = substrate.get_substrate_affect()
            return {
                "valence":    float(np.tanh(summary.get("valence", 0.0))),
                "arousal":    float((summary.get("arousal", 0.0) + 1.0) / 2.0),
                "dominance":  float(np.tanh(summary.get("dominance", 0.0))),
                "energy":     float(summary.get("global_energy", 0.5)),
                "volatility": float(min(1.0, summary.get("volatility", 0.0) / 10.0)),
            }
        except Exception:
            return {"valence": 0.0, "arousal": 0.3, "dominance": 0.0, "energy": 0.5, "volatility": 0.0}
    orchestrator.get_substrate_affect = get_substrate_affect

    _original_update = orchestrator._update_liquid_pacing
    def _update_with_substrate_crossfeed():
        _original_update()
        try:
            sub_affect = get_substrate_affect()
            affect_engine = getattr(orchestrator, 'affect_engine', None)
            if affect_engine and hasattr(affect_engine, 'modify'):
                import asyncio
                asyncio.create_task(
                    affect_engine.modify(
                        dv=sub_affect["valence"] * 0.01,
                        da=sub_affect["arousal"] * 0.01,
                        de=sub_affect["energy"]  * 0.005,
                        source="liquid_substrate"
                    )
                )
        except Exception as e:
            logger.debug("Substrate cross-feed error: %s", e)
    orchestrator._update_liquid_pacing = _update_with_substrate_crossfeed

    _original_gather = orchestrator._gather_agentic_context
    async def _gather_with_substrate(message: str) -> Dict[str, Any]:
        ctx = await _original_gather(message)
        try:
            sub_affect = get_substrate_affect()
            ctx["substrate"] = {
                "neural_valence":   round(sub_affect["valence"],   3),
                "neural_arousal":   round(sub_affect["arousal"],   3),
                "neural_dominance": round(sub_affect["dominance"], 3),
                "neural_energy":    round(sub_affect["energy"],    3),
                "neural_volatility": round(sub_affect["volatility"], 3),
            }
        except Exception as e:
            logger.debug("Substrate formatting error: %s", e)
        return ctx
    orchestrator._gather_agentic_context = _gather_with_substrate
    logger.info("✅ LiquidSubstrate bridged to orchestrator.")

def format_substrate_for_prompt(substrate_ctx: Dict[str, Any]) -> str:
    if not substrate_ctx: return ""
    v = substrate_ctx.get("neural_valence", 0.0)
    a = substrate_ctx.get("neural_arousal", 0.3)
    vo = substrate_ctx.get("neural_volatility", 0.0)
    valence_word = "positive" if v > 0.1 else ("negative" if v < -0.1 else "neutral")
    arousal_word = "heightened" if a > 0.6 else ("low" if a < 0.2 else "moderate")
    volatile_note = " (volatile, shifting rapidly)" if vo > 0.5 else ""
    return (f"[Neural substrate state: {valence_word} valence, "
            f"{arousal_word} arousal{volatile_note}. "
            f"Let this subtly colour your tone without overriding your reasoning.]")