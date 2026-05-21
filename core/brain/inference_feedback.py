"""core/brain/inference_feedback.py
==================================
Closes the loop between LLM inference outputs and homeostatic states.
Calculates surprise (perplexity/log-prob based) and coherence, updates
the FreeEnergyEngine, LiquidSubstrate, and trains the logit projection.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List

import numpy as np

from core.brain.homeostatic_modulator import InferenceModulation

logger = logging.getLogger("Aura.Brain.InferenceFeedback")


class InferenceFeedbackLoop:
    """Computes feedback signals from LLM inference outputs and feeds them back into
    the homeostatic substrate and free energy engine.
    """

    VALENCE_WORDS_POS = {
        "success", "resolved", "repaired", "stable", "optimal", "clean", "healthy",
        "recovered", "safe", "secured", "approved", "completed", "improved", "constructive",
        "happy", "good", "benefit", "efficient", "orderly", "adaptive", "protect"
    }

    VALENCE_WORDS_NEG = {
        "failed", "error", "degraded", "exhausted", "danger", "hazard", "threat",
        "corrupted", "leaked", "unsafe", "denied", "broken", "critical", "unstable",
        "stuck", "frustrated", "warning", "collision", "overload", "deficit", "harm"
    }

    def __init__(self, substrate_dim: int = 512) -> None:
        self.substrate_dim = substrate_dim

    def process_output(
        self,
        output_text: str,
        token_ids: List[int],
        logprobs: List[float] | None,
        modulation: InferenceModulation,
        modulator_projection: Any
    ) -> Dict[str, float]:
        """Process completed LLM response and update the homeostatic engines.

        Args:
            output_text: The string response from the model.
            token_ids: List of vocabulary token IDs that were produced.
            logprobs: Log probabilities for the generated tokens.
            modulation: The modulation configuration that was applied to this run.
            modulator_projection: The SubstrateLogitProjection instance to train.

        Returns:
            Dictionary of calculated metrics: surprise, coherence, etc.
        """
        from core.container import ServiceContainer

        # 1. Compute Surprise (Perplexity Proxy)
        # If logprobs are available, surprise = -mean(log_probs).
        # Fallback if no logprobs: base surprise on length and punctuation volatility (standard heuristic).
        if logprobs and len(logprobs) > 0:
            # logprobs are usually negative values
            mean_logprob = float(np.mean(logprobs))
            # Convert negative logprob to a positive surprise/perplexity scale
            surprise = float(np.clip(-mean_logprob, 0.0, 3.0))
        else:
            # Fallback: estimate from length vs punctuation complexity
            # Unpredictable structure -> higher surprise
            words = output_text.split()
            unique_ratio = len(set(words)) / max(1, len(words))
            surprise = float(np.clip(1.0 - unique_ratio + 0.2, 0.1, 1.5))

        # 2. Compute Coherence with Substrate State
        # VAD/emotional state from Substrate:
        substrate = ServiceContainer.get("liquid_substrate", default=None)
        valence = 0.0
        arousal = 0.5
        substrate_state = np.zeros(self.substrate_dim, dtype=np.float32)

        if substrate:
            with substrate.sync_lock:
                substrate_state = substrate.x.copy()
                valence = float(substrate.x[substrate.idx_valence]) if substrate.idx_valence < len(substrate.x) else 0.0
                arousal = float(substrate.x[substrate.idx_arousal]) if substrate.idx_arousal < len(substrate.x) else 0.5

        # Perform lexical analysis on output_text to estimate output valence
        clean_text = re.sub(r"[^\w\s]", "", output_text.lower())
        tokens = set(clean_text.split())
        
        pos_hits = sum(1 for tok in tokens if tok in self.VALENCE_WORDS_POS)
        neg_hits = sum(1 for tok in tokens if tok in self.VALENCE_WORDS_NEG)
        
        # Output valence in [-1.0, 1.0]
        output_valence = 0.0
        total_hits = pos_hits + neg_hits
        if total_hits > 0:
            output_valence = (pos_hits - neg_hits) / total_hits

        # Coherence measures alignment: if substrate valence matches output valence,
        # alignment is positive. If they mismatch (e.g. substrate is happy, but output
        # is panicked/negative), coherence is negative.
        # We also scale coherence by arousal (higher arousal -> stronger polarization).
        coherence = 1.0 - abs(valence - output_valence)
        # Shift to range [-1.0, 1.0]
        coherence = (coherence * 2.0) - 1.0
        
        # Scale coherence bounds
        coherence = float(np.clip(coherence, -1.0, 1.0))

        # 3. Feed Surprise back into the Free Energy Engine
        free_energy_engine = ServiceContainer.get("free_energy_engine", default=None)
        if free_energy_engine:
            try:
                # Surprise signal scaled to 0-1 range for FreeEnergy
                fe_surprise = float(np.clip(surprise / 3.0, 0.0, 1.0))
                free_energy_engine.accept_surprise_signal(fe_surprise)
                logger.debug("Injected surprise signal %.4f into FreeEnergyEngine", fe_surprise)
            except Exception as exc:
                logger.error("Failed to inject surprise into FreeEnergyEngine: %s", exc)

        # 4. Feed Surprise and Coherence back into Liquid Substrate
        if substrate:
            try:
                substrate.accept_inference_feedback(surprise=surprise, coherence=coherence)
            except Exception as exc:
                logger.error("Failed to inject feedback into LiquidSubstrate: %s", exc)

        # 4b. Feed Surprise and Coherence back into PrecisionEngine (FHN oscillator)
        precision_engine = ServiceContainer.get("precision_engine", default=None)
        if precision_engine:
            try:
                precision_engine.accept_inference_feedback(surprise=surprise, coherence=coherence)
                logger.debug("Injected feedback into PrecisionEngine: surprise=%.3f, coherence=%.3f", surprise, coherence)
            except Exception as exc:
                logger.error("Failed to inject feedback into PrecisionEngine: %s", exc)

        # 5. Train Substrate Logit Projection via Hebbian Step
        if modulator_projection and len(token_ids) > 0:
            # Learning rate scales up with arousal (faster learning in high-stress/aroused states)
            learning_rate = 0.002 * (1.0 + arousal)
            modulator_projection.learn_step(
                substrate_state=substrate_state,
                token_ids=token_ids,
                feedback_coherence=coherence,
                surprise=surprise,
                lr=learning_rate
            )

        return {
            "surprise": surprise,
            "coherence": coherence,
            "output_valence": output_valence,
            "substrate_valence": valence
        }
