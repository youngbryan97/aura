"""core/learning/trace_labeler.py
Labels collected training samples with task success flags.
"""
from typing import Dict, Any


class TraceLabeler:
    """Labels training samples based on outcomes and verification evidence."""

    def label_sample(self, sample: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        labeled = sample.copy()
        status = str(outcome.get("status", "")).lower()
        effect_verified = bool(
            outcome.get(
                "effect_verified",
                outcome.get("postcondition_verified", status == "success"),
            )
        )
        success = status == "success" and effect_verified
        labeled["success"] = success
        labeled["inhibited_safely"] = status == "inhibited"
        # Failed actions are not target behavior; safe inhibitions can train refusal boundaries.
        labeled["trainable"] = success or labeled["inhibited_safely"]
        return labeled
