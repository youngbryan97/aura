"""core/sleep/world_model_training.py
Offline world model training and causal graph alignment.
"""
import logging
import time
from typing import Any

logger = logging.getLogger("Sleep.WorldModelTraining")


class WorldModelTrainer:
    """Consolidates causal graph edges during sleep cycles."""

    def train_world_model(self, causal_logs: Any) -> None:
        if not isinstance(causal_logs, dict):
            logger.warning("WorldModelTrainer skipped non-dict world model: %s", type(causal_logs).__name__)
            return

        training = causal_logs.setdefault(
            "_sleep_world_model_training",
            {
                "cycles": 0,
                "edges_reinforced": 0,
                "last_updated": None,
            },
        )
        training["cycles"] = int(training.get("cycles", 0)) + 1
        training["last_updated"] = time.time()

        edges = causal_logs.get("causal_edges") or causal_logs.get("edges") or {}
        reinforced = 0
        if isinstance(edges, dict):
            for edge_id, edge in edges.items():
                if not isinstance(edge, dict):
                    continue
                observations = float(edge.get("observations", edge.get("count", 0)) or 0)
                confidence = float(edge.get("confidence", 0.5) or 0.5)
                edge["confidence"] = min(1.0, confidence + min(0.05, observations * 0.005))
                edge["last_sleep_reinforced"] = training["last_updated"]
                edge["edge_id"] = edge.get("edge_id", edge_id)
                reinforced += 1

        training["edges_reinforced"] = int(training.get("edges_reinforced", 0)) + reinforced
        logger.info("WorldModelTrainer reinforced %d causal edges.", reinforced)
