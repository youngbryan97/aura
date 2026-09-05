"""
core/resilience/eternal_record.py
─────────────────────────────────
Captures immutable snapshots of the Knowledge Graph and system state, 
archiving them as "Eternal Records" in the persistent brain directory.
"""

import json
import logging
import shutil
import time
from pathlib import Path

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.EternalRecord")

class EternalRecord:
    def __init__(self, brain_dir: Path):
        self.brain_dir = brain_dir
        self.records_dir = brain_dir / "eternal_records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        logger.info("🏺 Eternal Record Archive initialized at %s", self.records_dir)

    def create_snapshot(self, kg_path: Path):
        """Create a full physical and logical snapshot of the system's knowledge."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        snapshot_name = f"record_{timestamp}"
        snapshot_dir = self.records_dir / snapshot_name
        snapshot_dir.mkdir(exist_ok=True)

        try:
            # 1. Archive the Knowledge Graph (SQLite)
            if kg_path.exists():
                shutil.copy2(kg_path, snapshot_dir / "knowledge_graph.db")
                logger.info("✓ Knowledge Graph snapshot archived: %s", snapshot_name)

            # 2. Capture System Metadata
            metadata = {
                "timestamp": time.time(),
                "human_time": time.ctime(),
                "aura_version": "v1.0.Singularity",
                "phase": "Phase 21: The Singularity Threshold",
                "status": "CONVERGENCE_ACTIVE"
            }
            get_file_write_gateway().write_text(
                snapshot_dir / "metadata.json",
                json.dumps(metadata, indent=4),
                source="resilience.eternal_record.metadata",
            )

            return snapshot_dir
        except OSError as e:
            record_degradation('eternal_record', e)
            logger.error("Failed to create Eternal Record snapshot: %s", e)
            return None

    def list_records(self):
        return sorted([d.name for d in self.records_dir.iterdir() if d.is_dir()])
