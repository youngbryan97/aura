"""
core/resilience/cognitive_wal.py
────────────────────────────────
Ensures Aura never loses her train of thought during a power loss or system crash.
Implements a Write-Ahead Log (WAL) for cognitive intents.
"""
from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log
from pathlib import Path
import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.Resilience.WAL")

class CognitiveWAL:
    def __init__(self, filepath: Optional[str] = None):
        if filepath:
            self.filepath = Path(filepath)
        else:
            from core.config import config
            self.filepath = config.paths.data_dir / "memory" / "wal.jsonl"
            
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._pending_intents: Dict[str, Dict] = {}

    def log_intent(self, turn_id: str, action: str, target: str, context: Optional[Dict] = None, blocking: bool = False):
        """Write the thought to disk BEFORE executing it."""
        entry = {
            "time": time.time(),
            "id": turn_id,
            "action": action,
            "target": target,
            "context": context,
            "status": "pending",
            "critical": blocking
        }
        self._pending_intents[turn_id] = entry
        
        try:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(entry) + "\n")
                if blocking:
                    f.flush()
                    os.fsync(f.fileno())  # Force OS to write to disk
        except Exception as e:
            record_degradation('cognitive_wal', e)
            logger.error("Failed to write to WAL: %s", e)

    def mark_complete(self, turn_id: str):
        """Called only when the thought successfully completes."""
        if turn_id in self._pending_intents:
            entry = self._pending_intents.pop(turn_id)
            entry["status"] = "committed"
            entry["time"] = time.time()
            
            try:
                with open(self.filepath, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                record_degradation('cognitive_wal', e)
                logger.error("Failed to commit WAL entry: %s", e)

    def recover_state(self) -> List[Dict]:
        """
        Run at boot. Identifies intents that were logged but not committed.
        """
        if not os.path.exists(self.filepath):
            return []

        intents = {}
        try:
            with open(self.filepath, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        intent_id = entry.get("id")
                        if entry.get("status") == "pending":
                            intents[intent_id] = entry
                        elif entry.get("status") == "committed":
                            if intent_id in intents:
                                del intents[intent_id]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            record_degradation('cognitive_wal', e)
            logger.error("Failed to read WAL during recovery: %s", e)
            return []

        if intents:
            logger.info("💾 WAL: Recovered %s interrupted intents.", len(intents))
        return list(intents.values())

    def clear(self):
        """Prune old committed entries to keep the file small.
        
        Stability fix: preserves pending intents instead of truncating everything.
        Only removes committed/resolved entries.
        """
        try:
            if not os.path.exists(self.filepath):
                return
            if os.path.getsize(self.filepath) <= 1024 * 1024:  # 1MB
                return
                
            logger.info("💾 WAL: Pruning committed entries (preserving %d pending).", len(self._pending_intents))
            
            # Collect only pending entries from file + in-memory
            pending_entries = []
            try:
                with open(self.filepath, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("status") == "pending":
                                # Check if it's still pending (not committed later)
                                intent_id = entry.get("id")
                                if intent_id in self._pending_intents:
                                    pending_entries.append(line.strip())
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                record_degradation('cognitive_wal', e)
                capture_and_log(e, {'module': __name__})
            
            # Atomic rewrite with only pending entries
            tmp_path = str(self.filepath) + ".tmp"
            with open(tmp_path, "w") as f:
                for entry_line in pending_entries:
                    f.write(entry_line + "\n")
            
            import shutil
            shutil.move(tmp_path, str(self.filepath))
            logger.info("💾 WAL: Pruned successfully. %d pending intents preserved.", len(pending_entries))
            
        except Exception as e:
            record_degradation('cognitive_wal', e)
            logger.error("WAL clear failed: %s", e)

# Singleton instance
cognitive_wal = CognitiveWAL()


def log_intent(turn_id: str, action: str, target: str, context: Optional[Dict] = None, blocking: bool = False):
    return cognitive_wal.log_intent(turn_id, action, target, context, blocking=blocking)


def mark_complete(turn_id: str):
    return cognitive_wal.mark_complete(turn_id)


def recover_state() -> List[Dict]:
    return cognitive_wal.recover_state()


def clear():
    return cognitive_wal.clear()
