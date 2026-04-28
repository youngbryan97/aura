from core.runtime.errors import record_degradation
import json
import logging
import shutil
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import zlib

from core.config import config

logger = logging.getLogger("Core.Resilience.StateManager")


class _SafeEncoder(json.JSONEncoder):
    """Custom encoder that handles Enums, numpy types, and other non-serializable objects."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError as _e:
            logger.debug('Ignored ImportError in state_manager.py: %s', _e)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


class StateManager:
    """Manages system state snapshots for resilience and recovery.
    Saves critical data (memory, configuration, active tasks) to disk.
    """
    
    def __init__(self):
        self.snapshot_dir = Path(config.paths.data_dir) / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
    async def save_snapshot_async(self, orchestrator_state: Dict[str, Any], reason: str = "periodic") -> bool:
        """Asynchronously save a snapshot using a background thread."""
        from core.utils.executor import run_in_thread
        return await run_in_thread(self.save_snapshot, orchestrator_state, reason)

    def save_snapshot(self, orchestrator_state: Dict[str, Any], reason: str = "periodic") -> bool:
        """Save a snapshot of the current system state.
        
        Args:
            orchestrator_state: Dict containing current orchestrator data (memory, goals, etc)
            reason: Why the snapshot is being taken (periodic, shutdown, error)

        """
        try:
            timestamp = int(time.time())
            iso_time = datetime.now().isoformat()
            
            snapshot = {
                "meta": {
                    "timestamp": timestamp,
                    "iso_time": iso_time,
                    "reason": reason,
                    "version": "3.4"
                },
                "data": orchestrator_state
            }
            
            # 1. Save "latest" snapshot (for quick recovery)
            latest_path = self.snapshot_dir / "latest_snapshot.json"
            
            # Write to temp file first for atomicity (with Checksum)
            temp_path = latest_path.with_suffix(".tmp")
            
            # 1a. Serialize payload to bytes
            data_bytes = json.dumps(snapshot, indent=2, cls=_SafeEncoder).encode('utf-8')
            checksum = zlib.crc32(data_bytes) & 0xffffffff # Force unsigned
            
            with open(temp_path, 'wb') as f:
                f.write(checksum.to_bytes(4, 'big'))
                f.write(data_bytes)
            
            if reason == "existential":
                existential_path = self.snapshot_dir / "existential_snapshot.json"
                shutil.copy2(temp_path, existential_path)
                logger.info("🛡️ Hardened Existential Snapshot secured.")

            # specific snapshot for history if it's significant
            if reason in ["shutdown", "error", "manual"]:
                history_path = self.snapshot_dir / f"snapshot_{timestamp}_{reason}.json"
                shutil.copy2(temp_path, history_path)
                
            # Rename temp to latest
            temp_path.replace(latest_path)
            
            # Wire EternalRecord for long-term persistence
            if reason in ["manual", "existential", "shutdown"]:
                try:
                    from core.resilience.eternal_record import EternalRecord
                    # Use the parent of snapshots dir as brain_dir
                    brain_dir = self.snapshot_dir.parent
                    recorder = EternalRecord(brain_dir)
                    
                    # Snapshot the Knowledge Graph if it exists
                    kg_path = Path(config.paths.data_dir) / "knowledge_graph.db"
                    recorder.create_snapshot(kg_path)
                    logger.info("🏺 Eternal Record snapshot triggered via StateManager (%s)", reason)
                except Exception as er_err:
                    record_degradation('state_manager', er_err)
                    logger.error("Failed to trigger Eternal Record: %s", er_err)

            logger.debug("State snapshot saved (%s).", reason)
            return True
            
        except Exception as e:
            record_degradation('state_manager', e)
            logger.error("Failed to save state snapshot: %s", e)
            return False

    def load_last_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the most recent snapshot."""
        return self._load_from_path(self.snapshot_dir / "latest_snapshot.json")

    def load_existential_snapshot(self) -> Optional[Dict[str, Any]]:
        """Phase 18.3: Load the hardened identity snapshot."""
        return self._load_from_path(self.snapshot_dir / "existential_snapshot.json")

    def _initiate_autopsy(self, corrupted_file_path: Path):
        """Archives corrupted data for later analysis without halting the system."""
        autopsy_dir = self.snapshot_dir / "autopsy"
        autopsy_dir.mkdir(exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        target_path = autopsy_dir / f"corrupted_state_{timestamp}_{corrupted_file_path.name}"
        try:
            shutil.move(str(corrupted_file_path), str(target_path))
            logger.critical("🚨 DATA CORRUPTION DETECTED: Snapshot quarantined to %s", target_path)
        except Exception as e:
            record_degradation('state_manager', e)
            logger.error("Failed to quarantine corrupted file %s: %s", corrupted_file_path, e)

    def _load_from_path(self, path: Path) -> Optional[Dict[str, Any]]:
        """Generic loader logic with Checksum verification."""
        try:
            if not path.exists():
                logger.debug("Snapshot path %s does not exist.", path)
                return None
                
            with open(path, 'rb') as f:
                header = f.read(4)
                
                # Fallback for old un-checksummed JSON strings (starts with '{' == 123)
                if len(header) > 0 and header[0] == 123:
                    f.seek(0)
                    data_bytes = f.read()
                    logger.warning("Loading unchecksummed legacy snapshot at %s", path)
                else:
                    checksum_from_file = int.from_bytes(header, 'big')
                    data_bytes = f.read()
                    calculated_checksum = zlib.crc32(data_bytes) & 0xffffffff # Force unsigned
                    
                    if checksum_from_file != calculated_checksum:
                        self._initiate_autopsy(path)
                        raise ValueError(f"State checksum mismatch in {path.name}! File corrupted.")
                        
            snapshot = json.loads(data_bytes.decode('utf-8'))
                
            meta = snapshot.get("meta", {})
            data = snapshot.get("data", {})
            
            logger.info("Loaded snapshot from %s (Reason: %s)", meta.get('iso_time'), meta.get('reason'))
            return data
            
        except Exception as e:
            record_degradation('state_manager', e)
            logger.error("Failed to load snapshot from %s: %s", path, e)
            return None

    def get_snapshot_history(self) -> list:
        """List available snapshots."""
        snapshots = []
        for f in self.snapshot_dir.glob("snapshot_*.json"):
            try:
                snapshots.append({
                    "path": str(f),
                    "name": f.name,
                    "size": f.stat().st_size,
                    "time": f.stat().st_mtime
                })
            except Exception as exc:
                record_degradation('state_manager', exc)
                logger.debug("Suppressed: %s", exc)

        return sorted(snapshots, key=lambda x: x['time'], reverse=True)