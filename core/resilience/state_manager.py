from core.runtime.errors import record_degradation
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import zlib

from core.config import config
from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway

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
            
            # 1a. Serialize payload to bytes
            data_bytes = json.dumps(snapshot, indent=2, cls=_SafeEncoder).encode('utf-8')
            checksum = zlib.crc32(data_bytes) & 0xffffffff # Force unsigned
            payload = checksum.to_bytes(4, 'big') + data_bytes
            
            with local_internal_governed_scope(
                f"resilience.state_manager.{reason}",
                receipt_prefix=f"state-manager-file-write:{reason}",
            ):
                if reason == "existential":
                    existential_path = self.snapshot_dir / "existential_snapshot.json"
                    get_file_write_gateway().write_bytes(
                        existential_path,
                        payload,
                        source="resilience.state_manager.existential_snapshot",
                    )
                    logger.info("🛡️ Hardened Existential Snapshot secured.")

                # specific snapshot for history if it's significant
                if reason in ["shutdown", "error", "manual"]:
                    history_path = self.snapshot_dir / f"snapshot_{timestamp}_{reason}.json"
                    get_file_write_gateway().write_bytes(
                        history_path,
                        payload,
                        source="resilience.state_manager.history_snapshot",
                    )
                    
                get_file_write_gateway().write_bytes(
                    latest_path,
                    payload,
                    source="resilience.state_manager.latest_snapshot",
                )
            
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
                    except (ImportError, AttributeError, RuntimeError) as er_err:
                        record_degradation('state_manager', er_err)
                        logger.error("Failed to trigger Eternal Record: %s", er_err)

            logger.debug("State snapshot saved (%s).", reason)
            return True
            
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('state_manager', e)
            logger.error("Failed to save state snapshot: %s", e)
            return False

    def load_last_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the most recent snapshot."""
        latest_path = self.snapshot_dir / "latest_snapshot.json"
        data = self._load_from_path(latest_path)
        if data is not None:
            return data

        recovered = self._load_most_recent_history_snapshot()
        if recovered is not None:
            logger.warning(
                "Recovered state from historical snapshot after latest snapshot was unavailable or invalid."
            )
        return recovered

    def load_existential_snapshot(self) -> Optional[Dict[str, Any]]:
        """Phase 18.3: Load the hardened identity snapshot."""
        return self._load_from_path(self.snapshot_dir / "existential_snapshot.json")

    def _initiate_autopsy(self, corrupted_file_path: Path):
        """Archives corrupted data for later analysis without halting the system."""
        autopsy_dir = self.snapshot_dir / "autopsy"
        autopsy_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        target_path = autopsy_dir / f"corrupted_state_{timestamp}_{corrupted_file_path.name}"
        try:
            shutil.move(str(corrupted_file_path), str(target_path))
            logger.critical("🚨 DATA CORRUPTION DETECTED: Snapshot quarantined to %s", target_path)
        except (OSError, IOError) as e:
            record_degradation('state_manager', e)
            logger.error("Failed to quarantine corrupted file %s: %s", corrupted_file_path, e)

    def _history_snapshot_paths(self) -> Iterable[Path]:
        """Return historical snapshot candidates newest-first."""
        candidates = []
        for path in self.snapshot_dir.glob("snapshot_*.json"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except (OSError, RuntimeError, ValueError) as exc:
                record_degradation(
                    "state_manager_history_snapshot_scan",
                    exc,
                    severity="warning",
                    action="skipped unreadable historical snapshot metadata",
                    extra={"path": str(path)},
                )
        for _mtime, path in sorted(candidates, key=lambda item: item[0], reverse=True):
            yield path

    def _promote_recovered_snapshot(self, source_path: Path) -> None:
        """Repair latest_snapshot.json from a verified historical snapshot."""
        latest_path = self.snapshot_dir / "latest_snapshot.json"
        try:
            payload = source_path.read_bytes()
            with local_internal_governed_scope(
                "resilience.state_manager.recovered_latest",
                receipt_prefix="state-manager-file-write:recovered-latest",
            ):
                get_file_write_gateway().write_bytes(
                    latest_path,
                    payload,
                    source="resilience.state_manager.recovered_latest_snapshot",
                )
        except (OSError, RuntimeError, AttributeError, ValueError) as exc:
            record_degradation(
                "state_manager_latest_repair",
                exc,
                severity="warning",
                action="loaded historical state but could not repair latest snapshot",
                extra={"source_path": str(source_path), "latest_path": str(latest_path)},
            )
            logger.warning("Loaded historical state but could not repair latest snapshot: %s", exc)

    def _load_most_recent_history_snapshot(self) -> Optional[Dict[str, Any]]:
        """Recover from the newest valid historical snapshot if latest is missing or corrupt."""
        for path in self._history_snapshot_paths():
            data = self._load_from_path(path)
            if data is None:
                continue
            self._promote_recovered_snapshot(path)
            logger.info("Loaded recovery snapshot from %s", path.name)
            return data
        return None

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
                        logger.error("State checksum mismatch in %s. Snapshot quarantined.", path.name)
                        return None
                        
            try:
                snapshot = json.loads(data_bytes.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as decode_error:
                self._initiate_autopsy(path)
                record_degradation(
                    "state_manager_snapshot_decode",
                    decode_error,
                    severity="warning",
                    action="quarantined unreadable snapshot and continued without recovery state",
                    extra={"path": str(path)},
                )
                logger.error("Snapshot %s was unreadable and has been quarantined: %s", path, decode_error)
                return None
                
            meta = snapshot.get("meta", {})
            data = snapshot.get("data", {})
            
            logger.info("Loaded snapshot from %s (Reason: %s)", meta.get('iso_time'), meta.get('reason'))
            return data
            
        except (OSError, ConnectionError, TimeoutError, zlib.error, ValueError) as e:
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
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('state_manager', exc)
                logger.debug("Suppressed: %s", exc)

        return sorted(snapshots, key=lambda x: x['time'], reverse=True)
