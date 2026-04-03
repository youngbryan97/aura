"""Memory Management System with Atomic Operations and Corruption Recovery.

This module provides persistent memory storage with automatic pruning,
corruption detection, and atomic write operations to prevent data loss.
"""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("Kernel.Memory")


from core.memory.base import MemoryEvent, MemoryType


class Memory:
    """Robust memory management system with:
    1. Atomic write operations to prevent corruption
    2. Automatic corruption detection and recovery
    3. Thread-safe operations
    4. Memory pruning to prevent unbounded growth
    5. Checksum validation for data integrity
    """
    
    # Class constants
    MAX_EPISODIC_ENTRIES = 1000
    AUTO_SAVE_THRESHOLD = 100
    BACKUP_COUNT = 3
    
    def __init__(self, storage_file: str = "autonomy_engine/memory/memory_v9.json"):
        """Initialize memory system.
        
        Args:
            storage_file: Path to memory storage file

        """
        self.storage_file = Path(storage_file)
        self._lock = threading.RLock()
        self._dirty_count = 0
        
        # Default data structure
        self.data = {
            "episodic": [],
            "semantic": {},
            "goals": [],
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_modified": None,
                "checksum": None
            }
        }
        
        # Ensure directory exists
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory
        self.load()
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 checksum of memory data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _validate_data_structure(self, data: Dict[str, Any]) -> bool:
        """Validate memory data structure integrity."""
        required_keys = {"episodic", "semantic", "goals"}
        if not all(key in data for key in required_keys):
            return False
        
        # Validate episodic is a list
        if not isinstance(data["episodic"], list):
            return False
        
        # Validate semantic is a dict
        if not isinstance(data["semantic"], dict):
            return False
        
        # Validate goals is a list
        if not isinstance(data["goals"], list):
            return False
        
        return True
    
    def _create_backup(self) -> None:
        """Create a backup of the current memory file."""
        if not self.storage_file.exists():
            return
        
        # Rotate backups (keep last N)
        backup_dir = self.storage_file.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{self.storage_file.stem}_{timestamp}.json"
        
        try:
            with self._lock:
                # Read current file
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Write backup
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.debug("Created memory backup: %s", backup_file.name)
                
                # Clean old backups
                backups = sorted(backup_dir.glob("*.json"))
                if len(backups) > self.BACKUP_COUNT:
                    for old_backup in backups[:-self.BACKUP_COUNT]:
                        old_backup.unlink()
                        
        except Exception as e:
            logger.error("Failed to create backup: %s", e)
    
    def _atomic_write(self, data: Dict[str, Any]) -> bool:
        """Atomic write with rollback capability.
        
        Args:
            data: Data to write
            
        Returns:
            True if successful, False otherwise

        """
        temp_file = self.storage_file.with_suffix(".tmp")
        backup_file = self.storage_file.with_suffix(".bak")
        
        try:
            # Update metadata
            data["metadata"]["last_modified"] = datetime.now().isoformat()
            data["metadata"]["checksum"] = None # Reset before calculation for consistency
            data["metadata"]["checksum"] = self._calculate_checksum(data)
            
            # Write to temporary file
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Create backup of current file if it exists
            if self.storage_file.exists():
                self.storage_file.replace(backup_file)
            
            # Move temp file to final location
            temp_file.replace(self.storage_file)
            
            # Clean up backup if everything succeeded
            if backup_file.exists():
                backup_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error("Atomic write failed: %s", e)
            
            # Attempt rollback
            try:
                if backup_file.exists():
                    if self.storage_file.exists():
                        self.storage_file.unlink()
                    backup_file.replace(self.storage_file)
            except Exception as rollback_error:
                logger.critical("Rollback failed: %s", rollback_error)
            
            # Clean up temp file
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as exc:
                    logger.debug("Suppressed: %s", exc)

            return False
    
    def load(self) -> None:
        """Load memory from storage file with corruption recovery.
        
        Attempts to load from primary file, falls back to backup if corrupted.
        """
        try:
            if not self.storage_file.exists():
                logger.info("Memory file not found. Initializing new memory.")
                self.save()
                return
            
            with self._lock:
                # Try to load main file
                data = self._load_file(self.storage_file)
                
                if data is None:
                    # Try backup file
                    backup_file = self.storage_file.with_suffix(".bak")
                    if backup_file.exists():
                        logger.warning("Loading from backup due to corruption")
                        data = self._load_file(backup_file)
                
                if data is None:
                    logger.error("All memory files corrupted. Starting fresh.")
                    self._create_backup()  # Backup corrupted file
                else:
                    self.data = data
                    logger.info("Memory loaded successfully (%d events)", len(self.data['episodic']))
                    
        except Exception as e:
            logger.error("Memory load failed: %s", e)
            # Continue with default data
    
    def _load_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and validate a single memory file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if not self._validate_data_structure(data):
                logger.error("Invalid memory structure in %s", file_path.name)
                return None
            
            # Validate checksum if present
            if "metadata" in data and "checksum" in data["metadata"]:
                stored_checksum = data["metadata"]["checksum"]
                data["metadata"]["checksum"] = None  # Remove for calculation
                calculated_checksum = self._calculate_checksum(data)
                
                if stored_checksum != calculated_checksum:
                    logger.error("Checksum mismatch in %s", file_path.name)
                    return None
                
                # Restore checksum
                data["metadata"]["checksum"] = stored_checksum
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error("JSON decode error in %s: %s", file_path.name, e)
            return None
        except Exception as e:
            logger.error("Error reading %s: %s", file_path.name, e)
            return None
    
    def save(self) -> bool:
        """Save memory to storage with atomic write.
        
        Returns:
            True if save successful, False otherwise

        """
        with self._lock:
            try:
                # Create backup before save
                self._create_backup()
                
                # Prune episodic memory if needed
                self._prune_episodic()
                
                # Perform atomic write
                success = self._atomic_write(self.data)
                
                if success:
                    self._dirty_count = 0
                    logger.debug("Memory saved successfully")
                else:
                    logger.error("Memory save failed")
                
                return success
                
            except Exception as e:
                logger.error("Save operation failed: %s", e, exc_info=True)
                return False
    
    def _prune_episodic(self) -> None:
        """Prune episodic memory to prevent unbounded growth."""
        episodic = self.data["episodic"]
        if len(episodic) > self.MAX_EPISODIC_ENTRIES:
            # Keep most recent entries
            self.data["episodic"] = episodic[-self.MAX_EPISODIC_ENTRIES:]
            logger.info("Pruned episodic memory to %s entries", self.MAX_EPISODIC_ENTRIES)
    
    def log_event(self, event: Union[Dict[str, Any], MemoryEvent]) -> bool:
        """Log a memory event with auto-save threshold.
        
        Args:
            event: Event data as dict or MemoryEvent instance
            
        Returns:
            True if event logged successfully

        """
        try:
            with self._lock:
                # Convert dict to MemoryEvent if needed
                if isinstance(event, dict):
                    event = MemoryEvent.from_dict(event)
                
                # Add to episodic memory
                self.data["episodic"].append(event.to_dict())
                
                # Increment dirty counter
                self._dirty_count += 1
                
                # Auto-save if threshold reached
                if self._dirty_count >= self.AUTO_SAVE_THRESHOLD:
                    self.save()
                
                logger.debug("Logged event: %s", event.event_type)
                return True
                
        except ValueError as e:
            logger.error("Invalid event data: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to log event: %s", e)
            return False
    
    def update_semantic(self, key: str, value: Any) -> bool:
        """Update semantic memory.
        
        Args:
            key: Semantic key
            value: Value to store
            
        Returns:
            True if update successful

        """
        if not isinstance(key, str) or not key.strip():
            logger.error("Invalid semantic key")
            return False
        
        try:
            with self._lock:
                self.data["semantic"][key] = value
                self._dirty_count += 1
                
                # Auto-save if threshold reached
                if self._dirty_count >= self.AUTO_SAVE_THRESHOLD:
                    self.save()
                
                logger.debug("Updated semantic memory: %s", key)
                return True
                
        except Exception as e:
            logger.error("Failed to update semantic memory: %s", e)
            return False
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent memory events.
        
        Args:
            count: Number of events to retrieve
            
        Returns:
            List of recent events

        """
        with self._lock:
            events = self.data["episodic"][-count:]
            return events.copy() if events else []
    
    def get_semantic(self, key: str, default: Any = None) -> Any:
        """Get value from semantic memory.
        
        Args:
            key: Semantic key
            default: Default value if key not found
            
        Returns:
            Value or default

        """
        with self._lock:
            return self.data["semantic"].get(key, default)
    
    def clear_episodic(self) -> bool:
        """Clear all episodic memory."""
        with self._lock:
            try:
                self.data["episodic"] = []
                self._dirty_count += 1
                self.save()
                logger.info("Cleared episodic memory")
                return True
            except Exception as e:
                logger.error("Failed to clear episodic memory: %s", e)
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            return {
                "episodic_entries": len(self.data["episodic"]),
                "semantic_entries": len(self.data["semantic"]),
                "goal_entries": len(self.data["goals"]),
                "last_modified": self.data["metadata"].get("last_modified"),
                "dirty_count": self._dirty_count
            }

# --- Standalone Atomic Utilities (ISSUE 14) ---

def atomic_write(file_path: str, content: str) -> None:
    """Thread-safe atomic write utility using a temporary file and atomic rename."""
    path = Path(file_path)
    temp_path = path.with_suffix(".tmp")
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file using context manager (ensures closure)
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
            
        # Atomic rename
        temp_path.replace(path)
    except Exception as e:
        logger.error("Standalone atomic write failed for %s: %s", file_path, e)
        if temp_path.exists():
            temp_path.unlink()
        raise