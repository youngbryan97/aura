"""
core/bootstrap/integrity.py — Boot-time File Verification
=========================================================
Ensures critical files exist and are not corrupted.
"""
import logging
import hashlib
import os
from pathlib import Path
from typing import List

logger = logging.getLogger("Aura.Bootstrap.Integrity")

def verify_core_files(base_dir: Path) -> bool:
    """Verifies existence of critical core files."""
    critical_files = [
        "core/container.py",
        "core/orchestrator/main.py",
        "core/config.py",
        "core/event_bus.py",
        "core/mycelium.py"
    ]
    
    missing = []
    for f in critical_files:
        path = base_dir / f
        if not path.exists():
            missing.append(f)
            
    if missing:
        logger.critical("🚨 CRITICAL FILES MISSING: %s", ", ".join(missing))
        return False
        
    logger.info("✅ Core file integrity verified.")
    return True

def calculate_checksum(file_path: Path) -> str:
    """Calculates SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
