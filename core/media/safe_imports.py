"""core/media/safe_imports.py — macOS Compatibility Layer
Ensures media stack doesn't crash on macOS when multiple backends collide.
"""
import sys
import os
import logging

logger = logging.getLogger("Aura.MediaSafe")

def prevent_collisions():
    """Apply environment and import shims for macOS media stability."""
    if sys.platform != "darwin":
        return

    # Patch 24: Fix media stack collisions on macOS
    # Prevents "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized."
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Prevents CV2/AVFoundation deadlocks
    os.environ["OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION"] = "1"
    
    logger.info("🍎 macOS Media Collision Guards applied.")

# Auto-apply on import
prevent_collisions()
