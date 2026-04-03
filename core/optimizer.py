import json
import logging
import os
import shutil
from typing import Any, Dict, List

try:
    from core.brain.cognitive_patch import CognitivePatchStrategy
except ImportError:
    CognitivePatchStrategy = None  # type: ignore

try:
    from core.patch_library import AVAILABLE_PATCHES, PipInstallPatch
except ImportError:
    AVAILABLE_PATCHES = []  # type: ignore
    PipInstallPatch = None  # type: ignore

logger = logging.getLogger("Kernel.Optimizer")

class Optimizer:
    def __init__(self, data_file="autonomy_engine/data/hard_examples.json"):
        self.data_file = data_file
        
    async def run(self):
        """Main optimization loop:
        1. Read Hard Examples.
        2. Group by reason.
        3. Match with PatchLibrary.
        4. Apply fixes.
        """
        if not os.path.exists(self.data_file):
            return

        try:
            with open(self.data_file, 'r') as f:
                failures = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Corrupted hard_examples.json. resetting.")
            return

        if not failures:
            return

        logger.info("Optimizer analyzing %d failures...", len(failures))
        
        fixed_count = 0
        
        # Analyze unique failure reasons to avoid redundant patching
        unique_reasons = set(f.get("reason", "") + " " + str(f.get("outcome", "")) for f in failures)
        
        for signature in unique_reasons:
            handled = False
            for patch in AVAILABLE_PATCHES:
                if patch.match(signature):
                    logger.info("Strategy Match: %s for failure '%s...'", patch.name, signature[:50])
                    
                    # Special handling for patches that need the signature (like pip install)
                    if isinstance(patch, PipInstallPatch):
                        success = await patch.apply(signature)
                    else:
                        success = await patch.apply()
                        
                    if success:
                        fixed_count += 1
                        handled = True
                        break # One patch per issue type
            
            if not handled:
                logger.info("No heuristic match. Escalate to Cognitive Engine...")
                # Fallback: Ask the Brain
                cog_patch = CognitivePatchStrategy()
                if await cog_patch.apply(signature):
                     fixed_count += 1
                     handled = True

        if fixed_count > 0:
            logger.info("Optimizer applied %s patches. Archiving failures.", fixed_count)
            self._archive_dataset()
            
    def _archive_dataset(self):
        # Move hard_examples to archive to prevent re-processing same events
        if os.path.exists(self.data_file):
            archive_path = self.data_file + ".processed"
            # Append mode for archive? For now just overwrite or rotate.
            # Simple rotation:
            if os.path.exists(archive_path):
                os.remove(archive_path)
            shutil.move(self.data_file, archive_path)
            # Create empty new file
            with open(self.data_file, 'w') as f:
                json.dump([], f)