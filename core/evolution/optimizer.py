import json
import logging
import os
import shutil
from typing import Any, Dict, List

from core.runtime.file_write_gateway import get_file_write_gateway

try:
    from core.brain.cognitive_patch import CognitivePatchStrategy
except ImportError:
    CognitivePatchStrategy = None  # type: ignore

from core.self_modification.patch_library import PatchStrategy, get_patches

logger = logging.getLogger("Kernel.Optimizer")

class Optimizer:
    def __init__(self, data_file="autonomy_engine/data/hard_examples.json", *, patches: List[PatchStrategy] | None = None):
        self.data_file = data_file
        self.patches = list(patches) if patches is not None else get_patches()

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
            for patch in self.patches:
                if patch.match(signature):
                    logger.info("Strategy Match: %s for failure '%s...'", patch.name, signature[:50])

                    success = await patch.apply(signature)

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
            get_file_write_gateway().write_text(
                self.data_file,
                json.dumps([]),
                source="optimizer.archive_dataset",
            )
