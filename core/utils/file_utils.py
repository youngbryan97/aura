import json
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


def atomic_write_json(file_path: str, data: Any, indent: int = 2) -> None:
    """
    Writes JSON data to a file atomically using a temporary file and os.replace.
    Ensures that partial writes don't corrupt the target file.
    """
    file_path = str(file_path) # Ensure path is string
    dir_name = os.path.dirname(file_path)
    
    # Create directory if it doesn't exist
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    fd, temp_path = tempfile.mkstemp(dir=dir_name or ".", prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            # Ensure data is written to disk
            os.fsync(f.fileno())
        
        # Atomic rename (on POSIX, this is atomic)
        os.replace(temp_path, file_path)
    except Exception as e:
        logger.error(f"Atomic write to {file_path} failed: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as _e:
                logger.debug('Ignored OSError in file_utils.py: %s', _e)
        raise
