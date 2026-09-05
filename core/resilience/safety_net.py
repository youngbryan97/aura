from core.runtime.errors import record_degradation
import datetime
import io
import logging
import sys
import traceback
from pathlib import Path
from core.config import config
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("SafetyNet")

def panic_handler(exc_type, exc_value, exc_traceback):
    """The final safety net. Catches unhandled crashes.
    """
    # Ignore KeyboardInterrupt (Ctrl+C) so you can still stop her
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # 1. Log the crash to console
    logger.critical("🔥 UNHANDLED EXCEPTION. SYSTEM CRITICAL.", exc_info=(exc_type, exc_value, exc_traceback))
    
    # 2. Write a Post-Mortem Report (for you to debug later)
    crash_dir = config.paths.data_dir / "crashes"
    try:
        crash_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = crash_dir / f"crash_{timestamp}.txt"
        
        buffer = io.StringIO()
        buffer.write("=== AURA POST-MORTEM ===\n")
        buffer.write(f"Time: {timestamp}\n")
        buffer.write(f"Error: {exc_value}\n")
        buffer.write("\nTraceback:\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=buffer)
        get_file_write_gateway().write_text(
            report_path,
            buffer.getvalue(),
            source="resilience.safety_net.post_mortem",
        )
        
        logger.info("Crash report saved to %s", report_path)
    except (OSError, IOError) as e:
        record_degradation('safety_net', e)
        logger.critical("Failed to write crash report: %s", e)

    # 3. (Optional) Emergency Memory Dump could go here

def install():
    sys.excepthook = panic_handler
    logger.info("Safety Net installed. Unhandled errors will be caught.")
