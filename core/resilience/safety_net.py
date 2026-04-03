import datetime
import logging
import sys
import traceback
from pathlib import Path
from core.config import config

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
        
        with open(report_path, "w") as f:
            f.write("=== AURA POST-MORTEM ===\n")
            f.write(f"Time: {timestamp}\n")
            f.write(f"Error: {exc_value}\n")
            f.write("\nTraceback:\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        
        logger.info("Crash report saved to %s", report_path)
    except Exception as e:
        logger.critical("Failed to write crash report: %s", e)

    # 3. (Optional) Emergency Memory Dump could go here

def install():
    sys.excepthook = panic_handler
    logger.info("Safety Net installed. Unhandled errors will be caught.")