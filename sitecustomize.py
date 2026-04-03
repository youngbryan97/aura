import sys
import os
from pathlib import Path

# Aura Site-Customize: Early Path Locking
# This script is automatically imported by Python at startup.
# It ensures the virtual environment and project root are always in sys.path.

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Look for venv
VENV_PATH = PROJECT_ROOT / ".venv"
if not VENV_PATH.exists():
    VENV_PATH = PROJECT_ROOT / ".venv_aura"

if VENV_PATH.exists():
    # Construct the platform-specific site-packages path
    lib_dir = VENV_PATH / "lib"
    if lib_dir.exists():
        # Scan for all python3.x directories to ensure compatibility if runtime version changes
        for py_dir in lib_dir.glob("python3.*"):
            site_packages = py_dir / "site-packages"
            if site_packages.exists() and str(site_packages) not in sys.path:
                sys.path.insert(0, str(site_packages))
                # Trigger site-packages processing (for .pth files)
                import site
                site.addsitedir(str(site_packages))

# Set stable thread counts for Apple Silicon (Safe Baseline)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
