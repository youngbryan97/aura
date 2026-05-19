#!/usr/bin/env python3
"""Cross-platform diagnostic bootstrap preflight setup for Aura.

Verifies environment health, imports, database schemas, and hardware
capabilities to ensure clean and portable runs across macOS, Linux, and Windows.
"""

from __future__ import annotations

import sys
import os
import platform
import sqlite3
import json
import time
from pathlib import Path
from typing import Any, Dict

# Ensure core is on the path if run directly
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

# Path configs
DB_PATH = BASE_DIR / "tests" / "test_projects.db"
REPORT_PATH = BASE_DIR / "artifacts" / "unity" / "latest" / "PREFLIGHT_HEALTH.json"


def _check_system() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_version_ok": sys.version_info >= (3, 10),
    }


def _check_dependencies() -> dict[str, Any]:
    deps = ["pydantic", "psutil", "pytest", "asyncio"]
    status: dict[str, Any] = {}
    all_ok = True
    
    for dep in deps:
        try:
            __import__(dep)
            status[dep] = "available"
        except ImportError:
            status[dep] = "missing"
            all_ok = False
            
    # Check PyAutoGUI
    try:
        __import__("pyautogui")
        status["pyautogui"] = "available"
    except ImportError:
        status["pyautogui"] = "limited (desktop clicks disabled)"
        
    status["core_dependencies_ok"] = all_ok
    return status


def _check_database() -> dict[str, Any]:
    report = {
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
        "schema_ok": False,
        "detail": "",
    }
    
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        cursor = conn.cursor()
        
        # Test projects table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projects'")
        projects_exist = cursor.fetchone() is not None
        
        # Test tasks table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
        tasks_exist = cursor.fetchone() is not None
        
        if not (projects_exist and tasks_exist):
            # Try initializing DB via ProjectStore to repair schema
            from core.data.project_store import ProjectStore
            store = ProjectStore(str(DB_PATH))
            report["detail"] = "Database schema initialized/repaired successfully."
        else:
            report["detail"] = "Existing projects and tasks tables verified."
            
        report["schema_ok"] = True
        conn.close()
    except Exception as e:
        report["detail"] = f"Database diagnostic failed: {e}"
        report["schema_ok"] = False
        
    return report


def _check_hardware_accelerator() -> dict[str, Any]:
    is_mac = platform.system() == "Darwin"
    is_arm = "arm" in platform.machine().lower() or "aarch64" in platform.machine().lower()
    
    report = {
        "is_apple_silicon": is_mac and is_arm,
        "mlx_available": False,
        "torch_available": False,
        "torch_device": "cpu",
        "guidance": "",
    }
    
    # Check MLX
    try:
        __import__("mlx.core")
        report["mlx_available"] = True
    except ImportError:
        pass
        
    # Check PyTorch
    try:
        torch = __import__("torch")
        report["torch_available"] = True
        if torch.cuda.is_available():
            report["torch_device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            report["torch_device"] = "mps"
    except ImportError:
        pass
        
    # Determine guidance
    if report["is_apple_silicon"] and not report["mlx_available"]:
        report["guidance"] = "Recommendation: Install 'mlx' to leverage Apple Silicon local ML capabilities."
    elif not report["torch_available"]:
        report["guidance"] = "Recommendation: Run 'pip install torch' to access complete cognitive model verification."
    else:
        report["guidance"] = "Hardware acceleration configuration is healthy."
        
    return report


def main() -> int:
    print("=" * 60)
    print(" 🛠️  Aura Sovereign Stack: Preflight Bootstrap Diagnostic")
    print("=" * 60)
    
    system = _check_system()
    print(f"System:      {system['os']} {system['release']} ({system['machine']})")
    print(f"Python:      {system['python_version']} [{'PASS' if system['python_version_ok'] else 'FAIL'}]")
    
    deps = _check_dependencies()
    print(f"Core Deps:   {'PASS' if deps['core_dependencies_ok'] else 'FAIL (check local venv)'}")
    for k, v in deps.items():
        if k != "core_dependencies_ok":
            print(f"  - {k}: {v}")
            
    db = _check_database()
    print(f"Database:    {'PASS' if db['schema_ok'] else 'FAIL'}")
    print(f"  - Path:    {db['db_path']}")
    print(f"  - Status:  {db['detail']}")
    
    ml = _check_hardware_accelerator()
    print(f"Accelerator: Device={ml['torch_device']} (MLX={ml['mlx_available']}, PyTorch={ml['torch_available']})")
    if ml["guidance"]:
        print(f"  - Note:    {ml['guidance']}")
        
    # Compile full telemetry
    report = {
        "generated_at_unix": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": system,
        "dependencies": deps,
        "database": db,
        "accelerator": ml,
        "overall_healthy": system["python_version_ok"] and deps["core_dependencies_ok"] and db["schema_ok"],
    }
    
    # Save report
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n✓ Diagnostic health report saved to: {REPORT_PATH}")
    except OSError as e:
        print(f"\n⚠️ Failed to write diagnostic health report: {e}")
        
    print("=" * 60)
    return 0 if report["overall_healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
