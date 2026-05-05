#!/usr/bin/env python3
import subprocess
import time
import os
from pathlib import Path

GAUNTLET_TESTS = {
    "Closure Gauntlet": "tests/test_audit_chain.py",
    "Activation Audit": "tests/test_causal_exclusion.py",
    "Headless Environment Stress": "tests/test_headless_live_boot.py",
    "Replay Learning Improvement": "tests/test_canary_replay_real.py",
    "Abstraction Transfer": "tests/test_grounding_and_plasticity.py",
    "Self-Mod Rollback Drill": "tests/test_restore_drill.py",
    "Production 32B CAA Validation": "tests/steering/test_caa_32b.py",
    "Long-Run Stability Trace": "tests/test_long_run_model.py",
    "External Task Performance": "tests/test_agent_workspace_integrations.py"
}

def run_gauntlet():
    root_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(root_dir)
    
    results_file = root_dir / "tests" / "CAPABILITY_GAUNTLET_RESULTS.md"
    
    with open(results_file, "w") as f:
        f.write("# Aura Capability Gauntlet Results\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Status:** Execution Complete\n\n")
        
        f.write("## Execution Summary\n\n")
        
        for name, path in GAUNTLET_TESTS.items():
            print(f"Running: {name} ({path})...")
            
            start_time = time.time()
            try:
                # We use pytest to run each file. We capture output.
                # Adding -v to get more verbose output but limiting to stdout.
                # If a test fails because of missing dependencies or long runtimes, we still capture it.
                # Find correct pytest path
                pytest_cmd = ["pytest"]
                if (root_dir / ".venv" / "bin" / "pytest").exists():
                    pytest_cmd = [str(root_dir / ".venv" / "bin" / "pytest")]

                result = subprocess.run(
                    pytest_cmd + ["-v", "--tb=short", path],
                    capture_output=True,
                    text=True,
                    timeout=300 # 5 minute timeout per suite
                )
                duration = time.time() - start_time
                status = "✅ PASS" if result.returncode == 0 else "❌ FAIL"
                
                f.write(f"### {name}\n")
                f.write(f"- **File:** `{path}`\n")
                f.write(f"- **Status:** {status}\n")
                f.write(f"- **Duration:** {duration:.2f}s\n\n")
                
                f.write("```text\n")
                # Truncate output to avoid massive files
                output = result.stdout + "\n" + result.stderr
                if len(output) > 2000:
                    output = output[:1000] + "\n...[TRUNCATED]...\n" + output[-1000:]
                f.write(output)
                f.write("\n```\n\n")
                
                print(f"  -> {status} in {duration:.2f}s")
                
            except subprocess.TimeoutExpired:
                f.write(f"### {name}\n")
                f.write(f"- **File:** `{path}`\n")
                f.write(f"- **Status:** ⚠️ TIMEOUT (>300s)\n\n")
                print(f"  -> ⚠️ TIMEOUT")
            except Exception as e:
                f.write(f"### {name}\n")
                f.write(f"- **File:** `{path}`\n")
                f.write(f"- **Status:** 💥 ERROR ({e})\n\n")
                print(f"  -> 💥 ERROR: {e}")

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    run_gauntlet()
