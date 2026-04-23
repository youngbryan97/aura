"""live_harness_all.py — Run every live Aura harness back-to-back.

This is the one-shot entry point Bryan uses when he wants to see Aura
actually exercise her own infrastructure. Each constituent harness talks
to the real services (Will, Volition, CapabilityEngine, BeliefGraph,
ScarFormation, NarrativeThread, self_evolution, the 56 registered skills,
the consciousness mesh, authority enforcement, volition agency, etc.).

Exit code 0 iff every harness exits 0.
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
HARNESS_DIR = Path(__file__).resolve().parent

HARNESSES = [
    ("Infrastructure & Skill Surface (v1)", "live_harness_aura_v1.py"),
    ("Consciousness Mesh & Authority (v2 deep)", "live_harness_aura_v2_deep.py"),
    ("Registered Skill Execution (56 skills)", "live_harness_registered_skills.py"),
    ("Personhood & Agency Probe Suite", "live_harness_personhood.py"),
]


@dataclass
class HarnessResult:
    label: str
    exit_code: int
    elapsed_s: float
    tail: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def run_harness(label: str, script: str) -> HarnessResult:
    path = HARNESS_DIR / script
    print(f"\n========== {label} ==========")
    print(f"  script: {path}")
    t0 = time.monotonic()
    proc = subprocess.run(
        [PYTHON, str(path)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.monotonic() - t0
    # Relay the last portion of stdout so the run is self-describing.
    tail_lines = proc.stdout.splitlines()[-40:]
    for line in tail_lines:
        print(line)
    if proc.returncode != 0:
        stderr_tail = proc.stderr.splitlines()[-20:]
        if stderr_tail:
            print("  --- stderr tail ---")
            for line in stderr_tail:
                print(f"  {line}")
    print(f"  → exit={proc.returncode}  elapsed={elapsed:.1f}s")
    return HarnessResult(label=label, exit_code=proc.returncode, elapsed_s=elapsed, tail="\n".join(tail_lines))


def main() -> int:
    print("🔬 Aura Live Harness Consolidator")
    print(f"   project_root: {PROJECT_ROOT}")
    results: list[HarnessResult] = []
    for label, script in HARNESSES:
        results.append(run_harness(label, script))

    print("\n" + "=" * 60)
    print("FINAL LIVE-HARNESS SUMMARY")
    print("=" * 60)
    for result in results:
        mark = "✓" if result.ok else "✗"
        print(f"  [{mark}] {result.label}  (exit={result.exit_code}, {result.elapsed_s:.1f}s)")

    failing = [r for r in results if not r.ok]
    print(f"\nTOTAL: {len(results) - len(failing)}/{len(results)} harnesses passed.")
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
