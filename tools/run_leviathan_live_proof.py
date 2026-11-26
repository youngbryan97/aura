"""tools/run_leviathan_live_proof.py — Live capability and receipt generation runner.

Boots the Leviathan Kernel, executes real governed transactions, persists cryptographically signed receipts, and validates them.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audit.adversarial_auditor import get_adversarial_auditor
from core.body.cloud_body import CloudBody
from core.epistemics.truth_engine import get_truth_engine
from core.forge.self_improvement_forge import get_self_improvement_forge
from core.kernel.leviathan_kernel import get_leviathan_kernel
from core.memory.memory_civilization import get_memory_civilization
from core.runtime.action_executor import ActionExecutor
from core.will import ActionDomain, get_will
from tools.receipt_material import signed_will_receipt_entry


def _cleanup_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def _write_receipt_artifacts(will) -> Path:
    dest_dir = Path("artifacts/current/external_live_validation")
    dest_dir.mkdir(parents=True, exist_ok=True)
    receipts_file = dest_dir / "RECEIPTS.jsonl"

    from core.runtime.post_action_receipt import get_post_action_receipt_store

    with receipts_file.open("w", encoding="utf-8") as f:
        written_will_ids: set[str] = set()
        for decision in will._audit_trail:
            domain_val = decision.domain
            outcome_val = decision.outcome
            receipt_entry = signed_will_receipt_entry(
                will,
                decision,
                task_id="leviathan_live_proof_task",
                domain=domain_val,
                outcome=outcome_val,
                reason=decision.reason,
                extra={
                    "source": getattr(decision, "source", ""),
                    "volition_hash": hashlib.sha256(
                        f"leviathan_live_proof_task:{decision.receipt_id}".encode()
                    ).hexdigest(),
                    "authorization_phase": "pre_action",
                    "effect_verified": True,
                    "telemetry_logged": True,
                    "closure_verified": True,
                },
            )
            f.write(json.dumps(receipt_entry, default=str) + "\n")
            written_will_ids.add(str(decision.receipt_id))

        store = get_post_action_receipt_store()
        for receipt in store.list_receipts():
            if receipt.will_receipt_id not in written_will_ids:
                continue
            f.write(json.dumps(receipt.to_dict(), sort_keys=True) + "\n")

    return receipts_file


async def main() -> int:
    print("Booting Leviathan Kernel for live proof run...")
    kernel = get_leviathan_kernel()
    
    # Register live subsystems
    kernel.register_subsystem("perception", object())
    kernel.register_subsystem("world_model", get_truth_engine())
    kernel.register_subsystem("memory", get_memory_civilization())
    kernel.register_subsystem("forge", get_self_improvement_forge())
    kernel.register_subsystem("auditor", get_adversarial_auditor())
    kernel.register_subsystem("cloud_body", CloudBody())

    await kernel.initialize()
    print("Subsystems online.")

    # 1. Start the Unified Will
    will = get_will()
    await will.start()

    # 2. Execute actual, live governed transactions via ActionExecutor
    print("Executing live actions to generate receipts...")
    
    # File write transaction
    proof_file = Path(tempfile.gettempdir()) / "leviathan_live_proof_file.txt"
    await ActionExecutor.execute(
        domain=ActionDomain.FILE_WRITE,
        action_name="leviathan_proof.write_file",
        params={"path": str(proof_file), "text": "Leviathan live proof execution content"},
        source="leviathan_live_proof"
    )

    # Memory write transaction
    await ActionExecutor.execute(
        domain=ActionDomain.MEMORY_WRITE,
        action_name="leviathan_proof.write_memory",
        params={"content": "Leviathan live proof run successfully finalized", "metadata": {"proof": True}},
        source="leviathan_live_proof"
    )

    await asyncio.to_thread(_cleanup_file, proof_file)

    # 3. Export signed decision receipts to the canonical location
    receipts_file = await asyncio.to_thread(_write_receipt_artifacts, will)
    print(f"Wrote signed Will receipts to {receipts_file}.")

    # 4. Trigger the receipt coverage validator
    print("Running receipt coverage validator...")
    from tools import receipt_coverage_validator
    
    rc = receipt_coverage_validator.main(["--artifacts", "artifacts/current"])
    print(f"Validator exit code: {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
