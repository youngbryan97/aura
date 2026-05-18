import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Add base dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.world_model.acg import ActionConsequenceGraph


async def test_acg():
    print("Testing Action-Consequence Graph (ACG)...")

    # Use a temp path for testing
    test_path = Path(tempfile.gettempdir()) / "test_causal_graph.json"
    if test_path.exists():
        test_path.unlink()

    acg = ActionConsequenceGraph(persist_path=str(test_path))

    # 1. Record an action
    target_path = Path(tempfile.gettempdir()) / "test.txt"
    action = {"tool": "write_file", "params": {"TargetFile": str(target_path)}}
    context = "Testing initial causal link"
    outcome = {"ok": True, "message": "File written"}
    success = True

    acg.record_outcome(action, context, outcome, success)

    # 2. Check persistence
    if not test_path.exists():
        raise RuntimeError("ACG failed to save to disk")
    data = json.loads(await asyncio.to_thread(test_path.read_text))
    if len(data) != 1:
        raise RuntimeError("Incorrect number of links saved")
    if data[0]["action"] != "write_file":
        raise RuntimeError("Assertion failed")

    print("✓ ACG Recording & Persistence: PASS")

    # 3. Query consequences
    matches = acg.query_consequences("write_file", {"TargetFile": str(target_path)})
    if len(matches) != 1:
        raise RuntimeError("Failed to retrieve matched action")
    if not matches[0]["outcome"]["ok"]:
        raise RuntimeError("Assertion failed")

    print("✓ ACG Query & Retrieval: PASS")

    # 4. Param mismatch test
    misses = acg.query_consequences(
        "write_file",
        {"TargetFile": str(Path(tempfile.gettempdir()) / "different.txt")},
    )
    # Since my overlap logic is lenient (matches/common > 0.5), and there is only 1 common key here...
    # If keys are same but values different, it should be a miss.
    # Actually _params_overlap simple check: matches is 0, len(common) is 1. 0/1 is 0. <= 0.5. Miss.
    if misses:
        raise RuntimeError("ACG matched action despite param mismatch")

    print("✓ ACG Parameter Filtering: PASS")

    print("\n--- ACG VERIFICATION SUCCESSFUL ---")

if __name__ == "__main__":
    asyncio.run(test_acg())
