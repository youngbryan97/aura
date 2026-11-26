import asyncio
import logging
import sys
import os

# Ensure core is loadable
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Import the refactored skills
from core.skills.sovereign_terminal import SovereignTerminalSkill
from core.skills.file_operation import FileOperationSkill
from core.skills.computer_use import ComputerUseSkill

async def main():
    print("=== Aura Headless Autonomous Boot Validation ===\n")
    
    # Context
    context = {"mode": "headless_test"}
    
    # 1. Test Persistent Terminal & Truncation
    print("\n--- Test 1: Sovereign Terminal (Anti-Hang & Truncation) ---")
    term = SovereignTerminalSkill()
    # Trigger a command that produces a lot of output
    res1 = await term.safe_execute({"action": "execute", "command": "seq 1 5000", "timeout": 3}, context)
    if res1.get("ok"):
        stdout = res1.get("stdout", "")
        if "TRUNCATED" in stdout:
            print("Smart Truncation Success.")
        else:
            print("Smart Truncation Failed (Output length:", len(stdout), ")")
    else:
        print("Terminal failed:", res1.get("error"))

    # 2. Test Semantic File Read
    print("\n--- Test 2: Semantic File Editor (Read) ---")
    f_op = FileOperationSkill()
    test_file = "test_semantic.txt"
    await f_op.safe_execute({"action": "write", "path": test_file, "content": "Line 1\nLine 2\nLine 3\n"}, context)
    
    read_res = await f_op.safe_execute({"action": "read", "path": test_file}, context)
    if read_res.get("ok") and "0001:" in read_res.get("content", ""):
         print("Line-Indexed Read Success.")
    else:
         print("Line-Indexed Read Failed.")

    # 3. Test File Syntax Patching
    print("\n--- Test 3: Semantic Patch & Syntax Validation ---")
    py_test_file = "test_syntax.py"
    await f_op.safe_execute({"action": "write", "path": py_test_file, "content": "def hello():\n    print('world')\n"}, context)
    
    # Introduce syntax error
    patch_res = await f_op.safe_execute({
        "action": "patch", 
        "path": py_test_file, 
        "start_line": 2, 
        "end_line": 2, 
        "content": "    print('world'"  # missing closing paren
    }, context)
    
    if patch_res.get("ok") == False and "Syntax Error introduced" in patch_res.get("error", ""):
        print("Pre-Commit Syntax Validation Success (Blocked bad save).")
    else:
        print("Syntax Validation Failed:", patch_res)
        
    # Clean up
    os.remove(test_file)
    if os.path.exists(py_test_file):
        os.remove(py_test_file)
        
    # 4. Verified Computer Use
    print("\n--- Test 4: Verified Computer Use (Headless Fallback) ---")
    comp = ComputerUseSkill()
    # Read screen text should fallback gracefully in headless or return actual text
    comp_res = await comp.safe_execute({"action": "read_screen_text", "target": ""}, context)
    if comp_res.get("ok"):
        text = comp_res.get("text", "")
        print(f"State Verification OK: {text[:50]}...")
    else:
        # Might fail gracefully if accessibility permissions are denied
        if "permission" in comp_res:
            print("State Verification Handled via Permission Error (Normal for non-root headless).")
    
    print("\n--- Test 5: Stateful Active Coding Sandbox ---")
    from core.skills.active_coding import RunCodeSkill
    active_code = RunCodeSkill()
    # Test state persistence: create a variable, then read it
    res_code_1 = await active_code.safe_execute({"code": "x = 55", "stateful": True}, context)
    res_code_2 = await active_code.safe_execute({"code": "print(x)", "stateful": True}, context)
    
    if res_code_2.get("ok") and "55" in res_code_2.get("stdout", ""):
        print("Stateful Execution Intact (Variables Resisted Wipe).")
    else:
        print("Stateful Execution Failed:", res_code_2)

    print("\n--- Test 6: MemoryOps Letta Architecture (MemFS) ---")
    from core.skills.memory_ops import MemoryOpsSkill
    mem_ops = MemoryOpsSkill()
    
    # Core Append
    mem_append = await mem_ops.safe_execute({"action": "core_append", "block": "user", "content": "I like dark mode."}, context)
    # Core Replace
    mem_replace = await mem_ops.safe_execute({"action": "core_replace", "block": "user", "old_content": "I like dark mode.", "content": "I like light mode."}, context)
    
    # Read manually to verify
    mem_path = mem_ops.mem_fs_dir / "user.txt"
    if mem_path.exists() and "light mode" in mem_path.read_text():
        print("Letta Core Memory Block Edited Successfully in MemFS.")
    else:
        print("MemFS Editing Failed.")
        
    print("\n--- Test 7: Belief Ops MemFS Integration ---")
    from core.skills.belief_ops import AddBeliefSkill, QueryBeliefsSkill
    add_belief = AddBeliefSkill()
    q_belief = QueryBeliefsSkill()
    
    await add_belief.safe_execute({"source": "Tester", "relation": "loves", "target": "robustness"}, context)
    q_res = await q_belief.safe_execute({"subject": "Tester", "limit": 10}, context)
    if q_res.get("ok") and "robustness" in q_res.get("summary", ""):
         print("Beliefs successfully stored and retrieved from MemFS.")
    else:
         print("BeliefOps Integration Failed:", q_res)

    print("\n--- Test 8: Web Search Deep Crawl Verification ---")
    from core.skills.web_search import EnhancedWebSearchSkill
    search_skill = EnhancedWebSearchSkill()
    # Keep network local to avoid external latency, but call safely to see tool binds.
    search_res = await search_skill.safe_execute({"query": "What is Python?", "deep": True, "num_results": 2}, context)
    if search_res.get("ok") or "error" in search_res:
         print("Web Search Deep Execution Path Handled.")
    else:
         print("Web Search Deep Failed.")
         
    print("\n=== All Primary Autonomous Architectures Verified ===")

if __name__ == "__main__":
    asyncio.run(main())


# `main()` above carries four real checks that print/and assert nothing,
# so a regression would have printed into a log nobody reads. It was also
# never collected. These are the same properties as pytest tests, against a
# tmp path rather than the repo root — the original wrote test_semantic.txt
# into the working directory and never removed it.

import pytest


def test_terminal_truncates_a_flood_of_output():
    """Anti-hang: unbounded stdout is how a skill wedges the runtime.

    This used to run ``seq 1 5000`` through the whole skill and assert on the
    result. It stopped passing when shell execution became properly governed —
    first the user-confirmation overlay refused it, and underneath that the
    authority gateway wants a standing grant, which is correct and must not be
    waived to keep a test green.

    The property under test never needed a subprocess. ``_smart_truncate`` is
    the anti-hang mechanism; driving four authority layers to reach it only
    made the test fail for reasons that were not about truncation.
    """
    term = SovereignTerminalSkill()
    flood = "\n".join(str(i) for i in range(5000))
    assert len(flood) > 5000

    truncated = term._smart_truncate(flood)
    assert "TRUNCATED" in truncated
    assert len(truncated) < len(flood)
    # Head and tail both survive: the tail is where a traceback's cause lands,
    # and keeping only the head is how a truncated log loses the error.
    assert truncated.startswith("0\n1\n")
    assert truncated.rstrip().endswith("4999")


def test_short_output_is_returned_whole():
    term = SovereignTerminalSkill()
    assert term._smart_truncate("short") == "short"
    assert term._smart_truncate("") == ""


@pytest.mark.asyncio
async def test_a_shell_execution_is_governed():
    """The refusal that broke the old test is governance working, so it is pinned."""
    term = SovereignTerminalSkill()
    result = await term.safe_execute(
        {"action": "execute", "command": "seq 1 5", "timeout": 10},
        {"mode": "headless_test"},
    )
    assert not result.get("ok")
    assert "Authority refused" in str(result.get("error", ""))


@pytest.fixture
def workspace_file():
    """A uniquely named file inside the workspace, removed afterwards.

    The skill sandboxes to the workspace, so tmp_path is correctly refused.
    The original script wrote test_semantic.txt into the repo root and never
    cleaned it up.
    """
    import os
    import uuid

    name = f"_headless_boot_probe_{uuid.uuid4().hex[:8]}.txt"
    yield name
    for candidate in (name, os.path.join(os.getcwd(), name)):
        try:
            os.remove(candidate)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_the_file_skill_refuses_paths_outside_the_workspace(tmp_path):
    """Fail-closed sandboxing, pinned. A skill that can write anywhere is a
    skill that can overwrite the runtime that is running it."""
    f_op = FileOperationSkill()
    result = await f_op.safe_execute(
        {"action": "write", "path": str(tmp_path / "escape.txt"), "content": "x"},
        {"mode": "headless_test"},
    )
    assert result.get("ok") is False
    assert "outside workspace" in str(result.get("error", "")).lower()


@pytest.mark.asyncio
async def test_reads_are_line_indexed(workspace_file):
    """A semantic editor must cite lines, or an edit cannot be targeted."""
    f_op = FileOperationSkill()
    context = {"mode": "headless_test"}

    written = await f_op.safe_execute(
        {"action": "write", "path": workspace_file, "content": "Line 1\nLine 2\nLine 3\n"},
        context,
    )
    assert written.get("ok"), written.get("error")

    read = await f_op.safe_execute({"action": "read", "path": workspace_file}, context)
    assert read.get("ok"), read.get("error")
    assert "0001:" in read.get("content", "")


@pytest.mark.asyncio
async def test_a_write_round_trips_its_content(workspace_file):
    f_op = FileOperationSkill()
    context = {"mode": "headless_test"}

    await f_op.safe_execute(
        {"action": "write", "path": workspace_file, "content": "alpha\nbeta\n"}, context
    )
    read = await f_op.safe_execute({"action": "read", "path": workspace_file}, context)
    assert "alpha" in read.get("content", "")
    assert "beta" in read.get("content", "")


@pytest.mark.asyncio
async def test_a_failing_command_reports_rather_than_hangs():
    term = SovereignTerminalSkill()
    result = await term.safe_execute(
        {"action": "execute", "command": "exit 3", "timeout": 10},
        {"mode": "headless_test"},
    )
    assert result is not None
    assert "ok" in result
