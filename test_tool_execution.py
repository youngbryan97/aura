#!/usr/bin/env python3
"""
Test tool execution with fixed TaskTracker memory management

Verifies tools work cleanly without crashes or hangs
"""

import sys
import asyncio
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    asyncio.TimeoutError,
)

async def test_file_write_read(work_dir: Path):
    """Test basic file write/read tool operations"""
    from core.capability_engine import execute_tool
    test_file = work_dir / "test_aura_write.txt"
    
    print("Testing File Write Operation...")
    
    # Test write
    result = await execute_tool(
        tool_name="write_file",
        parameters={
            "file_path": str(test_file),
            "content": "Hello from Aura v62 - memory leak fixed!",
            "mode": "w"
        }
    )
    
    if result.get("ok"):
        print(f"   ✓ File write succeeded")
    else:
        print(f"   ✗ File write failed: {result.get('error')}")
        return False
    
    print("Testing File Read Operation...")
    
    # Test read
    result = await execute_tool(
        tool_name="read_file",
        parameters={
            "file_path": str(test_file)
        }
    )
    
    if result.get("ok"):
        content = result.get("content", "")
        print(f"   ✓ File read succeeded: {len(content)} bytes")
        if "memory leak fixed" in content:
            print(f"   ✓ Content verified")
            return True
    else:
        print(f"   ✗ File read failed: {result.get('error')}")
    
    return False

async def test_multiple_tools_sequential(work_dir: Path):
    """Test multiple tool calls in sequence"""
    from core.capability_engine import execute_tool
    
    print("\nTesting 10 Sequential Tool Calls...")
    
    for i in range(10):
        result = await execute_tool(
            tool_name="write_file",
            parameters={
                "file_path": str(work_dir / f"test_sequential_{i}.txt"),
                "content": f"Test {i} - memory stable",
                "mode": "w"
            }
        )
        if not result.get("ok"):
            print(f"   ✗ Call {i} failed")
            return False
        if i % 3 == 0:
            print(f"   ✓ Call {i} succeeded")
    
    print(f"   ✓ All 10 sequential calls succeeded")
    return True

async def main():
    print("=" * 60)
    print("TESTING TOOL EXECUTION WITH FIXED MEMORY MANAGEMENT")
    print("=" * 60)
    
    try:
        with tempfile.TemporaryDirectory(prefix="aura_tool_execution_") as tmp:
            work_dir = Path(tmp)
            test1 = await test_file_write_read(work_dir)
            test2 = await test_multiple_tools_sequential(work_dir)
        
        print("\n" + "=" * 60)
        if test1 and test2:
            print("ALL TOOL TESTS PASSED")
            print("   - File operations work cleanly")
            print("   - Sequential tool calls stable")
            print("   - No crashes or hangs observed")
            return 0
        else:
            print("SOME TESTS FAILED")
            return 1
    except SCRIPT_RECOVERABLE_ERRORS as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
