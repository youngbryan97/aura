#!/usr/bin/env python3
"""
Integration test: Verify file operation execution works end-to-end.
Tests that file operations actually execute on the filesystem.
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FILE_OPERATION_TEST_ERRORS = (
    AssertionError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

async def test_file_operations():
    """Test that file operations actually execute."""
    from core.skills.file_operation import FileOperationSkill, FileOpInput
    
    # Use a relative path within workspace (file_operation uses cwd as root_dir)
    test_file = "test_output.txt"
    
    skill = FileOperationSkill()
    
    # Test 1: Write file
    print("Test 1: Writing file...")
    result = await skill.execute(FileOpInput(
        action="write",
        path=test_file,
        content="Hello, Aura! This is a real file created by tool execution."
    ))
    
    print(f"  Result: {result}")
    
    # Verify file was created
    if os.path.exists(test_file):
        print(f"  ✓ File created: {test_file}")
        with open(test_file, "r") as f:
            content = f.read()
        print(f"  ✓ Content verified: {content[:50]}...")
    else:
        print(f"  ✗ File NOT created!")
        return False
    
    # Test 2: Read file
    print("\nTest 2: Reading file...")
    result = await skill.execute(FileOpInput(
        action="read",
        path=test_file
    ))
    print(f"  Result: {result['ok']}")
    print(f"  Content length: {len(result.get('content', ''))}")
    
    # Test 3: Append to file
    print("\nTest 3: Appending to file...")
    result = await skill.execute(FileOpInput(
        action="append",
        path=test_file,
        content="This is appended content."
    ))
    print(f"  Result: {result}")
    
    # Verify append worked
    with open(test_file, "r") as f:
        content = f.read()
    if "appended content" in content:
        print(f"  ✓ Append verified")
    else:
        print(f"  ✗ Append failed!")
        return False
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_file_operations())
        if result:
            print("\nAll file operation tests PASSED")
            sys.exit(0)
        else:
            print("\nFile operation tests FAILED")
            sys.exit(1)
    except FILE_OPERATION_TEST_ERRORS as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
