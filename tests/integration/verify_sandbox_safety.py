################################################################################

"""
Verification script for CodeRepairSandbox.
"""
import sys
import os
from pathlib import Path

# Add current dir to path to find autonomy_engine
sys.path.append(os.getcwd())

try:
    from security.code_sandbox import CodeRepairSandbox, SecurityLevel
except ImportError:
    from autonomy_engine.security.code_sandbox import CodeRepairSandbox, SecurityLevel

def test_sandbox():
    print("Initializing Sandbox...")
    sandbox = CodeRepairSandbox(security_level=SecurityLevel.RESTRICTED)
    
    # Test 1: Valid Code
    print("\n--- Test 1: Valid Code ---")
    valid_code = "def hello():\n    print('Hello World')\n"
    res1 = sandbox.verify_patch(Path("dummy.py"), valid_code)
    print(f"Result: {res1}")
    if res1["syntax_valid"] and res1["static_check_passed"]:
        print("✅ PASS: Valid code accepted")
    else:
        print("❌ FAIL: Valid code rejected")

    # Test 2: Syntax Error
    print("\n--- Test 2: Syntax Error ---")
    invalid_code = "def hello()\n    print('Missing colon')\n"
    res2 = sandbox.verify_patch(Path("dummy.py"), invalid_code)
    print(f"Result: {res2}")
    if not res2["syntax_valid"]:
        print("✅ PASS: Syntax error caught")
    else:
        print("❌ FAIL: Syntax error missed")
        
    print("\nSandbox Verification Complete.")

if __name__ == "__main__":
    test_sandbox()


##
