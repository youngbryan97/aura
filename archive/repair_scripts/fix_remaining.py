import os

sys_exit_files = [
    "verify_cognition.py",
    "tests/chaos_test.py",
    "tests/run_tool_audit.py"
]

for f in sys_exit_files:
    if os.path.exists(f):
        with open(f, 'r') as file:
            content = file.read()
        content = content.replace("sys.exit(", "raise SystemExit(")
        with open(f, 'w') as file:
            file.write(content)

