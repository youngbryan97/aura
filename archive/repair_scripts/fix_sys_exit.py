import os

files = [
    "scripts/verify_telemetry.py",
    "scripts/tunnel_manager.py",
    "scripts/trigger_lora_bake.py",
    "scripts/one_off/final_verification.py",
    "scripts/one_off/verify_stabilization.py",
    "scripts/one_off/aura_diagnostic.py"
]

for f in files:
    with open(f, 'r') as file:
        content = file.read()
    content = content.replace("sys.exit(", "raise SystemExit(")
    with open(f, 'w') as file:
        file.write(content)

