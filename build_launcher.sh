#!/bin/bash
# Builds a simple macOS native .app wrapper for Aura

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Building Aura.app in $SCRIPT_DIR..."

cat << APPLESCRIPT > launcher.applescript
try
    tell application "Terminal"
        activate
        do script "cd '$SCRIPT_DIR' && source .venv/bin/activate && ./run_aura.sh"
    end tell
on error errMsg
    display dialog "Failed to launch Aura: " & errMsg buttons {"OK"} default button 1
end try
APPLESCRIPT

# Compile directly to a macOS App Bundle
osacompile -o "Aura.app" launcher.applescript
rm launcher.applescript

echo "✅ Aura.app created successfully on your Desktop/aura folder!"
echo "Double click Aura.app to boot Aura natively."
