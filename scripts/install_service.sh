#!/bin/bash
# scripts/install_service.sh
# ── Aura Service Installer ───────────────────────────
# Registers Aura as a macOS launchd service for automatic
# recovery on boot and crash-resilient process supervision.

set -e

echo "Installing Aura as a background launchd service..."

SERVICE_NAME="com.aura.sovereign"
PLIST_PATH="$HOME/Library/LaunchAgents/$SERVICE_NAME.plist"
LOG_DIR="$HOME/.aura/logs"
AURA_ROOT="$(pwd)"

mkdir -p "$LOG_DIR"

cat <<EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$SERVICE_NAME</string>
    <key>ProgramArguments</key>
    <array>
        <string>$AURA_ROOT/.venv/bin/python3</string>
        <string>-m</string>
        <string>core.orchestrator.main</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$AURA_ROOT</string>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/aura_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/aura_stderr.log</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>Crashed</key>
        <true/>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>$AURA_ROOT</string>
    </dict>
</dict>
</plist>
EOF

echo "Created launchd plist at $PLIST_PATH"

launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load -w "$PLIST_PATH"

echo "Service registered and started."
echo "You can check logs at: $LOG_DIR/aura_stdout.log"
