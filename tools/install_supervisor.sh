#!/bin/bash
# Install the Aura launchd supervisor: "active essentially forever."
#
# The protection stack (watchdog lethal ladder, external sentinel) is
# designed to KILL Aura safely when something runs away — a categorized
# exit with a tombstone beats a frozen Mac. This supervisor closes the
# loop: launchd restarts her within seconds, so a lethal-ladder exit
# becomes a ~30-second blip with continuity intact (state vault +
# memory persist across restarts) instead of a dead app.
#
# KeepAlive semantics:
#   - SuccessfulExit=false: restart on ANY non-zero exit (incl. the
#     watchdog's categorized exit 70 and sentinel SIGKILL), but NOT
#     after a clean operator shutdown (aura_main.py --stop exits 0).
#   - ThrottleInterval=15: never crash-loop faster than every 15s.
#
# Usage:
#   bash tools/install_supervisor.sh            # install + load
#   bash tools/install_supervisor.sh --remove   # unload + remove
#
# Logs: ~/Library/Logs/aura-supervisor.{out,err}.log

set -euo pipefail

LABEL="com.aura.supervisor"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
AURA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${AURA_ROOT}/.venv/bin/python"
LOG_DIR="$HOME/Library/Logs"

if [[ "${1:-}" == "--remove" ]]; then
    launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Aura supervisor removed."
    exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "${PYTHON_BIN} not found — run make setup first." >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
        <string>${AURA_ROOT}/aura_main.py</string>
        <string>--headless</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${AURA_ROOT}</string>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>15</integer>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/aura-supervisor.out.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/aura-supervisor.err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>AURA_SUPERVISED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
PLIST_EOF

launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Aura supervisor installed and loaded (${LABEL})."
echo "   Crash recovery: any non-zero exit restarts her within ~15s."
echo "   Clean stop: 'python aura_main.py --stop' exits 0 → no restart."
echo "   Remove:     bash tools/install_supervisor.sh --remove"
