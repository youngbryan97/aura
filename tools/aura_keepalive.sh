#!/bin/bash
# ==============================================================================
# Aura Keepalive Supervisor — bounded auto-recovery for the live kernel
# ==============================================================================
# The launcher starts the kernel once and detaches; nothing brings it back if it
# dies. On 2026-07-03 a sustained-load OOM (managed RSS past the 30GB lethal
# ceiling) killed the kernel and it stayed down — the desktop sat on
# "Connecting to runtime..." until a manual relaunch.
#
# This supervisor watches the health port and relaunches the kernel when it dies
# UNEXPECTEDLY, with crash-loop protection so it can never spin forever
# reloading a 20GB model. It is intentionally SEPARATE from launch_aura.sh so it
# adds zero risk to the critical launch path.
#
# Usage:
#   tools/aura_keepalive.sh            # supervise on the default port (8000)
#   AURA_PORT=8000 tools/aura_keepalive.sh
#
# Clean stop (so it does NOT fight you):
#   touch ~/.aura/keepalive.stop       # then stop Aura however you like
#   (the supervisor exits within one poll interval and clears the flag)
#
# Tunables (env):
#   AURA_PORT                 default 8000
#   AURA_KEEPALIVE_INTERVAL_S default 20   (poll cadence)
#   AURA_KEEPALIVE_GRACE_S    default 240  (boot grace before first health demand)
#   AURA_KEEPALIVE_MAX_PER_HR default 5    (crash-loop ceiling; then give up)
# ==============================================================================
set -uo pipefail

AURA_ROOT="$(cd -P "$(dirname "$0")/.." && pwd -P)"
cd "$AURA_ROOT" || exit 1

PORT="${AURA_PORT:-8000}"
INTERVAL_S="${AURA_KEEPALIVE_INTERVAL_S:-20}"
GRACE_S="${AURA_KEEPALIVE_GRACE_S:-240}"
MAX_PER_HR="${AURA_KEEPALIVE_MAX_PER_HR:-5}"
STOP_FLAG="$HOME/.aura/keepalive.stop"
LOG_DIR="$HOME/.aura/logs"
LOG="$LOG_DIR/keepalive.log"
mkdir -p "$LOG_DIR"

# Restart timestamps (epoch seconds) within the trailing hour.
RESTARTS=()

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee -a "$LOG" ; }

health_ready() {
  # 0 = ready, non-zero = not ready/unreachable.
  curl -fs -m 5 "http://127.0.0.1:${PORT}/api/health/boot" 2>/dev/null \
    | grep -q '"ready": *true'
}

prune_restart_window() {
  local now cutoff kept
  now=$(date +%s); cutoff=$((now - 3600)); kept=()
  for t in "${RESTARTS[@]:-}"; do
    [ -n "$t" ] && [ "$t" -ge "$cutoff" ] && kept+=("$t")
  done
  RESTARTS=("${kept[@]:-}")
}

relaunch() {
  local now; now=$(date +%s)
  prune_restart_window
  local count=0
  for t in "${RESTARTS[@]:-}"; do [ -n "$t" ] && count=$((count+1)); done
  if [ "$count" -ge "$MAX_PER_HR" ]; then
    log "CRASH-LOOP GUARD: ${count} restarts in the last hour (ceiling ${MAX_PER_HR}). Not relaunching. Manual attention needed."
    return 1
  fi
  RESTARTS+=("$now")
  log "Kernel down — relaunching via launch_aura.sh (restart #$((count+1)) this hour)."
  ./launch_aura.sh >/dev/null 2>&1 || true
  return 0
}

log "Aura keepalive supervisor started (port=${PORT}, interval=${INTERVAL_S}s, grace=${GRACE_S}s, ceiling=${MAX_PER_HR}/hr). Touch ${STOP_FLAG} to stop."

# Initial boot grace so we don't fight an in-progress launch.
DOWN_STREAK=0
SLEPT=0
while [ "$SLEPT" -lt "$GRACE_S" ]; do
  [ -f "$STOP_FLAG" ] && { rm -f "$STOP_FLAG"; log "Stop flag seen during grace — exiting."; exit 0; }
  health_ready && break
  sleep "$INTERVAL_S"; SLEPT=$((SLEPT + INTERVAL_S))
done

while true; do
  if [ -f "$STOP_FLAG" ]; then
    rm -f "$STOP_FLAG"
    log "Stop flag seen — supervisor exiting cleanly (will not relaunch)."
    exit 0
  fi

  if health_ready; then
    DOWN_STREAK=0
  else
    DOWN_STREAK=$((DOWN_STREAK + 1))
    # Require two consecutive misses so a single slow health poll (e.g. the
    # kernel busy mid-generation) never triggers a needless relaunch.
    if [ "$DOWN_STREAK" -ge 2 ]; then
      if ! relaunch; then
        # Crash-loop ceiling hit — stop supervising rather than thrash.
        exit 1
      fi
      DOWN_STREAK=0
      # Give the fresh kernel its boot grace before demanding health again.
      sleep "$GRACE_S"
    fi
  fi
  sleep "$INTERVAL_S"
done
