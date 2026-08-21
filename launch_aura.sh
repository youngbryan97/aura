#!/bin/bash
# ==============================================================================
# Aura Unified Launch Script
# Guarantees Aura loads from the LIVE SOURCE CODE, bypassing any PyInstaller builds
# ==============================================================================

# Dynamically resolve root path relative to this script
export AURA_ROOT="$(cd -P "$(dirname "$0")" && pwd -P)"
if [ -z "${AURA_MODELS_DIR:-}" ]; then
    AURA_GIT_COMMON_DIR="$(git -C "$AURA_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    if [ -n "$AURA_GIT_COMMON_DIR" ] && [ "$(basename "$AURA_GIT_COMMON_DIR")" = ".git" ]; then
        AURA_PRIMARY_ROOT="$(dirname "$AURA_GIT_COMMON_DIR")"
        if [ -d "$AURA_PRIMARY_ROOT/models" ]; then
            export AURA_MODELS_DIR="$AURA_PRIMARY_ROOT/models"
        fi
        if [ -d "$AURA_PRIMARY_ROOT/training/fused-model" ]; then
            export AURA_FUSED_MODEL_ROOT="$AURA_PRIMARY_ROOT/training/fused-model"
        fi
    fi
fi
cd "$AURA_ROOT" || exit 1

print_usage() {
    cat <<'EOF'
Usage: ./launch_aura.sh [options] [aura_main.py options]

Options:
  --open-gui-window, --gui-window  Open only the desktop GUI window.
  --port PORT                      Bind Aura's local API to PORT (1-65535).
  --reboot                         Replace an existing Aura runtime.
  -h, --help                       Show this help without changing runtime state.

All unrecognized options are passed through to aura_main.py.
EOF
}

OPEN_GUI_WINDOW=0
REBOOT_MODE=0
AURA_PORT=8000
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        --open-gui-window|--gui-window)
            OPEN_GUI_WINDOW=1
            shift
            ;;
        --port)
            if [[ -n "$2" ]]; then
                AURA_PORT="$2"
                shift 2
            else
                echo "❌ Missing value for --port"
                exit 1
            fi
            ;;
        --reboot)
            REBOOT_MODE=1
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${PASSTHROUGH_ARGS[@]}"

if ! [[ "$AURA_PORT" =~ ^[0-9]+$ ]] || [ "$AURA_PORT" -lt 1 ] || [ "$AURA_PORT" -gt 65535 ]; then
    echo "❌ Invalid --port value: $AURA_PORT (expected 1-65535)"
    exit 1
fi

echo -e "  \033[1;36m     ▄████████    ███    ███   ████████▄      ▄████████\033[0m"
echo -e "  \033[1;36m    ███    ███    ███    ███   ███   ▀███    ███    ███\033[0m"
echo -e "  \033[1;35m    ███    ███    ███    ███   ███    ███    ███    ███\033[0m"
echo -e "  \033[1;35m    ███    ███    ███    ███   ███    ███    ███    ███\033[0m"
echo -e "  \033[1;34m  ▀███████████    ███    ███   ████████▀   ▀███████████\033[0m"
echo -e "  \033[1;34m    ███    ███    ███    ███   ███   ▀███    ███    ███\033[0m"
echo -e "  \033[1;34m    ███    ███    ███    ███   ███    ███    ███    ███\033[0m"
echo -e "  \033[1;34m    ███    ███    ▀████████▀   ███    ███    ███    ███\033[0m"
echo -e "  \033[1;30m┌─────────────────────────────────────────────────────────────┐\033[0m"
echo -e "  \033[1;30m│\033[0m  \033[1;35m✨ A U R A   L U N A   C O G N I T I V E   R U N T I M E ✨\033[0m  \033[1;30m│\033[0m"
echo -e "  \033[1;30m├─────────────────────────────────────────────────────────────┤\033[0m"
echo -e "  \033[1;30m│\033[0m  🧠 \033[1;36mStateful Mind\033[0m | 🛡️ \033[1;32mSealed Governance\033[0m | 🧬 \033[1;34mLive Morphogenesis\033[0m \033[1;30m│\033[0m"
echo -e "  \033[1;30m└─────────────────────────────────────────────────────────────┘\033[0m"
echo ""
echo -e "🌸 \033[1;32mInitializing Aura\033[0m (Live Source Mode)..."

# Version Lock: Aura requires Python 3.12 for binary compatibility with its native extensions (grpc, mlx).
# We prefer the venv python if it matches 3.12, otherwise we search for system python3.12.
if [ -f ".venv/bin/python3" ] && .venv/bin/python3 --version | grep -q "3.12"; then
    PYTHON_CMD=".venv/bin/python3"
elif command -v python3.12 &>/dev/null; then
    echo "⚠️  Venv mismatch or missing. Using system python3.12 directly."
    PYTHON_CMD="python3.12"
else
    echo "❌ CRITICAL: Python 3.12 not found. Aura's native AI cores will fail on higher versions."
    echo "Please install python 3.12 or fix your .venv."
    exit 1
fi

echo "📍 Using Interpreter: $($PYTHON_CMD --version) at $PYTHON_CMD"

# A signed Aura.app launch is pinned to the workspace it belongs to. Verify
# that contract before cleanup so an app from a DIFFERENT checkout can never
# terminate a valid current runtime.
#
# It used to be pinned to the exact HEAD and dirty file state as well, and any
# difference exited 1 with "Rebuild the installed app before launch". Aura
# commits to her own repository, so that made refusing to start the normal
# outcome and a human rebuild the only way out. Identity is what protects the
# running instance; how far the workspace has moved since the bundle was built
# is a fact to report, not a reason to refuse to start. The runtime measures
# and publishes it as source_drift / source_current.
if [ "${AURA_LAUNCHED_FROM_APP:-0}" = "1" ]; then
    echo "🔏 Verifying signed app source identity..."
    if ! "$PYTHON_CMD" -m core.runtime.launch_provenance preflight --root "$AURA_ROOT"; then
        echo "❌ Aura.app does not belong to this workspace (root, bundle identity, or signature)."
        echo "   This is an identity failure, not staleness — rebuild from THIS checkout."
        exit 1
    fi
    # Keep the launcher binary itself current. This is the one artifact that
    # genuinely goes stale, because it is compiled code rather than a pointer
    # to live source. Aura rebuilds it herself rather than waiting to be asked.
    #
    # Install first: a build staged during the previous session becomes the
    # resident app now. Then rebuild if the source has moved again, staging it
    # for next time rather than replacing a bundle that is currently executing.
    "$PYTHON_CMD" -m core.runtime.app_bundle_sync --root "$AURA_ROOT" --install-staged || true
    "$PYTHON_CMD" -m core.runtime.app_bundle_sync --root "$AURA_ROOT" || true
fi

LOG_DIR="${HOME}/.aura/logs"
mkdir -p "$LOG_DIR"
LAUNCH_LOG="${LOG_DIR}/desktop-launch.log"

rotate_launch_log() {
    # Every boot appends raw stdout here forever; without a bound the file
    # reached 145MB. Rotate at boot past the size gate into a gzip'd ring.
    local max_bytes=$((20 * 1024 * 1024))
    local keep=5
    [ -f "$LAUNCH_LOG" ] || return 0
    local size
    size=$(stat -f%z "$LAUNCH_LOG" 2>/dev/null || echo 0)
    [ "$size" -ge "$max_bytes" ] || return 0
    local i
    for (( i=keep-1; i>=1; i-- )); do
        [ -f "${LAUNCH_LOG}.${i}.gz" ] && mv -f "${LAUNCH_LOG}.${i}.gz" "${LAUNCH_LOG}.$((i+1)).gz"
    done
    mv -f "$LAUNCH_LOG" "${LAUNCH_LOG}.1"
    gzip -f "${LAUNCH_LOG}.1" 2>/dev/null || true
}

backup_env_file() {
    # .env carries the local shared secret (AURA_API_TOKEN) and was once
    # destroyed with no recovery path. Keep a bounded backup ring; skip
    # writes when content is unchanged. Values are never printed.
    local env_file="${AURA_ROOT}/.env"
    [ -e "$env_file" ] || return 0
    local ring_dir="${HOME}/.aura/backups/env"
    mkdir -p "$ring_dir" 2>/dev/null || return 0
    chmod 700 "$ring_dir" 2>/dev/null || true
    local newest
    newest=$(ls -t "$ring_dir"/env-*.bak 2>/dev/null | head -1)
    if [ -n "$newest" ] && cmp -s "$env_file" "$newest"; then
        return 0
    fi
    local dest="${ring_dir}/env-$(date +%Y%m%d_%H%M%S)-$$.bak"
    while [ -e "$dest" ]; do
        dest="${ring_dir}/env-$(date +%Y%m%d_%H%M%S)-$$-${RANDOM}.bak"
    done
    cp -L "$env_file" "$dest" 2>/dev/null || return 0
    chmod 600 "$dest" 2>/dev/null || true
    ls -t "$ring_dir"/env-*.bak 2>/dev/null | tail -n +11 | while IFS= read -r old; do
        rm -f "$old"
    done
}

check_env_file() {
    if [ ! -e "${AURA_ROOT}/.env" ]; then
        echo "⚠️  No .env at ${AURA_ROOT}/.env — GUI and server share AURA_API_TOKEN via this file."
        echo "   Recovery ring (if any): ${HOME}/.aura/backups/env/"
    elif ! grep -q '^AURA_API_TOKEN=' "${AURA_ROOT}/.env" 2>/dev/null; then
        echo "⚠️  .env present but AURA_API_TOKEN is missing — the desktop GUI may fail to authenticate."
    fi
}

rotate_launch_log
backup_env_file
check_env_file

resolve_launch_log() {
    if touch "$LAUNCH_LOG" 2>/dev/null; then
        echo "$LAUNCH_LOG"
        return
    fi

    local fallback_log="${TMPDIR:-/tmp}/aura-desktop-launch.log"
    if touch "$fallback_log" 2>/dev/null; then
        echo "$fallback_log"
        return
    fi

    echo "/dev/null"
}

ACTIVE_LAUNCH_LOG="$(resolve_launch_log)"

if [ "$OPEN_GUI_WINDOW" = "1" ]; then
    echo "🪟 Opening Aura desktop window..."
    export PYTHONUNBUFFERED=1
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    export OBJC_PRINT_LOAD_METHODS=NO
    if [ "$AURA_ATTACH_LAUNCHER" = "1" ]; then
        exec "$PYTHON_CMD" -u aura_main.py --gui-window --port "$AURA_PORT" "$@"
    else
        nohup "$PYTHON_CMD" -u aura_main.py --gui-window --port "$AURA_PORT" "$@" >>"$ACTIVE_LAUNCH_LOG" 2>&1 &
        GUI_PID=$!
        disown "$GUI_PID" 2>/dev/null || true
        echo ""
        echo "✨ Aura desktop window opening (PID: $GUI_PID)"
        echo "📜 Logs: $ACTIVE_LAUNCH_LOG"
        exit 0
    fi
fi

# 1. Cleanup Phase — use bounded Python cleanup so the launcher can't hang on pkill/lsof
#
# --reboot is documented as "Replace an existing Aura runtime", but the cleanup
# refuses to touch a VERIFIED live runtime unless AURA_CLEANUP_FORCE is set, and
# nothing was setting it. So --reboot logged
#   "Verified live Aura runtime detected (PID: …); skipping aggressive
#    pre-launch process cleanup"
# and then started a SECOND desktop runtime beside the first: two 32B models,
# ~20GB each, on a 64GB host. That is the duplicate-runtime memory cascade, and
# it arrived through the one flag whose entire job was to prevent it.
#
# The guard itself is right — an unasked-for launch must never kill a healthy
# instance. Explicitly asking to reboot is the authorization it was waiting for.
if [ "$REBOOT_MODE" = "1" ]; then
    export AURA_CLEANUP_FORCE=1
fi
echo "🧹 Cleaning up existing instances..."
if ! "$PYTHON_CMD" aura_cleanup.py; then
    echo "⚠️  Cleanup reported an issue; continuing with launch."
fi

# 2. Launch Phase
echo "🚀 Starting Aura Desktop..."
export MLX_METAL_SYNC=1
: "${AURA_FORCE_CAMERA:=0}"             # Camera stays off by default on macOS boot
: "${AURA_ENABLE_UVLOOP:=0}"            # macOS desktop path prefers stock asyncio
: "${AURA_ATTACH_LAUNCHER:=0}"          # Set to 1 to keep the shell attached for live logs
export AURA_FORCE_CAMERA
export AURA_ENABLE_UVLOOP
export AURA_ATTACH_LAUNCHER
export AURA_LOCAL_BACKEND=mlx        # Aura's in-process fine-tuned MLX mind.
# ── Substrate interoception + epistemic reach (felt thought) ─────────────────
: "${AURA_INTEROCEPTION:=1}"             # Feel the decode: per-token surprisal/entropy tap
: "${AURA_EPISTEMIC_REACH:=1}"           # Felt doubt may verify claims externally (governed)
: "${AURA_REACH_READ_HOSTS:=en.wikipedia.org}"  # Operator READ allowlist for governed reach
: "${AURA_EPISTEMIC_REACH_PER_HOUR:=6}"  # Hard hourly budget for external verifications
export AURA_INTEROCEPTION
export AURA_EPISTEMIC_REACH
export AURA_REACH_READ_HOSTS
export AURA_EPISTEMIC_REACH_PER_HOUR
: "${AURA_SAFE_BOOT_DESKTOP:=0}" # Recovery-only; normal desktop launches are the full runtime
: "${AURA_ENABLE_BACKGROUND_COGNITION:=1}"
: "${AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM:=1}"
: "${AURA_BACKGROUND_BOOT_GRACE_S:=60}"
if [ "${AURA_LAUNCHED_FROM_APP:-0}" = "1" ]; then
    : "${AURA_DESKTOP_RESOURCE_GUARD:=1}"
    : "${AURA_EXTERNAL_GUI_OWNER:=1}"
    # The .app should not inherit a stale diagnostic setting that disables the
    # one bounded full-mind repair pass. Use
    # AURA_DESKTOP_FORCE_DISABLE_SECONDARY_MODEL_REPAIR=1 for explicit
    # diagnostics.
    unset AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR
    if [ "$REBOOT_MODE" != "1" ]; then
        : "${AURA_CLEANUP_RECENT_GRACE_S:=45}"
    fi
fi
if [ "${AURA_DESKTOP_RESOURCE_GUARD:-0}" = "1" ] || [ "${AURA_SAFE_BOOT_DESKTOP:-0}" = "1" ]; then
    : "${AURA_EAGER_CORTEX_WARMUP:=auto}"
    # Prewarm the 32B Cortex in the BACKGROUND shortly after boot so the first
    # conversational message is responsive instead of waiting minutes for a cold
    # 32B load. This stays RAM-gated by the admission snapshot (it will not warm
    # under genuine memory pressure), so it improves latency without risking the
    # memory-pressure failures the resource guard prevents. Previously this was
    # "auto", which disabled prewarm entirely and left the chat
    # lane cold until a foreground demand that did not reliably trigger a load.
    : "${AURA_DEFERRED_CORTEX_PREWARM:=1}"
    : "${AURA_ENABLE_PERMANENT_SWARM:=0}"
    : "${AURA_DESKTOP_METAL_CACHE_RATIO:=0.16}"
    : "${AURA_DESKTOP_METAL_CACHE_CAP_GB:=10}"
    : "${AURA_DESKTOP_MLX_MEMORY_RATIO:=0.54}"
    : "${AURA_DESKTOP_MLX_MEMORY_CAP_GB:=34}"   # in-process MLX 32B headroom
    : "${AURA_DESKTOP_MLX_MEMORY_FLOOR_GB:=18}"
    : "${AURA_DESKTOP_PROCESS_RSS_RATIO:=0.62}"
    : "${AURA_DESKTOP_PROCESS_RSS_CAP_GB:=40}"   # kernel holds the in-process MLX model
    : "${AURA_DESKTOP_PROCESS_RSS_FLOOR_GB:=24}"
    : "${AURA_PROCESS_RSS_LIMIT_GB:=40}"   # kernel holds the in-process MLX model
    : "${AURA_MEMWATCH_SOFT_MB:=37888}"
    : "${AURA_MEMWATCH_HARD_MB:=41984}"
    : "${AURA_MEMWATCH_LETHAL_MB:=43008}"
    : "${AURA_MEMORY_SENTINEL_INTERVAL_S:=0.5}"
    : "${AURA_GOVERNOR_PRUNE_MB:=37888}"
    : "${AURA_GOVERNOR_UNLOAD_MB:=39936}"
    : "${AURA_GOVERNOR_CRITICAL_MB:=40960}"
    : "${AURA_LOCAL_RUNTIME_SINGLETON:=1}"
    : "${AURA_LOCAL_PARALLEL_SLOTS:=1}"
    : "${AURA_ENABLE_LOCAL_DEEP_SOLVER:=0}"
    : "${AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB:=24}"
    : "${AURA_MLX_32B_PROJECTED_FOOTPRINT_GB:=auto}"
    : "${AURA_MLX_32B_PROCESS_RESERVE_GB:=3}"
    : "${AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB:=52}"
    : "${AURA_MLX_72B_PROJECTED_FOOTPRINT_GB:=auto}"
    : "${AURA_MLX_72B_PROCESS_RESERVE_GB:=5}"
    : "${AURA_FOREGROUND_CHAT_MAX_TOKENS:=2048}"
    echo "🛡️ Full desktop runtime enabled with RAM-aware Cortex and process guards."
else
    : "${AURA_ENABLE_PERMANENT_SWARM:=1}"   # Multi-agent internal debate
fi
export AURA_SAFE_BOOT_DESKTOP
export AURA_ENABLE_BACKGROUND_COGNITION
export AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM
export AURA_BACKGROUND_BOOT_GRACE_S
export AURA_DESKTOP_RESOURCE_GUARD
export AURA_EXTERNAL_GUI_OWNER
export AURA_CLEANUP_RECENT_GRACE_S
export AURA_EAGER_CORTEX_WARMUP
export AURA_DEFERRED_CORTEX_PREWARM
export AURA_ENABLE_PERMANENT_SWARM
export AURA_DESKTOP_METAL_CACHE_RATIO
export AURA_DESKTOP_METAL_CACHE_CAP_GB
export AURA_DESKTOP_MLX_MEMORY_RATIO
export AURA_DESKTOP_MLX_MEMORY_CAP_GB
export AURA_DESKTOP_MLX_MEMORY_FLOOR_GB
export AURA_DESKTOP_PROCESS_RSS_RATIO
export AURA_DESKTOP_PROCESS_RSS_CAP_GB
export AURA_DESKTOP_PROCESS_RSS_FLOOR_GB
export AURA_PROCESS_RSS_LIMIT_GB
export AURA_MEMWATCH_SOFT_MB
export AURA_MEMWATCH_HARD_MB
export AURA_MEMWATCH_LETHAL_MB
export AURA_MEMORY_SENTINEL_INTERVAL_S
export AURA_GOVERNOR_PRUNE_MB
export AURA_GOVERNOR_UNLOAD_MB
export AURA_GOVERNOR_CRITICAL_MB
export AURA_LOCAL_RUNTIME_SINGLETON
export AURA_LOCAL_PARALLEL_SLOTS
export AURA_ENABLE_LOCAL_DEEP_SOLVER
export AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB
export AURA_MLX_32B_PROJECTED_FOOTPRINT_GB
export AURA_MLX_32B_PROCESS_RESERVE_GB
export AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB
export AURA_MLX_72B_PROJECTED_FOOTPRINT_GB
export AURA_MLX_72B_PROCESS_RESERVE_GB
export AURA_FOREGROUND_CHAT_MAX_TOKENS
export AURA_LOCAL_BACKEND
export PYTHONUNBUFFERED=1              # Always flush startup logs for launcher diagnostics
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # Suppress ObjC fork warnings
# Suppress duplicate ObjC class warnings from cv2/av FFmpeg dylib overlap (cosmetic, not a crash)
export OBJC_PRINT_LOAD_METHODS=NO

# Launch directly using the discovered loader. By default we detach so Terminal
# is not left parented to Aura for the entire session.
if [ "$AURA_ATTACH_LAUNCHER" = "1" ]; then
    "$PYTHON_CMD" -u aura_main.py --desktop --port "$AURA_PORT" "$@" &
else
    nohup "$PYTHON_CMD" -u aura_main.py --desktop --port "$AURA_PORT" "$@" >>"$ACTIVE_LAUNCH_LOG" 2>&1 &
fi
AURA_PID=$!

cleanup_attached_launcher() {
    local status=${1:-0}
    trap - INT TERM EXIT
    if [ -n "${AURA_PID:-}" ] && kill -0 "$AURA_PID" 2>/dev/null; then
        kill -TERM "$AURA_PID" 2>/dev/null || true
        wait "$AURA_PID" 2>/dev/null || true
    fi
    "$PYTHON_CMD" aura_cleanup.py >/dev/null 2>&1 || true
    exit "$status"
}

echo ""
echo "✨ Aura Luna launching (PID: $AURA_PID)"
echo ""
echo "💡 PRO-TIP: Add this alias to your ~/.zshrc to launch Aura from anywhere:"
echo "   alias aura=\"$AURA_ROOT/launch_aura.sh\""
echo ""
if [ "$AURA_ATTACH_LAUNCHER" = "1" ]; then
    echo "📜 Attached launcher mode active. Press Ctrl+C to stop following the process."
    trap 'cleanup_attached_launcher 130' INT
    trap 'cleanup_attached_launcher 143' TERM
    trap 'cleanup_attached_launcher $?' EXIT
    wait "$AURA_PID"
    LAUNCH_STATUS=$?
    cleanup_attached_launcher "$LAUNCH_STATUS"
else
    disown "$AURA_PID" 2>/dev/null || true
    echo "📜 Logs: $ACTIVE_LAUNCH_LOG"
fi
