#!/bin/bash
# scripts/nethack_runner.sh

set -euo pipefail

export AURA_TEST_MODE=1
: "${AURA_MODEL:=Aura-Cortex}"
export AURA_MODEL
export AURA_LOCAL_BACKEND=mlx
export AURA_NETHACK_LOG=~/.aura/logs/nethack/kernel_trace.jsonl
: "${AURA_NETHACK_STEPS:=5000}"
: "${AURA_NETHACK_SAFE_MAX_STEPS:=5000}"
: "${AURA_NETHACK_LONG_RUN_CONFIRM_FILE:=$HOME/.aura/run/allow_long_nethack}"

mkdir -p ~/.aura/logs/nethack/

echo "Launching Aura NetHack Gameplay..." > ~/.aura/logs/nethack/runner.log
date >> ~/.aura/logs/nethack/runner.log

if [[ "${AURA_SAFE_BOOT_DESKTOP:-0}" == "1" || "${AURA_LAUNCHED_FROM_APP:-0}" == "1" ]]; then
    if [[ "${AURA_ALLOW_DESKTOP_NETHACK:-0}" != "1" && "${AURA_ALLOW_DESKTOP_LONGRUNS:-0}" != "1" ]]; then
        echo "Refusing NetHack strict-real run during desktop-safe Aura session." >> ~/.aura/logs/nethack/runner.log
        echo "Set AURA_ALLOW_DESKTOP_NETHACK=1 for an intentional operator-started proof run." >> ~/.aura/logs/nethack/runner.log
        exit 64
    fi
fi

if ! [[ "${AURA_NETHACK_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "Refusing invalid AURA_NETHACK_STEPS=${AURA_NETHACK_STEPS}." >> ~/.aura/logs/nethack/runner.log
    exit 64
fi

if ! [[ "${AURA_NETHACK_SAFE_MAX_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "Refusing invalid AURA_NETHACK_SAFE_MAX_STEPS=${AURA_NETHACK_SAFE_MAX_STEPS}." >> ~/.aura/logs/nethack/runner.log
    exit 64
fi

confirm_file="${AURA_NETHACK_LONG_RUN_CONFIRM_FILE/#\\~/$HOME}"
required_confirmation="allow-long-nethack:${PWD}:${AURA_NETHACK_STEPS}"
if [[ "${AURA_NETHACK_STEPS}" -gt "${AURA_NETHACK_SAFE_MAX_STEPS}" ]]; then
    if [[ ! -f "${confirm_file}" || "$(cat "${confirm_file}")" != "${required_confirmation}" ]]; then
        echo "Refusing ${AURA_NETHACK_STEPS} NetHack steps without one-shot confirmation file." >> ~/.aura/logs/nethack/runner.log
        echo "Write '${required_confirmation}' to ${confirm_file} for an intentional long proof run." >> ~/.aura/logs/nethack/runner.log
        exit 64
    fi
    rm -f "${confirm_file}"
fi

if [[ "${AURA_NETHACK_STEPS}" -gt 50000 && "${AURA_NETHACK_UNSAFE_RAM_CONFIRM:-}" != "I_ACCEPT_64GB_RAM_RISK" ]]; then
    echo "Refusing ${AURA_NETHACK_STEPS} NetHack steps without AURA_NETHACK_UNSAFE_RAM_CONFIRM=I_ACCEPT_64GB_RAM_RISK." >> ~/.aura/logs/nethack/runner.log
    exit 64
fi

# Run challenges/nethack_challenge.py only after resource guards pass.
.venv/bin/python challenges/nethack_challenge.py \
    --mode strict_real \
    --steps "${AURA_NETHACK_STEPS}" \
    --trace ~/.aura/logs/nethack/kernel_trace.jsonl \
    --log-level INFO >> ~/.aura/logs/nethack/runner.log 2>&1
