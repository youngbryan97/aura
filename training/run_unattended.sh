#!/usr/bin/env bash
# Unattended-training wrapper for Aura's LoRA pipeline (macOS only).
# Survives lid-close via `caffeinate`, re-spawns the Python orchestrator
# on non-zero exits up to MAX_RETRIES.
set -uo pipefail

TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${TRAINING_DIR}/.." && pwd)"
LOG_DIR="${TRAINING_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/unattended_$(date +%Y%m%d_%H%M%S).log"

MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_PAUSE="${RETRY_PAUSE:-30}"

PYTHON_BIN="${AURA_PYTHON:-${REPO_DIR}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="$(command -v python3 || command -v python)"

ulimit -n 4096 || true

if ! command -v caffeinate >/dev/null 2>&1; then
    echo "ERROR: caffeinate not found — wrapper is macOS-only." | tee -a "${LOG_FILE}"
    exit 2
fi

ORCH="${TRAINING_DIR}/run_unattended.py"
if [[ ! -f "${ORCH}" ]]; then
    echo "ERROR: orchestrator missing at ${ORCH}" | tee -a "${LOG_FILE}"
    exit 2
fi

ARGS=("$@")

{
    echo "============================================================"
    echo " AURA UNATTENDED TRAINING WRAPPER"
    echo " started_at  : $(date -Iseconds)"
    echo " python      : ${PYTHON_BIN}"
    echo " log_file    : ${LOG_FILE}"
    echo " max_retries : ${MAX_RETRIES}"
    echo " ulimit -n   : $(ulimit -n)"
    echo " args        : ${ARGS[*]:-(none)}"
    echo "============================================================"
} | tee -a "${LOG_FILE}"

attempt=0
while :; do
    attempt=$((attempt + 1))
    echo "[wrapper] attempt ${attempt}/${MAX_RETRIES} at $(date -Iseconds)" | tee -a "${LOG_FILE}"

    set +e
    if [[ ${#ARGS[@]} -gt 0 ]]; then
        caffeinate -i -m -s -d "${PYTHON_BIN}" "${ORCH}" "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    else
        caffeinate -i -m -s -d "${PYTHON_BIN}" "${ORCH}" 2>&1 | tee -a "${LOG_FILE}"
    fi
    rc="${PIPESTATUS[0]}"
    set -e

    echo "[wrapper] orchestrator rc=${rc}" | tee -a "${LOG_FILE}"
    [[ "${rc}" -eq 0 ]] && { echo "[wrapper] clean exit." | tee -a "${LOG_FILE}"; exit 0; }
    if [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
        echo "[wrapper] exhausted ${MAX_RETRIES} retries — giving up." | tee -a "${LOG_FILE}"
        exit "${rc}"
    fi
    echo "[wrapper] sleeping ${RETRY_PAUSE}s before retry…" | tee -a "${LOG_FILE}"
    sleep "${RETRY_PAUSE}"
done
