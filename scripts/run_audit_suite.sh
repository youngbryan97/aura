#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-full}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

run() {
  echo "+ $*"
  "$@"
}

echo "Aura Luna audit suite (${MODE})"
echo "Repository: $ROOT_DIR"
echo "Python: $PYTHON"

case "$MODE" in
  quick)
    run env AURA_PYTEST_FORCE_EXIT_AFTER_SUMMARY=1 "$PYTHON" -m pytest tests/test_audit_contracts.py crucible_test.py -q
    ;;
  full)
    run env AURA_PYTEST_FORCE_EXIT_AFTER_SUMMARY=1 "$PYTHON" -m pytest -q
    if command -v npm >/dev/null 2>&1; then
      run npm --prefix interface/static/shell run build
      run npm --prefix interface/static/memory run build
    else
      echo "npm not found; skipping frontend builds." >&2
    fi
    ;;
  *)
    echo "Usage: $0 [quick|full]" >&2
    exit 2
    ;;
esac

echo "Audit suite complete."
