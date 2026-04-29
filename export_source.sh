#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/scripts/export_architecture_source.py" \
  --root "$SCRIPT_DIR" \
  --output-dir "$HOME/Downloads" \
  --char-limit 4000000 \
  --copy-limit 1000
