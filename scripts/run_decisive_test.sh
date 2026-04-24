#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python tests/run_decisive_test.py
python tests/run_scale_sweep.py
