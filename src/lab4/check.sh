#!/bin/bash
# check.sh — validate Lab 4 output against golden results.
# Usage: ./check.sh [RESULT_DIR] [GOLDEN_DIR]
# See `python3 scripts/check_result.py --help` for details.
set -euo pipefail

ROOT_DIR="$(pwd)"

exec "${PYTHON:-python3}" "$ROOT_DIR/scripts/check_result.py" \
    --time-tolerance "${TIME_TOLERANCE:-1e-8}" \
    "$@"
