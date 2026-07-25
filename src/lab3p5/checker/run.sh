#!/bin/bash
# FusedAddRmsNorm — functional / correctness test (NO profiling).
#
# Runs `checker/test_op.py` (generates inputs from seed, fp32 golden, compares).
# Use `bash checker/profile.sh` for msprof performance numbers.
#
# Usage (from anywhere — the script resolves the kit root itself):
#   bash checker/run.sh                 # Ascend C: all cases
#   LANG=triton bash checker/run.sh     # Triton: all cases (no build)
#   LANG=tilelang bash checker/run.sh   # TileLang: all cases (no build)
#   bash checker/run.sh <case_num>      # single case
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Every `hpc submit` starts a fresh container, so always activate the complete
# CANN/Python/custom-OPP environment in this shell.
# shellcheck disable=SC1091
source "$ROOT/env.sh"

LANG="${LANG:-ascendc}"
# Resolve the backend. `LANG` doubles as the POSIX locale var on locale-enabled
# images (here it's exported as zh_CN.UTF-8), so a strict `== ascendc` check
# would wrongly skip the Ascend C build. test_op.py only special-cases
# triton/tilelang and treats everything else as ascendc — mirror that: only
# those two values take the no-build path; any other value (incl. a locale)
# is the Ascend C default. This keeps `LANG=triton bash run.sh` (README) working.
case "$LANG" in
  triton|tilelang) BACKEND="$LANG"; NEED_BUILD=0 ;;
  *)               BACKEND="ascendc"; NEED_BUILD=1 ;;
esac
export PYTHONPATH="$ROOT:$ROOT/checker:${PYTHONPATH:-}"
CASE_NUM="${1:-}"

# Ascend C needs the op + wheel installed in the current HPC container. A wheel
# left in the shared directory by an earlier job is not enough.
if (( NEED_BUILD )); then
    if ! python3 -c "import custom_ops_lib" >/dev/null 2>&1; then
        bash checker/build.sh
    fi
    # build.sh runs in a subshell, so re-source the generated custom OPP env.
    # shellcheck disable=SC1091
    source "$ROOT/env.sh"
fi

echo "=== [run] correctness ($BACKEND) ==="
if [[ -n "$CASE_NUM" ]]; then
    python3 -u checker/test_op.py "$CASE_NUM"
else
    python3 -u checker/test_op.py
fi
echo "=== [run] correctness done ==="
