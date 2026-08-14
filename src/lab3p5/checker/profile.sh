#!/bin/bash
# FusedAddRmsNorm — single-operator performance test (msprof op).
#
# Measures the student's operator only. NO correctness check here — use run.sh
# for that.
#
# Performance is measured on case 2 only (B=256, H=1024; one-based case ID).
# Usage (from anywhere — the script resolves the kit root itself):
#   bash checker/profile.sh               # Ascend C, case 2
#   LANG=triton bash checker/profile.sh   # Triton, case 2
#   LANG=tilelang bash checker/profile.sh # TileLang, case 2
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if (( $# != 0 )); then
    echo "[ERROR] profile.sh always profiles public case 2; no case argument is accepted." >&2
    exit 2
fi
# Every `hpc submit` starts a fresh container, so always activate the complete
# CANN/Python/custom-OPP environment in this shell.
# shellcheck disable=SC1091
source "$ROOT/env.sh"

LANG="${LANG:-ascendc}"
# Resolve the backend. `LANG` doubles as the POSIX locale var on locale-enabled
# images (here it's exported as zh_CN.UTF-8), so a strict `== ascendc` check
# would wrongly skip the Ascend C build. See checker/run.sh for the rationale.
case "$LANG" in
  triton|tilelang) BACKEND="$LANG"; NEED_BUILD=0 ;;
  *)               BACKEND="ascendc"; NEED_BUILD=1 ;;
esac
export PYTHONPATH="$ROOT:$ROOT/checker:${PYTHONPATH:-}"
PROFILE_CASE_NUM=2
PROFILE_SHAPE="256x1024"
PYTHON_BIN="$(python3 -c 'import sys; print(sys.executable)')"

# Ascend C needs the op + wheel installed in the current HPC container. A wheel
# left in the shared directory by an earlier job does not imply that its Python
# extension is installed in this fresh container.
if (( NEED_BUILD )); then
    if ! python3 -c "import custom_ops_lib" >/dev/null 2>&1; then
        bash checker/build.sh
    fi
    # build.sh runs in a subshell, so re-source the generated custom OPP env.
    # shellcheck disable=SC1091
    source "$ROOT/env.sh"
fi

echo "############################## STUDENT OP ($BACKEND) ##############################"
echo "=== [profile] student:$BACKEND case $PROFILE_CASE_NUM ($PROFILE_SHAPE) under msprof op ==="
PROF_DIR="prof_out"
rm -rf "$PROF_DIR"
mkdir -p "$PROF_DIR"
if ! LANG="$BACKEND" timeout 180 msprof op \
    --warm-up=10 \
    --output="$ROOT/$PROF_DIR" \
    "$PYTHON_BIN" checker/test_op.py --profile "$PROFILE_CASE_NUM"; then
    echo "[ERROR] msprof op failed for student:$BACKEND" >&2
    exit 1
fi
if ! STUDENT_US=$(python3 checker/get_time.py "$ROOT/$PROF_DIR"); then
    echo "[ERROR] failed to read Task Duration for student:$BACKEND" >&2
    exit 1
fi

echo ""
echo "=========================== SUMMARY ==========================="
printf "  student  (%s): %s us\n" "$BACKEND" "$STUDENT_US"

if [[ "$STUDENT_US" == "0.0000" ]]; then
    echo "  [ERROR] student op reported zero Task Duration"
    exit 1
fi
echo "Performance profiling complete."
