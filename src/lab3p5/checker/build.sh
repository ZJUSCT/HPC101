#!/bin/bash
# Build + install the FusedAddRmsNorm Ascend C op AND the pybind extension.
# Run AFTER `source env.sh` (from the kit root). Lives under `checker/` now;
# ROOT still resolves to the kit root (parent of this dir) so paths match the
# runtime layout the README documents.
#
# IMPORTANT: this script runs in a subshell, so the custom_opp set_env.bash it
# sources (registers the aclnn op with libopapi.so via ASCEND_CUSTOM_OPP_PATH
# / LD_LIBRARY_PATH) does NOT leak to the caller. After this returns, either
# `source ./env.sh` again, or `source $CUSTOM_OPP_HOME/vendors/customize/bin/
# set_env.bash` to make aclnnFusedAddRmsNorm resolvable at runtime. run.sh does
# the former automatically.
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/checker"

OP_DIR="$ROOT/src/ascendc"
export CUSTOM_OPP_HOME="${CUSTOM_OPP_HOME:-$HOME/custom_opp}"
# env.sh must be sourced first so the system CANN (ASCEND_TOOLKIT_HOME) is on
# PATH — we don't second-guess its location here.
: "${ASCEND_TOOLKIT_HOME:?env.sh not sourced — run 'source ./env.sh' first (ASCEND_TOOLKIT_HOME unset)}"

echo "=== [build] patch op CMakePresets -> system CANN ==="
python3 - <<PY
import json, os
p = os.path.join("$OP_DIR", "CMakePresets.json")
cfg = json.load(open(p))
cv = cfg["configurePresets"][0]["cacheVariables"]
cv["ASCEND_CANN_PACKAGE_PATH"] = {"type": "PATH", "value": "$ASCEND_TOOLKIT_HOME"}
cv["ASCEND_PYTHON_EXECUTABLE"] = {"type": "STRING", "value": os.environ.get("FRAMEWORK_PYTHON", "python3")}
json.dump(cfg, open(p, "w"), indent=4)
print("[build] CANN ->", cv["ASCEND_CANN_PACKAGE_PATH"]["value"])
PY

echo "=== [build] build + install custom op ==="
cd "$OP_DIR"
rm -rf build_out
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}"
export DDK_PATH="$ASCEND_HOME_PATH"
export NPU_HOST_LIB="$ASCEND_HOME_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/devlib"
bash build_op.sh 2>&1 | tail -6
RUN_PKG="$(ls build_out/custom_opp_*.run 2>/dev/null | head -1)"
[[ -z "$RUN_PKG" ]] && { echo "[build] no .run produced"; exit 1; }
"$RUN_PKG" --quiet --install-path="$CUSTOM_OPP_HOME" 2>&1 | tail -2
# set_env.bash (generated above) already prepends op_api/lib to LD_LIBRARY_PATH
# and sets ASCEND_CUSTOM_OPP_PATH — no manual LD_LIBRARY_PATH poking needed.
# shellcheck disable=SC1091
source "$CUSTOM_OPP_HOME/vendors/customize/bin/set_env.bash"
export CUSTOM_OPP_SOURCED=1

echo "=== [build] build + install pybind wheel ==="
cd "$OP_DIR"
# setup.py + extension/custom_op.cpp + common/pytorch_npu_helper.hpp all live
# under src/ascendc/ now (setup.py is a sibling of extension/ and common/).
python3 setup.py build bdist_wheel
pip3 install dist/custom_ops-*.whl --force-reinstall --no-deps

echo "[build] done."
