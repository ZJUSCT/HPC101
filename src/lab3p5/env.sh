#!/usr/bin/env bash
# lab3.5 student kit — environment activation.
# Source from the kit root:  source ./env.sh
#
# Activates the system CANN toolkit and ensures python3 is on PATH. Relies only
# on system-level paths under /usr/local/Ascend and /usr/local/python* — no
# hard-coded home directories, no competition-specific conda env.
#
# NOTE: this file is *sourced* into an interactive shell, so do NOT add `set -e`
# here — it would leak into the caller's shell and terminate it on the first
# non-zero return.

_KIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. System CANN toolkit -------------------------------------------------
# The image may expose `msprof` without the full runtime library paths (notably
# libhccl.so). Therefore command availability is not a sufficient activation
# check. Source set_env.sh once per shell, preferring the `cann` symlink.
if [[ -z "${_LAB3P5_CANN_ENV_SOURCED:-}" ]]; then
    for _set_env in \
        /usr/local/Ascend/cann/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
        /usr/local/Ascend/cann-8.5.0/set_env.sh; do
        if [[ -f "$_set_env" ]]; then
            # shellcheck disable=SC1090
            source "$_set_env"
            _LAB3P5_CANN_ENV_SOURCED=1
            break
        fi
    done
    unset _set_env
fi

# --- 2. Logical NPU device ---------------------------------------------------
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"

# --- 3. python3 --------------------------------------------------------------
# The system interpreter lives outside the default PATH on this image; put it
# back if python3 isn't already resolvable. The kit depends on torch/torch_npu,
# which live in the cp311 site-packages under /usr/local/python3.11.14 — so prefer
# that interpreter before falling back to /usr/bin/python3 (the OS 3.10, which
# has none of the kit packages). /usr/local/bin is kept for cmake/ninja/pip.
if ! command -v python3 >/dev/null 2>&1 || ! python3 -c "import torch, torch_npu" >/dev/null 2>&1; then
    for _py in /usr/local/python3.11.14/bin /usr/local/bin /usr/bin; do
        if [[ -x "$_py/python3" ]]; then
            export PATH="$_py:$PATH"
            break
        fi
    done
    unset _py
fi

# --- 4. Custom operator package (built by build.sh) -------------------------
# set_env.bash is generated at build time and already exports
# ASCEND_CUSTOM_OPP_PATH plus prepends op_api/lib to LD_LIBRARY_PATH — so we
# only source it. Guard with a flag so re-sourcing doesn't pile up duplicates.
# CUSTOM_OPP_HOME matches build.sh's --install-path (overridable).
export CUSTOM_OPP_HOME="${CUSTOM_OPP_HOME:-$HOME/custom_opp}"
if [[ -z "${CUSTOM_OPP_SOURCED:-}" ]] && [[ -f "$CUSTOM_OPP_HOME/vendors/customize/bin/set_env.bash" ]]; then
    # shellcheck disable=SC1091
    source "$CUSTOM_OPP_HOME/vendors/customize/bin/set_env.bash"
    export CUSTOM_OPP_SOURCED=1
fi

# --- 5. Make the kit importable --------------------------------------------
# test_op.py imports `case_specs`, `custom_ops_lib`, and `src.<lang>` by name.
# `src` lives under the kit root; `case_specs`/`test_op`/`get_time` live under
# the sibling `checker/` dir. Put both on PYTHONPATH so a bare
# `python3 checker/test_op.py` resolves every import.
export PYTHONPATH="$_KIT_ROOT:$_KIT_ROOT/checker${PYTHONPATH:+:$PYTHONPATH}"

# --- 6. Sanity checks (loud, not silent) -----------------------------------
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    echo "[env.sh] ERROR: ASCEND_TOOLKIT_HOME is unset — no CANN set_env.sh found under /usr/local/Ascend" >&2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "[env.sh] ERROR: python3 not found on PATH" >&2
fi

echo "[env.sh] CANN=${ASCEND_TOOLKIT_HOME:-?}  device=${ASCEND_DEVICE_ID}  python=$(command -v python3 || echo '?')"
unset _KIT_ROOT
