/**
 * @file fused_add_rms_norm_tiling.h
 * @brief Tiling data definition for the FusedAddRmsNorm operator
 *
 * Op: FusedAddRmsNorm(x, residual, weight, eps, enable_pdl) -> (y, residual_out)
 *   residual_out = x + residual
 *   y = residual_out / sqrt(mean(residual_out^2, dim=-1) + eps) * weight
 *
 * Inputs/outputs are FP16 ND tensors of shape (B, H); weight is (H,). The kernel
 * is row-parallel (one row per work item, split across AIV cores) and computes
 * in FP32 for precision, casting back to FP16 on the way out.
 *
 * Only op_host includes this header. The kernel accesses the tiling fields
 * through the GET_TILING_DATA(tilingData, tiling) macro (it must NOT include
 * this file — that would pull in graph/types.h and break the kernel compile).
 */
#ifndef FUSED_ADD_RMS_NORM_TILING_H
#define FUSED_ADD_RMS_NORM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
// 32B UB alignment unit == 16 FP16 elements == 8 FP32 elements.
constexpr int32_t ALIGN_NUM = 16;

BEGIN_TILING_DATA_DEF(FusedAddRmsNormTilingData)
    // Number of rows (B) in the (B, H) input.
    TILING_DATA_FIELD_DEF(int32_t, batchSize);
    // Hidden size (H) — the per-row reduction length.
    TILING_DATA_FIELD_DEF(int32_t, hiddenSize);
    // hiddenSize rounded up to a multiple of ALIGN_NUM (FP16 32B alignment).
    // UB tiles are sized to this; vector ops run on alignedHidden, reduce runs
    // on hiddenSize (the tail is masked out / ignored).
    TILING_DATA_FIELD_DEF(int32_t, alignedHidden);
    // 32B alignment unit in elements (== ALIGN_NUM, mirrored for the kernel).
    TILING_DATA_FIELD_DEF(int32_t, alignNum);
    // eps added inside the sqrt for numerical stability.
    TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedAddRmsNorm, FusedAddRmsNormTilingData)
}
#endif  // FUSED_ADD_RMS_NORM_TILING_H
