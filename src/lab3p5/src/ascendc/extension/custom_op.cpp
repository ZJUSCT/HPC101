/*
 * FusedAddRmsNorm — Ascend C (aclnn) entry for the student framework.
 *
 * This file is the ONLY C++ the student must touch for the Ascend C backend.
 * It wires the op's aclnn API into a torch custom op so test_op.py can call it.
 *
 * The actual kernel lives in op/op_host + op/op_kernel (the FusedAddRmsNorm
 * aclnn op you build with build.sh). Here we just launch it via EXEC_NPU_CMD,
 * exactly like the S9 samples (Concat/Transpose).
 */
#include <torch/extension.h>
#include "../common/pytorch_npu_helper.hpp"

using namespace at;

/**
 * FusedAddRmsNorm host launch.
 *   residual_out = residual + x
 *   y = residual_out / sqrt(mean(residual_out^2, dim=-1) + eps) * weight
 *
 * Args:
 *   x        : (B, H) float16
 *   residual : (B, H) float16
 *   weight   : (H,)    float16
 *   eps      : float
 * Returns:
 *   tuple(y, residual_out) — both (B, H) float16.
 *
 * NOTE: this default impl just calls the aclnn API you registered. If you
 * prefer a hand-written Ascend C kernel (op/op_kernel/*.cpp), that path is
 * wired too — just make sure aclnnFusedAddRmsNorm resolves to your kernel
 * after build.sh + install.
 */
std::tuple<Tensor, Tensor> fused_add_rmsnorm_impl_npu(
    const Tensor &x, const Tensor &residual, const Tensor &weight,
    double eps) {
    Tensor y = at::empty_like(x);
    Tensor residual_out = at::empty_like(x);

    // aclnnFusedAddRmsNorm(x, residual, weight, eps, enable_pdl, y, residual_out)
    // Profiling warm-up is handled by `msprof op --warm-up`, so a normal op
    // invocation launches exactly one FusedAddRmsNorm kernel.
    bool enable_pdl = false;
    EXEC_NPU_CMD(aclnnFusedAddRmsNorm,
                 x, residual, weight, eps, enable_pdl,
                 y, residual_out);
    return std::make_tuple(y, residual_out);
}

TORCH_LIBRARY(fusedaddrmsnorm, m) {
    m.def("fused_add_rmsnorm(Tensor x, Tensor residual, Tensor weight, "
          "float eps) -> (Tensor y, Tensor residual_out)");
}

TORCH_LIBRARY_IMPL(fusedaddrmsnorm, PrivateUse1, m) {
    m.impl("fused_add_rmsnorm", &fused_add_rmsnorm_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_rmsnorm", &fused_add_rmsnorm_impl_npu,
          "FusedAddRmsNorm (residual+x then RMSNorm)");
}
