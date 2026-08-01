"""Build the student's custom-op extension into a wheel and install it.

Mirrors the S9 `Transpose/setup.py` exactly. Produces a `custom_ops_lib`
module that test_op.py imports.
"""
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))

exts = []
ext1 = NpuExtension(
    name="custom_ops_lib",
    sources=["./extension/custom_op.cpp"],
    extra_compile_args=[
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        # Make `../common/pytorch_npu_helper.hpp` resolvable from extension/custom_op.cpp.
        '-I' + os.path.dirname(os.path.abspath(__file__)),
    ],
)
exts.append(ext1)

setup(
    name="custom_ops",
    version='1.0',
    keywords='custom_ops',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)
