# BICG-Stable 算法

## BICGSTAB Algorithm

1. $r_0 = b - \mathbf{A}x_0$
2. Choose an arbitrary vector $\hat{r_0}$ such that $(\hat{r_0}, r_0)\ne 0$, eg. $\hat{r_0} = r_0$
3. $\rho_0 = (\hat{r}_0, r_0)$
4. $p_0 = r_0$
5. For $i = 1, 2, 3, \ldots$
    1. $y = \mathbf{M^{-1}} p_{i-1}$
    2. $v = \mathbf{A}{y}$
    3. $\alpha = \rho_{i-1} / (\hat{r}_0, v)$
    4. $h = x_{i-1} + \alpha y$
    5. $s = r_{i-1} - \alpha v$
    6. If $s$ is within the accuracy tolerance then $x_{i-1} = h$ and quit.
    7. $z = \mathbf{M^{-1}} s$
    8. $t = \mathbf{A}z$
    9. $\omega = (t, s) / (t, t)$
    10. $x_i = h + \omega z$
    11. $r_i = s - \omega t$
    12. If $r_i$ is within the accuracy tolerance then quit.
    13. $\rho_i = (\hat{r}_0, r_i)$
    14. $\beta = (\rho_i / \rho_{i-1}) (\alpha / \omega)$
    15. $p_i = r_i + \beta (p_{i-1} - \omega v)$

## Baseline 代码介绍

```text
.
├── CMakeLists.txt     # CMake 构建文件
├── compile.sh         # 编译脚本
├── include            # 头文件目录
│   └── judger.h       # 评测器头文件
├── run.sh             # 运行脚本
└── src                # 源代码目录
    ├── bicgstab       # BICGSTAB 算法实现
    │   ├── solver.c   # C 版本
    │   └── solver.f90 # Fortran 版本 (如果有)
    ├── judger.cpp     # 评测器实现
    └── main.cpp       # 主程序
```

**严禁修改计时区**，因此对于 `src/main.cpp` 的修改，请仅限于添加 MPI 相关的代码。还有 `src/judger.cpp`, `include/judger.h` 不可以修改，其他部分都可以进行修改。

如果希望提交到 OJ 测评，请注意：

- `src/judger.cpp`, `include/judger.h` 在 OJ 测评时会被替换，因此修改并不会生效。
- 如果有新增文件并希望可以被 OJ 测评，请将新文件放在 `src` 或 `include` 文件夹中。OJ 会使用的文件为：`src` 文件夹，`include` 文件夹，`CMakeLists.txt`, `run.sh`, `compile.sh`，其他的文件不会生效。

在使用 OpenMP 的时候，请记得在 `CMakeLists.txt` 中添加 `-fopenmp` 或者 `-qopenmp` 选项哦。

数据文件在 `/river/hpc101/2025/lab4/data` 中，你可以在代码的根目录下，通过 `ln -s /river/hpc101/2025/lab4/data data` 来引用它们。如果直接复制数据的话，请注意不要向 OJ 或者学在浙大提交数据文件。
