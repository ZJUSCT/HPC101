# Lab4: AMSS-NCKU

Trimmed AMSS-NCKU numerical relativity lab for the single configuration in
`AMSS_NCKU_Input.py`:

- CPU evolution executable: `ABE`
- GPU evolution executable: `ABEGPU`
- Initial data: `Ansorg-TwoPuncture`
- Evolution equation: vacuum `BSSN`
- Grid: `Patch`, cell-centered, equatorial symmetry
- Finite difference: 4th order

Both CPU and GPU evolution paths are kept. Old verification scripts and
reference case files have been removed from this directory.

## Environment

The container is a Debian 13 box with the toolchain preinstalled. Two
architectures are shipped:

| Architecture | Hardware | Default toolchain | Executables |
| ------------ | -------- | ----------------- | ----------- |
| `linux/arm64` | Kunpeng 920B | GNU 14 + OpenMPI 5, `AMSS_ENABLE_GPU=OFF` | `TwoPunctureABE`, `ABE` |
| `linux/amd64` | x86_64 + NVIDIA V100 | GNU 13 + OpenMPI 5 + CUDA 12.4, `sm_70` | `TwoPunctureABE`, `ABE`, `ABEGPU` |

x86/V100 uses **CUDA 12.4** (Debian 13 package), **not** CUDA 13 — CUDA 13
dropped `sm_70` support. NVIDIA driver is **not** in the image; the host
injects it via NVIDIA Container Toolkit.

### Available compilers (advanced comparison)

The image ships several compiler/MPI combos beyond the default. Switch
via `CXX` / `FC` / `CUDACXX` / `MPI_CXX_COMPILER` env vars before
`compile.sh`.

- amd64: `icpx`/`ifx` (after `. /etc/profile.d/oneapi.sh`),
  `clang++`/`flang-19`, MPICH (`mpicxx.mpich` / `mpiexec.mpich`)
- arm64: `armclang++`/`armflang` (after `module load acfl/24.10.1`),
  MPICH (`mpicxx.mpich` / `mpiexec.mpich`)

CMake caches compilers, so switch toolchains with a fresh build dir.

### Switching MPI

Default is OpenMPI. To switch to MPICH (advanced comparison only):

```bash
export MPI_CXX_COMPILER=/usr/bin/mpicxx.mpich
export MPIEXEC_EXECUTABLE=/usr/bin/mpiexec.mpich
export AMSS_MPIEXEC=/usr/bin/mpiexec.mpich
rm -rf $AMSS_BUILD_DIR      # CMake cache is OpenMPI-specific; MUST use a fresh build dir
./compile.sh
```

Switching MPI **requires** a fresh `AMSS_BUILD_DIR` — the existing
`CMakeCache.txt` records the previous MPI's wrapper path and will fail
at configure time.

### CUDA-aware MPI

`AMSS_MPI_CUDA_AWARE` defaults to `0`. Debian OpenMPI/MPICH do not
advertise CUDA-aware support and the current AMSS code path uses
host-staging. To experiment, pass `-DAMSS_MPI_CUDA_AWARE=1` to
`compile.sh` and verify your MPI actually supports device buffers.

### V100 driver and `sm_70`

x86/V100 nodes must run NVIDIA driver ≥ `525.60.13` (CUDA 12.x minor
compatibility floor). The image's `nvcc` targets `sm_70`; verify with
`cuobjdump` if a kernel fails to load on V100.

## Build

```bash
./compile.sh
```

This builds:

- `build/ABE`
- `build/ABEGPU` (only when `AMSS_ENABLE_GPU=ON`, which is the default on
  the amd64 image and `OFF` on the arm64 image)
- `build/TwoPunctureABE`

For a faster debug build:

```bash
./compile.sh -DAMSS_OPT='-O0'
```

## Run

```bash
./run.sh
```

Optional TwoPuncture cache:

```bash
./run.sh --twop-cache
```

The run driver writes results under:

```text
GW250118/AMSS_NCKU_output/
GW250118/figure/
```

## Correctness check

```bash
./check.sh
```

`check.sh` resolves `RESULT_DIR` against `AMSS_OUTPUT_ROOT` (or the lab
root if unset), and `GOLDEN_DIR` against the lab root. The shipped
`golden/` directory is used by default. Pass an explicit `RESULT_DIR` to
check a non-default run directory. See `python3 scripts/check_result.py --help`
for details.

## Main Files

- `AMSS_NCKU_Input.py`: the fixed run parameters
- `AMSS_NCKU_Program.py`: run driver
- `scripts/setup.py`, `scripts/numerical_grid.py`,
  `scripts/generate_TwoPuncture_input.py`, `scripts/renew_puncture_parameter.py`:
  parfile generation
- `scripts/makefile_and_run.py`: launches `TwoPunctureABE` and the configured
  `ABE` or `ABEGPU` executable through MPI
- `scripts/plot_xiaoqu.py`, `scripts/plot_GW_strain_amplitude_xiaoqu.py`:
  post-run plots
- `scripts/check_result.py`: validates simulation output against golden results
- `src/`: source files still needed to compile `ABE`, `ABEGPU`, and `TwoPunctureABE`
