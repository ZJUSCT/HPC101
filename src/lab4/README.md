# amss-ncku-lab

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

## Build

```bash
./compile.sh
```

This builds:

- `build/ABE`
- `build/ABEGPU`
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
GW190521/AMSS_NCKU_output/
GW190521/figure/
```

## Main Files

- `AMSS_NCKU_Input.py`: the fixed run parameters
- `AMSS_NCKU_Program.py`: run driver
- `setup.py`, `numerical_grid.py`, `generate_TwoPuncture_input.py`,
  `renew_puncture_parameter.py`: parfile generation
- `plot_xiaoqu.py`, `plot_GW_strain_amplitude_xiaoqu.py`: post-run plots
- `src/`: source files still needed to compile `ABE`, `ABEGPU`, and `TwoPunctureABE`
