# LBM-3D-LES-air-over-waves (Taichi)

This repository contains a Taichi-based 3D D3Q19 lattice Boltzmann solver for
the original air-over-moving-waves problem and the current single-fluid water
free-surface experiments.

## Active Files

- `LBM_3D_LES_air_over_waves_v1.py`: original reference source.
- `LBM_3D_LES_air_over_waves_CODEX.py`: active development solver.
- `lbm_lattice.py`: shared D3Q19 lattice constants.
- `lbm_env.py`: small environment-variable parsing helpers.
- `run_ab_case.py`: diagnostic run harness that records frames and metrics.
- `AGENTS.md`: project physics and boundary-condition rules.
- `WATER_FREE_SURFACE_IMPLEMENTATION.md`: development log for the water and
  free-surface path.

## Numerical Model

- Lattice: D3Q19.
- Equilibrium: standard low-Mach, second-order Hermite expansion.
- Streaming: pull scheme.
- Collision: selectable `regularized` or retained KBC-style collision path.
- LES: optional Smagorinsky-style local eddy-viscosity update.
- Solid geometry: per-link solid/open flags and open fractions from the moving
  boundary representation.
- Water free surface: height-function kinematic path, plus an experimental VOF
  path under `LBM_FREE_SURFACE_TRACKING=vof`.

## Current Free-Surface Pressure Setup

For water free-surface runs, the default pressure formulation is:

```powershell
$env:LBM_PRESSURE_FORMULATION = "hydrostatic_balanced"
```

This stores the LBM density as the pressure perturbation about the flat
still-water hydrostatic state, while the diagnostic plot reconstructs total
gauge pressure for display. This is a physical hydrostatic decomposition, not a
damping or smoothing model.

The active free-surface boundary options are:

```powershell
$env:LBM_FREE_SURFACE_BOUNDARY = "cell"        # default
$env:LBM_FREE_SURFACE_BOUNDARY = "hydrostatic"
```

The old experimental `link`, `fsk_link`, and `fsl` modes are not active solver
options.

## Minimal Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install taichi numpy matplotlib
```

## Example Water Gaussian Run

```powershell
$env:LBM_PHYSICS_MODE = "water"
$env:LBM_FREE_SURFACE_INITIAL = "gaussian"
$env:LBM_GAUSSIAN_AMPLITUDE_M = "8"
$env:LBM_X_BOUNDARY = "solid"
$env:LBM_TI_ARCH = "gpu"
$env:LBM_PRESSURE_DIAGNOSTIC = "total_pressure"

.\.venv\Scripts\python.exe .\run_ab_case.py `
  --module .\LBM_3D_LES_air_over_waves_CODEX.py `
  --label water_gaussian_amp8 `
  --steps 10000 `
  --plot-freq 500 `
  --plot-start 0 `
  --output-dir .\test_runs\water_gaussian_amp8 `
  --random-seed 12345
```

Leave Matplotlib interactive windows enabled when visually assessing pressure
and vorticity behavior.

## Example Disconnected VOF Run

This starts from a flat pool by setting the Gaussian amplitude to zero, then
adds a finite VOF water block above the free surface:

```powershell
$env:LBM_PHYSICS_MODE = "water"
$env:LBM_FREE_SURFACE_TRACKING = "vof"
$env:LBM_FREE_SURFACE_INITIAL = "gaussian"
$env:LBM_GAUSSIAN_AMPLITUDE_M = "0"
$env:LBM_PRESSURE_FORMULATION = "vof_component_balanced"
$env:LBM_VOF_INITIAL_SHAPE = "block"
$env:LBM_VOF_BLOCK_SIZE_M = "4"
$env:LBM_VOF_BLOCK_BOTTOM_M = "12"
$env:LBM_X_BOUNDARY = "solid"
$env:LBM_VOF_THIN_GAP_BRIDGE = "1"          # optional unresolved one-cell coalescence bridge
$env:LBM_VOF_THIN_GAP_BRIDGE_STRENGTH = "4" # conservative test value; very large values are diagnostic
$env:LBM_VOF_COLLAPSE_TRAPPED_GAS = "1"     # optional unresolved trapped-air removal
$env:LBM_VOF_COLLAPSE_INTERVAL = "200"
$env:LBM_VOF_COLLAPSE_MAX_VOLUME_M3 = "8"
$env:LBM_TI_ARCH = "gpu"

.\.venv\Scripts\python.exe .\run_ab_case.py `
  --module .\LBM_3D_LES_air_over_waves_CODEX.py `
  --label vof_falling_block_amp0 `
  --steps 1500 `
  --plot-freq 250 `
  --plot-start 0 `
  --output-dir .\test_runs\vof_falling_block_amp0 `
  --random-seed 12345
```

For detached VOF water, use `vof_component_balanced`. This balances the initial
flat pool against the still-water hydrostatic reference while retaining ordinary
gravity on detached water above the pool. The old full `total_pressure` setup is
kept only as an explicit diagnostic via `LBM_ALLOW_TOTAL_PRESSURE_VOF_DIAGNOSTIC=1`.

The optional trapped-gas collapse pass is a low-overhead unresolved-gas closure.
It flood-fills exterior gas on the GPU, collapses only trapped gas below
`LBM_VOF_COLLAPSE_MAX_VOLUME_M3`, and removes the same tracked water mass from
exterior VOF interface cells. It is not a column-height collapse.
