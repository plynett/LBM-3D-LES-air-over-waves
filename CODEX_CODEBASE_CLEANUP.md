# Codebase Cleanup Review

## Scope

Reviewed the active project files:

- `LBM_3D_LES_air_over_waves_CODEX.py`
- `run_ab_case.py`
- `lbm_lattice.py`
- `lbm_env.py`
- `README.md`
- `WATER_FREE_SURFACE_IMPLEMENTATION.md`

The original reference files were not modified.

## Removed Relics

- Removed a hard-disabled 3D vorticity isosurface plotting block from the active
  solver. It was guarded by `plot_isosurfaces = 0`, never executed, and forced
  optional `skimage` / `Poly3DCollection` imports.
- Removed the unused free-surface link-fraction diagnostic path:
  `free_surface_open_frac`, `near_free_surface`, and
  `build_free_surface_link_fractions()`. These fields were updated every step
  but not read by the solver or diagnostic harness.
- Removed the inactive `LBM_FREE_SURFACE_BOUNDARY=link` option. It did not
  select a distinct boundary closure in the active solver.
- Removed the unused `env_float()` helper.
- Replaced the duplicated hardcoded D3Q19 opposite-direction map with the shared
  `D3Q19_OPP` constant from `lbm_lattice.py`.
- Removed the stale solver-local Matplotlib diagnostic implementation. Direct
  solver runs now bind to the shared `run_ab_case.py` visualizer, so the 2D
  diagnostic figure, CSV metrics, and render snapshots have one active owner.
- Moved Matplotlib ownership into `run_ab_case.py` instead of borrowing
  `module.plt` from the solver.
- Consolidated inline PyVista render argument construction into one
  `RENDER_CLI_ENV_ARGS` mapping and removed obsolete render env helper code.

## Left In Place

- `run_ab_case.py` diagnostics were left in place. They are host-side only and
  do not feed back into the simulation.
- `height`, `height_static`, `prescribed_solitary`, and `vof` free-surface
  branches were retained because they are reachable model paths.
- Air-mode features such as the inlet/fringe/top drive, KBC collision path, and
  optional LES/backscatter support were retained because they are part of the
  original solver capability or current diagnostics.
- Historical notes in `WATER_FREE_SURFACE_IMPLEMENTATION.md` were retained, with
  a cleanup note clarifying that old link-closure modes are no longer active
  run options.

## Physics Impact

The cleanup removes unused code paths and stale documentation only. It does not
change viscosity, relaxation time, gravity, boundary-wall forcing, collision
physics, free-surface pressure reconstruction, or diagnostic pressure formulas.
