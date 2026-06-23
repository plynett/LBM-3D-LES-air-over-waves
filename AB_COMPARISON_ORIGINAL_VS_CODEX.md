# A/B Comparison: Original vs CODEX Moving-Boundary Implementation

Files compared:

- Original solver: `LBM_3D_LES_air_over_waves_v1.py`
- CODEX solver: `LBM_3D_LES_air_over_waves_CODEX.py`
- A/B harness: `run_ab_case.py`

The original solver file was not modified. The A/B harness imports either solver, overrides only run length/plot frequency, and replaces `visualize()` with the same diagnostic plotting function for both cases.

## Test Definition

Both cases used:

```text
steps = 10000
plot_freq = 500
plot_start = 0
random_seed = 12345
Re = 3.958e7
tau0 = 0.5000002274
```

Diagnostic band:

```text
0 < phi <= 10 dx
```

Diagnostics:

- pressure RMS in the boundary band,
- pressure high-pass RMS in the boundary band,
- vorticity RMS in the boundary band,
- vorticity high-pass RMS in the boundary band,
- max speed,
- CODEX-only cut-link/refill counts.

The visible Matplotlib figure was shown during both runs. Frames and metrics were saved.

## Output Files

Seeded original:

- frames: `test_runs/ab_seeded_original/`
- metrics: `test_runs/ab_seeded_original/original_seeded_metrics.csv`

Seeded CODEX:

- frames: `test_runs/ab_seeded_codex/`
- metrics: `test_runs/ab_seeded_codex/codex_seeded_metrics.csv`

Comparison plot:

- `test_runs/ab_seeded_comparison_metrics.png`

Preliminary unseeded runs were also saved under:

- `test_runs/ab_original/`
- `test_runs/ab_codex/`
- `test_runs/ab_comparison_metrics.png`

The seeded runs are the authoritative comparison.

## Seeded Results Summary

Mean over the 20 plotted samples:

```text
metric                  original      CODEX        CODEX/original
pressure RMS            65.93 Pa      98.37 Pa     1.49
pressure high-pass RMS  26.37 Pa      60.20 Pa     2.33
vorticity RMS           7.616e-03     7.261e-03    0.95
vorticity high-pass RMS 1.995e-03     2.519e-03    1.26
```

Final sample at step 10000:

```text
metric                  original      CODEX        CODEX/original
pressure RMS            46.11 Pa      74.88 Pa     1.62
pressure high-pass RMS  29.02 Pa      46.63 Pa     1.61
vorticity RMS           6.059e-03     5.669e-03    0.94
vorticity high-pass RMS 2.073e-03     2.248e-03    1.08
max speed               36.34 m/s     38.79 m/s    1.07
```

## Interpretation

The CODEX moving-boundary implementation is more geometrically explicit, but it is not better by the requested noise metrics.

The original boundary produces substantially lower pressure high-pass noise in the 10-cell moving-boundary band. CODEX is about 2.33x worse on average and about 1.61x worse at step 10000.

The vorticity high-pass comparison is closer, but still worse for CODEX: about 1.26x worse on average and about 1.08x worse at step 10000.

The lower total vorticity RMS in CODEX at the final step does not offset the high-pass result, because the user's stated concern is grid-size vorticity variation near the moving boundary.

## Physical Conclusion

This attempt should not replace the original boundary condition yet.

The likely issue is not the idea of using SDF/cut-link geometry itself; that remains the right direction for arbitrary geometry. The likely issue is the specific interpolated moving bounce-back realization:

- the moving-wall correction scaling may not match the chosen BFL interpolation branch,
- the pull-stream formulation may be applying the correction with the wrong population/time level,
- the SDF cell-state override is only partial because the older height-gated per-link fields still exist,
- the transition refill may be injecting local pressure disturbances when cells are uncovered,
- the original equilibrium blending may have been acting as an implicit pressure regularization, even though it is not a clean physical wall condition.

## Repair Run

The CODEX boundary implementation was repaired after this comparison by removing the failed SDF/cut-link replacement and restoring the original continuous moving-boundary path:

- no cell-center SDF fluid/solid override,
- no solid-to-fluid refill pulses,
- no stream-side BFL-like reconstruction in the existing pull-stream loop,
- restored original closed-link equilibrium conditioning,
- restored original open-fraction streaming blend,
- restored original post-stream moving-wall reflection blend.

Fixed CODEX run:

```text
module      = LBM_3D_LES_air_over_waves_CODEX.py
label       = codex_fixed
steps       = 10000
plot_freq   = 500
random_seed = 12345
output      = test_runs/ab_fixed_codex/
```

The visible Matplotlib figure was shown while running. The physical parameters were unchanged:

```text
Re   = 3.958e7
tau0 = 0.5000002274
```

Mean over the 20 plotted samples:

```text
metric                  original      fixed CODEX   fixed/original
pressure RMS            65.93 Pa      65.93 Pa      1.00
pressure high-pass RMS  26.37 Pa      26.37 Pa      1.00
vorticity RMS           7.616e-03     7.616e-03     1.00
vorticity high-pass RMS 1.995e-03     1.995e-03     1.00
max speed               36.77 m/s     36.77 m/s     1.00
```

Final sample at step 10000:

```text
metric                  original      fixed CODEX   delta
pressure RMS            46.1117 Pa    46.1117 Pa    0
pressure high-pass RMS  29.0178 Pa    29.0178 Pa    0
vorticity RMS           6.05869e-03   6.05869e-03   0
vorticity high-pass RMS 2.07349e-03   2.07349e-03   0
max speed               36.3443 m/s   36.3443 m/s   0
```

## Updated Conclusion

The failed CODEX boundary replacement has been removed. The current CODEX copy now matches the original boundary behavior and keeps only diagnostics/runtime controls around it.

For the next arbitrary-surface attempt, the constraint is sharper: preserve the original continuous link-fraction behavior and derive any non-single-valued obstacle treatment so that mass flux, no penetration, no slip/moving-wall momentum exchange, and pull-stream indexing remain consistent in the actual moving-bed geometry. Do not use added damping, altered viscosity, altered Reynolds number, or smoothed flow fields to quiet the boundary.

## Link-Wise Wall Velocity Candidate

After restoring the original boundary path, CODEX tested one conservative physical improvement:

- keep original vertical link-fraction geometry,
- keep original open-fraction streaming and moving-wall population formulas,
- sample prescribed wall velocity at each link endpoint instead of using one cell-centered wall velocity for every direction.

This is controlled by:

```text
LBM_BOUNDARY_GEOMETRY=vertical
LBM_WALL_VELOCITY_SAMPLING=link
```

Visible 10,000-step seeded result:

```text
metric                  original cell  link velocity  link/original
mean pressure RMS       65.9334 Pa     64.6924 Pa     0.9812
mean pressure HP RMS    26.3737 Pa     25.8870 Pa     0.9815
mean vorticity RMS      7.61599e-03    7.62416e-03    1.0011
mean vorticity HP RMS   1.99453e-03    1.99186e-03    0.9987

step 10000:
pressure HP RMS         29.0178 Pa     28.4186 Pa     0.9794
vorticity HP RMS        2.07349e-03    2.07653e-03    1.0015
```

The signed-distance geometry variants were also tested and rejected because they increased high-pass noise. The active CODEX candidate is therefore the original boundary with link-wise wall velocity interpolation.

## Arbitrary Geometry Cube Test

The CODEX solver was extended from a single-valued moving bed to a combined bed/cube geometry using per-link wall source selection:

- moving bed links use the original moving-wall treatment,
- stationary cube links use zero wall velocity,
- nearest wall source is selected per lattice link from the bed gap and cube SDF,
- no refill, damping, viscosity, relaxation-time, forcing, or flow-speed changes.

The requested cube size was revised from 10 m to 6 m on a side. The accepted test uses the original `Ly=10 m` spanwise domain with a centered 6 m cube.

Visible 10,000-step result:

```text
LBM_OBSTACLE_MODE=bed_cube
LBM_CUBE_SIDE_M=6.0
grid                 600 x 20 x 40
cube center          (150.0, 5.0, 11.0) m
Re                   3.958e7
tau0                 0.5000002274
refill               0 throughout

mean pressure HP RMS  30.8414 Pa
mean vorticity HP RMS 2.32031e-03

step 10000:
pressure HP RMS       28.9914 Pa
vorticity HP RMS      2.00757e-03
max speed             38.7738 m/s
```

The run completed cleanly. The cube causes an early local acceleration above the nominal low-Mach target, but the final max speed returned below `0.1` LB without altering the physical case.

## Pitched Cube Follow-Up

The subsequent 45 degree rotation requested for the cube must be interpreted carefully. A yaw rotation about the vertical z axis is a valid 3D rotation, but it does not visibly rotate the cube in the x-z diagnostic slice. The corrected visible rotation test used pitch about the spanwise y axis:

```text
LBM_OBSTACLE_MODE=bed_cube
LBM_CUBE_SIDE_M=6.0
LBM_CUBE_YAW_DEG=0.0
LBM_CUBE_PITCH_DEG=45.0
steps=20000
plot_freq=1000
random_seed=12345
```

Visible 20,000-step result:

```text
grid                  600 x 20 x 40
cube center           (150.0, 5.0, 12.243) m
Re                    3.958e7
tau0                  0.5000002274
refill                0 throughout

mean pressure HP RMS  31.1776 Pa
mean vorticity HP RMS 2.49501e-03

tail mean, steps >= 10000:
pressure HP RMS       29.1048 Pa
vorticity HP RMS      2.19125e-03

step 20000:
pressure HP RMS       27.6227 Pa
vorticity HP RMS      1.97006e-03
max speed             35.6283 m/s
```

The final frame shows the cube as a 45 degree diamond in the x-z slice. The physical setup was not altered; the early local peak speed remains a modeling concern to track separately from boundary-noise appearance.
