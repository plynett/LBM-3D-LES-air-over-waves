# CODEX Moving Boundary Notes

Working file: `LBM_3D_LES_air_over_waves_CODEX.py`

Original file preserved: `LBM_3D_LES_air_over_waves_v1.py`

## Objective

Maintain a CODEX working copy while preserving physical fidelity. Any new arbitrary-geometry boundary method must first outperform or at least match the original moving-boundary behavior without changing viscosity, relaxation time, forcing, flow speed, or adding artificial damping.

The intended solid-wall condition remains no slip and no penetration at the moving bed:

```text
u_fluid(x_wall) = u_wall(x_wall)
```

where `x_wall` is the physical wall location represented by the lattice geometry model.

## Current Boundary Direction

The previous CODEX SDF/cut-link replacement is superseded. It produced higher pressure and vorticity high-pass noise than the original in the 10-cell moving-boundary band.

The active CODEX boundary has been restored to the original moving-boundary path:

- `build_lattice_open_from_free_surface_periodic()` uses the original half-link open fractions and keeps `lo[0]` based on whether any link remains open.
- `update_bed_populations_and_reinit()` uses the original continuous closed-link population conditioning toward moving-wall equilibrium.
- `stream()` uses the original pull-stream blend:

```text
f_new(x,q) = frac_open * f(x-c_q,q) + (1-frac_open) * f(x,opp(q))
```

- `apply_open_boundary_conditions()` again applies the original post-stream moving-wall reflection blend:

```text
f_wall = f_pre(opp(q)) - 6*w_opp*rho0*(c_opp . u_wall)
f_new(q) = frac_open*f_new(q) + (1-frac_open)*f_wall
```

The CODEX copy still includes runtime overrides and boundary-noise diagnostics, but the active moving-boundary physics now matches the original implementation.

## Removed Failed Change

The following changes were removed because they made the implementation worse:

- hard cell-center SDF fluid/solid state override,
- moving solid-to-fluid refill events,
- stream-side BFL-like reconstruction in the existing pull-stream loop,
- disabling the original post-stream moving-wall blend.

The key lesson is that the original is better because it avoids discrete cell-state switching and represents the moving bed through continuous link fractions. A future arbitrary-surface method should preserve that continuity and must be derived carefully for the existing pull-stream time/indexing convention.

## Boundary Diagnostics

Each plot reports diagnostics in the moving-boundary band:

```text
0 < phi <= 10 * dx
```

Reported quantities:

- pressure RMS after subtracting local band mean,
- pressure high-pass RMS,
- vorticity RMS after subtracting local band mean,
- vorticity high-pass RMS,
- cut-link count,
- solid-to-fluid refill count.

The live plot draws the bed line and the `+10 dx` assessment band.

## Historical SDF Smoke Run (Superseded)

This run used the failed SDF/cut-link replacement and is kept only as a record of what not to keep.

Command:

```powershell
$env:LBM_STEPS='200'; $env:LBM_PLOT_FREQ='50'; $env:LBM_PLOT_START='0'; .\.venv\Scripts\python.exe .\LBM_3D_LES_air_over_waves_CODEX.py
```

Result:

- Taichi started on CUDA.
- Simulation completed 200 steps.
- Physical Reynolds number remained `3.958e+07`.
- `tau0` remained `0.5000002274`.
- The visible figure was shown through the normal Matplotlib plotting path.

Boundary-band diagnostics trended downward over the short smoke run:

```text
step ~50:  p_rms=8.718e+01 Pa, p_hp_rms=9.169e+01 Pa, vort_rms=1.151e-02, vort_hp_rms=3.738e-03
step ~100: p_rms=7.137e+01 Pa, p_hp_rms=7.127e+01 Pa, vort_rms=1.152e-02, vort_hp_rms=3.518e-03
step ~150: p_rms=5.887e+01 Pa, p_hp_rms=5.790e+01 Pa, vort_rms=1.084e-02, vort_hp_rms=2.979e-03
step ~200: p_rms=5.054e+01 Pa, p_hp_rms=4.897e+01 Pa, vort_rms=1.085e-02, vort_hp_rms=2.915e-03
```

## Open Review Items

- Do not reintroduce hard cell-center fluid/solid switching for the moving boundary.
- Any future arbitrary-surface method must preserve continuous link-fraction behavior and exact mass/momentum consistency in the existing pull-stream update.
- For non-single-valued boundaries, prefer a documented immersed/cut-cell LBM method that reconstructs boundary populations without adding dissipation or changing the physical Reynolds number.

## Historical SDF 10,000-Step Visible Run (Superseded)

This run used the failed SDF/cut-link replacement and is kept only as a record of the regression.

Command:

```powershell
$env:LBM_STEPS='10000'; $env:LBM_PLOT_FREQ='500'; $env:LBM_PLOT_START='0'; .\.venv\Scripts\python.exe .\LBM_3D_LES_air_over_waves_CODEX.py
```

Result:

- Taichi started on CUDA.
- Simulation completed 10,000 steps.
- Physical Reynolds number remained `3.958e+07`.
- `tau0` remained `0.5000002274`.
- The visible Matplotlib figure was shown while running.

Selected boundary-band diagnostics:

```text
step   500: p_rms=3.361e+01 Pa, p_hp=3.112e+01 Pa, vort_rms=1.001e-02, vort_hp=2.201e-03
step  2500: p_rms=7.973e+01 Pa, p_hp=5.669e+01 Pa, vort_rms=7.712e-03, vort_hp=2.048e-03
step  5000: p_rms=1.090e+02 Pa, p_hp=7.631e+01 Pa, vort_rms=7.443e-03, vort_hp=3.284e-03
step  6500: p_rms=1.506e+02 Pa, p_hp=7.550e+01 Pa, vort_rms=6.713e-03, vort_hp=2.719e-03
step  7500: p_rms=1.256e+02 Pa, p_hp=5.235e+01 Pa, vort_rms=6.333e-03, vort_hp=2.514e-03
step 10000: p_rms=7.261e+01 Pa, p_hp=4.697e+01 Pa, vort_rms=5.505e-03, vort_hp=2.161e-03
```

Velocity note:

```text
step 10000: max U = 0.0971 LB = 38.85 m/s
```

The max lattice speed briefly exceeded `0.1` around the middle of the run, then returned below `0.1` by the final step. The pressure-band metric rose during the transient and then relaxed. The vorticity-band high-pass metric finished near its early-run value and below its mid-run maximum.

## Fixed CODEX 10,000-Step Visible Run

Command:

```powershell
.\.venv\Scripts\python.exe .\run_ab_case.py --module .\LBM_3D_LES_air_over_waves_CODEX.py --label codex_fixed --steps 10000 --plot-freq 500 --plot-start 0 --output-dir .\test_runs\ab_fixed_codex --random-seed 12345
```

Result:

- Taichi started on CUDA.
- Simulation completed 10,000 steps.
- Physical Reynolds number remained `3.958e+07`.
- `tau0` remained `0.5000002274`.
- The visible Matplotlib figure was shown while running.
- `refill=0` throughout because the restored boundary does not create solid-to-fluid refill events.

The fixed CODEX metrics are identical to the seeded original metrics at all saved samples:

```text
mean pressure RMS            = 65.9334 Pa
mean pressure high-pass RMS  = 26.3737 Pa
mean vorticity RMS           = 7.61599e-03
mean vorticity high-pass RMS = 1.99453e-03

step 10000:
pressure RMS                 = 46.1117 Pa
pressure high-pass RMS       = 29.0178 Pa
vorticity RMS                = 6.05869e-03
vorticity high-pass RMS      = 2.07349e-03
max speed                    = 36.3443 m/s
```

Conclusion: the CODEX implementation is repaired back to original boundary behavior plus diagnostics/runtime controls. It is not yet a successful arbitrary-surface method.

## Geometry-Only Variants (Rejected)

Two signed-distance geometry variants were tested while keeping the original population treatment:

- `LBM_BOUNDARY_GEOMETRY=phi`: use signed-distance-like `phi` for both open fractions and open/closed link flags.
- `LBM_BOUNDARY_GEOMETRY=phi_fraction`: use `phi` only for open fractions, but keep original vertical open/closed link flags.

Both completed visible 10,000-step runs at the original physical parameters, but both increased high-pass noise relative to the original vertical geometry.

Mean over 20 plotted samples:

```text
metric                  original      phi          phi_fraction
pressure high-pass RMS  26.3737 Pa    27.0752 Pa   27.0650 Pa
vorticity high-pass RMS 1.99453e-03   2.01178e-03  2.02212e-03
```

Conclusion: signed-distance geometry is retained as an optional diagnostic path only. It is not the default accepted boundary geometry.

## Link-Wise Wall Velocity Sampling (Accepted Candidate)

The accepted current CODEX change keeps the original vertical link-fraction geometry and original population treatment, but samples the prescribed moving-wall velocity at each lattice link endpoint instead of using the same cell-centered `(j,i)` wall velocity for every direction.

Runtime settings:

```text
LBM_BOUNDARY_GEOMETRY=vertical
LBM_WALL_VELOCITY_SAMPLING=link
```

This is a kinematic consistency improvement, not a damping change. It changes only the spatial interpolation point for `u_wall` in:

- closed-link equilibrium conditioning in `update_bed_populations_and_reinit()`,
- post-stream moving-wall reflection correction in `apply_open_boundary_conditions()`.

10,000-step visible seeded result:

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

Conclusion: keep link-wise wall velocity sampling as the active CODEX candidate. It modestly improves pressure high-pass noise and does not materially change vorticity high-pass noise.

## Arbitrary Geometry Extension: Stationary Cube

The CODEX solver now supports an additional stationary cube obstacle through `LBM_OBSTACLE_MODE`:

```text
LBM_OBSTACLE_MODE=bed       # original moving bed only
LBM_OBSTACLE_MODE=bed_cube  # moving bed plus stationary cube
LBM_OBSTACLE_MODE=cube      # stationary cube only
```

The cube is represented by an analytic signed-distance function:

```text
LBM_CUBE_SIDE_M      default 6.0
LBM_CUBE_CENTER_X_M  default 0.5 * Lx
LBM_CUBE_CENTER_Y_M  default 0.5 * Ly
LBM_CUBE_CENTER_Z_M  default chosen above the moving bed
LBM_CUBE_YAW_DEG     default 0.0, yaw about vertical z axis
LBM_CUBE_PITCH_DEG   default 0.0, pitch about spanwise y axis
```

Implementation notes:

- The original bed open-fraction treatment remains active for the moving bed.
- Each lattice link now stores `lattice_wall_type`: none, moving bed, or stationary cube.
- For each link, the nearest solid source is selected from the bed gap and cube SDF.
- Moving-bed links use the prescribed moving-wall velocity, sampled at the link endpoint.
- Cube links use zero wall velocity.
- No solid-to-fluid refill is introduced.
- No viscosity, relaxation time, forcing, or flow speed changes were made.

The user revised the cube test from 10 m to 6 m side length. A 10 m cube in the original `Ly=10 m` domain would fill the periodic span, so the accepted test uses a 6 m cube in the original domain.

Visible 10,000-step test:

```text
LBM_OBSTACLE_MODE=bed_cube
LBM_CUBE_SIDE_M=6.0
LBM_BOUNDARY_GEOMETRY=vertical
LBM_WALL_VELOCITY_SAMPLING=link
steps=10000
plot_freq=500
random_seed=12345
```

Result:

```text
Grid                 600 x 20 x 40
Cube center          (150.0, 5.0, 11.0) m
Cube side            6.0 m
Re                   3.958e7
tau0                 0.5000002274
refill               0 throughout

mean pressure RMS            137.083 Pa
mean pressure high-pass RMS  30.8414 Pa
mean vorticity RMS           8.30208e-03
mean vorticity high-pass RMS 2.32031e-03
mean max speed               40.7789 m/s

step 10000:
pressure RMS                 85.3867 Pa
pressure high-pass RMS       28.9914 Pa
vorticity RMS                6.02904e-03
vorticity high-pass RMS      2.00757e-03
max speed                    38.7738 m/s
```

The run completed without NaNs/crashes and with no refill events. Early in the run, local acceleration around the cube briefly pushed `Umax` above the usual low-Mach comfort range, but it relaxed below `0.1` LB by the end without changing the physical setup.

## Pitched Cube 20,000-Step Run

The first longer "rotated cube" run used yaw about the vertical axis. That rotation is real in 3D, but it does not make the cube visibly rotate in the x-z diagnostic slice used for these figures. The corrected visible rotation test pitches the cube by 45 degrees about the spanwise y axis:

```text
LBM_OBSTACLE_MODE=bed_cube
LBM_CUBE_SIDE_M=6.0
LBM_CUBE_YAW_DEG=0.0
LBM_CUBE_PITCH_DEG=45.0
LBM_BOUNDARY_GEOMETRY=vertical
LBM_WALL_VELOCITY_SAMPLING=link
steps=20000
plot_freq=1000
random_seed=12345
```

Result:

```text
Grid                 600 x 20 x 40
Cube center          (150.0, 5.0, 12.243) m
Cube side            6.0 m
Cube yaw             0 degrees
Cube pitch           45 degrees
Re                   3.958e7
tau0                 0.5000002274
refill               0 throughout

mean pressure RMS            116.332 Pa
mean pressure high-pass RMS  31.1776 Pa
mean vorticity RMS           6.94155e-03
mean vorticity high-pass RMS 2.49501e-03
mean max speed               40.8463 m/s

tail mean, steps >= 10000:
pressure high-pass RMS       29.1048 Pa
vorticity high-pass RMS      2.19125e-03
max speed                    38.4438 m/s

step 20000:
pressure RMS                 69.2412 Pa
pressure high-pass RMS       27.6227 Pa
vorticity RMS                5.31422e-03
vorticity high-pass RMS      1.97006e-03
max speed                    35.6283 m/s
```

The run completed cleanly and the visible figure showed the cube as a 45 degree diamond in the x-z slice. The pitched cube produced a stronger initial acceleration (`Umax` peaked near `0.1566` LB early), then relaxed to about `0.0882` LB by step 20000 without changing viscosity, relaxation time, forcing, or flow speed.
