# Water Free-Surface Implementation Notes

Working file: `LBM_3D_LES_air_over_waves_CODEX.py`

Date: 2026-06-14

## Purpose

This change adds an opt-in water/free-surface path while preserving the existing air-over-moving-solid-boundary behavior by default.

The existing arbitrary solid boundary remains the solid-wall treatment. It is still responsible for:

- stationary or moving bed geometry,
- cube/obstacle geometry,
- no-penetration and no-slip solid-wall momentum exchange.

The water free surface is handled separately because it is not a solid wall. It should satisfy:

- kinematic condition: the interface moves with the water,
- dynamic normal-stress condition: pressure at the free surface is atmospheric, neglecting surface tension for now,
- approximately zero tangential shear because the gas phase is not solved.

## Runtime Modes

New environment controls:

```text
LBM_PHYSICS_MODE=air|water
LBM_FREE_SURFACE_MODE=none|height_static|height_kinematic|prescribed_solitary
LBM_FREE_SURFACE_INITIAL=flat|solitary|gaussian
LBM_X_BOUNDARY=periodic|solid
LBM_COLLISION=kbc|regularized
LBM_BED_PROFILE=moving_sine|flat
LBM_WATER_DEPTH_M=10.0
LBM_FREE_SURFACE_LEVEL_M=<bed level + depth>
LBM_SOLITARY_AMPLITUDE_M=0.5
LBM_SOLITARY_X0_M=0.25*Lx
LBM_GAUSSIAN_AMPLITUDE_M=3.0
LBM_GAUSSIAN_CENTER_X_M=0.5*Lx
LBM_GAUSSIAN_SIGMA_M=12.5
LBM_WATER_CURRENT_MPS=0.0
```

Default behavior is unchanged for the air model:

```text
LBM_PHYSICS_MODE=air
LBM_FREE_SURFACE_MODE=none
LBM_BED_PROFILE=moving_sine
```

Water mode defaults to:

```text
LBM_PHYSICS_MODE=water
LBM_FREE_SURFACE_MODE=height_kinematic
LBM_BED_PROFILE=flat
rho = 998.2 kg/m^3
nu = 1.004e-6 m^2/s
g = 9.81 m/s^2
collision = regularized
```

For the Gaussian release case, `LBM_X_BOUNDARY` defaults to `solid`; other modes keep the existing periodic `x` default unless overridden.

## Implemented Physics

### Water Properties

The physical Reynolds number is still computed from the physical configuration:

```text
Re = U_ref * Lz / nu_water
```

For the default water solitary setup:

```text
U_ref = sqrt(g * (h + a)) = 10.149 m/s
Lz    = 20 m
nu    = 1.004e-6 m^2/s
Re    = 2.022e8
tau0  = 0.5000000445
```

No artificial viscosity, relaxation-time increase, local damping, or flow-field filtering was added.

### Collision Model

Water mode now defaults to a regularized collision operator:

```text
LBM_COLLISION=regularized
```

The physical molecular viscosity is still set by the same `tau0` computed from the physical water configuration. The regularized operator projects the non-equilibrium populations onto the second-order hydrodynamic stress tensor before relaxation, removing non-hydrodynamic lattice ghost modes generated near the free surface. This is a collision-model change, not a viscosity increase, damping layer, or flow-field filter.

The old collision path remains available for comparison:

```text
LBM_COLLISION=kbc
```

### Gravity

Gravity is included as a body force using a Guo-style forcing term in collision, with the corresponding half-step force correction in the macroscopic velocity:

```text
g_lb = g * dt^2 / dx
```

Hydrostatic density is initialized consistently with the weakly compressible LBM pressure relation:

```text
p_lb = c_s^2 rho_lb
delta_rho = g_lb * depth_lattice / c_s^2
```

### Free Surface

The first implementation is a single-valued height-function free surface. This is suitable for still water, Airy waves, and non-overturning solitary-wave development tests, but it is not yet the final overturning-capable VOF/PLIC free-surface method.

Fields added:

```text
free_surface_h[j,i]
free_surface_h_last[j,i]
free_surface_fill[k,j,i]
free_surface_type[k,j,i] = gas/liquid/interface
free_surface_type_prev[k,j,i]
free_surface_flux_x[j,i]
free_surface_flux_y[j,i]
```

Gas cells are inactive. Interface/gas-side missing populations are reconstructed from atmospheric pressure and local interface velocity rather than solid-wall bounce-back:

```text
f_q = f_eq_q(rho_atm, u_interface)
    + f_eq_qbar(rho_atm, u_interface)
    - f_qbar
```

This is intentionally separate from the solid moving-wall treatment.

Cells that transition from gas to liquid/interface are refilled from physically adjacent wet cells. The refill uses hydrostatic density at the new cell elevation and velocity extrapolated only from cells that were already wet before the classification change. The distributions are initialized to equilibrium for that state. This prevents newly wetted cells from retaining gas-equilibrium populations without adding artificial dissipation to existing water cells.

### Kinematic Surface Update

The first attempted pointwise update used surface velocity directly:

```text
h_t = w_s - u_s h_x - v_s h_y
```

It was replaced by a conservative column-flux form:

```text
h_t = -d/dx int u dz - d/dy int v dz
```

This is not smoothing. It is a conservative height-function form of the kinematic condition.

For `LBM_X_BOUNDARY=solid`, the height-function update enforces zero depth-integrated normal flux at `x=0` and `x=Lx` by using wall-face fluxes of zero in the finite-volume divergence. The LBM populations at those end walls use stationary solid-wall bounce-back for incoming directions whose upstream cell would lie outside the domain.

The end walls are also represented in the directional solid-link mask (`lattice_open`, `lattice_open_frac`, `lattice_wall_type`) as `WALL_XMIN` and `WALL_XMAX`, so they pass through the same post-stream solid reflection path as other walls. For water, stationary wall links are not pre-reinitialized to `rho0` equilibrium, because doing so violates the hydrostatic pressure field along a vertical wall.

## Tests Run

### Flat Water 10,000-Step Test

Command:

```powershell
$env:LBM_PHYSICS_MODE='water'
$env:LBM_FREE_SURFACE_MODE='height_static'
$env:LBM_FREE_SURFACE_INITIAL='flat'
$env:LBM_OBSTACLE_MODE='bed'
$env:LBM_BED_PROFILE='flat'
.\.venv\Scripts\python.exe .\run_ab_case.py --module .\LBM_3D_LES_air_over_waves_CODEX.py --label water_flat_10k --steps 10000 --plot-freq 2000 --plot-start 0 --output-dir .\test_runs\water_flat_10k --random-seed 12345
```

Result:

```text
completed 10000 steps
Re = 2.022e8
tau0 = 0.5000000445
surface range = [10.000, 10.000] m at every plotted sample
reported water volume = 30000.000 m^3 at every plotted sample
max speed at step 10000 = 0.01 m/s
refill = 0
```

The visible plot showed a flat cyan free surface, inactive gas above, hydrostatic pressure below, and essentially zero vorticity.

### Solitary Initial-Condition Smoke Test

Command:

```powershell
$env:LBM_PHYSICS_MODE='water'
$env:LBM_FREE_SURFACE_MODE='height_kinematic'
$env:LBM_FREE_SURFACE_INITIAL='solitary'
$env:LBM_SOLITARY_AMPLITUDE_M='0.5'
$env:LBM_OBSTACLE_MODE='bed'
$env:LBM_BED_PROFILE='flat'
.\.venv\Scripts\python.exe .\run_ab_case.py --module .\LBM_3D_LES_air_over_waves_CODEX.py --label water_solitary_2k --steps 2000 --plot-freq 500 --plot-start 0 --output-dir .\test_runs\water_solitary_2k --random-seed 12345
```

Result:

```text
completed 2000 steps
Re = 2.022e8
tau0 = 0.5000000445
refill = 0
reported water volume = 30513.309 m^3 at all plotted samples
surface range at step 500  = [10.003, 10.515] m
surface range at step 2000 = [9.603, 10.593] m
max speed at step 2000 = 1.62 m/s
```

Interpretation:

The run is stable and volume-conserving to the printed precision. It is not yet a validated solitary-wave model. The initially prescribed solitary shape adjusts and develops a depression/secondary wave, which indicates that the initial pressure/velocity/free-surface state is not yet a fully consistent solitary-wave solution for this LBM/free-surface discretization.

### Gaussian Hump With Solid End Walls

Command:

```powershell
$env:LBM_PHYSICS_MODE='water'
$env:LBM_FREE_SURFACE_MODE='height_kinematic'
$env:LBM_FREE_SURFACE_INITIAL='gaussian'
$env:LBM_GAUSSIAN_AMPLITUDE_M='3.0'
$env:LBM_GAUSSIAN_CENTER_X_M='150.0'
$env:LBM_GAUSSIAN_SIGMA_M='12.5'
$env:LBM_WATER_CURRENT_MPS='0.0'
$env:LBM_X_BOUNDARY='solid'
$env:LBM_OBSTACLE_MODE='bed'
$env:LBM_BED_PROFILE='flat'
.\.venv\Scripts\python.exe .\run_ab_case.py --module .\LBM_3D_LES_air_over_waves_CODEX.py --label water_gaussian_solid_walls_sigma12p5_3k --steps 3000 --plot-freq 500 --plot-start 0 --output-dir .\test_runs\water_gaussian_solid_walls_sigma12p5_3k --random-seed 12345
```

Initial result with the old BGK/KBC-style collision:

```text
completed 3000 steps
Re = 2.250e8
tau0 = 0.5000000400
Gaussian hump = 3.0 m amplitude, x0 = 150 m, sigma = 12.5 m, zero initial velocity
x boundary = solid
reported water volume = 30939.986 m^3 at all plotted samples
surface range at step 1000 = [10.000, 11.513] m
surface range at step 2000 = [9.548, 11.615] m
surface range at step 3000 = [6.596, 11.385] m
max speed at step 2000 = 2.45 m/s
max speed at step 3000 = 18.41 m/s
```

Interpretation:

The hump began splitting into left- and right-traveling disturbances and remained roughly symmetric through about 2000 steps. The volume was conserved to the printed precision. By 3000 steps, however, the center-region free surface developed an unphysical deep trough with grid-scale vorticity/pressure structure and a large speed spike. This was not considered a valid long-run result.

Corrected 10,000-step command:

```powershell
$env:LBM_PHYSICS_MODE='water'
$env:LBM_FREE_SURFACE_MODE='height_kinematic'
$env:LBM_FREE_SURFACE_INITIAL='gaussian'
$env:LBM_GAUSSIAN_AMPLITUDE_M='3.0'
$env:LBM_GAUSSIAN_CENTER_X_M='150.0'
$env:LBM_GAUSSIAN_SIGMA_M='12.5'
$env:LBM_WATER_CURRENT_MPS='0.0'
$env:LBM_X_BOUNDARY='solid'
$env:LBM_COLLISION='regularized'
$env:LBM_OBSTACLE_MODE='bed'
$env:LBM_BED_PROFILE='flat'
.\.venv\Scripts\python.exe .\run_ab_case.py --module .\LBM_3D_LES_air_over_waves_CODEX.py --label water_gaussian_regularized_sigma12p5_10k --steps 10000 --plot-freq 1000 --plot-start 0 --output-dir .\test_runs\water_gaussian_regularized_sigma12p5_10k --random-seed 12345
```

Corrected result:

```text
completed 10000 steps
Re = 2.250e8
tau0 = 0.5000000400
collision = regularized
reported water volume = 30939.984-30939.988 m^3 across plotted samples
max speed at step 3000 = 2.25 m/s
max speed at step 7000 = 1.20 m/s after wall interaction
max speed at step 10000 = 1.28 m/s
surface range at step 3000 = [9.658, 11.655] m
surface range at step 7000 = [9.912, 13.114] m near reflected wall waves
surface range at step 10000 = [9.509, 11.493] m
vort_hp_rms at step 3000 = 1.24e-4
vort_hp_rms at step 10000 = 1.06e-4
```

Interpretation:

The regularized collision removes the prior center-collapse failure without changing the physical viscosity or Reynolds number. The Gaussian release splits into left- and right-traveling disturbances, reflects from the solid end walls, and remains bounded through 10,000 steps. Volume is conserved to the displayed precision. Some height-function stair-stepping remains at the free surface, so this is a stable development test rather than a final validation of wave phase speed or amplitude preservation.

### 8 m Gaussian Hump Stress Test

Command:

```powershell
$env:LBM_PHYSICS_MODE='water'
$env:LBM_FREE_SURFACE_MODE='height_kinematic'
$env:LBM_FREE_SURFACE_INITIAL='gaussian'
$env:LBM_GAUSSIAN_AMPLITUDE_M='8.0'
$env:LBM_GAUSSIAN_CENTER_X_M='150.0'
$env:LBM_GAUSSIAN_SIGMA_M='12.5'
$env:LBM_WATER_CURRENT_MPS='0.0'
$env:LBM_X_BOUNDARY='solid'
$env:LBM_COLLISION='regularized'
$env:LBM_OBSTACLE_MODE='bed'
$env:LBM_BED_PROFILE='flat'
.\.venv\Scripts\python.exe .\run_ab_case.py --module .\LBM_3D_LES_air_over_waves_CODEX.py --label water_gaussian_amp8_solidlinks_fix_10k --steps 10000 --plot-freq 1000 --plot-start 0 --output-dir .\test_runs\water_gaussian_amp8_solidlinks_fix_10k --random-seed 12345
```

Result:

```text
completed 10000 steps
Re = 2.647e8
tau0 = 0.5000000340
collision = regularized
reported water volume = 32506.627-32506.631 m^3 through step 6000
surface range at step 6000 = [9.485, 14.645] m
surface range at step 7000 = [9.516, 19.750] m
surface range at step 10000 = [9.185, 13.684] m
max speed at step 7000 = 4.30 m/s
max speed at step 10000 = 3.90 m/s
```

Interpretation:

The corrected solid-link end-wall treatment remains bounded through 10,000 steps. At step 7000, the reflected waves at the solid end walls still drive the free surface to `19.750 m`, which is the current height cap for a 20 m domain (`Lz - 0.5 dx`). After that contact with the computational ceiling, the run is no longer a clean physical validation: water volume drops slightly and near-wall vorticity increases. However, the previous late-time velocity blow-up is removed. The correct next diagnostic is still to rerun the 8 m release in a taller domain, not to add damping or reduce the physical amplitude.

### Free-Surface Noise Investigation

A new diagnostic was added to `run_ab_case.py` for pressure and vorticity high-pass RMS in a narrow band around the free surface, plus a one-dimensional high-pass metric for the plotted surface elevation. This is separate from the existing 10-cell solid-boundary band metric, which can miss grid-scale noise generated at the free surface.

The relevant physical free-surface conditions remain:

```text
p = p_atm at the liquid-gas interface, neglecting surface tension
the interface moves with the liquid normal velocity
tangential gas shear is neglected in this single-fluid model
```

Literature checked during this pass:

- Free-surface LBM applies missing gas-to-interface populations using the gas pressure/density and interface-cell velocity; see the discussion summarized in Schwarzmeier/Ruede, "Analysis and comparison of boundary condition variants in the free-surface lattice Boltzmann method" and the earlier Koerner/Thuerey/Ruede model.
- Boundary-condition accuracy depends strongly on how the free boundary position/orientation is represented. Research summaries note that the widely used Koerner-style free-surface boundary is first-order and that higher-order variants account for boundary position and orientation.
- Refilling newly wetted cells is a known source of artifacts; published comparisons test averaging, extrapolation, and equilibrium-based refill schemes across standing waves, dam breaks, and related free-surface benchmarks.

Tests on the 3 m Gaussian release with solid end walls:

```text
Baseline cell pressure BC, equilibrium refill, 2000 steps:
  fs_p_hp_rms   at step 2000 = 4.92e+02 Pa
  fs_vort_hp    at step 2000 = 5.49e-04
  eta_hp_rms    at step 2000 = 3.35e-02 m

Hydrostatic node-density target in the pressure reconstruction, 2000 steps:
  fs_p_hp_rms   at step 2000 = 8.87e+02 Pa
  fs_vort_hp    at step 2000 = 1.71e-03
  eta_hp_rms    at step 2000 = 1.07e-01 m

Non-equilibrium-preserving refill, original pressure BC, 2000 steps:
  fs_p_hp_rms   at step 2000 = 4.13e+02 Pa
  fs_vort_hp    at step 2000 = 5.47e-04
  eta_hp_rms    at step 2000 = 3.80e-02 m
```

Interpretation:

- The standard gas-pressure free-surface boundary remains the best of the tested local pressure reconstructions. A hydrostatic density target inside the missing-population reconstruction made the free-surface vorticity noise materially worse, likely because it double-counts pressure recovery already produced by the body force and density field.
- A first attempt at interpolated/fractional free-surface pressure reconstruction was not retained. A direction-dependent link hydrostatic target and velocity extrapolation increased the surface-band vorticity. A later GPU-safe local fraction probe ran, but the actual interpolated-population blend triggered a repeatable CUDA/Taichi kernel failure before timestep diagnostics. This should be revisited only as a deliberate PLIC/cut-link implementation, not as an ad hoc branch in the current height-function boundary.
- Non-equilibrium-preserving refill is physically cleaner than pure equilibrium refill because it does not erase nearby viscous/non-equilibrium stress when a gas cell becomes wet. It did not materially reduce the visible stair-step vorticity in the 2000-step Gaussian test, so it is available as `LBM_FREE_SURFACE_REFILL=noneq` but is not treated as a solved fix.
- The visual noise is tightly tied to the stair-stepped height-function interface and cell conversion, not to bulk pressure or the solid boundary.

Recommended physical next step:

Replace the current height-function-only interface treatment with a proper single-fluid VOF/free-surface LBM interface layer:

```text
mass m = epsilon * rho per cell
population-based mass exchange across links
interface normals from volume fraction/height reconstruction
PLIC or equivalent sub-cell interface geometry
gas-pressure missing-population reconstruction applied only on reconstructed interface links
excess/deficit mass redistribution on interface/liquid/gas state changes
```

This is the same conceptual lesson as the solid moving boundary: the surface should be represented geometrically at sub-cell resolution before applying the physical boundary condition. Unlike local damping or filtering, this changes the numerical realization of the free-surface physics rather than suppressing the resulting artifacts.

### Experimental VOF/FSLBM Implementation

An opt-in VOF/free-surface LBM path was added:

```powershell
$env:LBM_FREE_SURFACE_TRACKING='vof'
```

The default remains:

```powershell
$env:LBM_FREE_SURFACE_TRACKING='height'
```

because the VOF path is not yet validated for long runs.

Implemented pieces:

```text
free_surface_mass[k,j,i]        tracked liquid mass in lattice units
free_surface_mass_delta[k,j,i]  population-based interface mass exchange
free_surface_mass_excess[k,j,i] boundedness/conversion excess or deficit
```

The VOF update follows the single-fluid FSLBM structure:

1. Initialize liquid mass from the initial fill fraction and LBM density.
2. Exchange mass only for interface cells.
3. Use streamed populations for incoming/outgoing mass:
   `dm = f_incoming_from_neighbor - f_outgoing_to_neighbor`.
4. Use full link weight for liquid-interface exchange.
5. Use mean fill fraction for interface-interface exchange.
6. Convert cells from interface to gas/liquid based on `epsilon = m/rho`.
7. Redistribute overfill/underfill excess locally using the interface normal from the fill-gradient.
8. Enforce a one-cell interface layer by marking liquid cells adjacent to gas as full interface cells. This changes only the boundary interpretation, not mass or fill.
9. Reconstruct the plotted `free_surface_h` from column-integrated fill for diagnostics.

This is a real VOF/FSLBM path, but still lacks the more complete production pieces: robust mass redistribution over multiple passes, PLIC/cut-link reconstruction for the pressure boundary, and validation-specific state-change handling.

#### VOF Test Results

Command pattern:

```powershell
$env:LBM_PHYSICS_MODE='water'
$env:LBM_FREE_SURFACE_MODE='height_kinematic'
$env:LBM_FREE_SURFACE_TRACKING='vof'
$env:LBM_FREE_SURFACE_INITIAL='gaussian'
$env:LBM_GAUSSIAN_AMPLITUDE_M='3.0'
$env:LBM_GAUSSIAN_CENTER_X_M='150.0'
$env:LBM_GAUSSIAN_SIGMA_M='12.5'
$env:LBM_WATER_CURRENT_MPS='0.0'
$env:LBM_X_BOUNDARY='solid'
$env:LBM_COLLISION='regularized'
$env:LBM_OBSTACLE_MODE='bed'
$env:LBM_BED_PROFILE='flat'
.\.venv\Scripts\python.exe .\run_ab_case.py --module .\LBM_3D_LES_air_over_waves_CODEX.py --label water_gaussian_vof_normal_10k --steps 10000 --plot-freq 1000 --plot-start 0 --output-dir .\test_runs\water_gaussian_vof_normal_10k --random-seed 12345
```

Short-run result:

```text
VOF normal-weighted conversion, step 2000:
  max speed     = 1.52 m/s
  volume        = 31025.020 m^3
  vof_mass      = 249252.547 lattice mass units
  vof_excess    = 120 cells
  fs_p_hp_rms   = 3.98e+02 Pa
  fs_vort_hp    = 5.53e-04
  eta_hp_rms    = 5.14e-02 m
```

The velocity field remained bounded through 2000 steps. The surface looked somewhat smoother in places, but the free-surface vorticity high-pass was not clearly better than the previous height-function result.

Long-run result:

```text
VOF normal-weighted conversion, step 8000:
  max speed     = 3.11 m/s
  fs_vort_hp    = 5.25e-04

VOF normal-weighted conversion, step 9000:
  max speed     = 19.36 m/s
  fs_vort_hp    = 5.86e-04

VOF normal-weighted conversion, step 10000:
  max speed     = 35.65 m/s
  volume        = 31038.115 m^3
  vof_mass      = 249199.641 lattice mass units
  vof_excess    = 440 cells
  fs_p_hp_rms   = 4.41e+02 Pa
  fs_vort_hp    = 6.06e-04
  eta_hp_rms    = 1.51e-01 m
```

Interpretation:

- The VOF path is stable over short development tests but fails the required long-run standard for this project.
- The late-time velocity growth begins after reflected waves interact with the VOF conversion layer.
- The failure is not a reason to add damping. It points to incomplete FSLBM state-change handling and pressure-boundary geometry.
- The current VOF path should be considered experimental and opt-in only.

Recommended next VOF work:

1. Implement a PLIC/cut-link free-surface pressure boundary so gas-pressure reconstruction uses interface position and normal rather than only cell type.
2. Quantify mass conservation by separating liquid, interface, gas, pending excess, and orphan-redistributed mass over long runs.
3. Validate the bounded VOF path against Airy-wave dispersion and a small-amplitude standing/reflecting wave before using it with obstacles.
4. Re-run the 3 m Gaussian case after the PLIC pressure boundary and compare free-surface vorticity high-pass metrics against the current VOF and height-function baselines.

#### VOF State-Change Update

The VOF path was updated to handle the late-time tiny-interface-cell failure without changing the physical viscosity, relaxation time, gravity, velocity scale, or collision physics.

Implementation changes:

```text
free_surface_mass_excess_dir[k,j,i]  +1 sends mass toward gas, -1 toward liquid
capacity-aware excess redistribution  avoids placing added mass where no VOF capacity exists unless using explicit topology fallback
orphan interface cleanup              merges unresolved detached interface components back into nearby resolved wet cells
```

The important physical distinction is that a near-empty interface cell becoming gas has positive residual water mass that must move toward the liquid side. The previous sign-only excess logic could not distinguish this from overfilled-cell excess that should move toward the gas side.

The orphan cleanup is a topology/geometry rule, not flow damping:

```text
LBM_VOF_EMPTY_FILL_THRESHOLD=0.0
LBM_VOF_ORPHAN_EMPTY_THRESHOLD=0.49
LBM_VOF_ORPHAN_SEARCH_RADIUS=4
LBM_VOF_ORPHAN_MAX_WEAK_NEIGHBORS=1
```

It only acts on detached or weakly connected sub-cell VOF components that have no resolved neighboring wet cell. Removed mass is conservatively redistributed to nearby wet cells along D3Q19 lattice-link directions. This follows the same spirit as published FSLBM lonely-interface-cell cleanup, but it should not be interpreted as a validated spray/droplet model.

The earlier nonzero `LBM_VOF_EMPTY_FILL_THRESHOLD=0.001` was found to be too
aggressive for the Gaussian wave test. It emptied thin but still resolved
interface volumes and generated repeated cell-state changes. The default is now
zero so a positive fill is preserved unless the conservative VOF update produces
a true boundedness/excess conversion.

The 3 m Gaussian release with solid end walls was rerun for 10,000 visible steps:

```text
label = water_gaussian_vof_orphan_weak1_10k
step 10000:
  max speed             = 14.16 m/s
  surface range         = [7.698, 11.719] m
  water volume          = 31010.959 m^3
  VOF mass              = 249006.266 lattice mass units
  pending VOF excess    = 1.905e-03 over 7 cells
  orphan cleanups       = 1 on final plotted step
  Umax location/type    = (276.75, 2.75, 10.25) m, interface
  Umax fill             = 0.405
  Umax wet/full/gas nbr = 4 / 2 / 14
  fs_p_hp_rms           = 2.299e+03 Pa
  fs_vort_hp_rms        = 2.257e-03
```

Interpretation:

- The VOF run is now bounded through 10,000 steps for this Gaussian release. The previous detached-cell runaway to O(100 m/s) is removed.
- The solution is still not a validated free-surface solver. Free-surface high-pass vorticity is higher than desired, and late-time reflected-wave geometry creates intermittent orphan cleanups.
- The next physics/numerics step is still a PLIC/cut-link free-surface pressure boundary. The current gas-pressure reconstruction is cell-state based, so it does not yet apply the dynamic free-surface condition at a reconstructed sub-cell interface location.

## Current Limitations

- The default free-surface tracking remains the height-function method. The opt-in VOF/FSLBM path is more general than the height function, but it is not yet a full VOF/PLIC interface and should not be used as a validated overturning-wave or spray/droplet solver.
- The solitary-wave initialization uses a shallow-water depth-uniform horizontal velocity approximation. It is not a high-order fully consistent solitary-wave velocity/pressure field.
- The 3 m, sigma = 12.5 m Gaussian zero-velocity release is stable through 10,000 steps with regularized collision, but wave celerity, reflection quality, and amplitude error still need quantitative validation.
- The 8 m, sigma = 12.5 m Gaussian release reaches the top height cap in the current 20 m domain after solid-wall reflection. A taller vertical domain is required before judging long-time validity of this larger-amplitude case.
- Free-surface validation has not yet been done against Airy-wave dispersion, solitary-wave celerity, phase error, or amplitude preservation.
- The water Reynolds number is physically very high, so `tau0` is extremely close to 0.5. This is physically faithful but numerically delicate.
- Water free-surface validation currently runs with LES and stochastic backscatter off by default. High-Re obstacle flows will need a separate, physically justified SGS treatment review.

## Recommended Next Step

Validate the free-surface solver before adding obstacles:

1. Flat still-water run for at least 10,000 steps with volume, max velocity, and surface range tracked.
2. Small-amplitude Airy wave with measured phase speed and amplitude decay.
3. Improve solitary-wave initial velocity/pressure consistency.
4. Solitary wave propagation over flat bed.
5. Add submerged cube only after the free-surface-only validation is credible.

## FSL Link-Interpolated Free-Surface Closure

Experimental free-surface pressure closures were tested:

```text
LBM_FREE_SURFACE_BOUNDARY=fsk_link
LBM_FREE_SURFACE_BOUNDARY=fsl
```

These modes are now disabled in the active configuration path. `fsk_link` kept the original Korner/FSLBM missing-PDF pressure closure but
extrapolates the interface velocity to the reconstructed VOF cut-link location.
`fsl` implements a Bogner-Ammer-Rude-style linear link closure for missing
gas-to-liquid PDFs. In both cases the intended physical target was unchanged:

```text
gas pressure at the interface = atmospheric pressure
gas phase is not solved
only PDFs missing because their source is gas are reconstructed
available liquid-side PDFs are not overwritten
interface velocity is extrapolated from the liquid side to the same cut-link location
```

The sub-cell link distance is estimated from the local VOF indicator crossing
`phi=0.5` between the active wet/interface cell and the gas-side source cell. This
is a grid-scale geometry interpretation, not smoothing of the flow field. The
implementation keeps the previous `cell` and `hydrostatic` modes available for
A/B comparison.

Research basis:

- Bogner, Ammer, and Rude (2015) show that the original Korner free-surface
  anti-bounce-back closure is first-order except for a mid-link interface, and
  derive a second-order link-interpolated closure using the free-surface position.
- Schwarzmeier and Rude (2023) compare free-surface boundary-condition variants
  and find that reconstructing only missing PDFs is generally more accurate than
  normal-based variants that overwrite available information.

Current limitation:

- The implemented `fsl` mode uses the simplified zero prescribed shear-rate tensor
  form. It does not yet extrapolate and impose the full viscous normal-stress
  tensor at the interface. That is a real physical limitation for strongly sheared
  free-surface flow, not a tuning knob.
- Initial smoke tests showed the full `fsl` formula can inject substantially more
  free-surface vorticity in this regularized, forced, high-Re solver. Treat `fsl`
  as failed in this implementation, not merely experimental. The lower-risk
  candidate for long testing was `fsk_link`, because it preserved the original
  pressure closure and only changed the boundary velocity evaluation point, but
  it also failed the long-run standard.
- The 10,000-step `fsk_link` candidate did not pass validation. It was plausible
  through about 2000 steps, but developed a free-surface/interface velocity spike
  by 3000 steps and NaNs by about 3600 steps:

```text
LBM_FREE_SURFACE_BOUNDARY=fsk_link
step 1000: Umax=1.79 m/s, fs_p_hp=3.74e+02 Pa, fs_vort_hp=6.21e-04
step 2000: Umax=2.22 m/s, fs_p_hp=7.79e+02 Pa, fs_vort_hp=7.86e-04
step 3000: Umax=163.84 m/s, fs_p_hp=2.68e+03 Pa, fs_vort_hp=1.75e-03
step 3600: NaNs
```

Interpretation: in the current VOF/refill implementation, extrapolating the
interface velocity along cut links is not yet consistent with the state-change
and missing-PDF reconstruction logic. The stable reference remains the original
`cell` closure.

Corrected direction:

- Use the existing moving-solid boundary architecture as the foundation:
  per-link geometry, link fractions, near-boundary tagging, and boundary
  application order.
- Do not reuse the no-slip moving-wall reflection itself for a free surface.
  The free-surface physics remains prescribed gas pressure, kinematic interface
  motion, and negligible gas shear.
- Build the next free-surface treatment as an adaptation of the solid boundary
  framework rather than a standalone pressure-closure branch.

### Solid-Boundary-Foundation Free-Surface Attempt

Date: 2026-06-15

A second attempt was made to fold the working solid-boundary architecture into
the free-surface treatment. The physical target was unchanged:

```text
kinematic condition: the interface moves with the liquid mass/volume update
dynamic condition: atmospheric gas pressure at the free surface
gas phase: inactive; only missing gas-to-liquid PDFs are reconstructed
available liquid/interface PDFs are not overwritten
```

Implementation lessons:

- A new `LBM_FREE_SURFACE_BOUNDARY=link` option was added for this experiment.
- The reconstructed free-surface link fraction is now stored on the pull-streaming
  source side, `x - 0.5*c_q`, not the outgoing `x + 0.5*c_q` side. This matches
  the population whose source may be gas in the pull-stream update.
- Gating the pressure reconstruction with this geometric fraction was wrong. If
  the upstream node is gas, the PDF is physically unavailable in a single-fluid
  free-surface model and must be reconstructed regardless of the local height
  reconstruction.
- Moving the free-surface pressure reconstruction into a post-stream pass, after
  the solid boundary pass, did not improve the solution. It changed the ordering
  near wall/free-surface intersections and produced a slightly worse long run.
- The retained active pressure boundary is therefore still the standard FSLBM
  only-missing-PDF closure applied during streaming. The `link` mode currently
  keeps the corrected pull-side link geometry available for diagnostics/future
  PLIC work, but it does not overwrite the validated `cell` closure behavior.

Visible tests run:

```text
cell baseline smoke, 20 steps:
  fs_p_hp=3.781e+02 Pa, fs_vort_hp=7.220e-05

bad geometric-gated post link, 20 steps:
  fs_p_hp=2.295e+03 Pa, fs_vort_hp=1.233e-04

post-staged missing-PDF link, 10,000 steps:
  Umax=18.89 m/s, surface=[8.083,11.661] m
  fs_p_hp=1.790e+03 Pa, fs_vort_hp=1.970e-03

current cell closure, 10,000 steps:
  Umax=9.89 m/s, surface=[7.872,11.674] m
  fs_p_hp=1.608e+03 Pa, fs_vort_hp=1.859e-03

cleaned link mode smoke, 20 steps:
  fs_p_hp=3.781e+02 Pa, fs_vort_hp=7.220e-05
```

Conclusion:

The known FSLBM result from Schwarzmeier/Rude is consistent with these tests:
reconstructing only missing PDFs is safer than normal/cut-link variants that
discard available liquid information. The solid-boundary framework is still the
right foundation for the future arbitrary free-surface geometry, but the next
valid step is not a post-stream pressure overwrite. It is a true PLIC/cut-link
interface reconstruction that supplies accurate source-side link distances and
normals while retaining the only-missing-PDF free-surface closure unless a
validated higher-order pressure closure is implemented.

### Stable VOF Gaussian Result After State-Change Correction

Date: 2026-06-15

Two state-change/refill issues were tested after the failed post-stream link
closure:

1. Newly wetted cells now extrapolate velocity and optional non-equilibrium
   content from neighboring liquid volume using fill-weighted interpolation. A
   nearly empty interface sliver no longer contributes with the same weight as a
   full liquid neighbor during refill.
2. The default `LBM_VOF_EMPTY_FILL_THRESHOLD` was changed from `0.001` to `0.0`.
   The previous threshold was deleting small positive interface volumes, which
   produced repeated topology changes and late-time velocity spikes.

The second change was the dominant correction. With `LBM_VOF_EMPTY_FILL_THRESHOLD=0`,
the 3 m Gaussian release with solid end walls completed 10,000 visible steps
without the previous late-time interface-cell blow-up:

```text
label = water_gaussian_vof_no_empty_threshold_10k
step 10000:
  max speed             = 2.09 m/s
  surface range         = [9.468, 11.583] m
  water volume          = 31035.961 m^3
  VOF mass              = 249185.828 lattice mass units
  pending VOF excess    = 0.0
  orphan cleanups       = 0
  fs_p_hp_rms           = 6.015e+02 Pa
  fs_vort_hp_rms        = 4.873e-04
```

Comparison to the previous default VOF 10,000-step run:

```text
previous default:
  max speed      = 9.89 m/s
  fs_p_hp_rms    = 1.608e+03 Pa
  fs_vort_hp_rms = 1.859e-03

corrected state-change default:
  max speed      = 2.09 m/s
  fs_p_hp_rms    = 6.015e+02 Pa
  fs_vort_hp_rms = 4.873e-04
```

Interpretation:

This is a material improvement and is consistent with the diagnosis that the
dominant noise source was not insufficient dissipation, but incorrect
state-change handling at small positive fill. The result is still not a final
validated free-surface solver: volume/mass drift and wave phase/amplitude error
still need formal Airy/standing-wave validation, and a true PLIC/cut-link
interface geometry is still required before overturning waves or complex
obstacle/free-surface interaction should be trusted.

### Reduced Domain / Finer Grid Run

Date: 2026-06-15

Requested configuration:

```text
Lx = 150 m     half the original x-domain
Ly = 2 m       one fifth of the original y-domain
Lz = 20 m
dx = 0.25 m    half the original grid spacing
grid = 600 x 8 x 80
```

Run command used the corrected VOF default (`LBM_VOF_EMPTY_FILL_THRESHOLD=0.0`),
3 m Gaussian hump, solid x walls, and regularized collision:

```text
label = water_gaussian_vof_halfx_yfifth_dxhalf_10k
steps = 10000
dt    = 1.660333e-03 s
```

Result at step 10000:

```text
max speed             = 6.98 m/s
surface range         = [9.595, 11.562] m
water volume          = 3202.375 m^3
VOF mass              = 205509.891 lattice mass units
pending VOF excess    = 0.0
orphan cleanups       = 0
fs_p_hp_rms           = 3.268e+02 Pa
fs_vort_hp_rms        = 8.784e-04
```

Interpretation:

The finer grid/reduced-domain run remained bounded through 10,000 steps with no
orphan-interface cleanup at the final sample and low free-surface pressure
high-pass. The free-surface vorticity high-pass was higher than the previous
coarser 10,000-step run, and the final max speed rose near the right wall late in
the run. Because halving `dx` also halves `dt`, this 10,000-step run corresponds
to about 16.6 s of physical time, not the 33.2 s of the original 10,000-step
coarse-grid case.

### Revert of Lateral-Wall Experiment and Refactor

Date: 2026-06-15

The attempted generalization to explicit solid `y` walls and lateral
free-slip/no-slip modes was reverted after testing showed a clear stability
regression relative to the previous working state. The reverted experiment
produced large side-wall/bottom-edge velocities and, for the free-slip variant,
late free-surface collapse. Those changes are not part of the active solver.

Conservative refactor performed after the revert:

```text
lbm_env.py      environment parsing helpers
lbm_lattice.py  D3Q19 weights and velocity-component constants
```

The active solver still uses the previous boundary behavior: `LBM_X_BOUNDARY`
controls the x boundary, while y remains periodic. No viscosity, relaxation
time, forcing, free-surface closure, or boundary physics was intentionally
changed by this refactor.

Smoke test:

```text
label = refactor_revert_smoke_100
steps = 100
arch = CUDA
grid = 600 x 8 x 80
Gaussian amplitude = 3 m
result = completed with visible frames at steps 50 and 100
```

### Standard X Solid-Wall Boundary

Date: 2026-06-15

Implemented an explicit standard LBM solid-wall boundary for the left and right
domain faces when `LBM_X_BOUNDARY=solid`.

Physical conditions:

```text
u_normal = 0 at x-min/x-max walls
u_tangent = 0 at x-min/x-max walls
mass flux through wall = 0
wall velocity = 0
```

Numerical treatment:

```text
After pull streaming, incoming populations from outside the x domain are
reconstructed by local opposite-population bounce-back:

left wall  (i=0):    f_new[q] = f[opp[q]] for cx[q] > 0
right wall (i=Nx-1): f_new[q] = f[opp[q]] for cx[q] < 0
```

The implementation is in `apply_x_solid_wall_bounceback()`. It intentionally
does not add damping, filtering, viscosity changes, or any moving-wall momentum
correction.

Smoke test:

```text
label = xwall_standard_bounceback_3m_5k
steps = 5000
arch = CUDA
grid = 600 x 8 x 80
Gaussian amplitude = 3 m
sigma = 12.5 m
step 5000:
  max speed      = 1.69 m/s
  surface range  = [9.731, 11.708] m
  water volume   = 3195.523 m^3
  fs_p_hp_rms    = 2.584e+02 Pa
  fs_vort_hp_rms = 7.296e-04
```

### Pressure Diagnostic Correction

Date: 2026-06-15

The water/free-surface pressure plot now separates the weakly compressible LBM
pressure into total pressure and hydrostatic-subtracted dynamic pressure:

```text
p_total = rho_phys * vel_scale^2 * c_s^2 * (rho_lb - rho0)
p_dynamic = p_total - rho_phys * g * max(eta(x,y) - z, 0)
```

This is a diagnostic-only change. It does not modify populations, collision,
viscosity, forcing, free-surface motion, or boundary conditions.

Reason:

The previous bottom panel plotted `p_total` relative to `rho0`. For water waves
this includes the hydrostatic column, so the color map was dominated by `rho g h`
rather than the dynamic/acoustic pressure error of interest. The run harness now
plots `p_dynamic` while still reporting total-pressure high-pass diagnostics
separately.

Short visible CUDA check with `LBM_USE_LES=1`, 3 m Gaussian,
`LBM_X_BOUNDARY=solid`, 500 steps:

```text
step 250:
  dynamic near-wall pressure RMS       = 1.678e+04 Pa
  dynamic near-wall pressure high-pass = 3.516e+03 Pa
  total near-wall pressure high-pass   = 4.622e+03 Pa
  free-surface dynamic pressure hp     = 1.696e+02 Pa

step 500:
  dynamic near-wall pressure RMS       = 2.839e+04 Pa
  dynamic near-wall pressure high-pass = 5.924e+03 Pa
  total near-wall pressure high-pass   = 2.652e+03 Pa
  free-surface dynamic pressure hp     = 2.626e+02 Pa
```

Interpretation:

The old plot overstated the problem by mixing hydrostatic and dynamic pressure.
However, the corrected dynamic-pressure panel still shows a coherent
column-scale pressure mismatch under the hump. This is not an LES-only issue; it
also appears with LES disabled. The likely source is the current cell-based
free-surface pressure closure, which applies the gas-pressure reconstruction at
cell/interface state rather than at a reconstructed sub-cell free-surface
location. That can leave the weakly compressible pressure field inconsistent
with the instantaneous free-surface elevation and excite acoustic ringing.

### Gaussian Initial-Condition Pressure Review

Date: 2026-06-15

The Gaussian release pressure issue was rechecked as an initial-condition
problem.

Implemented fix:

With Guo forcing, the physical velocity is reconstructed as:

```text
u = (sum_q f_q c_q + 0.5 F) / rho
```

Therefore a zero-physical-velocity water column under gravity must not be
initialized with zero raw population momentum. The initial equilibrium
populations now use:

```text
u_population,z = u_physical,z - 0.5 F_z / rho = u_physical,z + 0.5 g_lb
```

for active water cells. This keeps the requested Gaussian initial condition at
zero physical velocity after the first macroscopic reconstruction. No viscosity,
relaxation time, forcing, or damping was changed.

Short visible check, `LBM_USE_LES=1`, 3 m Gaussian, plotting every step:

```text
label = init_force_consistent_les_stepwise
step 1:
  dynamic near-wall pressure RMS       = 5.846e+01 Pa
  dynamic near-wall pressure high-pass = 3.430e+01 Pa
  free-surface dynamic pressure hp     = 6.152e+01 Pa
  max speed                            = 0.01 m/s
```

This removed the immediate gravity half-step pressure/velocity startup error,
but it did not materially improve the longer pressure behavior.

Long visible check, `LBM_USE_LES=1`, 3 m Gaussian, 10,000 steps:

```text
label = pressure_long_les_10k_current
step 1000:
  dynamic near-wall pressure RMS       = 3.088e+04 Pa
  dynamic near-wall pressure high-pass = 4.351e+03 Pa
  free-surface dynamic pressure hp     = 4.412e+02 Pa

step 5000:
  dynamic near-wall pressure RMS       = 1.276e+04 Pa
  dynamic near-wall pressure high-pass = 3.127e+03 Pa
  free-surface dynamic pressure hp     = 3.494e+02 Pa

step 10000:
  dynamic near-wall pressure RMS       = 1.362e+04 Pa
  dynamic near-wall pressure high-pass = 4.689e+03 Pa
  free-surface dynamic pressure hp     = 3.518e+02 Pa
```

Current diagnosis:

The corrected Guo-force initialization is physically required and remains in the
solver, but it is not a solution to the pressure oscillation seen over hundreds
to thousands of steps. The 250/500-step metrics before and after the change were
nearly identical, and the 10,000-step run still shows O(10^4 Pa) dynamic
pressure RMS and O(10^3 Pa) high-pass pressure in the 10-cell near-wall band.

A finite-depth linear-wave pressure initialization was also tested and then
removed from the active code because it did not materially improve the 500-step
pressure behavior. The remaining pressure growth is strongly correlated with VOF
state changes and free-surface reconstruction, not LES. The next correction
should focus on making the initial VOF/interface state and the first
mass-exchange/classification pass well-balanced, rather than changing
dissipation or turbulence modeling.

### Long-Run Pressure Residual Diagnostic

Date: 2026-06-15

A finite-depth linear wave-pressure residual diagnostic was added to
`run_ab_case.py`. For flat-bottom Gaussian tests it computes a diagnostic
pressure reference from the instantaneous free-surface shape:

```text
p_linear = rho g (base_head + eta_k cosh(k(z+h))/cosh(kh))
```

using cosine modes for solid left/right walls. The plotted pressure can be
switched with:

```text
LBM_PRESSURE_DIAGNOSTIC=linear_residual
```

This changes only the diagnostic plot and CSV metrics; it does not alter the
simulation.

Visible 1,000-step LES run with the residual diagnostic:

```text
label = pressure_linear_residual_les_1k
step 1000:
  dynamic near-wall pressure RMS       = 3.088e+04 Pa
  dynamic near-wall pressure high-pass = 4.351e+03 Pa
  linear-residual high-pass            = 4.335e+03 Pa
  free-surface linear-residual hp      = 4.469e+02 Pa
```

An initialization projection that applies the same gas-pressure
missing-population closure to t=0 interface populations was tested with:

```text
LBM_INIT_FREE_SURFACE_POPULATIONS=1
```

It did not materially change the pressure residual:

```text
label = init_fs_projection_residual_les_1k
step 1000:
  dynamic near-wall pressure RMS       = 3.089e+04 Pa
  dynamic near-wall pressure high-pass = 4.355e+03 Pa
  linear-residual high-pass            = 4.338e+03 Pa
  free-surface linear-residual hp      = 4.088e+02 Pa
```

Interpretation:

The O(10^4 Pa) coherent pressure variation is the same order as the physical
gravity-wave pressure scale for a 3 m water-surface displacement:

```text
rho g a ~= 998.2 * 9.81 * 3 = 2.94e+04 Pa
```

That component is not removable without changing the physical initial-value
problem. The remaining numerical target is the O(10^3 Pa) high-pass/residual
component, especially near the free surface and in the near-wall band. Removing
that should be pursued through better well-balanced free-surface/VOF
initialization and boundary reconstruction, not by damping or changing
viscosity.

### Hydrostatic-Balanced Pressure Formulation

Date: 2026-06-21

The large pressure oscillation was isolated with a flat-water rest test. A flat
free surface with physical gravity, water viscosity, and the same grid produced
the same O(10^3)-O(10^4 Pa) pressure ringing as the Gaussian hump. Turning
gravity off as a diagnostic removed the ringing, which showed that the VOF
bookkeeping was not the primary source. The failure was the weakly compressible
LBM initialization/forcing of the hydrostatic column: the code stored the full
hydrostatic pressure in density while also applying gravity as a body force, but
`f = f_eq(rho,u)` is not an exact discrete hydrostatic equilibrium for this
forced LBM.

The solver now supports:

```text
LBM_PRESSURE_FORMULATION=hydrostatic_balanced
```

This is the default for water free-surface runs. It is a physics-preserving
hydrostatic decomposition, not a damping model. The LBM density stores the
pressure perturbation about the flat still-water hydrostatic state:

```text
p_total = rho g (H0 - z) + p'
momentum equation: du/dt = -grad(p')/rho
free-surface pressure condition: p' = rho g eta
```

Consequences:

- A flat water column initializes as uniform density with zero body force in the
  LBM update, while physical gravity is still used for celerity and the
  free-surface pressure condition.
- A Gaussian hump initializes with the finite-depth linear-wave perturbation
  pressure when `LBM_INITIAL_PRESSURE_MODE=linear_wave`.
- The old full hydrostatic density/body-force formulation remains available as
  `LBM_PRESSURE_FORMULATION=total_pressure` for comparison.

Validation runs, all at physical water properties with Re = 2.25e8:

```text
flat_rest_hydrostatic_les_500_pf25_agg
  step>=100 mean p_linear_resid_hp = 4.220e+03 Pa
  step>=100 mean max speed         = 3.004e-01 m/s

flat_rest_balanced_les_500_pf25_agg
  step>=100 mean p_linear_resid_hp = 2.815e+00 Pa
  step>=100 mean max speed         = 2.285e-04 m/s

linear_init_residual_les_500_pf25_agg
  step>=100 mean Gaussian p_linear_resid_hp = 4.129e+03 Pa

gaussian3_balanced_linear_les_500_pf25_agg
  step>=100 mean Gaussian p_linear_resid_hp = 7.734e+02 Pa

gaussian3_balanced_linear_noLES_500_pf25_agg
  same pressure behavior as the LES run, confirming that the fix is not an LES
  damping effect.
```

Remaining issue:

The free-surface-only residual high-pass for the Gaussian case is still
O(1.4e3)-O(1.6e3 Pa). The saved frames show this as a coherent interface
pressure-closure mismatch rather than the previous domain-filling acoustic
oscillation. The next improvement should focus on sub-cell free-surface pressure
reconstruction and VOF/interface consistency, not on changing viscosity,
relaxation time, or adding filters.

## Cleanup note

The historical sections above describe several attempted free-surface link
closures (`link`, `fsk_link`, and `fsl`). Those branches are not active solver
configuration choices after the cleanup pass. The current code accepts only
`LBM_FREE_SURFACE_BOUNDARY=cell` and `LBM_FREE_SURFACE_BOUNDARY=hydrostatic`.
The earlier link-closure notes are retained as research history, not as current
run instructions.

## Disconnected VOF falling-block diagnostic

A first disconnected-flow VOF diagnostic was added with:

```text
LBM_FREE_SURFACE_TRACKING=vof
LBM_FREE_SURFACE_INITIAL=gaussian
LBM_GAUSSIAN_AMPLITUDE_M=0
LBM_VOF_INITIAL_SHAPE=block
LBM_VOF_BLOCK_SIZE_M=4
LBM_VOF_BLOCK_BOTTOM_M=12
LBM_PRESSURE_FORMULATION=total_pressure
```

The block initial condition overlays a rectangular water volume on top of the
flat-pool height-function initialization. Cell fill is initialized from exact
axis-aligned cell/block volume overlap, so partial cells at the block boundary
come from geometry rather than field smoothing.

The pressure formulation is intentionally `total_pressure` for this diagnostic.
The current hydrostatic-balanced formulation subtracts the flat-pool
hydrostatic state and sets the body force to zero; that is not a complete model
for detached water unless the free-surface pressure closure is generalized to
local disconnected interface geometry.

Runs completed:

```text
vof_falling_block_amp0
  domain 80 x 4 x 24 m, dx=1 m, steps=500, plot_freq=50
  bounded through first impact

vof_falling_block_amp0_long
  same case, steps=1500, plot_freq=250
  bounded through 11.09 s physical time
  VOF mass at frames: 3291.09, 3289.57, 3289.78, 3289.89, 3289.96, 3289.95
  VOF orphan count: 0 at plotted frames
  intermittent boundedness/state-change excess: 0-4 cells at plotted frames
```

Observed behavior:

- The block falls under gravity, impacts the flat pool, and produces a
  complex/non-single-valued VOF interface without immediate velocity blow-up.
- The VOF contour overlay is now used in plots for VOF runs because the
  column-integrated `free_surface_h` line is not a physical free surface for
  disconnected water.
- The interface remains visibly grid-stair-stepped after impact. The pressure
  and vorticity high-pass diagnostics remain finite but show interface-scale
  structure.
- Mass behavior is not yet production-quality. The VOF mass drifts slightly
  after impact, and small boundedness-excess events appear intermittently.

Next likely VOF issues:

- Replace column-height-dependent assumptions in VOF diagnostics/refill with
  local interface geometry wherever detached water is possible.
- Revisit the cell-state mass exchange and interface-layer enforcement so
  fill, mass, and type remain more tightly conservative during topology changes.
- Add a local PLIC/cut-link pressure boundary before treating the disconnected
  VOF path as physically validated.

### Disconnected VOF column-height and coalescence update

A later review found a real column-height relic in the disconnected VOF path:
`update_free_surface_height_from_vof()` projects all liquid in a vertical column
onto a single `free_surface_h`. That projection is still useful as a volume
diagnostic, but it is not a physical surface for detached drops or overturning
water.

Solver-side changes:

- VOF gas-to-wet refill no longer uses the column-projected hydrostatic density
  target. It initializes density from already-wet local neighbors, with a
  `rho0` fallback only when no wet donor exists.
- Disconnected VOF runs are blocked from `LBM_FREE_SURFACE_BOUNDARY=hydrostatic`
  because that mode still depends on a single column height.
- `run_ab_case.py` no longer subtracts a column-height hydrostatic pressure for
  VOF diagnostics. VOF pressure plots use total pressure, and free-surface-band
  metrics are based on actual partial-fill interface cells.
- A VOF coalescence-link repair was added after refill. If a link that was
  gas-facing in the previous topology becomes wet-connected, the stale
  atmospheric-boundary population is replaced from the newly connected wet
  donor along the same lattice characteristic.
- An opt-in one-cell thin-gap bridge was added:
  `LBM_VOF_THIN_GAP_BRIDGE=1`. It treats a one-cell gas gap between wet VOF
  regions as an unresolved coalescence link and blends the atmospheric
  free-surface closure with wet characteristic streaming. The blend is scaled by
  `LBM_VOF_THIN_GAP_BRIDGE_STRENGTH` and clipped to 1. This is a grid-scale
  geometry interpretation for a single-fluid model that does not resolve trapped
  air; it is not added dissipation.

Key diagnostics, all with the block bottom at 16 m over a flat 10 m pool:

```text
vof_block_high_gap_diagfix
  total_pressure, no thin-gap bridge
  step 50: pool pressure remains a horizontal hydrostatic field before contact
  step 500: fs_p_total_hp = 4.39e3 Pa, Umax = 8.19 m/s

vof_block_high_gap_coalesce_repair
  coalescence-link repair active, no thin-gap bridge
  step 500: fs_p_total_hp = 4.18e3 Pa, Umax = 7.76 m/s

vof_block_high_gap_bridge
  thin-gap bridge strength 1
  step 500: fs_p_total_hp = 3.74e3 Pa, Umax = 7.61 m/s

vof_block_high_gap_bridge4_confirm
  thin-gap bridge strength 4
  step 500: fs_p_total_hp = 3.48e3 Pa, Umax = 7.36 m/s
```

A full thin-gap collapse diagnostic (`LBM_VOF_THIN_GAP_BRIDGE_STRENGTH=100`,
clipped to 1) reduced the step-500 interface pressure high-pass further to
about `2.39e3 Pa`, but the 1000-step run showed more aggressive late-time
topology changes. This should be treated as a diagnostic upper bound, not a
validated default.

Rejected diagnostic:

```text
vof_block_high_gap_hydrobal_bridge4
  LBM_PRESSURE_FORMULATION=hydrostatic_balanced
```

The attempt used local VOF link elevations for the free-surface pressure target
and initialized detached water in the hydrostatic perturbation variable. It was
not physically consistent with the current VOF mass/capacity model, which uses
LBM density as the mass capacity. The run immediately produced O(1e4) VOF
excess counts and large pressure/vorticity artifacts. Disconnected VOF therefore
remains on `total_pressure` until the VOF mass variable is decoupled from
pressure perturbation density or the balanced formulation is otherwise
rederived.

### Disconnected VOF Initial Pressure Fix

Date: 2026-06-22

The falling-block VOF case reintroduced startup pressure ringing when it used
the full `total_pressure` formulation. The problem was the same physical
initialization issue found earlier for the connected water case: a weakly
compressible LBM column initialized with full hydrostatic density and an
explicit gravity body force is not an exact discrete rest state.

Implemented correction:

```text
LBM_PRESSURE_FORMULATION=vof_component_balanced
```

This is now the default for `LBM_VOF_INITIAL_SHAPE=block`. If an old run script
sets `LBM_PRESSURE_FORMULATION=total_pressure`, the solver auto-corrects to
`vof_component_balanced` unless
`LBM_ALLOW_TOTAL_PRESSURE_VOF_DIAGNOSTIC=1` is set explicitly.

Physical meaning:

- The original still-water pool below `LBM_FREE_SURFACE_LEVEL_M` is advanced in
  a hydrostatic reference pressure variable, so a flat pool initializes as a
  balanced rest state.
- Detached water above the still-water level keeps the ordinary gravity body
  force, so a falling block/drop is not artificially suspended.
- The total pressure diagnostic is reconstructed as
  `p_total = p_variable + rho*g*reference_head`, using the local reference
  field rather than a single column-height surface.

This is a pressure decomposition and initialization/configuration correction,
not added damping, filtering, viscosity adjustment, or Reynolds-number change.

Validation run:

```text
vof_block_component_balanced_hpfix_500
  domain 80 x 4 x 24 m, dx=1 m
  block bottom = 16 m, block size = 4 m
  old total_pressure env deliberately set; solver auto-corrected
  step 50:
    p_total_hp_rms    = 3.878e2 Pa
    fs_p_total_hp_rms = 1.083e2 Pa
    visible pressure field: flat hydrostatic pool, no pressure footprint under
    the suspended block
  step 200 before strong impact/merger:
    p_total_hp_rms    = 3.867e2 Pa
    fs_p_total_hp_rms = 1.484e2 Pa
```

Diagnostic correction:

The pressure high-pass metric in `run_ab_case.py` and the direct solver plot
previously used `np.roll` in the vertical direction. That wraps the bottom of a
linear hydrostatic column against the top and falsely reports O(1e4 Pa)
grid-scale pressure noise. The diagnostic now uses linear ghost values at
nonperiodic vertical and solid-x boundaries, so a linear hydrostatic pressure
column has zero high-pass residual up to roundoff.

### Low-Overhead Trapped-Gas Collapse

Date: 2026-06-22

The connected-component trapped-gas pressure prototype was reverted because its
host-side labeling required GPU-to-CPU field copies every step and slowed the
solver dramatically. The replacement is an opt-in unresolved-gas closure:

```text
LBM_VOF_COLLAPSE_TRAPPED_GAS=1
LBM_VOF_COLLAPSE_INTERVAL=200
LBM_VOF_COLLAPSE_MAX_VOLUME_M3=8
LBM_VOF_COLLAPSE_FLOOD_SWEEPS=<default Nx+Ny+Nz>
```

Physical interpretation:

- The solver still does not solve an air phase.
- Gas connected to the top exterior boundary remains atmospheric gas.
- Small sealed gas pockets are treated as unresolved and removed.
- The removed void volume is not replaced by newly created water. The closure
  fills trapped gas fraction and removes the same tracked LBM mass from
  exterior gas-connected VOF interface cells.
- Large trapped gas volumes are left alone by setting
  `LBM_VOF_COLLAPSE_MAX_VOLUME_M3` below their candidate volume.

Implementation:

- `initialize_vof_exterior_gas_marks()` seeds gas cells connected to the top
  boundary.
- `propagate_vof_exterior_gas_marks()` performs GPU-only 6-neighbor flood
  propagation; no NumPy connected-component labeling is used.
- `summarize_vof_trapped_gas_collapse()` measures total non-exterior gas volume
  and exterior interface mass capacity.
- `apply_vof_trapped_gas_collapse()` performs the conservative redistribution
  only when the candidate trapped volume is positive, below the configured
  threshold, and there is enough exterior interface mass to remove.

This is explicitly not the rejected column-collapse approach. It does not sort
water vertically or force a single-valued water column. It only changes VOF
cells identified by gas connectivity.

Diagnostics added to `run_ab_case.py`:

```text
vof_collapse_cells
vof_collapse_candidate_volume
vof_collapse_applied_volume
```

Smoke test:

```text
vof_collapse_smoke_1
  domain 20 x 4 x 10 m, dx=1 m
  flat VOF pool at 5 m
  collapse enabled, interval=1, flood_sweeps=40
  result: completed 1 step and plotted visibly
  candidate/applied collapse volume = 0/0 m^3, as expected for a flat pool
```

### Detached Floater Handling

Date: 2026-06-22

Observed issue:

In violent VOF runs, very small detached water fragments can remain above the
still-water level as isolated or weakly connected interface cells. Gravity is
already active for water above the still-water level under
`vof_component_balanced` pressure, so the failure is not missing body force.
The failure is topological: the standard FSLBM mass exchange only moves liquid
volume across links to wet neighbors. A lone interface cell surrounded by gas
has velocity and gravity, but no wet-neighbor exchange path through the inactive
gas phase.

Implemented controls:

```text
LBM_VOF_DETACHED_ADVECTION=1
LBM_VOF_DETACHED_MAX_WET_NEIGHBORS=18
LBM_VOF_DETACHED_MAX_RESOLVED_NEIGHBORS=0
LBM_VOF_DETACHED_RESIDUAL_FILL_THRESHOLD=0.01
```

Physical/numerical interpretation:

- Resolved connected water continues to use the population-based FSLBM mass
  exchange.
- Only unresolved detached cells with no resolved wet neighbor are eligible.
- The added transfer is a conservative VOF kinematic flux from the detached
  cell into adjacent gas cells selected by the local velocity direction.
- A positivity CFL limiter prevents a cell from losing more represented liquid
  than it contains in one step.
- Tiny residual tails below the residual fill threshold are geometrically
  compacted into the velocity-selected neighbor. This is an interface-geometry
  cleanup for unresolved spray remnants, not a viscosity or pressure damping
  term.

Diagnostics added:

```text
vof_detached_count
vof_detached_mass
vof_compact_count
vof_compact_mass
```

Focused visible test:

```text
vof_single_cell_floater_compact
  domain 40 x 4 x 20 m, dx=1 m
  flat pool at z=10 m
  1 m VOF block centered at z=15.5 m
  result: the detached floater was removed by step 300
```

Limitation:

This is not a resolved droplet/spray model. In the one-cell test the unresolved
1 m3 block disappears from the visible VOF field and the final flat-pool volume
returns to the original pool volume. Treat this path as a cleanup for numerical
single-cell floaters, not as physically validated droplet impact or spray
dynamics. Larger detached water bodies must be resolved by enough cells to
retain a meaningful interface and should not rely on this cleanup.

### Solitary-Wave Initial Condition

Date: 2026-06-22

Added a first-class ratio control for solitary waves:

```text
LBM_SOLITARY_HEIGHT_DEPTH_RATIO=0.3
```

When this is set, the crest height is computed as:

```text
H = (H/h) h
```

and takes precedence over `LBM_SOLITARY_AMPLITUDE_M`. The implemented initial
surface is the standard long-wave solitary form:

```text
eta(x,0) = H sech^2(kappa (x - x0))
kappa = sqrt(3 H / (4 h^3))
C = sqrt(g (h + H))
```

The initial horizontal velocity defaults to a mass-consistent long-wave relation
for a right- or left-moving pulse:

```text
u(x,z,0) = direction * C eta / (h + eta)
```

with `direction=+1` for propagation in positive x. The default
`LBM_SOLITARY_INITIAL_VELOCITY=shallow_water_divfree` also initializes the
vertical velocity from local incompressibility:

```text
w(x,z,0) = -(z - z_bed) du/dx
```

This enforces bottom no-penetration and the free-surface kinematic condition for
a permanent long-wave pulse:

```text
eta_t + u_s eta_x = w_s
```

The older `LBM_SOLITARY_INITIAL_VELOCITY=shallow_water` mode is retained only as
a diagnostic comparison. It initializes `u(x)` but leaves `w=0`; for a spatially
varying `u`, that violates local incompressibility and creates an immediate
pressure-adjustment artifact in the weakly compressible LBM.

Example visible run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_solitary_wave.ps1
```

For the default script:

```text
h = 10 m
H/h = 0.3
H = 3 m
C = sqrt(9.81 * 13) = 11.29 m/s
1/kappa = 21.08 m
```

### Solitary-Wave Pressure Initialization Check

Date: 2026-06-22

Diagnostic scale requested for the pressure-impulse investigation:

```text
h = 10 m
H = 5 m
H/h = 0.5
dx = 0.5 m
1/kappa = 16.33 m
```

This gives more than 10 grid points across the crest region.

Findings:

- `LBM_SOLITARY_INITIAL_VELOCITY=none`: no domain-scale pressure bounce. Total
  pressure remains hydrostatic plus the crest pressure perturbation.
- `LBM_SOLITARY_INITIAL_VELOCITY=shallow_water`: produces the pressure impulse.
  The cause is an incompatible initial velocity field: `u(x)` varies in space
  but `w=0`, so the initial state is not locally divergence-free and does not
  satisfy the free-surface kinematic condition.
- `LBM_SOLITARY_INITIAL_VELOCITY=shallow_water_divfree`: removes the pressure
  impulse while keeping the intended right-moving long-wave velocity scale.
  The 500-step visible test showed a smooth total-pressure field, `Umax` about
  4.4 to 5.3 m/s, and pressure high-pass RMS of order `2e2` to `4e2 Pa`, similar
  to the no-velocity case.

Sources checked:

- Caiazzo,
  [Analysis of lattice Boltzmann initialization routines](https://www.wias-berlin.de/people/caiazzo/PAPERS/Caiazzo_InitAlg_for_LBM.pdf):
  LBM can produce initial layers when macroscopic initial conditions are mapped
  inconsistently to kinetic populations; pressure and non-equilibrium parts must
  be treated consistently.
- Song et al. 2023,
  [Numerical Generation of Solitary Wave and Its Propagation Characteristics in a Step-Type Flume](https://researchonline.ljmu.ac.uk/id/eprint/26059/1/Numerical%20Generation%20of%20Solitary%20Wave%20and%20Its%20Propagation.pdf):
  stable solitary-wave generation in CFD commonly uses wave-maker boundary
  motion and higher-order solitary-wave theory; first-order solutions create
  larger trailing waves than ninth-order Fenton-based generation.
- Fenton,
  [A ninth-order solution for the solitary wave](https://www.johndfenton.com/Papers/Fenton72-A-ninth-order-solution-for-the-solitary-wave.pdf):
  higher-order solitary-wave theory is the appropriate path if we want a more
  exact nonlinear initial condition rather than a boundary-generated wave.

Remaining limitation:

`shallow_water_divfree` is a physically consistent long-wave initialization, but
it is not a full high-order nonlinear solitary-wave solution. For high-quality
solitary-wave validation at `H/h >= 0.3`, the next step should be either a
Fenton/Grimshaw-style higher-order initial condition including velocity and
pressure, or a moving/piston boundary wave maker.

### Rectangular-Prism Solid Obstacle

Date: 2026-06-22

The solid "cube" geometry is now an oriented rectangular prism. The boundary
condition path is unchanged: the same SDF/cut-link solid boundary treatment is
used, but the box half extents can differ by axis.

New size controls:

```text
LBM_CUBE_SIZE_X_M=<streamwise length>
LBM_CUBE_SIZE_Y_M=<spanwise length>
LBM_CUBE_SIZE_Z_M=<vertical length>
```

Backward compatibility:

```text
LBM_CUBE_SIDE_M=<legacy equal side length>
```

If a per-axis size is not provided, it falls back to `LBM_CUBE_SIDE_M`. Existing
orientation and placement variables are unchanged:

```text
LBM_CUBE_CENTER_X_M
LBM_CUBE_CENTER_Y_M
LBM_CUBE_CENTER_Z_M
LBM_CUBE_YAW_DEG
LBM_CUBE_PITCH_DEG
```

### Snapshot-Based PyVista Render

Date: 2026-06-22

High-quality 3D rendering is handled outside Matplotlib. The run harness writes
compact render snapshots at plotted frames and can immediately render each
snapshot with PyVista/VTK using a true 3D camera, depth buffering, opaque solids,
and translucent water materials.

`run_lbm_case.py` controls this with script constants:

```text
RUN_RENDER_DURING = True
RUN_RENDER_EVERY_PLOT = 1
RENDER_CAMERA_M = "155,-30,28"
RENDER_TARGET_M = "75,10,11.5"
RENDER_ZOOM = "1.1"
```

When enabled, each rendered plotted step writes:

```text
<label>_render_<step>.npz
<label>_render_<step>_pyvista.png
```

The standalone renderer can still render an existing snapshot:

```powershell
.\.venv\Scripts\python.exe .\render_lbm_snapshot.py `
  .\test_runs\<label>\<label>_render_00001.npz `
  --out .\test_runs\<label>\<label>_render.png `
  --camera 160,10,20 --target 70,10,11.5 --zoom 1.25 `
  --stride 1 --upsample 2 --smooth-iter 30 --water-opacity 0.48
```

The renderer extracts the VOF water-air isosurface with marching cubes. Solid
cells are inpainted only in the visualization volume before marching cubes so
fluid-solid interfaces are not drawn as false water-air surfaces. Solid objects
are rendered separately and opaquely. Mesh smoothing and interpolation are
visualization-only and do not modify VOF fill, mass, velocity, density, pressure,
or any solver field.
