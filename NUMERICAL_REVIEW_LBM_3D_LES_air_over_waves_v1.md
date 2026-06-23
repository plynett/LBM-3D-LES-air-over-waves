# Numerical Review: `LBM_3D_LES_air_over_waves_v1.py`

Scope: this review covers `LBM_3D_LES_air_over_waves_v1.py` only. The `LBM_3D_LES_air_over_3D_object_v1.py` file is intentionally out of scope per user instruction.

Project rule check: this review follows `AGENTS.md`. In particular, stability or noise fixes must not be obtained by changing the physical Reynolds number, adding artificial dissipation, filtering the flow field, or otherwise hiding boundary-condition errors.

## Executive Summary

The code is a weakly compressible D3Q19 LBM with BGK-like collision, Smagorinsky LES, a prescribed moving single-valued bottom geometry, periodic x/y indexing, a top velocity boundary, and a weak forcing fringe near `i = 0`.

The physical-to-lattice mapping is internally consistent in the narrow sense that the molecular base Reynolds number satisfies

```text
Re_phys = U_ref_phys * Lz_m / nu_air
Re_lb   = u_lid * Nz / nu0
```

For the current air case:

```text
U_ref_phys = 30 m/s
Lz_m       = 20 m
dx         = 0.5 m
Nz         = 40
nu_air     = 1.516e-5 m^2/s
u_lid      = 0.075

Re         = 3.96e7
nu0_lb     = 7.58e-8
tau0       = 0.5000002274
Ma_ref     = 0.130
dt_phys    = 0.00125 s
```

That is a very high-Re, very under-resolved VLES/LES problem. The molecular relaxation time is only about `2.27e-7` above `0.5`, which is close to single-precision resolution. The code therefore relies heavily on SGS viscosity, boundary treatment, positivity limiting, and sponge/forcing behavior for stability. That is not automatically wrong, but every such mechanism must be physically justified and validated because it can easily become hidden damping.

The largest boundary-condition concern is the moving bottom. The code combines three different wall treatments on the same links: pre-collision equilibrium reinitialization, stream-time open-fraction bounce fallback, and post-stream moving-wall bounce-back. This does not correspond cleanly to a standard interpolated bounce-back or momentum-exchange moving-wall method. It likely violates local mass/momentum consistency near the boundary and can plausibly generate pressure or vorticity noise.

## Solver Structure

Key functions:

- `calculate_feq`: standard second-order low-Mach D3Q19 equilibrium.
- `update_wave_bed_and_velocities`: prescribes a moving bottom height `water_h(y,x,t)` and wall velocity fields `u_wave`, `v_wave`, `w_wave`.
- `compute_phi_slope_corrected`: builds a slope-corrected signed-distance-like field, used only by the wall-model part of LES.
- `build_lattice_open_from_free_surface_periodic`: builds per-link open/closed flags and open fractions from the single-valued height field.
- `init_fields`: initializes log-law velocity, turbulence perturbations, and equilibrium populations.
- `update_bed_populations_and_reinit`: blends closed-link populations toward wall equilibrium before macroscopic reconstruction.
- `macro_step`: computes density and velocity moments, then overwrites top-layer velocity.
- `compute_LES`: finite-difference Smagorinsky eddy viscosity plus a near-wall log-law augmentation.
- `collide_KBC`: BGK-like collision with a positivity limiter, optional stochastic backscatter, and a top sponge.
- `stream`: pull streaming with open-fraction blending on partially blocked links.
- `apply_open_boundary_conditions`: applies a weak x-fringe forcing, top equilibrium overwrite, and moving-bed bounce-back correction.

## High-Re Treatment

### What the Code Does

The base molecular viscosity is set from the physical Reynolds number:

```python
Re_phys_air = U_ref_phys * Lz_m / nu_air
nu0 = u_lid * Nz / Re_phys_air
tau0 = 0.5 + 3 * nu0
```

This is the correct relationship for a single physical Reynolds number under the chosen LBM scaling.

LES then increases local effective viscosity:

```python
nu_eff = nu0 + nu_t
tau_eff = 0.5 + 3.0 * nu_eff
omegaLoc = 1.0 / tau_eff
```

The turbulence model uses finite-difference velocity gradients:

```python
nu_t = (Cs * Delta)**2 * S_mag
```

and a near-wall log-law estimate that can increase `nu_t`.

The collision kernel also applies:

- A top sponge with `tau_sponge_min = 0.95` in the top layers.
- A positivity limiter that scales the post-collision non-equilibrium part toward equilibrium if any population would become too small.
- Optional stochastic backscatter through `C_backscatter = 0.02`.

### Physical/Numerical Concerns

1. `tau0` is extremely close to `0.5`.

   With `tau0 - 0.5 = 2.27e-7`, the molecular viscosity is barely resolved in `ti.f32`. This is physically consistent with the stated Reynolds number, but numerically fragile. It means the meaningful viscosity in practice will be mostly SGS viscosity, wall-model viscosity, and any limiter/sponge effects.

2. The Smagorinsky strain-rate floor adds implicit eddy viscosity.

   The code computes:

   ```python
   S_mag = sqrt(2.0 * S_contr + 1e-6)
   ```

   Even in a uniform flow this creates `S_mag = 0.001`, so `nu_t = (0.2)^2 * 0.001 = 4e-5`, which is more than 500 times `nu0`. That is an artificial SGS-viscosity floor unless explicitly justified as a physical model. It changes the effective Reynolds number in low-strain regions.

3. The finite-difference LES gradient is boundary-sensitive.

   Gradients are computed by central differences using neighboring `ux/uy/uz`, including values in/near solid or partially blocked cells. For cut-cell or immersed surfaces, this can make the SGS stress depend on artificial solid-cell values rather than the resolved fluid strain. A more standard LBM-LES approach often uses the non-equilibrium stress tensor locally, reducing stencil contamination near boundaries.

4. The wall model is not geometrically consistent for sloped/moving walls.

   It detects a near-wall cell only when the cell directly below is solid:

   ```python
   if k > 0 and lattice_open[k - 1, j, i][0] == 0:
   ```

   This is a vertical-bottom assumption. For sloped walls, corners, or future arbitrary solids, the wall normal and tangential velocity must come from the local geometry, not from "cell below is solid."

5. The wall-model units and shear estimate need review.

   `dist_to_wall_m` is in physical meters, while `speed_rel` is in lattice units. The code then uses:

   ```python
   strain_approx = speed_rel / 0.5
   nu_wm = tau_w / strain_approx
   ```

   This is dimensionally ambiguous. A wall model can be implemented in lattice units, but all distance, velocity, and stress quantities must be mapped consistently.

6. The top sponge changes local physics.

   The top sponge relaxes with `tau = 0.95`, far from the physical molecular `tau0`. If the top is meant to be an absorbing numerical layer, this must be documented as a boundary treatment, kept away from regions of interest, and not treated as part of the physical high-Re solution. If the top is part of the modeled flow, this is artificial dissipation.

7. The collision is not a true KBC implementation.

   The function name says `collide_KBC`, but the implementation is BGK relaxation plus positivity limiting and optional stochastic stress injection. True KBC/entropic MRT methods decompose moments and use an entropy-based stabilizer. The current limiter may be useful diagnostically, but it locally changes non-equilibrium stress and therefore the effective transport.

8. Stochastic backscatter is not validated.

   Backscatter can be physically motivated in LES, but here it is random, local, and not tied to a resolved energy budget. It can inject grid-scale noise and should not be considered a validated high-Re closure without benchmark evidence.

## Boundary Conditions Actually Implemented

### X Direction

Enforced condition:

- Periodic streaming in x.
- A weak forcing/fringe region at `i < 5`, only for `k > Nz // 2`, relaxes all populations toward a target equilibrium with `rho0` and the log-law inlet profile.

How it is enforced:

- In `stream`, source index uses:

  ```python
  src_i = (i - cx[q]) % Nx
  ```

- In `apply_open_boundary_conditions`, for selected cells:

  ```python
  f_new = (1 - sigma) * f_new + sigma * feq(rho0, u_inlet_profile + fluct, 0, 0)
  ```

Physical limitations/errors:

- This is not a physical inlet/outlet pair. The docstring says left Dirichlet inlet and right Neumann outlet, but the code is periodic in x.
- The fringe adds/removes mass and momentum by relaxing all populations toward `rho0` equilibrium. This is closer to distributed body forcing/relaxation than a boundary condition.
- There is no explicit flux condition ensuring global mass conservation or a target pressure gradient.
- This can be acceptable for a periodic wind-tunnel/channel forcing setup, but it should be described and validated that way.

### Y Direction

Enforced condition:

- Periodic in spanwise direction.

How it is enforced:

```python
src_j = (j - cy[q]) % Ny
```

Physical limitations/errors:

- This is physically reasonable for an ideal spanwise-periodic slice.
- The domain is only `Ny = 20` cells wide at `dx = 0.5 m`, so large 3D turbulent structures may be artificially constrained.

### Top Boundary

Enforced condition:

- Top layer velocity is set to `(u_top_lb, 0, 0)`.
- Top layer populations are overwritten with equilibrium at `(rho0, u_top_lb, 0, 0)`.
- The top two layers also experience a high-viscosity sponge-like BGK relaxation.

How it is enforced:

- In `macro_step`:

  ```python
  ux[Nz - 1, j, i] = u_top_lb
  uy[Nz - 1, j, i] = 0.0
  uz[Nz - 1, j, i] = 0.0
  ```

- In `collide_KBC`:

  ```python
  is_sponge = (k > Nz - sponge_thickness)
  tau_sponge_min = 0.95
  ```

- In `apply_open_boundary_conditions`:

  ```python
  f_new[k_top, j, i][q] = feq(rho0, u_top_lb, 0, 0, q)
  ```

Physical limitations/errors:

- This is not a free-slip/specular top despite the header comment. It is a velocity-Dirichlet moving-lid/equilibrium overwrite.
- It enforces no penetration through `uz = 0`, but it also fixes tangential velocity and density.
- It does not enforce a stress-free shear condition.
- Overwriting all populations removes non-equilibrium stress and turbulence at the top; an incoming-population reconstruction would be less intrusive.
- If the top is intended as a far-field/open atmospheric boundary, this is too reflective and too dissipative.

### Bottom Moving Boundary

Enforced intended physical condition:

- The region below a prescribed single-valued height `water_h(y,x,t)` is treated as solid.
- The wall velocity is `(u_wave, v_wave, w_wave)`.
- Intended solid-wall condition is no penetration plus no slip relative to the moving wall.

How geometry is enforced:

- `water_h` is prescribed as a traveling-wave-like height field.
- For each cell and lattice direction, the code samples the interface height at a half-link endpoint and computes:

  ```python
  gap = zq - eta
  frac = clamp(0.5 + gap / dx, 0, 1)
  lo[q] = 0 if eta > zq else 1
  ```

- `lattice_open_frac` stores the continuous open fraction.
- `lattice_open` stores hard open/closed flags.
- `near_obstacle` is set if any direction is blocked.

How wall velocity is prescribed:

- `update_wave_bed_and_velocities` computes a prescribed wall height and horizontal velocity model.
- Vertical wall velocity is computed from the kinematic relation:

  ```python
  w_s = eta_t + u_s * eta_x + v_s * eta_y
  ```

How populations are modified:

1. Before macro reconstruction, `update_bed_populations_and_reinit` blends closed-link populations toward wall equilibrium:

   ```python
   f[q] = frac * f[q] + (1 - frac) * feq(rho0, u_wall, q)
   ```

2. During streaming, `stream` blends between normal pull-streaming and a bounce-back-like local opposite population:

   ```python
   f_new[q] = frac * f[src, q] + (1 - frac) * f[dest, opp[q]]
   ```

3. After streaming, `apply_open_boundary_conditions` applies a moving-wall correction:

   ```python
   f_wall = f[dest, qbar] - 6 * w[qbar] * rho0 * (c_qbar dot u_wall)
   f_new[q] = frac * f_new[q] + (1 - frac) * f_wall
   ```

Physical limitations/errors:

- The same wall is imposed three times in one time step. This is not a clean standard boundary condition and may double-count or conflict in mass/momentum exchange.
- The pre-collision equilibrium reinitialization removes non-equilibrium stress near the wall and pins density to `rho0`. That is not a conservative no-slip wall treatment.
- The stream-time fallback is stationary bounce-back-like and does not include wall velocity; the moving-wall correction is added later. This split is not equivalent to a known interpolated moving-wall bounce-back formula.
- The open fraction is not the standard Bouzidi-Firdaouss-Lallemand wall distance `q` along a lattice link. It is a vertical-gap smooth gate evaluated at a half-link endpoint.
- Link reciprocity is not guaranteed. The same physical fluid-solid link can be viewed differently from the two adjacent cells.
- The method is limited to single-valued height functions `z = h(x,y)`. It cannot represent submerged boxes, overhangs, vertical faces, overturning waves, or arbitrary closed solids.
- For sloped walls, no-penetration should be enforced relative to the local wall normal. The current method uses discrete link bounce-back and a global wall velocity field; it does not explicitly project wall velocity into normal/tangential components.
- The wall velocity field is stored on the horizontal grid, not reconstructed at the actual link-wall intersection. For sloped walls this can misplace the local wall kinematics.
- Momentum exchange is not accumulated or checked, so the dynamic consistency of the moving boundary cannot be verified.

### Solid Cells

Enforced condition:

- Cells with no open links are marked solid for diagnostics/plotting.
- Populations in closed directions are repeatedly blended toward wall equilibrium.

Physical limitations/errors:

- Solid cells still participate in collision and streaming unless handled indirectly by open fractions.
- Macroscopic velocity in solid/partial cells is computed from populations; it is not consistently overwritten by wall velocity after `macro_step`.
- This can let wall-equilibrium populations participate in neighboring fluid updates in a way that does not correspond to a clean fluid-solid interface.

## Specific Boundary-Condition Findings

### 1. Documentation and implementation disagree

The header says:

- Left inlet Dirichlet.
- Right outlet Neumann.
- Top free-slip/specular.

The code implements:

- Periodic x.
- Periodic y.
- Weak x-fringe forcing, not an inlet.
- Top moving-lid/equilibrium overwrite plus sponge.

This mismatch matters because boundary interpretation drives validation.

### 2. The current bottom wall is not true interpolated bounce-back

Known curved-wall LBM methods such as Bouzidi-Firdaouss-Lallemand use the actual wall intersection distance along each lattice link and interpolate populations accordingly. The current `frac = 0.5 + gap/dx` is a smooth vertical-height gate, not a link-distance bounce-back rule. It may reduce binary chatter, but it does not guarantee second-order wall placement or correct momentum transfer.

### 3. The moving wall does not have a single conservative momentum-exchange path

For a moving solid wall, the fluid should receive the correct wall momentum through the boundary links, and the opposite momentum should be attributable to the wall. Current code:

- Pins some populations to equilibrium.
- Uses local opposite populations in streaming.
- Applies a second moving-wall correction post-stream.

This makes it hard to know which operation enforces no penetration, which enforces no slip, and whether the wall force is physically correct.

### 4. The wall model and wall boundary are disconnected

The log-law wall model increases `nu_t`, while the boundary population treatment independently imposes moving-wall bounce-back/equilibrium. A wall-modeled LES should define a consistent relationship between wall stress and near-wall velocity. Here the wall stress is not directly applied as a force or stress boundary; it is converted into local eddy viscosity and mixed with an independent no-slip-like boundary.

### 5. Corners and arbitrary surfaces are not supported by the current representation

Because geometry is `z = h(x,y)`, the method cannot distinguish:

- vertical sidewalls,
- underside faces,
- concave corners,
- submerged objects,
- overturning free surfaces.

Noise near corners in future object tests should therefore be expected unless the geometry representation and boundary method are replaced.

## What Is Physically Acceptable to Keep

- The physical-Re mapping should be kept. Do not change `tau0` except by changing the actual physical configuration or resolution/scaling with explicit documentation.
- Low-Mach scaling with `u_lid = 0.075` is reasonable, though max velocities must be monitored.
- Periodic `y` is reasonable for a spanwise-periodic domain.
- Periodic `x` plus a documented forcing/fringe can be reasonable if the physical problem is a periodic forced channel, not an inlet/outlet domain.
- A Smagorinsky SGS model is physically defensible for under-resolved high-Re flow, but the implementation should avoid artificial floors, stencil contamination, and unvalidated stochastic injection.
- Geometry smoothing at grid scale is allowed if it reconstructs the wall geometry, normals, and link cut fractions. It must not smooth the flow field or add dissipation.

## Recommended Corrections Before Water/Free-Surface Work

These are conceptual recommendations, not implemented changes.

1. Rename and separate boundary modes.

   Decide whether the current problem is a periodic forced channel, a wind-tunnel inlet/outlet, or a lid-driven/open-top channel. The code and documentation should match.

2. Replace the bottom boundary with a single standard method.

   Use one physically defined moving-wall method:

   - For height-function walls: link-wise interpolated moving bounce-back using true link-wall intersection distance.
   - For arbitrary solids: SDF/cut-link reconstruction with local wall normal, wall velocity at intersection, and link fraction.
   - Track momentum exchange so wall force and mass conservation can be diagnosed.

3. Remove or quarantine equilibrium reinitialization near the wall.

   Pre-collision blending of closed-link populations to `feq(rho0,u_wall)` should not be part of the production boundary method unless it is derived from a validated refill scheme for cells that change solid/fluid state.

4. Make the LES closure physically consistent.

   - Remove the `+1e-6` strain floor or replace it with a documented numerical tolerance that does not create meaningful eddy viscosity.
   - Consider non-equilibrium-stress-based strain for LBM-LES.
   - Rework wall modeling so wall stress is applied consistently with the boundary condition and local wall geometry.
   - Disable stochastic backscatter until validated.

5. Replace the top boundary if it is not a physical moving lid.

   If the top is open/far-field, use incoming-population reconstruction, convective/open, pressure, or stress/free-slip treatment as appropriate. Avoid overwriting all populations unless the intended physics is a strong reservoir/lid.

6. Add boundary diagnostics before changing physics.

   Track:

   - global mass drift,
   - density min/max and RMS,
   - max Mach number,
   - near-wall normal velocity error,
   - no-slip/tangential velocity error at reconstructed wall intersections,
   - momentum exchange / wall force,
   - limiter activation count,
   - SGS viscosity distribution relative to molecular viscosity.

7. Validate with canonical cases before free surface.

   Suggested order:

   - Laminar Couette flow with stationary/moving flat walls at low Re.
   - Pressure-driven Poiseuille/channel flow with known profile.
   - Oscillating/moving wall with known Stokes-layer behavior.
   - Flow past a cylinder or sphere at moderate Re with a curved-wall method.
   - Only then return to high-Re LES and moving wavy bed.

## Implications for Future Water + Free Surface

The current `water_h` field is not a physical free surface. It is used as a moving solid boundary below which cells are closed. A single-phase water free-surface LBM will require different physics:

- Kinematic surface evolution from the water velocity.
- Interface mass/volume fraction tracking or a level-set/VOF coupling.
- Atmospheric pressure boundary at the free surface.
- Dynamic stress condition: normal/tangential stress balance, with surface tension if included.
- No "solid bounce-back" at the water-air interface unless intentionally modeling a rigid lid or prescribed wavemaker.

Therefore, the bottom-boundary work should be made correct first, but it should not be mistaken for a free-surface implementation.

## References Used

- Bouzidi, Firdaouss, and Lallemand, "Momentum transfer of a Boltzmann-lattice fluid with boundaries", Physics of Fluids 13, 3452-3459 (2001). https://doi.org/10.1063/1.1399290
- Zou and He, "On pressure and velocity boundary conditions for the lattice Boltzmann BGK model", Physics of Fluids 9, 1591-1598 (1997). https://doi.org/10.1063/1.869307
- Ladd and Verberg, "Lattice-Boltzmann Simulations of Particle-Fluid Suspensions", Journal of Statistical Physics 104, 1191-1251 (2001). https://doi.org/10.1023/A:1010414013942
- Hou, Sterling, Chen, and Doolen, "A Lattice Boltzmann Subgrid Model for High Reynolds Number Flows" (1994/1996). https://arxiv.org/abs/comp-gas/9401004
- Bosch, Chikatamarla, and Karlin, "Entropic multi-relaxation lattice Boltzmann scheme for turbulent flows" (2015). https://arxiv.org/abs/1507.02518
