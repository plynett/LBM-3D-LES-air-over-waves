# LBM-3D-LES-air-over-waves (Taichi) — single-file solver

This repository contains a **single Python script** implementing a 3D **D3Q19 isothermal lattice Boltzmann method (LBM)** with an **LES closure** and a **free-surface / moving-bed boundary pipeline**. The intent is a practical, phase-resolved flow solver with robust near-interface behavior using **per-link “open fractions”** rather than binary wet/dry switching.

## Demo video
https://youtu.be/USMgdHAk_DQ

## Core numerical model (at a glance)

### LBM formulation
- **Lattice**: D3Q19
- **Equilibrium**: standard low-Mach, second-order Hermite expansion  
  `feq = w*rho*(1 + 3(c·u) + 4.5(c·u)^2 - 1.5|u|^2)`
- **Streaming**: **pull scheme** (gather from upstream neighbors)
- **Collision**: BGK/KBC-style relaxation (as implemented), with stability logic (e.g., limiters/positivity safeguards) where present in the script
- **Macroscopic reconstruction**: `rho` and `u` from zeroth/first moments of `f`

### LES closure (optional)
- Local eddy-viscosity / effective relaxation computed from a strain-rate measure (Smagorinsky-style approach, as implemented).
- The LES step updates the effective relaxation parameter used during collision.

## Free-surface / moving boundary handling

### Height-function free surface
- The free surface is represented by a **single-valued height field** `water_h(y,x) = η(x,y,t)` sampled at horizontal cell centers.
- Periodicity is assumed in **x/y** for sampling and indexing; **z** is bounded.

### Per-link openness (cut-link gating)
A key design feature is the construction of two per-link fields:
- `lattice_open[k,j,i][q] ∈ {0,1}`: binary open/blocked link flag
- `lattice_open_frac[k,j,i][q] ∈ [0,1]`: continuous “open fraction” used for smooth blending

For each cell-center `(xc,yc,zc)` and each direction `q`, the code samples the surface at the **link half-step endpoint**:
`(xq,yq,zq) = (xc,yc,zc) + 0.5*dx*(cx[q], cy[q], cz[q])`

Then:
- `η(xq,yq)` is obtained via **periodic bilinear interpolation** of `water_h`
- A vertical gap is formed: `gap = zq - η`
- A smooth open fraction is computed and clamped:
  `frac = clamp(0.5 + gap/dx, 0, 1)`

This yields a **one-cell-thick transition** that reduces numerical noise compared to hard wet/dry toggles.

### Streaming with open-fraction blending
During streaming, partially blocked links are handled via blending:
- open portion: streams from upstream population
- blocked portion: falls back to an opposite-direction population (bounce-back-like)

This provides a practical cut-link behavior driven by `lattice_open_frac`.

### Moving bed / moving-wall enforcement
The script supports a prescribed moving boundary velocity through:
- `u_wave`, `v_wave`, `w_wave` (defined on the horizontal grid)

Near boundary/interface cells (`near_obstacle==1`), a moving-wall IBB/bounce-back-style correction is applied and **blended by the open fraction** so the enforcement strengthens as links become more blocked.

## Code organization (within the single file)
Function names may vary slightly, but the solver follows this sequence:

- **Initialization**
  - constants/scales
  - wave bed kinematics + prescribed wall velocity
  - interface geometry + per-link openness
  - velocity field initialization (log-law + optional perturbations)
  - `f` initialized to equilibrium

- **Per timestep**
  1. Update bed geometry/velocity
  2. Rebuild interface geometry and `lattice_open / lattice_open_frac`
  3. Re-apply near-bed population constraints (blended)
  4. Macro reconstruction (`rho,u`)
  5. LES closure (optional)
  6. Collision
  7. Streaming (pull)
  8. Post-stream boundary conditions (fringe inlet / top / bed IBB)
  9. Commit `f_new -> f`

## Running
### Requirements
- Python 3.10+
- `taichi`
- `numpy`
- Any optional plotting/IO libs imported by the script (e.g., `matplotlib`)

Install minimal:
```bash
pip install taichi numpy
