# LBM-3D-LES-air-over-waves (Taichi) — single-file solver

This repository contains a single Python script implementing a 3D D3Q19 lattice Boltzmann method (LBM) with an LES closure and a free-surface / moving-bed geometry pipeline using per-link open fractions (`lattice_open_frac`) and binary open flags (`lattice_open`).

## Demo video
https://youtu.be/USMgdHAk_DQ

## What’s in the file
Key components (naming follows the script):
- **Equilibrium**: D3Q19 low-Mach isothermal `feq`
- **Geometry / interface**: `compute_phi_slope_corrected()`, `build_lattice_open_from_free_surface_periodic()`
- **Initialization**: `init_fields()`
- **Time loop**: `main()`
- **Core LBM steps**: `macro_step()`, `compute_LES()` (optional), `collide_KBC()`, `stream()` (pull), `apply_open_boundary_conditions()`, `copy_post_and_swap()`

## Requirements
- Python 3.10+
- `taichi`
- `numpy`
- (Optional for plotting/outputs) whatever the script imports (e.g., `matplotlib`)

Install:
```bash
pip install taichi numpy
