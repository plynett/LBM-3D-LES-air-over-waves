"""Generic editable runner for LBM_3D_LES_air_over_waves_CODEX.py.

Edit CASE_ENV below to define a case. Every supported environment switch is an
active dictionary entry; each line notes the solver/harness default.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TAICHI_CACHE_DIR = ROOT / ".taichi_cache" / "ticache"

# ---------------------------------------------------------------------------
# Case command settings. These are passed to run_ab_case.py after the solver is
# imported. The LBM_STEPS/LBM_PLOT_FREQ/LBM_PLOT_START environment variables are
# still listed below for direct solver runs, but these command values control
# this generic runner.
# ---------------------------------------------------------------------------
RUN_LABEL = "water_solitary_H5_dx05_divfree"  # default: required by run_ab_case
RUN_STEPS = 10000  # default: 10000 in run_ab_case, 400000 in direct solver
RUN_PLOT_FREQ = 200  # default: 500 in run_ab_case, 100 in direct solver
RUN_PLOT_START = 0  # default: 0
RUN_OUTPUT_DIR = ROOT / "test_runs" / RUN_LABEL  # default: required by run_ab_case
RUN_RANDOM_SEED = 12345  # default: 12345
RUN_MODULE = ROOT / "LBM_3D_LES_air_over_waves_CODEX.py"  # default: project CODEX solver

# ---------------------------------------------------------------------------
# Optional high-quality PyVista render during the solver run.
#
# When enabled, each plotted solver frame writes a .npz render snapshot and
# immediately renders a matching PyVista PNG before the time loop continues.
# This keeps render configuration with the case without putting renderer
# concerns into the physics kernels.
# ---------------------------------------------------------------------------
RUN_RENDER_DURING = False  # default: False
RUN_RENDER_EVERY_PLOT = 1  # default: 1

RENDER_SIZE = "2560,1440"  # default: 2560,1440
RENDER_CAMERA_M = "165,20,18"  # default: 155,-30,28
RENDER_TARGET_M = "75,30,10.5"  # default: 75,10,11.5
RENDER_ZOOM = "1.1"  # default: 1.1
RENDER_VIEW_ANGLE_DEG = "26"  # default: 26
RENDER_STRIDE = "1"  # default: 1
RENDER_UPSAMPLE = "2"  # default: 2
RENDER_INTERP_ORDER = "1"  # default: 1
RENDER_ISO_LEVEL = "0.5"  # default: 0.5
RENDER_SMOOTH_ITER = "30"  # default: 30
RENDER_SMOOTH_PASS_BAND = "0.05"  # default: 0.05
RENDER_WATER_OPACITY = "0.95"  # default: 0.5
RENDER_WATER_COLOR = "#3639e6"  # default: #36b7e6
RENDER_OBSTACLE_COLOR = "#2c2c30"  # default: #2c2c30
RENDER_BED_COLOR = "#4b4741"  # default: #4b4741
RENDER_OBSTACLE_SOURCE = "auto"  # default: auto; choices: auto | prism | mask | none
RENDER_NO_BED = False  # default: False
RENDER_SHOW_WINDOW = False  # default: False

# Clear inherited values before applying CASE_ENV. This keeps the backend choice
# in this file from being shadowed by an external shell environment.
CLEAR_ENV = [
    "MPLBACKEND",
]

# ---------------------------------------------------------------------------
# Environment settings.
#
# Every project-controlled environment variable currently read by the solver or
# harness is listed in this block. Values are active; comments show the default.
# ---------------------------------------------------------------------------
CASE_ENV = {
    # Runtime / plotting
    "TCL_LIBRARY": r"C:\ProgramData\anaconda3\envs\Taichi\tcl\tcl8.6",  # default: unset
    "TK_LIBRARY": r"C:\ProgramData\anaconda3\envs\Taichi\tcl\tk8.6",  # default: unset
    "MPLBACKEND": "TkAgg",  # default: unset
    "LBM_TI_CACHE_DIR": str(TAICHI_CACHE_DIR),  # default: .taichi_cache/ticache
    "LBM_RUN_LABEL": RUN_LABEL,  # default: direct; used only for direct solver runs
    "LBM_OUTPUT_DIR": str(RUN_OUTPUT_DIR),  # default: frames; used only for direct solver runs
    "LBM_STEPS": str(RUN_STEPS),  # default: 400000; run_ab_case command overrides this
    "LBM_PLOT_FREQ": str(RUN_PLOT_FREQ),  # default: 100; run_ab_case command overrides this
    "LBM_PLOT_START": str(RUN_PLOT_START),  # default: 0; run_ab_case command overrides this

    # Core runtime and physics mode
    "LBM_TI_ARCH": "gpu",  # default: gpu; choices: gpu | cpu
    "LBM_PHYSICS_MODE": "water",  # default: air; choices: air | water
    "LBM_U_TOP_MPS": "0",  # default: 30 in air, 0 in water
    "LBM_WATER_CURRENT_MPS": "0",  # default: 0 in water, LBM_U_TOP_MPS in air
    "LBM_WATER_NU_M2S": "1.004e-6",  # default: 1.004e-6
    "LBM_WATER_RHO_KGM3": "998.2",  # default: 998.2
    "LBM_GRAVITY_MPS2": "9.81",  # default: 9.81 in water, 0 in air

    # Domain and resolution
    "LBM_LX_M": "160",  # default: 300
    "LBM_LY_M": "1",  # default: 10
    "LBM_LZ_M": "20",  # default: 20
    "LBM_DX_M": "0.125",  # default: 0.5

    # Numerics and diagnostics
    "LBM_USE_LES": "1",  # default: 1 in air, 0 in water
    "LBM_COLLISION": "regularized",  # default: kbc in air, regularized in water
    "LBM_BACKSCATTER": "0.0",  # default: 0.02 in air, 0.0 in water
    "LBM_PRESSURE_DIAGNOSTIC": "total_pressure",  # default: total_pressure
    "LBM_INITIAL_PRESSURE_MODE": "hydrostatic",  # default: hydrostatic
    "LBM_PRESSURE_FORMULATION": "hydrostatic_balanced",  # default: total_pressure in air, hydrostatic_balanced in water, vof_component_balanced for disconnected VOF
    "LBM_ALLOW_TOTAL_PRESSURE_VOF_DIAGNOSTIC": "0",  # default: 0
    "LBM_SAVE_RENDER_SNAPSHOT": "0",  # default: 0; RUN_RENDER_DURING overrides to 1

    # Open/top air-flow options
    "LBM_ENABLE_TOP_DRIVE": "1",  # default: 1 in air, ignored in water
    "LBM_ENABLE_TOP_SPONGE": "1",  # default: 1 in air, ignored in water
    "LBM_ENABLE_X_FRINGE": "1",  # default: 1 in air with periodic x, ignored otherwise

    # Solid boundaries and solid geometry
    "LBM_X_BOUNDARY": "solid",  # default: periodic, except water+gaussian defaults solid
    "LBM_BOUNDARY_GEOMETRY": "phi_fraction",  # default: vertical; choices: vertical | phi_fraction | phi
    "LBM_WALL_VELOCITY_SAMPLING": "link",  # default: link; choices: cell | link
    "LBM_BED_PROFILE": "piecewise_linear",  # default: moving_sine in air, flat in water; choices: moving_sine | flat | piecewise_linear
    "LBM_BOTTOM_FILE": str(ROOT / "bottom.txt"),  # default: bottom.txt; used when LBM_BED_PROFILE=piecewise_linear
    "LBM_BED_AMP_M": "0.0",  # default: 1.5 for moving_sine, 0.0 for flat
    "LBM_BED_LEVEL_M": "0",  # default: 2*bed_amp+3*dx for moving_sine, 0 for flat
    "LBM_BED_WAVELENGTH_M": "30",  # default: 30
    "LBM_WATER_DEPTH_M": "10",  # default: 10
    "LBM_OBSTACLE_MODE": "bed",  # default: bed; choices: bed | bed_cube | cube
    #"LBM_CUBE_SIDE_M": "6",  # default: 6; fallback for prism dimensions
    "LBM_CUBE_SIZE_X_M": "70",  # default: LBM_CUBE_SIDE_M
    "LBM_CUBE_SIZE_Y_M": "8",  # default: LBM_CUBE_SIDE_M
    "LBM_CUBE_SIZE_Z_M": "1",  # default: LBM_CUBE_SIDE_M
    "LBM_CUBE_CENTER_X_M": "110",  # default: 0.5*LBM_LX_M
    "LBM_CUBE_CENTER_Y_M": "30",  # default: 0.5*LBM_LY_M
    "LBM_CUBE_CENTER_Z_M": "10.5",  # default: computed from domain, bed, pitch, and cube size
    "LBM_CUBE_YAW_DEG": "0",  # default: 0
    "LBM_CUBE_PITCH_DEG": "0",  # default: 0

    # Free-surface model
    "LBM_FREE_SURFACE_MODE": "height_kinematic",  # default: none in air, height_kinematic in water
    "LBM_FREE_SURFACE_TRACKING": "vof",  # default: height; choices: height | vof
    "LBM_FREE_SURFACE_BOUNDARY": "cell",  # default: cell; choices: cell | hydrostatic
    "LBM_FREE_SURFACE_REFILL": "equilibrium",  # default: equilibrium; choices: equilibrium | noneq | directional
    "LBM_FREE_SURFACE_INITIAL": "solitary",  # default: solitary for water height_kinematic/prescribed_solitary, otherwise flat
    "LBM_FREE_SURFACE_LEVEL_M": "10",  # default: LBM_BED_LEVEL_M + LBM_FREE_SURFACE_DEPTH_M
    "LBM_FREE_SURFACE_DEPTH_M": "10",  # default: LBM_WATER_DEPTH_M
    "LBM_INIT_FREE_SURFACE_POPULATIONS": "0",  # default: 0

    # Solitary-wave initial condition
    "LBM_SOLITARY_HEIGHT_DEPTH_RATIO": "0.5",  # default: unset; when set, amplitude = ratio*depth
    "LBM_SOLITARY_AMPLITUDE_M": "0.5",  # default: 0.5 when height/depth ratio is unset
    "LBM_SOLITARY_DIRECTION": "1",  # default: 1
    "LBM_SOLITARY_INITIAL_VELOCITY": "shallow_water_divfree",  # default: shallow_water_divfree
    "LBM_SOLITARY_X0_M": "55",  # default: 0.25*LBM_LX_M

    # Gaussian initial condition
    "LBM_GAUSSIAN_AMPLITUDE_M": "3.0",  # default: 3.0
    "LBM_GAUSSIAN_CENTER_X_M": "80",  # default: 0.5*LBM_LX_M
    "LBM_GAUSSIAN_SIGMA_M": "12.5",  # default: 12.5

    # Disconnected VOF initial water block
    "LBM_VOF_INITIAL_SHAPE": "none",  # default: none; choices: none | block
    "LBM_VOF_BLOCK_SIZE_M": "4.0",  # default: 4.0
    "LBM_VOF_BLOCK_SIZE_X_M": "4.0",  # default: LBM_VOF_BLOCK_SIZE_M
    "LBM_VOF_BLOCK_SIZE_Y_M": "4.0",  # default: min(LBM_VOF_BLOCK_SIZE_M, LBM_LY_M)
    "LBM_VOF_BLOCK_SIZE_Z_M": "4.0",  # default: LBM_VOF_BLOCK_SIZE_M
    "LBM_VOF_BLOCK_CENTER_X_M": "80",  # default: 0.5*LBM_LX_M
    "LBM_VOF_BLOCK_CENTER_Y_M": "10",  # default: 0.5*LBM_LY_M
    "LBM_VOF_BLOCK_CENTER_Z_M": "14.5",  # default: LBM_VOF_BLOCK_BOTTOM_M + 0.5*LBM_VOF_BLOCK_SIZE_Z_M
    "LBM_VOF_BLOCK_BOTTOM_M": "12.5",  # default: min(LBM_LZ_M-size_z-dx, LBM_FREE_SURFACE_LEVEL_M+2*dx)

    # VOF topology/state-change controls
    "LBM_VOF_LINK_APERTURE": "1",  # default: 1
    "LBM_VOF_EMPTY_FILL_THRESHOLD": "0.0",  # default: 0.0
    "LBM_VOF_ORPHAN_EMPTY_THRESHOLD": "0.1",  # default: 0.49
    "LBM_VOF_ORPHAN_SEARCH_RADIUS": "4",  # default: 4
    "LBM_VOF_ORPHAN_MAX_WEAK_NEIGHBORS": "1",  # default: 1
    "LBM_VOF_DETACHED_ADVECTION": "1",  # default: 1
    "LBM_VOF_DETACHED_MAX_WET_NEIGHBORS": "18",  # default: 18
    "LBM_VOF_DETACHED_MAX_RESOLVED_NEIGHBORS": "0",  # default: 0
    "LBM_VOF_DETACHED_RESIDUAL_FILL_THRESHOLD": "0.01",  # default: 0.01
    "LBM_VOF_THIN_GAP_BRIDGE": "0",  # default: 0
    "LBM_VOF_THIN_GAP_BRIDGE_STRENGTH": "1.0",  # default: 1.0

    # Low-overhead trapped-gas collapse
    "LBM_VOF_COLLAPSE_TRAPPED_GAS": "0",  # default: 0
    "LBM_VOF_COLLAPSE_INTERVAL": "200",  # default: 200
    "LBM_VOF_COLLAPSE_MAX_VOLUME_M3": "8.0",  # default: 8.0
    "LBM_VOF_COLLAPSE_FLOOD_SWEEPS": "800",  # default: Nx+Ny+Nz
}


def apply_environment() -> dict[str, str]:
    for key in CLEAR_ENV:
        os.environ.pop(key, None)

    active_env = {key: str(value) for key, value in CASE_ENV.items() if value is not None}
    if RUN_RENDER_DURING:
        active_env.update(
            {
                "LBM_SAVE_RENDER_SNAPSHOT": "1",
                "LBM_RENDER_SNAPSHOT_INLINE": "1",
                "LBM_RENDER_SNAPSHOT_INLINE_EVERY": str(RUN_RENDER_EVERY_PLOT),
                "LBM_RENDER_PLOT_FREQ": str(RUN_PLOT_FREQ),
                "LBM_RENDER_SIZE": RENDER_SIZE,
                "LBM_RENDER_CAMERA_M": RENDER_CAMERA_M,
                "LBM_RENDER_TARGET_M": RENDER_TARGET_M,
                "LBM_RENDER_ZOOM": str(RENDER_ZOOM),
                "LBM_RENDER_VIEW_ANGLE_DEG": str(RENDER_VIEW_ANGLE_DEG),
                "LBM_RENDER_STRIDE": str(RENDER_STRIDE),
                "LBM_RENDER_UPSAMPLE": str(RENDER_UPSAMPLE),
                "LBM_RENDER_INTERP_ORDER": str(RENDER_INTERP_ORDER),
                "LBM_RENDER_ISO_LEVEL": str(RENDER_ISO_LEVEL),
                "LBM_RENDER_SMOOTH_ITER": str(RENDER_SMOOTH_ITER),
                "LBM_RENDER_SMOOTH_PASS_BAND": str(RENDER_SMOOTH_PASS_BAND),
                "LBM_RENDER_WATER_OPACITY": str(RENDER_WATER_OPACITY),
                "LBM_RENDER_WATER_COLOR": RENDER_WATER_COLOR,
                "LBM_RENDER_OBSTACLE_COLOR": RENDER_OBSTACLE_COLOR,
                "LBM_RENDER_BED_COLOR": RENDER_BED_COLOR,
                "LBM_RENDER_OBSTACLE_SOURCE": RENDER_OBSTACLE_SOURCE,
                "LBM_RENDER_NO_BED": "1" if RENDER_NO_BED else "0",
                "LBM_RENDER_SHOW_WINDOW": "1" if RENDER_SHOW_WINDOW else "0",
            }
        )

    os.environ.update(active_env)
    return active_env


def find_python() -> Path:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if python.exists():
        return python
    return Path(sys.executable)


def print_command(cmd: list[str]) -> None:
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))


def main() -> int:
    active_env = apply_environment()

    module_path = Path(RUN_MODULE)
    if not module_path.is_absolute():
        module_path = ROOT / module_path

    output_dir = Path(RUN_OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    cmd = [
        str(find_python()),
        str(ROOT / "run_ab_case.py"),
        "--module",
        str(module_path),
        "--label",
        RUN_LABEL,
        "--steps",
        str(RUN_STEPS),
        "--plot-freq",
        str(RUN_PLOT_FREQ),
        "--plot-start",
        str(RUN_PLOT_START),
        "--output-dir",
        str(output_dir),
        "--random-seed",
        str(RUN_RANDOM_SEED),
    ]

    print("Running LBM case from run_lbm_case.py")
    print(f"Workspace: {ROOT}")
    print("Active environment:")
    for key in sorted(active_env):
        print(f"  {key}={active_env[key]}")
    print("Command:")
    print_command(cmd)
    if RUN_RENDER_DURING:
        print("Inline PyVista renderer:")
        print(f"  every plotted frame: {RUN_RENDER_EVERY_PLOT}")
        print(f"  camera={RENDER_CAMERA_M} target={RENDER_TARGET_M} zoom={RENDER_ZOOM}")
    print()
    sys.stdout.flush()

    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
