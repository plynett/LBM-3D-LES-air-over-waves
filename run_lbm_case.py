"""Generic editable runner for LBM_3D_LES_air_over_waves_CODEX.py.

Edit CASE_ENV below to define a case. Active variables are normal dictionary
entries; unused variables are listed as commented-out entries so they can be
enabled without hunting through the solver.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Case command settings. These are passed to run_ab_case.py after the solver is
# imported. The LBM_STEPS/LBM_PLOT_FREQ/LBM_PLOT_START environment variables are
# still listed below for direct solver runs, but these command values control
# this generic runner.
# ---------------------------------------------------------------------------
RUN_LABEL = "water_solitary_H5_dx05_divfree"
RUN_STEPS = 20000
RUN_PLOT_FREQ = 25
RUN_PLOT_START = 0
RUN_OUTPUT_DIR = ROOT / "test_runs" / RUN_LABEL
RUN_RANDOM_SEED = 12345
RUN_MODULE = ROOT / "LBM_3D_LES_air_over_waves_CODEX.py"

# ---------------------------------------------------------------------------
# Optional high-quality PyVista render during the solver run.
#
# When enabled, each plotted solver frame writes a .npz render snapshot and
# immediately renders a matching PyVista PNG before the time loop continues.
# This keeps render configuration with the case without putting renderer
# concerns into the physics kernels.
# ---------------------------------------------------------------------------
RUN_RENDER_DURING = True
RUN_RENDER_EVERY_PLOT = 1  # 1 = render every plotted frame; 2 = every other plotted frame

RENDER_SIZE = "2560,1440"
RENDER_CAMERA_M = "165,-4,18"
RENDER_TARGET_M = "75,10,11.5"
RENDER_ZOOM = "1.1"
RENDER_VIEW_ANGLE_DEG = "26"
RENDER_STRIDE = "1"
RENDER_UPSAMPLE = "2"
RENDER_INTERP_ORDER = "1"
RENDER_ISO_LEVEL = "0.5"
RENDER_SMOOTH_ITER = "30"
RENDER_SMOOTH_PASS_BAND = "0.05"
RENDER_WATER_OPACITY = "0.95"
RENDER_WATER_COLOR = "#36b7e6"
RENDER_OBSTACLE_COLOR = "#2c2c30"
RENDER_BED_COLOR = "#4b4741"
RENDER_OBSTACLE_SOURCE = "auto"  # auto | prism | mask | none
RENDER_NO_BED = False
RENDER_SHOW_WINDOW = False

# Clear these variables before applying CASE_ENV. Leave MPLBACKEND cleared if
# you want Matplotlib to open its normal interactive window.
CLEAR_ENV = [
    "MPLBACKEND",
]

# ---------------------------------------------------------------------------
# Environment settings.
#
# Every LBM_* environment variable currently read by the solver/harness is
# listed in this block. Uncomment and edit values to enable options.
# ---------------------------------------------------------------------------
CASE_ENV = {
    # Runtime / plotting
    "TCL_LIBRARY": r"C:\ProgramData\anaconda3\envs\Taichi\tcl\tcl8.6",
    "TK_LIBRARY": r"C:\ProgramData\anaconda3\envs\Taichi\tcl\tk8.6",
    # "MPLBACKEND": "TkAgg",

    # Core runtime and physics mode
    "LBM_TI_ARCH": "gpu",  # gpu | cpu
    "LBM_PHYSICS_MODE": "water",  # air | water
    # "LBM_U_TOP_MPS": "30",
    # "LBM_WATER_CURRENT_MPS": "0",
    # "LBM_WATER_NU_M2S": "1.004e-6",
    # "LBM_WATER_RHO_KGM3": "998.2",
    # "LBM_GRAVITY_MPS2": "9.81",

    # Domain and resolution
    "LBM_LX_M": "160",
    "LBM_LY_M": "20",
    "LBM_LZ_M": "20",
    "LBM_DX_M": "0.25",

    # Numerics and diagnostics
    "LBM_USE_LES": "1",
    # "LBM_COLLISION": "regularized",  # regularized | kbc
    # "LBM_BACKSCATTER": "0.0",
    "LBM_PRESSURE_DIAGNOSTIC": "total_pressure",
    # "LBM_INITIAL_PRESSURE_MODE": "hydrostatic",  # hydrostatic | linear_wave
    # "LBM_PRESSURE_FORMULATION": "hydrostatic_balanced",
    # "LBM_ALLOW_TOTAL_PRESSURE_VOF_DIAGNOSTIC": "0",

    # Render snapshots are enabled automatically when RUN_RENDER_DURING is true.
    # "LBM_SAVE_RENDER_SNAPSHOT": "1",

    # Open/top air-flow options
    # "LBM_ENABLE_TOP_DRIVE": "1",
    # "LBM_ENABLE_TOP_SPONGE": "1",
    # "LBM_ENABLE_X_FRINGE": "1",

    # Solid boundaries and solid geometry
    "LBM_X_BOUNDARY": "solid",  # periodic | solid
    "LBM_BOUNDARY_GEOMETRY": "phi_fraction",  # vertical | phi_fraction | phi
    # "LBM_WALL_VELOCITY_SAMPLING": "link",  # cell | link
    "LBM_BED_PROFILE": "flat",  # moving_sine | flat
    # "LBM_BED_AMP_M": "1.5",
    # "LBM_BED_LEVEL_M": "0",
    # "LBM_BED_WAVELENGTH_M": "30",
    # "LBM_WATER_DEPTH_M": "10",
    "LBM_OBSTACLE_MODE": "bed_cube",  # bed | bed_cube | cube
    # "LBM_CUBE_SIDE_M": "6",  # legacy fallback for all three prism dimensions
    "LBM_CUBE_SIZE_X_M": "80",
    "LBM_CUBE_SIZE_Y_M": "8",
    "LBM_CUBE_SIZE_Z_M": "1",
    "LBM_CUBE_CENTER_X_M": "110",
    "LBM_CUBE_CENTER_Y_M": "10",
    "LBM_CUBE_CENTER_Z_M": "11.5",
    "LBM_CUBE_YAW_DEG": "0",
    "LBM_CUBE_PITCH_DEG": "0",

    # Free-surface model
    "LBM_FREE_SURFACE_MODE": "height_kinematic",  # none | height_static | height_kinematic | prescribed_solitary
    "LBM_FREE_SURFACE_TRACKING": "vof",  # height | vof
    # "LBM_FREE_SURFACE_BOUNDARY": "cell",  # cell | hydrostatic
    # "LBM_FREE_SURFACE_REFILL": "equilibrium",  # equilibrium | noneq | directional
    "LBM_FREE_SURFACE_INITIAL": "solitary",  # flat | solitary | gaussian
    "LBM_FREE_SURFACE_LEVEL_M": "10",
    "LBM_FREE_SURFACE_DEPTH_M": "10",
    # "LBM_INIT_FREE_SURFACE_POPULATIONS": "0",

    # Solitary-wave initial condition
    "LBM_SOLITARY_HEIGHT_DEPTH_RATIO": "0.5",
    # "LBM_SOLITARY_AMPLITUDE_M": "3",
    "LBM_SOLITARY_DIRECTION": "1",
    "LBM_SOLITARY_INITIAL_VELOCITY": "shallow_water_divfree",  # none | shallow_water | shallow_water_divfree
    "LBM_SOLITARY_X0_M": "45",

    # Gaussian initial condition
    # "LBM_GAUSSIAN_AMPLITUDE_M": "8",
    # "LBM_GAUSSIAN_CENTER_X_M": "36",
    # "LBM_GAUSSIAN_SIGMA_M": "12.5",

    # Disconnected VOF initial water block
    # "LBM_VOF_INITIAL_SHAPE": "block",  # none | block
    # "LBM_VOF_BLOCK_SIZE_M": "4",
    # "LBM_VOF_BLOCK_SIZE_X_M": "4",
    # "LBM_VOF_BLOCK_SIZE_Y_M": "4",
    # "LBM_VOF_BLOCK_SIZE_Z_M": "4",
    # "LBM_VOF_BLOCK_CENTER_X_M": "80",
    # "LBM_VOF_BLOCK_CENTER_Y_M": "2",
    # "LBM_VOF_BLOCK_CENTER_Z_M": "15",
    # "LBM_VOF_BLOCK_BOTTOM_M": "12",

    # VOF topology/state-change controls
    # "LBM_VOF_EMPTY_FILL_THRESHOLD": "0.0",
    # "LBM_VOF_ORPHAN_EMPTY_THRESHOLD": "0.49",
    # "LBM_VOF_ORPHAN_SEARCH_RADIUS": "4",
    # "LBM_VOF_ORPHAN_MAX_WEAK_NEIGHBORS": "1",
    "LBM_VOF_DETACHED_ADVECTION": "1",
    # "LBM_VOF_DETACHED_MAX_WET_NEIGHBORS": "18",
    # "LBM_VOF_DETACHED_MAX_RESOLVED_NEIGHBORS": "0",
    # "LBM_VOF_DETACHED_RESIDUAL_FILL_THRESHOLD": "0.01",
    # "LBM_VOF_THIN_GAP_BRIDGE": "0",
    # "LBM_VOF_THIN_GAP_BRIDGE_STRENGTH": "1.0",

    # Low-overhead trapped-gas collapse
    "LBM_VOF_COLLAPSE_TRAPPED_GAS": "0",
    # "LBM_VOF_COLLAPSE_INTERVAL": "200",
    # "LBM_VOF_COLLAPSE_MAX_VOLUME_M3": "8.0",
    # "LBM_VOF_COLLAPSE_FLOOD_SWEEPS": "184",
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
