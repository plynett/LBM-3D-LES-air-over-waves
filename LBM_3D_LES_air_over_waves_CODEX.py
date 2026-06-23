"""D3Q19 LBM solver for air-over-waves and single-fluid water tests.

The active solver supports two related configurations:

- air mode: airflow over a moving solid surface, with the original periodic
  channel, moving-wall, and optional LES/backscatter features retained.
- water mode: single-fluid free-surface tests using either a height-function
  kinematic surface or the experimental VOF/FSLBM path.

The physical viscosity, velocity scales, gravity, and domain dimensions set the
lattice viscosity and relaxation time. Stability diagnostics must not change
those physical quantities except when explicitly run as diagnostics. See
AGENTS.md for the project physics rules.
"""
# -----------------------------------------------------------------------------
# Imports
# - Standard library: file paths and OS utilities for output management.
# - Taichi: GPU-accelerated compute kernels for the LBM.
# - NumPy: host-side diagnostics and array work.
# -----------------------------------------------------------------------------
import os
import taichi as ti
import numpy as np

from lbm_env import env_choice, env_int
from lbm_lattice import D3Q19_CX, D3Q19_CY, D3Q19_CZ, D3Q19_OPP, D3Q19_WEIGHTS

# -----------------------------------------------------------------------------
# Taichi runtime initialization
# - LBM_TI_ARCH=gpu selects the GPU backend (CUDA/Metal/Vulkan depending on platform).
# - LBM_TI_ARCH=cpu is useful for compiler diagnostics when GPU codegen fails.
# - If you need deterministic behavior for debugging, consider setting ti.init(..., random_seed=...).
# -----------------------------------------------------------------------------
taichi_arch_name = env_choice("LBM_TI_ARCH", "gpu", ("gpu", "cpu"))
ti.init(arch=ti.cpu if taichi_arch_name == "cpu" else ti.gpu)

# -----------------------------------------------------------------------------
# Configuration overview
# The solver module defines physics, numerical parameters, Taichi fields, and
# the LBM time loop; small shared helpers live in lbm_env.py and lbm_lattice.py.
# -----------------------------------------------------------------------------
# ==============================
# 0. User Input / Physical Params
# ==============================
# Physical scale parameters. These define the "Real World" problem we are solving.
physics_mode = env_choice("LBM_PHYSICS_MODE", "air", ("air", "water"))

free_surface_mode = os.environ.get(
    "LBM_FREE_SURFACE_MODE",
    "none" if physics_mode == "air" else "height_kinematic",
).lower()
if free_surface_mode not in ("none", "height_static", "height_kinematic", "prescribed_solitary"):
    raise ValueError(
        "LBM_FREE_SURFACE_MODE must be 'none', 'height_static', "
        "'height_kinematic', or 'prescribed_solitary'"
    )
water_free_surface_enabled = 1 if free_surface_mode != "none" else 0
free_surface_initial_condition = os.environ.get(
    "LBM_FREE_SURFACE_INITIAL",
    "solitary" if (physics_mode == "water" and free_surface_mode in ("height_kinematic", "prescribed_solitary")) else "flat",
).lower()
if free_surface_initial_condition not in ("flat", "solitary", "gaussian"):
    raise ValueError("LBM_FREE_SURFACE_INITIAL must be 'flat', 'solitary', or 'gaussian'")
vof_initial_shape = os.environ.get("LBM_VOF_INITIAL_SHAPE", "none").lower()
if vof_initial_shape not in ("none", "block"):
    raise ValueError("LBM_VOF_INITIAL_SHAPE must be 'none' or 'block'")
initial_pressure_mode = env_choice(
    "LBM_INITIAL_PRESSURE_MODE",
    "hydrostatic",
    ("hydrostatic", "linear_wave"),
)
default_pressure_formulation = "hydrostatic_balanced" if physics_mode == "water" else "total_pressure"
if physics_mode == "water" and vof_initial_shape != "none":
    default_pressure_formulation = "vof_component_balanced"
pressure_formulation = env_choice(
    "LBM_PRESSURE_FORMULATION",
    default_pressure_formulation,
    ("total_pressure", "hydrostatic_balanced", "vof_component_balanced"),
)
pressure_formulation_auto_corrected = False
free_surface_tracking_mode = os.environ.get(
    "LBM_FREE_SURFACE_TRACKING",
    "height",
).lower()
if free_surface_tracking_mode not in ("height", "vof"):
    raise ValueError("LBM_FREE_SURFACE_TRACKING must be 'height' or 'vof'")
free_surface_boundary_mode = os.environ.get(
    "LBM_FREE_SURFACE_BOUNDARY",
    "cell",
).lower()
if free_surface_boundary_mode not in ("cell", "hydrostatic"):
    raise ValueError("LBM_FREE_SURFACE_BOUNDARY must be 'cell' or 'hydrostatic'")
free_surface_refill_mode = os.environ.get(
    "LBM_FREE_SURFACE_REFILL",
    "equilibrium",
).lower()
if free_surface_refill_mode not in ("equilibrium", "noneq", "directional"):
    raise ValueError("LBM_FREE_SURFACE_REFILL must be 'equilibrium', 'noneq', or 'directional'")
initialize_free_surface_populations = os.environ.get(
    "LBM_INIT_FREE_SURFACE_POPULATIONS",
    "0",
).lower() in ("1", "true", "yes")
vof_empty_fill_threshold = float(os.environ.get("LBM_VOF_EMPTY_FILL_THRESHOLD", "0.0"))
if not (0.0 <= vof_empty_fill_threshold < 0.5):
    raise ValueError("LBM_VOF_EMPTY_FILL_THRESHOLD must be in [0, 0.5)")
vof_orphan_empty_threshold = float(os.environ.get("LBM_VOF_ORPHAN_EMPTY_THRESHOLD", "0.49"))
if not (0.0 <= vof_orphan_empty_threshold < 0.5):
    raise ValueError("LBM_VOF_ORPHAN_EMPTY_THRESHOLD must be in [0, 0.5)")
vof_orphan_search_radius = int(os.environ.get("LBM_VOF_ORPHAN_SEARCH_RADIUS", "4"))
if vof_orphan_search_radius < 1 or vof_orphan_search_radius > 4:
    raise ValueError("LBM_VOF_ORPHAN_SEARCH_RADIUS must be in [1, 4]")
vof_orphan_max_weak_neighbors = int(os.environ.get("LBM_VOF_ORPHAN_MAX_WEAK_NEIGHBORS", "1"))
if vof_orphan_max_weak_neighbors < 0 or vof_orphan_max_weak_neighbors > 18:
    raise ValueError("LBM_VOF_ORPHAN_MAX_WEAK_NEIGHBORS must be in [0, 18]")
vof_detached_advection_enabled = os.environ.get(
    "LBM_VOF_DETACHED_ADVECTION",
    "1",
).lower() in ("1", "true", "yes")
vof_detached_max_wet_neighbors = int(os.environ.get("LBM_VOF_DETACHED_MAX_WET_NEIGHBORS", "18"))
if vof_detached_max_wet_neighbors < 0 or vof_detached_max_wet_neighbors > 18:
    raise ValueError("LBM_VOF_DETACHED_MAX_WET_NEIGHBORS must be in [0, 18]")
vof_detached_max_resolved_neighbors = int(os.environ.get("LBM_VOF_DETACHED_MAX_RESOLVED_NEIGHBORS", "0"))
if vof_detached_max_resolved_neighbors < 0 or vof_detached_max_resolved_neighbors > 18:
    raise ValueError("LBM_VOF_DETACHED_MAX_RESOLVED_NEIGHBORS must be in [0, 18]")
vof_detached_residual_fill_threshold = float(os.environ.get("LBM_VOF_DETACHED_RESIDUAL_FILL_THRESHOLD", "0.01"))
if not (0.0 <= vof_detached_residual_fill_threshold < 0.5):
    raise ValueError("LBM_VOF_DETACHED_RESIDUAL_FILL_THRESHOLD must be in [0, 0.5)")
vof_thin_gap_bridge_enabled = os.environ.get(
    "LBM_VOF_THIN_GAP_BRIDGE",
    "0",
).lower() in ("1", "true", "yes")
vof_thin_gap_bridge_strength = float(os.environ.get("LBM_VOF_THIN_GAP_BRIDGE_STRENGTH", "1.0"))
if vof_thin_gap_bridge_strength < 0.0:
    raise ValueError("LBM_VOF_THIN_GAP_BRIDGE_STRENGTH must be nonnegative")
vof_collapse_trapped_gas_enabled = os.environ.get(
    "LBM_VOF_COLLAPSE_TRAPPED_GAS",
    "0",
).lower() in ("1", "true", "yes")

U_top_phys = 30. if physics_mode == "air" else 0.0      # [m/s] lid / free stream speed
U_top_phys = float(os.environ.get("LBM_U_TOP_MPS", U_top_phys))
water_current_phys = float(os.environ.get("LBM_WATER_CURRENT_MPS", 0.0 if physics_mode == "water" else U_top_phys))
Lx_m = 300.0            # [m] domain length
Ly_m = 10.0            # [m] spanwise width (new y-direction "slice" extent)
Lz_m = 20.0            # [m] domain height

# Spatial Resolution
# dx determines the fidelity of the simulation. 
# At dx=~1m, we are in the VLES (Very Large Eddy Simulation) regime.
# We resolve large wake structures, but the boundary layer is mostly sub-grid.
dx = 0.5              # [m] grid spacing in x and y
Lx_m = float(os.environ.get("LBM_LX_M", Lx_m))
Ly_m = float(os.environ.get("LBM_LY_M", Ly_m))
Lz_m = float(os.environ.get("LBM_LZ_M", Lz_m))
dx = float(os.environ.get("LBM_DX_M", dx))
nu_air = 1.516e-5      # [m^2/s] kinematic viscosity of air (Standard Atmosphere)
rho_air_standard_phys = 1.2    # [kg/m^3] standard density of air
nu_water = float(os.environ.get("LBM_WATER_NU_M2S", 1.004e-6))  # [m^2/s] fresh water near 20 C
rho_water_phys = float(os.environ.get("LBM_WATER_RHO_KGM3", 998.2))  # [kg/m^3]
fluid_name = "water" if physics_mode == "water" else "air"
nu_phys = nu_water if physics_mode == "water" else nu_air
rho_phys = rho_water_phys if physics_mode == "water" else rho_air_standard_phys
rho_air_phys = rho_phys  # backward-compatible name used by existing diagnostics/harness
gravity_phys = float(os.environ.get("LBM_GRAVITY_MPS2", 9.81 if physics_mode == "water" else 0.0))

# Lattice resolution
Nx = int(Lx_m / dx)  # Lattice grid resolution along x: derived from physical domain size / dx
Ny = int(Ly_m / dx)  # Lattice grid resolution along y: derived from physical domain size / dx
Nz = int(Lz_m / dx)  # Lattice grid resolution along z: derived from physical domain size / dx
vof_collapse_interval = env_int("LBM_VOF_COLLAPSE_INTERVAL", 200)
if vof_collapse_interval < 1:
    raise ValueError("LBM_VOF_COLLAPSE_INTERVAL must be positive")
vof_collapse_max_volume_m3 = float(os.environ.get("LBM_VOF_COLLAPSE_MAX_VOLUME_M3", "8.0"))
if vof_collapse_max_volume_m3 < 0.0:
    raise ValueError("LBM_VOF_COLLAPSE_MAX_VOLUME_M3 must be nonnegative")
vof_collapse_flood_sweeps = env_int("LBM_VOF_COLLAPSE_FLOOD_SWEEPS", max(1, Nx + Ny + Nz))
if vof_collapse_flood_sweeps < 1:
    raise ValueError("LBM_VOF_COLLAPSE_FLOOD_SWEEPS must be positive")

# Timesteps & plotting
steps = 400000  # number of LBM "time" steps to run
plot_freq = 100  # create a plot every this many steps
plot_step_start = 0  # start plotting after this maNz steps (plotting can be slow, so we skip the initial transient)
steps = env_int("LBM_STEPS", steps)
plot_freq = env_int("LBM_PLOT_FREQ", plot_freq)
plot_step_start = env_int("LBM_PLOT_START", plot_step_start)
boundary_geometry_mode = os.environ.get("LBM_BOUNDARY_GEOMETRY", "vertical").lower()
if boundary_geometry_mode not in ("vertical", "phi_fraction", "phi"):
    raise ValueError("LBM_BOUNDARY_GEOMETRY must be 'vertical', 'phi_fraction', or 'phi'")
use_phi_open_fraction = 1 if boundary_geometry_mode in ("phi_fraction", "phi") else 0
use_phi_open_flags = 1 if boundary_geometry_mode == "phi" else 0
default_x_boundary_mode = "solid" if (physics_mode == "water" and free_surface_initial_condition == "gaussian") else "periodic"
x_boundary_mode = env_choice("LBM_X_BOUNDARY", default_x_boundary_mode, ("periodic", "solid"))
wall_velocity_sampling = env_choice("LBM_WALL_VELOCITY_SAMPLING", "link", ("cell", "link"))
use_link_wall_velocity = 1 if wall_velocity_sampling == "link" else 0

# parameters for the solid bed and, in water mode, the physical free surface
bed_profile = os.environ.get("LBM_BED_PROFILE", "moving_sine" if physics_mode == "air" else "flat").lower()
if bed_profile not in ("moving_sine", "flat"):
    raise ValueError("LBM_BED_PROFILE must be 'moving_sine' or 'flat'")
bed_is_moving = 1 if bed_profile == "moving_sine" else 0

bed_amp_m = 1.5 if bed_profile == "moving_sine" else 0.0 # amplitude [m] of the solid-bed sine wave
bed_amp_m = float(os.environ.get("LBM_BED_AMP_M", bed_amp_m))
depth_swl_m = float(os.environ.get("LBM_WATER_DEPTH_M", 10.0))  # mean water depth [m]
wave_speed = np.sqrt(gravity_phys * depth_swl_m) if gravity_phys > 0.0 else np.sqrt(9.81 * depth_swl_m)
bed_wavelength_m = float(os.environ.get("LBM_BED_WAVELENGTH_M", 30.0))  # wavelength [m] of the solid-bed sine wave
wave_period = bed_wavelength_m / max(wave_speed, 1e-12)  # period [s]

offset_m = 2.0*bed_amp_m + 3.0*dx if bed_profile == "moving_sine" else 0.0
offset_m = float(os.environ.get("LBM_BED_LEVEL_M", offset_m))  # solid bed elevation [m]
U_wave_phys = wave_speed if bed_is_moving == 1 else 0.0 # max horizontal velocity of the solid moving bed [m/s]

free_surface_depth_m = float(os.environ.get("LBM_FREE_SURFACE_DEPTH_M", depth_swl_m))
free_surface_base_m = float(os.environ.get("LBM_FREE_SURFACE_LEVEL_M", offset_m + free_surface_depth_m))
if free_surface_depth_m <= 0.0:
    raise ValueError("LBM_FREE_SURFACE_DEPTH_M must be positive")
solitary_height_depth_ratio_env = os.environ.get("LBM_SOLITARY_HEIGHT_DEPTH_RATIO")
if solitary_height_depth_ratio_env is not None:
    solitary_height_depth_ratio = float(solitary_height_depth_ratio_env)
    if solitary_height_depth_ratio <= 0.0:
        raise ValueError("LBM_SOLITARY_HEIGHT_DEPTH_RATIO must be positive")
    solitary_amp_m = solitary_height_depth_ratio * free_surface_depth_m
else:
    solitary_amp_m = float(os.environ.get("LBM_SOLITARY_AMPLITUDE_M", 0.5))
    if solitary_amp_m <= 0.0:
        raise ValueError("LBM_SOLITARY_AMPLITUDE_M must be positive")
    solitary_height_depth_ratio = solitary_amp_m / free_surface_depth_m
solitary_x0_m = float(os.environ.get("LBM_SOLITARY_X0_M", 0.25 * Lx_m))
solitary_direction = float(os.environ.get("LBM_SOLITARY_DIRECTION", 1.0))
if solitary_direction == 0.0:
    raise ValueError("LBM_SOLITARY_DIRECTION must be nonzero")
solitary_direction = 1.0 if solitary_direction > 0.0 else -1.0
solitary_initial_velocity_mode = os.environ.get("LBM_SOLITARY_INITIAL_VELOCITY", "shallow_water_divfree").lower()
if solitary_initial_velocity_mode not in ("none", "shallow_water", "shallow_water_divfree"):
    raise ValueError(
        "LBM_SOLITARY_INITIAL_VELOCITY must be 'none', 'shallow_water', "
        "or 'shallow_water_divfree'"
    )
solitary_celerity_phys = np.sqrt(gravity_phys * (free_surface_depth_m + solitary_amp_m)) if gravity_phys > 0.0 else 0.0
solitary_kappa_m_inv = np.sqrt(3.0 * solitary_amp_m / (4.0 * free_surface_depth_m**3))
solitary_length_scale_m = 1.0 / solitary_kappa_m_inv
gaussian_amp_m = float(os.environ.get("LBM_GAUSSIAN_AMPLITUDE_M", 3.0))
gaussian_center_x_m = float(os.environ.get("LBM_GAUSSIAN_CENTER_X_M", 0.5 * Lx_m))
gaussian_sigma_m = float(os.environ.get("LBM_GAUSSIAN_SIGMA_M", 12.5))
gaussian_celerity_phys = np.sqrt(gravity_phys * (free_surface_depth_m + gaussian_amp_m)) if gravity_phys > 0.0 else 0.0
vof_block_size_m = float(os.environ.get("LBM_VOF_BLOCK_SIZE_M", 4.0))
vof_block_size_x_m = float(os.environ.get("LBM_VOF_BLOCK_SIZE_X_M", vof_block_size_m))
vof_block_size_y_m = float(os.environ.get("LBM_VOF_BLOCK_SIZE_Y_M", min(vof_block_size_m, Ly_m)))
vof_block_size_z_m = float(os.environ.get("LBM_VOF_BLOCK_SIZE_Z_M", vof_block_size_m))
vof_block_center_x_m = float(os.environ.get("LBM_VOF_BLOCK_CENTER_X_M", 0.5 * Lx_m))
vof_block_center_y_m = float(os.environ.get("LBM_VOF_BLOCK_CENTER_Y_M", 0.5 * Ly_m))
vof_block_bottom_default_m = min(
    Lz_m - vof_block_size_z_m - dx,
    free_surface_base_m + 2.0 * dx,
)
vof_block_bottom_m = float(os.environ.get("LBM_VOF_BLOCK_BOTTOM_M", vof_block_bottom_default_m))
vof_block_center_z_m = float(os.environ.get(
    "LBM_VOF_BLOCK_CENTER_Z_M",
    vof_block_bottom_m + 0.5 * vof_block_size_z_m,
))
if vof_initial_shape != "none" and free_surface_tracking_mode != "vof":
    raise ValueError("LBM_VOF_INITIAL_SHAPE requires LBM_FREE_SURFACE_TRACKING=vof")
vof_has_disconnected_feature = vof_initial_shape != "none"
allow_total_pressure_vof = os.environ.get(
    "LBM_ALLOW_TOTAL_PRESSURE_VOF_DIAGNOSTIC",
    "0",
).lower() in ("1", "true", "yes")
if vof_has_disconnected_feature and pressure_formulation == "total_pressure" and not allow_total_pressure_vof:
    # Full hydrostatic density plus a body force is not a well-balanced
    # still-water initialization in this weakly compressible LBM. Detached VOF
    # defaults to a component pressure reference: the original flat pool is
    # hydrostatically balanced, while detached water still receives gravity.
    pressure_formulation = "vof_component_balanced"
    pressure_formulation_auto_corrected = True
if vof_has_disconnected_feature and pressure_formulation == "hydrostatic_balanced":
    raise ValueError(
        "Disconnected VOF features with LBM_PRESSURE_FORMULATION=hydrostatic_balanced are not supported; "
        "use vof_component_balanced so the flat pool is balanced while detached water still falls."
    )
if vof_has_disconnected_feature and free_surface_boundary_mode == "hydrostatic":
    raise ValueError(
        "Disconnected VOF features require LBM_FREE_SURFACE_BOUNDARY=cell; "
        "the hydrostatic option still uses a column-height surface."
    )
vof_block_drop_height_m = max(vof_block_center_z_m - free_surface_base_m, 0.0)
vof_reference_speed_phys = (
    np.sqrt(2.0 * gravity_phys * vof_block_drop_height_m)
    if (gravity_phys > 0.0 and vof_initial_shape != "none")
    else 0.0
)

obstacle_mode = os.environ.get("LBM_OBSTACLE_MODE", "bed").lower()
if obstacle_mode not in ("bed", "bed_cube", "cube"):
    raise ValueError("LBM_OBSTACLE_MODE must be 'bed', 'bed_cube', or 'cube'")
include_bed_geometry = 1 if obstacle_mode in ("bed", "bed_cube") else 0
include_cube_geometry = 1 if obstacle_mode in ("bed_cube", "cube") else 0
cube_side_m = float(os.environ.get("LBM_CUBE_SIDE_M", 6.0))
cube_size_x_m = float(os.environ.get("LBM_CUBE_SIZE_X_M", cube_side_m))
cube_size_y_m = float(os.environ.get("LBM_CUBE_SIZE_Y_M", cube_side_m))
cube_size_z_m = float(os.environ.get("LBM_CUBE_SIZE_Z_M", cube_side_m))
if cube_size_x_m <= 0.0 or cube_size_y_m <= 0.0 or cube_size_z_m <= 0.0:
    raise ValueError("LBM_CUBE_SIZE_X_M, LBM_CUBE_SIZE_Y_M, and LBM_CUBE_SIZE_Z_M must be positive")
cube_half_x_m = 0.5 * cube_size_x_m
cube_half_y_m = 0.5 * cube_size_y_m
cube_half_z_m = 0.5 * cube_size_z_m
cube_center_x_m = float(os.environ.get("LBM_CUBE_CENTER_X_M", 0.5 * Lx_m))
cube_center_y_m = float(os.environ.get("LBM_CUBE_CENTER_Y_M", 0.5 * Ly_m))
cube_yaw_deg = float(os.environ.get("LBM_CUBE_YAW_DEG", 0.0))
cube_yaw_rad = np.deg2rad(cube_yaw_deg)
cube_yaw_cos = float(np.cos(cube_yaw_rad))
cube_yaw_sin = float(np.sin(cube_yaw_rad))
cube_pitch_deg = float(os.environ.get("LBM_CUBE_PITCH_DEG", 0.0))
cube_pitch_rad = np.deg2rad(cube_pitch_deg)
cube_pitch_cos = float(np.cos(cube_pitch_rad))
cube_pitch_sin = float(np.sin(cube_pitch_rad))
cube_vertical_half_extent_m = cube_half_z_m * abs(cube_pitch_cos) + cube_half_x_m * abs(cube_pitch_sin)
cube_center_z_default = min(
    Lz_m - cube_vertical_half_extent_m - dx,
    max(0.5 * Lz_m, offset_m + 2.0 * bed_amp_m + cube_vertical_half_extent_m + dx),
)
cube_center_z_m = float(os.environ.get("LBM_CUBE_CENTER_Z_M", cube_center_z_default))

WALL_NONE = 0
WALL_BED = 1
WALL_CUBE = 2
WALL_XMIN = 3
WALL_XMAX = 4

FS_GAS = 0
FS_LIQUID = 1
FS_INTERFACE = 2

# ==============================
# LBM & Numerical Stability Parameters
# ==============================
# LBM reference/stability parameter: controls weak-compressibility (Mach) and viscosity via τ and ν=(τ-1/2)/3.
rho0 = 1.0             # lattice reference density (arbitrary in incompressible flow, usually 1.0)

# Lattice Velocity Scaling
# LBM is a weakly compressible solver. To simulate incompressible flow, 
# the velocity in lattice units (dx/dt) must be small compared to the 
# lattice speed of sound (cs = 1/sqrt(3) approx 0.577).
# u_lid < 0.1 ensures Mach number < 0.17, keeping compressibility errors low (<3%).
# LBM reference/stability parameter: controls weak-compressibility (Mach) and viscosity via τ and ν=(τ-1/2)/3.
u_lid = 0.075 # use a relatively small value as with very high Reynolds numbers, flow can have strong velocity amplifications   

# -----------------------------------------------------------------------------
# Physical ↔ lattice-unit mapping
# LBM evolves in nondimensional lattice units. We keep a small reference lattice
# speed (u_lid) for low-Mach stability, then scale to physical velocities with:
#   vel_scale = U_ref_phys / u_lid   [m/s per lattice unit]
#   dt_phys   = u_lid * dx / U_ref_phys  [s per LBM step]
# U_ref_phys is chosen as the max of imposed top and moving-bed speeds so the
# mapping remains valid even when U_top_phys = 0.
# -----------------------------------------------------------------------------
U_ref_phys = max(
    abs(U_top_phys),
    abs(U_wave_phys),
    abs(water_current_phys),
    abs(solitary_celerity_phys),
    abs(gaussian_celerity_phys),
    abs(vof_reference_speed_phys),
    1e-6,
)   # [m/s] mapping velocity scale
vel_scale = U_ref_phys / u_lid          # [m/s] per lattice unit
dt_phys   = u_lid * dx / U_ref_phys     # [s] physical time per LBM step

# Top / Bottom boundary velocity in lattice units (can be 0.0 cleanly)
u_top_lb = U_top_phys / vel_scale
u_bed_lb = wave_speed / vel_scale 
u_current_lb = water_current_phys / vel_scale
g_lb = gravity_phys * dt_phys * dt_phys / dx
use_component_balanced_pressure = 1 if (
    water_free_surface_enabled == 1 and pressure_formulation == "vof_component_balanced"
) else 0
use_hydrostatic_balanced_pressure = 1 if (
    water_free_surface_enabled == 1
    and pressure_formulation in ("hydrostatic_balanced", "vof_component_balanced")
) else 0
g_body_lb = 0.0 if (
    water_free_surface_enabled == 1 and pressure_formulation == "hydrostatic_balanced"
) else g_lb

Re_phys_air = U_ref_phys * Lz_m / nu_phys  # Physical Reynolds number based on domain height
Re_phys = Re_phys_air
nu0 = u_lid * Nz / Re_phys_air
tau0 = 0.5 + 3 * nu0 # base relaxation time (molecular). 
                       # Stability Limit: Must be > 0.5. 
                       # Values closer to 0.5 imply lower viscosity (Higher Reynolds).
                       # Values > 1.0 are very viscous (Stokes flow).
Re_lb_base = u_lid * Nz / nu0  # must be same as physical Re
print(f"tau used in current simulation: {tau0:.10f}")

# ==============================
# -----------------------------------------------------------------------------
# Turbulence model
# Smagorinsky LES: adds eddy viscosity nu_t based on local strain magnitude |S|.
# A wall-model augmentation modifies nu_t near the bed using a log-law estimate.
# -----------------------------------------------------------------------------
# Turbulence Modeling (LES)
# ==============================
# The code is set up use a (Very) Large Eddy Simulation (VLES) approach for turbulence modeling,
# where the LES model is active everywhere in the domain. 
# A wall model is implemented as a modification to the local turbulent viscosity 
# near the wall, based on the log-law of the wall.
use_LES = os.environ.get("LBM_USE_LES", "1" if physics_mode == "air" else "0").lower() in ("1", "true", "yes") # set to False to disable LES
collision_model = os.environ.get("LBM_COLLISION", "regularized" if physics_mode == "water" else "kbc").lower()
if collision_model not in ("kbc", "regularized"):
    raise ValueError("LBM_COLLISION must be 'kbc' or 'regularized'")

# Smagorinsky Constant (Cs)
# Standard values are 0.1 - 0.2. 
# Lower values (0.1) allow more fluctuation but may be unstable.
# Higher values (0.16-0.2) are more dissipative (stable).
Cs = 0.2 
Delta = 1.0 # Filter width. In standard LBM-LES, usually set to 1.0 lattice unit.

# Max Turbulent Viscosity Clamp
# Prevents numerical explosions in high-shear regions by limiting how "thick" 
# the fluid can artificially become.
nu_t_max = 10. 

# ==============================
# Synthetic Turbulence Generation (Inlet)
# ==============================
# To sustain turbulence, we cannot feed laminar flow into the domain.
# We inject synthetic fluctuations at the inlet using a Langevin equation (Markov Chain).
TI = 0.1              # Turbulence Intensity (100*TI % fluctuation relative to mean wind)
Correlation = 0.0     # Temporal correlation (0 to 1). 
                      # 0.0 = White Noise (random every step).
                      # 0.9+ = "Frozen" turbulence (long streaks).

# ==============================
# Wall Model Parameters
# ==============================
# Since the boundary layer is likely thinner than 'dx', we cannot resolve the no-slip condition directly.
# We use the "Log Law of the Wall" to approximate the shear stress at the boundary.
kappa = 0.41          # Von Karman constant (universal turbulence constant)
z0_phys = 0.01      # Roughness element height (m). 
                      # 0.0001 = Sand, 0.01 = Grass, 1.0 = Urban.
                      # This controls the virtual drag of the surface.

# Backscatter Model Parameters (optional, for energy injection at small scales)
# Standard Smagorinsky is purely dissipative (drains energy). 
# In coarse VLES, this kills interesting eddies.
# Backscatter injects random energy back into the flow to trigger instabilities.
C_backscatter = float(os.environ.get("LBM_BACKSCATTER", 0.02 if physics_mode == "air" else 0.0))
enable_top_drive = 1 if (physics_mode == "air" and os.environ.get("LBM_ENABLE_TOP_DRIVE", "1") != "0") else 0
enable_x_fringe = 1 if (physics_mode == "air" and x_boundary_mode == "periodic" and os.environ.get("LBM_ENABLE_X_FRINGE", "1") != "0") else 0
enable_top_sponge = 1 if (physics_mode == "air" and os.environ.get("LBM_ENABLE_TOP_SPONGE", "1") != "0") else 0

# Global fields for intermediate calculations
tmp = ti.field(dtype=ti.f32, shape=Nz) # temporary array for inlet velocity profile
u_inlet_profile = ti.field(dtype=ti.f32, shape=Nz) # inlet velocity profile (log-law) in lattice units
nu_t_field = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # turbulent viscosity field for LES
Smag_stress_field = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # Smagorinsky stress field for diagnostics
u_inlet_fluct = ti.field(dtype=ti.f32, shape=(Nz, Ny)) # synthetic turbulence fluctuations for inlet (Langevin process)

# Physical coordinate grids for plotting
x_phys = (np.arange(Nx) + 0.5) * dx  # center of cells
y_phys = (np.arange(Ny) + 0.5) * dx
z_phys = (np.arange(Nz) + 0.5) * dx
Z_phys, Y_phys, X_phys = np.meshgrid(z_phys, y_phys, x_phys, indexing='ij')

# ==============================
# Taichi Fields
# ==============================
w = ti.field(dtype=ti.f32, shape=19) # D3Q19 weights
cx = ti.field(dtype=ti.i32, shape=19) # D3Q19 x-velocities
cy = ti.field(dtype=ti.i32, shape=19) # D3Q19 y-velocities
cz = ti.field(dtype=ti.i32, shape=19) # D3Q19 z-velocities
opp = ti.field(dtype=ti.i32, shape=19) # Opposite directions

rho = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # density field
rho_initial = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # initialized pressure/density target
pressure_reference_head = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # hydrostatic pressure reference head [m]
gravity_body_fraction = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # fraction of physical gravity retained in momentum equation
ux = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # x-velocity field
uy = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # y-velocity field
uz = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # z-velocity field  

u_wave = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # x-dir velocity along wave surface
v_wave = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # y-dir velocity along wave surface
w_wave = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # z-dir velocity along wave surface

water_h = ti.field(dtype=ti.f32, shape=(Ny, Nx))  # instantaneous water surface height (meters)
water_h_last = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # previous water surface height (meters)
phi = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # signed distance (positive = air)
free_surface_h = ti.field(dtype=ti.f32, shape=(Ny, Nx))  # physical water free-surface elevation [m]
free_surface_h_last = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # previous free-surface elevation [m]
free_surface_fill = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # liquid volume fraction for height-function free surface
free_surface_type = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx)) # gas/liquid/interface state for single-phase water
free_surface_type_prev = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx)) # previous gas/liquid/interface state
free_surface_flux_x = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # depth-integrated x flux for kinematic surface update [m^2/s]
free_surface_flux_y = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # depth-integrated y flux for kinematic surface update [m^2/s]
free_surface_mass = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # VOF/FSLBM liquid mass in lattice units
free_surface_mass_delta = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # conservative link mass exchange increment
free_surface_mass_excess = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # boundedness excess redistributed after conversion
free_surface_mass_excess_dir = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx)) # +1 toward gas side, -1 toward liquid side
free_surface_detached_mass_delta = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # conservative kinematic flux for unresolved detached VOF cells
vof_exterior_gas = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx)) # gas cells connected to the exterior atmosphere

# main distribution functions (ping-pong)
# 'f' holds the distribution functions at the current step
# 'f_new' holds the post-collision/post-streaming functions for the next step
f = ti.Vector.field(19, dtype=ti.f32, shape=(Nz, Ny, Nx))  # main distribution functions
f_new = ti.Vector.field(19, dtype=ti.f32, shape=(Nz, Ny, Nx))  # post-collision distribution functions
lattice_open = ti.Vector.field(19, dtype=ti.i32, shape=(Nz, Ny, Nx))  # boolean mark for open-channel cells (1=open,0=solid)
lattice_open_frac = ti.Vector.field(19, dtype=ti.f32, shape=(Nz, Ny, Nx))  # δ for each q (only meaningful when blocked)

lattice_wall_type = ti.Vector.field(19, dtype=ti.i32, shape=(Nz, Ny, Nx))  # nearest wall source per link

# LES fields
omegaLoc = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx))  # local relaxation parameter for LES (1/tau_effective)

# obstacle masks (Integer fields: 0 = Fluid, 1 = Solid)
near_obstacle = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx)) # near-obstacle flag
obstacle = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx))        # current obstacle mask
obstacle_prev = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx))    # previous obstacle mask
boundary_refill_count = ti.field(dtype=ti.i32, shape=())      # cells refilled after solid->fluid transition
free_surface_refill_count = ti.field(dtype=ti.i32, shape=())  # cells refilled after gas->liquid/interface transition
free_surface_coalescence_repair_count = ti.field(dtype=ti.i32, shape=()) # VOF links repaired after gas gap closure
free_surface_thin_gap_bridge_count = ti.field(dtype=ti.i32, shape=()) # one-cell VOF gas gaps bridged during streaming
free_surface_cut_link_count = ti.field(dtype=ti.i32, shape=()) # gas-side links reconstructed at the free surface
free_surface_vof_excess_count = ti.field(dtype=ti.i32, shape=()) # VOF cells with boundedness excess/deficit
free_surface_vof_orphan_count = ti.field(dtype=ti.i32, shape=()) # low-fill orphan interface cells removed by FSLBM cleanup
free_surface_vof_collapse_cell_count = ti.field(dtype=ti.i32, shape=()) # trapped-gas cells collapsed this step
free_surface_vof_collapse_candidate_volume_m3 = ti.field(dtype=ti.f32, shape=()) # trapped gas candidate volume [m^3]
free_surface_vof_collapse_applied_volume_m3 = ti.field(dtype=ti.f32, shape=()) # trapped gas volume actually collapsed [m^3]
free_surface_vof_collapse_removal_scale = ti.field(dtype=ti.f32, shape=()) # fraction removed from exterior interface mass
free_surface_vof_collapse_candidate_mass = ti.field(dtype=ti.f32, shape=()) # candidate collapse mass in lattice units
free_surface_vof_collapse_exterior_capacity = ti.field(dtype=ti.f32, shape=()) # exterior removable interface mass
free_surface_vof_detached_advect_count = ti.field(dtype=ti.i32, shape=()) # unresolved detached VOF cells with gas-link mass advection
free_surface_vof_detached_advect_mass = ti.field(dtype=ti.f32, shape=()) # total lattice mass moved by detached VOF advection
free_surface_vof_detached_compress_count = ti.field(dtype=ti.i32, shape=()) # tiny detached residual cells geometrically compacted
free_surface_vof_detached_compress_mass = ti.field(dtype=ti.f32, shape=()) # total lattice mass moved by detached residual compaction
boundary_cut_link_count = ti.field(dtype=ti.i32, shape=())    # cut links treated by interpolated moving bounce-back
wall_momentum_x = ti.field(dtype=ti.f32, shape=())            # diagnostic momentum exchange proxy
wall_momentum_y = ti.field(dtype=ti.f32, shape=())
wall_momentum_z = ti.field(dtype=ti.f32, shape=())


@ti.func # compute the equilibrium distribution function for given local density and velocity
def calculate_feq(k_rho, k_ux, k_uy, k_uz, q):
    # -----------------------------------------------------------------------------
    # Equilibrium distribution feq for D3Q19 (isothermal, low-Mach)
    # Inputs:
    #   k_rho : local density (lattice units)
    #   k_ux, k_uy, k_uz : local velocity components (lattice units)
    #   q     : lattice direction index (0..18)
    # Output:
    #   feq_q : equilibrium population for direction q
    #
    # Notes:
    # - This is the standard second-order Hermite expansion:
    #     feq = w*rho*(1 + 3(c·u) + 4.5(c·u)^2 - 1.5|u|^2)
    # - Keeping u_lid small ensures compressibility errors remain bounded.
    # -----------------------------------------------------------------------------
    # D3Q19 isothermal equilibrium
    # feq_q = w_q * rho * [1 + 3(c·u) + 4.5(c·u)^2 - 1.5|u|^2]
    k_u2 = k_ux * k_ux + k_uy * k_uy + k_uz * k_uz  # |u|^2
    k_cu = 3.0 * (cx[q] * k_ux + cy[q] * k_uy + cz[q] * k_uz) # 3(c·u)
    return k_rho * w[q] * (1.0 + k_cu + 0.5 * k_cu * k_cu - 1.5 * k_u2)


@ti.func
def local_g_body_lb(k: ti.i32, j: ti.i32, i: ti.i32) -> ti.f32:
    g_local = g_body_lb
    if ti.static(use_component_balanced_pressure == 1):
        g_local = g_lb * gravity_body_fraction[k, j, i]
    return g_local


@ti.func
def smoothstep01(x: ti.f32) -> ti.f32:
    # -----------------------------------------------------------------------------
    # Smooth ramp on [0,1] with C1 continuity
    return x * x * (3.0 - 2.0 * x)


@ti.func
def sample_phi_periodic(xp: ti.f32, yp: ti.f32, zp: ti.f32) -> ti.f32:
    # Periodic x/y, clamped-z trilinear sample of the signed-distance-like field.
    Lx = ti.cast(Nx, ti.f32) * dx
    Ly = ti.cast(Ny, ti.f32) * dx
    Nx_f = ti.cast(Nx, ti.f32)
    Ny_f = ti.cast(Ny, ti.f32)

    xw = xp - ti.floor(xp / Lx) * Lx
    yw = yp - ti.floor(yp / Ly) * Ly

    u = xw / dx - 0.5
    v = yw / dx - 0.5
    u = u - ti.floor(u / Nx_f) * Nx_f
    v = v - ti.floor(v / Ny_f) * Ny_f

    i0 = ti.floor(u, ti.i32)
    j0 = ti.floor(v, ti.i32)
    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny

    fx = u - ti.cast(i0, ti.f32)
    fy = v - ti.cast(j0, ti.f32)

    r = zp / dx - 0.5
    k0 = ti.floor(r, ti.i32)
    fz = r - ti.cast(k0, ti.f32)
    k1 = k0 + 1

    if k0 < 0:
        k0 = 0
        k1 = 0
        fz = 0.0
    if k1 >= Nz:
        k0 = Nz - 1
        k1 = Nz - 1
        fz = 0.0

    p00_0 = phi[k0, j0, i0] * (1.0 - fx) + phi[k0, j0, i1] * fx
    p01_0 = phi[k0, j1, i0] * (1.0 - fx) + phi[k0, j1, i1] * fx
    pxy_0 = p00_0 * (1.0 - fy) + p01_0 * fy

    p00_1 = phi[k1, j0, i0] * (1.0 - fx) + phi[k1, j0, i1] * fx
    p01_1 = phi[k1, j1, i0] * (1.0 - fx) + phi[k1, j1, i1] * fx
    pxy_1 = p00_1 * (1.0 - fy) + p01_1 * fy

    return pxy_0 * (1.0 - fz) + pxy_1 * fz


@ti.kernel
def update_pressure_reference_fields():
    # Pressure-variable reference for gravity-balanced water cases.
    #
    # hydrostatic_balanced:
    #   The whole water domain uses p' about the flat still-water column, so the
    #   explicit body-force gravity is removed globally.
    #
    # vof_component_balanced:
    #   Only the original still-water pool below the base free surface uses the
    #   hydrostatic reference. Detached water above that level is advanced with
    #   ordinary gravity, so a falling block/drop does not get artificially held.
    for k, j, i in pressure_reference_head:
        pressure_reference_head[k, j, i] = 0.0
        gravity_body_fraction[k, j, i] = 0.0

        if ti.static(water_free_surface_enabled == 1):
            if lattice_open[k, j, i][0] == 1 and free_surface_type[k, j, i] != FS_GAS:
                zc = (ti.cast(k, ti.f32) + 0.5) * dx
                if ti.static(pressure_formulation == "hydrostatic_balanced"):
                    pressure_reference_head[k, j, i] = free_surface_base_m - zc
                    gravity_body_fraction[k, j, i] = 0.0
                elif ti.static(pressure_formulation == "vof_component_balanced"):
                    if zc < free_surface_base_m:
                        pressure_reference_head[k, j, i] = free_surface_base_m - zc
                        gravity_body_fraction[k, j, i] = 0.0
                    else:
                        pressure_reference_head[k, j, i] = 0.0
                        gravity_body_fraction[k, j, i] = 1.0
                else:
                    gravity_body_fraction[k, j, i] = 1.0


def prepare_initial_density_field():
    rho_np = np.full((Nz, Ny, Nx), rho0, dtype=np.float32)
    if water_free_surface_enabled != 1:
        rho_initial.from_numpy(rho_np)
        return

    h_np = free_surface_h.to_numpy().astype(np.float64)
    bed_np = water_h.to_numpy().astype(np.float64)
    x = (np.arange(Nx, dtype=np.float64) + 0.5) * dx
    z = (np.arange(Nz, dtype=np.float64) + 0.5) * dx

    use_linear = (
        initial_pressure_mode == "linear_wave"
        and free_surface_initial_condition == "gaussian"
        and np.allclose(bed_np, bed_np[0, 0], atol=1.0e-12, rtol=0.0)
    )

    if use_linear:
        bed = float(bed_np[0, 0])
        depth0 = max(free_surface_base_m - bed, dx)
        eta = h_np.mean(axis=0) - free_surface_base_m
        if x_boundary_mode == "solid":
            length = Nx * dx
            n = np.arange(Nx, dtype=np.float64)
            kx = n * np.pi / length
            cos_matrix = np.cos(np.outer(kx, x))
            coeff = (2.0 / Nx) * (cos_matrix @ eta)
            coeff[0] = np.mean(eta)
            for kk, zc in enumerate(z):
                z_rel = min(max(zc - bed, 0.0), depth0)
                vertical = np.ones_like(kx)
                active = kx > 0.0
                kh = kx[active] * depth0
                kz = kx[active] * z_rel
                vertical[active] = (
                    np.exp(kz - kh) + np.exp(-kz - kh)
                ) / (1.0 + np.exp(-2.0 * kh))
                eta_head = coeff @ (vertical[:, None] * cos_matrix)
                pressure_head = eta_head
                if use_hydrostatic_balanced_pressure != 1:
                    pressure_head = np.maximum((free_surface_base_m - zc) + eta_head, 0.0)
                rho_line = rho0 + g_lb * pressure_head / dx / (1.0 / 3.0)
                rho_np[kk, :, :] = rho_line.astype(np.float32)[None, :]
        else:
            modes = np.fft.rfftfreq(Nx, d=dx) * 2.0 * np.pi
            eta_hat = np.fft.rfft(eta)
            for kk, zc in enumerate(z):
                z_rel = min(max(zc - bed, 0.0), depth0)
                vertical = np.ones_like(modes)
                active = modes > 0.0
                kh = modes[active] * depth0
                kz = modes[active] * z_rel
                vertical[active] = (
                    np.exp(kz - kh) + np.exp(-kz - kh)
                ) / (1.0 + np.exp(-2.0 * kh))
                eta_head = np.fft.irfft(eta_hat * vertical, n=Nx)
                pressure_head = eta_head
                if use_hydrostatic_balanced_pressure != 1:
                    pressure_head = np.maximum((free_surface_base_m - zc) + eta_head, 0.0)
                rho_line = rho0 + g_lb * pressure_head / dx / (1.0 / 3.0)
                rho_np[kk, :, :] = rho_line.astype(np.float32)[None, :]
    else:
        for kk, zc in enumerate(z):
            if use_component_balanced_pressure == 1:
                # Store pressure perturbation relative to the per-cell reference
                # field. The default falling-block diagnostic starts from a flat
                # pool, so this is zero in both the pool and detached block.
                head = h_np - free_surface_base_m
            elif use_hydrostatic_balanced_pressure == 1:
                head = h_np - free_surface_base_m
            else:
                head = np.maximum(h_np - zc, 0.0)
            rho_plane = rho0 + g_lb * head / dx / (1.0 / 3.0)
            rho_np[kk, :, :] = np.asarray(rho_plane, dtype=np.float32)

    rho_initial.from_numpy(rho_np)


@ti.func
def free_surface_hydrostatic_density_at_cell(k: ti.i32, j: ti.i32, i: ti.i32) -> ti.f32:
    # One pressure target per fluid node: atmospheric pressure at the local
    # height-function surface plus the vertical hydrostatic head down to the
    # node center. This avoids direction-by-direction pressure anisotropy.
    head = 0.0
    if ti.static(use_component_balanced_pressure == 1):
        zc = (ti.cast(k, ti.f32) + 0.5) * dx
        if gravity_body_fraction[k, j, i] < 0.5:
            head = zc - free_surface_base_m
        else:
            head = 0.0
    elif ti.static(use_hydrostatic_balanced_pressure == 1):
        # Hydrostatic-balanced formulation stores only the pressure
        # perturbation p' about the flat still-water hydrostatic state.
        # The free-surface normal stress is therefore p' = rho*g*eta.
        head = free_surface_h[j, i] - free_surface_base_m
    else:
        zc = (ti.cast(k, ti.f32) + 0.5) * dx
        head = free_surface_h[j, i] - zc
    if head < 0.0:
        if ti.static(use_hydrostatic_balanced_pressure != 1):
            head = 0.0
    rho_fs = rho0 + g_lb * (head / dx) / (1.0 / 3.0)
    return rho_fs


@ti.func
def free_surface_pressure_density_at_link(k: ti.i32, j: ti.i32, i: ti.i32, q: ti.i32) -> ti.f32:
    rho_fs = rho0
    if ti.static(use_component_balanced_pressure == 1):
        z_interface = (ti.cast(k, ti.f32) + 0.5) * dx - 0.5 * ti.cast(cz[q], ti.f32) * dx
        head = 0.0
        if gravity_body_fraction[k, j, i] < 0.5:
            head += z_interface - free_surface_base_m
        rho_fs = rho0 + g_lb * (head / dx) / (1.0 / 3.0)
    elif ti.static(use_hydrostatic_balanced_pressure == 1 and free_surface_tracking_mode == "vof"):
        # In the hydrostatic-balanced variable, atmospheric pressure on a
        # disconnected VOF surface corresponds to p' = -p_ref(z_interface).
        # Use the local gas-side link midpoint as the interface elevation; do
        # not project detached water into a single column height.
        z_interface = (ti.cast(k, ti.f32) + 0.5) * dx - 0.5 * ti.cast(cz[q], ti.f32) * dx
        head = z_interface - free_surface_base_m
        rho_fs = rho0 + g_lb * (head / dx) / (1.0 / 3.0)
    else:
        rho_fs = free_surface_hydrostatic_density_at_cell(k, j, i)
    return rho_fs

@ti.func
def cube_sdf_at_point(xp: ti.f32, yp: ti.f32, zp: ti.f32) -> ti.f32:
    Lx = ti.cast(Nx, ti.f32) * dx
    Ly = ti.cast(Ny, ti.f32) * dx

    dxp = xp - cube_center_x_m
    dyp = yp - cube_center_y_m
    if ti.static(x_boundary_mode == "periodic"):
        if dxp > 0.5 * Lx:
            dxp -= Lx
        if dxp < -0.5 * Lx:
            dxp += Lx
    if dyp > 0.5 * Ly:
        dyp -= Ly
    if dyp < -0.5 * Ly:
        dyp += Ly

    x_local = cube_yaw_cos * dxp + cube_yaw_sin * dyp
    y_local = -cube_yaw_sin * dxp + cube_yaw_cos * dyp
    z1 = zp - cube_center_z_m

    x_pitch = cube_pitch_cos * x_local - cube_pitch_sin * z1
    z_local = cube_pitch_sin * x_local + cube_pitch_cos * z1

    px = ti.abs(x_pitch)
    py = ti.abs(y_local)
    pz = ti.abs(z_local)

    qx = px - cube_half_x_m
    qy = py - cube_half_y_m
    qz = pz - cube_half_z_m

    ox = ti.max(qx, 0.0)
    oy = ti.max(qy, 0.0)
    oz = ti.max(qz, 0.0)
    outside = ti.sqrt(ox * ox + oy * oy + oz * oz)
    inside = ti.min(ti.max(qx, ti.max(qy, qz)), 0.0)
    return outside + inside


@ti.func
def sample_wall_velocity_lb(xp: ti.f32, yp: ti.f32):
    # Periodic bilinear interpolation of the prescribed wall velocity at the
    # reconstructed wall-link intersection. Velocities are stored in LB units.
    Lx = ti.cast(Nx, ti.f32) * dx
    Ly = ti.cast(Ny, ti.f32) * dx
    Nx_f = ti.cast(Nx, ti.f32)
    Ny_f = ti.cast(Ny, ti.f32)

    xw = xp - ti.floor(xp / Lx) * Lx
    yw = yp - ti.floor(yp / Ly) * Ly

    u = xw / dx - 0.5
    v = yw / dx - 0.5
    u = u - ti.floor(u / Nx_f) * Nx_f
    v = v - ti.floor(v / Ny_f) * Ny_f

    i0 = ti.floor(u, ti.i32)
    j0 = ti.floor(v, ti.i32)
    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny

    fx = u - ti.cast(i0, ti.f32)
    fy = v - ti.cast(j0, ti.f32)

    ux00 = u_wave[j0, i0]
    ux10 = u_wave[j0, i1]
    ux01 = u_wave[j1, i0]
    ux11 = u_wave[j1, i1]

    uy00 = v_wave[j0, i0]
    uy10 = v_wave[j0, i1]
    uy01 = v_wave[j1, i0]
    uy11 = v_wave[j1, i1]

    uz00 = w_wave[j0, i0]
    uz10 = w_wave[j0, i1]
    uz01 = w_wave[j1, i0]
    uz11 = w_wave[j1, i1]

    ux0 = ux00 * (1.0 - fx) + ux10 * fx
    ux1 = ux01 * (1.0 - fx) + ux11 * fx
    uy0 = uy00 * (1.0 - fx) + uy10 * fx
    uy1 = uy01 * (1.0 - fx) + uy11 * fx
    uz0 = uz00 * (1.0 - fx) + uz10 * fx
    uz1 = uz01 * (1.0 - fx) + uz11 * fx

    return ti.Vector([
        ux0 * (1.0 - fy) + ux1 * fy,
        uy0 * (1.0 - fy) + uy1 * fy,
        uz0 * (1.0 - fy) + uz1 * fy,
    ])


@ti.kernel
def init_constants():
    # -----------------------------------------------------------------------------
    # Populate D3Q19 lattice constants:
    #   - w[q]    : quadrature weights
    #   - (cx,cy,cz)[q] : discrete velocity set (integers)
    #   - opp[q]  : opposite-direction lookup (for bounce-back/reflection)
    #
    # Design:
    # - Using ti.static([...]) makes these compile-time constants in Taichi kernels.
    # - The final loop writes values into Taichi fields for GPU access.
    # -----------------------------------------------------------------------------
    # All of these are pure Python lists; ti.static(range(...)) makes the loop
    # a compile-time unrolled loop, so the list indexing is done on the Python side.
    # D3Q19 weights
    w_vals = ti.static(D3Q19_WEIGHTS)

    cx_vals = ti.static(D3Q19_CX)

    cy_vals = ti.static(D3Q19_CY)

    cz_vals = ti.static(D3Q19_CZ)

    # Opposites: for each q, opp[q] has (cx,cy,cz) = -(cx,cy,cz)
    opp_vals = ti.static(D3Q19_OPP)
    for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
        w[q]   = w_vals[q]  # Intermediate scalar 'w' for this kernel block
        cx[q]  = cx_vals[q]  # Intermediate scalar 'cx' for this kernel block
        cy[q]  = cy_vals[q]  # Intermediate scalar 'cy' for this kernel block
        cz[q]  = cz_vals[q]  # Intermediate scalar 'cz' for this kernel block
        opp[q] = opp_vals[q]  # Intermediate scalar 'opp' for this kernel block

    # Initialize fields:s
    for k, j, i in rho:  # k: z index (0..Nz-1), j: y index (0..Ny-1), i: x index
        rho[k, j, i] = rho0 # initial density
        ux[k, j, i] = 0.0  # Intermediate scalar 'ux' for this kernel block
        uy[k, j, i] = 0.0  # Intermediate scalar 'uy' for this kernel block
        uz[k, j, i] = 0.0  # Intermediate scalar 'uz' for this kernel block
        near_obstacle[k, j, i] = 0  # Intermediate scalar 'near_obstacle' for this kernel block
        obstacle[k, j, i] = 0  # Intermediate scalar 'obstacle' for this kernel block
        obstacle_prev[k, j, i] = 0  # Intermediate scalar 'obstacle_prev' for this kernel block
        free_surface_fill[k, j, i] = 1.0
        free_surface_type[k, j, i] = FS_LIQUID
        free_surface_type_prev[k, j, i] = FS_LIQUID
        free_surface_mass[k, j, i] = rho0
        free_surface_mass_delta[k, j, i] = 0.0
        free_surface_mass_excess[k, j, i] = 0.0
        free_surface_mass_excess_dir[k, j, i] = 0
        vof_exterior_gas[k, j, i] = 0
        f[k, j, i] = ti.Vector.zero(ti.f32, 19) # initialize to zero; will be set to equilibrium later
        f_new[k, j, i] = ti.Vector.zero(ti.f32, 19)  # Initialize vector accumulator (all zeros)
        lattice_wall_type[k, j, i] = ti.Vector.zero(ti.i32, 19)
        # Local relaxation rate ω=1/τ: encodes ν_eff=ν0+ν_t so collision adapts to resolved shear (LES).
        omegaLoc[k, j, i] = 1.0 / tau0

    for j, i in u_wave:  # Parallel loop over index space (Taichi SPMD)
        u_wave[j, i] = 0.0  # Intermediate scalar 'u_wave' for this kernel block
        v_wave[j, i] = 0.0  # Intermediate scalar 'v_wave' for this kernel block
        w_wave[j, i] = 0.0  # Intermediate scalar 'w_wave' for this kernel block
        water_h[j, i] = offset_m  # Intermediate scalar 'water_h' for this kernel block
        water_h_last[j, i] = offset_m  # Intermediate scalar 'water_h_last' for this kernel block
        free_surface_h[j, i] = free_surface_base_m
        free_surface_h_last[j, i] = free_surface_base_m
        free_surface_flux_x[j, i] = 0.0
        free_surface_flux_y[j, i] = 0.0

    boundary_refill_count[None] = 0
    free_surface_refill_count[None] = 0
    free_surface_coalescence_repair_count[None] = 0
    free_surface_thin_gap_bridge_count[None] = 0
    free_surface_cut_link_count[None] = 0
    free_surface_vof_excess_count[None] = 0
    free_surface_vof_orphan_count[None] = 0
    free_surface_vof_collapse_cell_count[None] = 0
    free_surface_vof_collapse_candidate_volume_m3[None] = 0.0
    free_surface_vof_collapse_applied_volume_m3[None] = 0.0
    free_surface_vof_collapse_removal_scale[None] = 0.0
    free_surface_vof_collapse_candidate_mass[None] = 0.0
    free_surface_vof_collapse_exterior_capacity[None] = 0.0
    free_surface_vof_detached_advect_count[None] = 0
    free_surface_vof_detached_advect_mass[None] = 0.0
    free_surface_vof_detached_compress_count[None] = 0
    free_surface_vof_detached_compress_mass[None] = 0.0
    boundary_cut_link_count[None] = 0
    wall_momentum_x[None] = 0.0
    wall_momentum_y[None] = 0.0
    wall_momentum_z[None] = 0.0

@ti.kernel
def update_wave_bed_and_velocities(t: ti.f32):
    # -----------------------------------------------------------------------------
    # Prescribed moving “water surface” geometry and kinematics
    # When coupling with the wave model, this routine would be replaced
    # by a call to the wave solver to get the instantaneous surface height
    # and horizontal velocities at time t.
    #
    # This routine defines:
    #   - water_h(j,i): the instantaneous surface height (meters)
    #   - (u_wave, v_wave, w_wave): the imposed interface velocity (LB units)
    #
    # Critical modeling detail:
    # - The geometry water_h is prescribed as a *traveling* waveform
    # - The interface velocity components are prescribed to satisfy a kinematic
    #   boundary condition so that eta(x,t), u_s, and w_s are internally consistent:
    #       w_s = eta_t + u_s * eta_x + v_s * eta_y
    # - These interface velocities are then used by the immersed bounce-back (IBB)
    #   moving-wall correction to inject momentum into the air.
    #
    # Implementation notes:
    # - Two harmonics are included (eta1, eta2) with different phase speeds, creating
    #   a more complex ripple field than a single sinusoid.
    # - All velocities are stored in lattice units by dividing by vel_scale.
    # -----------------------------------------------------------------------------
    
    k = 2.0 * 3.141592653589793 / bed_wavelength_m  # rad/m
    c = wave_speed                                  # m/s
    bed_time_ramp = 2.0 * wave_period                # time to ramp up bed motion
    bed_amp_now = 0.0
    if ti.static(bed_is_moving == 1):
        bed_amp_now = bed_amp_m * smoothstep01(t / bed_time_ramp) if t < bed_time_ramp else bed_amp_m  # Ramp-up amplitude for prescribed bed motion (m)
    for j in range(Ny):  # Parallel loop over index space (Taichi SPMD)
        for i in range(Nx):  # Parallel loop over index space (Taichi SPMD)
            x0 = ti.cast(i, ti.f32) * dx            # physical x (m)
            # store last water height for time derivative
            water_h_last[j, i] = water_h[j, i]  # Intermediate scalar 'water_h_last' for this kernel block

            eta1 = 0.0
            eta2 = 0.0
            eta = 0.0
            eta_x = 0.0
            phase = 0.0
            phase2 = 0.0
            if ti.static(bed_is_moving == 1):
                # Traveling solid-bed wave: eta(x,t) = a sin(k(x - c t))
                x_phase = x0 - c * t  # Intermediate scalar 'x_phase' for this kernel block
                x_phase2 = x0 - 0.5 * c * t  # second harmonic at half speed
                phase   = 0.5*k * x_phase  # Phase argument for first harmonic (rad)
                phase2  = k * x_phase2  # Phase argument for second harmonic (rad)
                eta1 = bed_amp_now * ti.sin(phase)  # First harmonic elevation component (m)
                eta2 = bed_amp_now * ti.sin(phase2)  # Second harmonic elevation component (m)
                eta = eta1 + eta2  # m
                eta_x1 = bed_amp_now * 0.5*k * ti.cos(phase)  # Intermediate scalar 'eta_x1' for this kernel block
                eta_x2 = bed_amp_now * k * ti.cos(phase2)  # Intermediate scalar 'eta_x2' for this kernel block
                eta_x = eta_x1 + eta_x2  # Interface slope d eta / dx

            # Update the prescribed surface height
            bed_h = offset_m + eta  # Prescribed bed/interface height (m) including vertical offset
            water_h[j, i] = bed_h  # Intermediate scalar 'water_h' for this kernel block

            # Surface slopes (analytic, consistent with eta)
            eta_x1 = bed_amp_now * 0.5*k * ti.cos(phase)  # Intermediate scalar 'eta_x1' for this kernel block
            eta_x2 = bed_amp_now * k * ti.cos(phase2)  # Intermediate scalar 'eta_x2' for this kernel block
            eta_x = eta_x1 + eta_x2  # Interface slope ∂η/∂x (dimensionless)
            eta_y = 0.0  # Interface slope ∂η/∂y (dimensionless)

            # Time derivative of eta at fixed x (analytic):
            eta_t = (water_h[j,i] - water_h_last[j,i]) / dt_phys  # m/s

            # --- Choose the SURFACE (water-particle) horizontal velocity model ---
            # These can eventually be adjusted to model some type of air-water slip
            h0 = depth_swl_m  # Intermediate scalar 'h0' for this kernel block
            u_s = c * eta1 / (h0 + eta1) + 0.5 * c * eta2 / (h0 + eta2)   # m/s
            v_s = 0.0  # Prescribed surface spanwise velocity v(x,t) (m/s)

            # --- Enforce kinematic BC to get the vertical component ---
            # This is important! The IBB moving wall condition needs a consistent
            # set of (u_s, v_s, w_s) that satisfy the kinematic BC.
            w_s = eta_t + u_s * eta_x + v_s * eta_y  # m/s

            # Store boundary velocities in LBM units
            u_wave[j, i] = u_s / vel_scale  # Intermediate scalar 'u_wave' for this kernel block
            v_wave[j, i] = v_s / vel_scale  # Intermediate scalar 'v_wave' for this kernel block
            w_wave[j, i] = w_s / vel_scale  # Intermediate scalar 'w_wave' for this kernel block


@ti.func
def solitary_surface_height_at_x(xp: ti.f32, t: ti.f32) -> ti.f32:
    Lx = ti.cast(Nx, ti.f32) * dx
    h0 = free_surface_depth_m
    kappa_sol = ti.sqrt(3.0 * solitary_amp_m / (4.0 * h0 * h0 * h0))
    center = solitary_x0_m + solitary_direction * solitary_celerity_phys * t
    xrel = xp - center
    if ti.static(x_boundary_mode == "periodic"):
        if xrel > 0.5 * Lx:
            xrel -= Lx
        if xrel < -0.5 * Lx:
            xrel += Lx
    arg = kappa_sol * xrel
    exp_p = ti.exp(arg)
    exp_m = ti.exp(-arg)
    sech = 2.0 / (exp_p + exp_m)
    return free_surface_base_m + solitary_amp_m * sech * sech


@ti.func
def gaussian_surface_height_at_x(xp: ti.f32) -> ti.f32:
    Lx = ti.cast(Nx, ti.f32) * dx
    xrel = xp - gaussian_center_x_m
    if ti.static(x_boundary_mode == "periodic"):
        if xrel > 0.5 * Lx:
            xrel -= Lx
        if xrel < -0.5 * Lx:
            xrel += Lx
    arg = xrel / gaussian_sigma_m
    return free_surface_base_m + gaussian_amp_m * ti.exp(-0.5 * arg * arg)


@ti.kernel
def initialize_free_surface_height(t: ti.f32):
    for j, i in free_surface_h:
        x = (ti.cast(i, ti.f32) + 0.5) * dx
        h = free_surface_base_m
        if ti.static(free_surface_initial_condition == "solitary" or free_surface_mode == "prescribed_solitary"):
            h = solitary_surface_height_at_x(x, t)
        elif ti.static(free_surface_initial_condition == "gaussian"):
            h = gaussian_surface_height_at_x(x)
        free_surface_h[j, i] = h
        free_surface_h_last[j, i] = h


@ti.kernel
def classify_free_surface_cells():
    for k, j, i in free_surface_fill:
        old_type = free_surface_type[k, j, i]
        fill = 1.0
        ctype = FS_LIQUID
        if ti.static(water_free_surface_enabled == 1):
            z_bottom = ti.cast(k, ti.f32) * dx
            h = free_surface_h[j, i]
            fill = (h - z_bottom) / dx
            if fill < 0.0:
                fill = 0.0
            if fill > 1.0:
                fill = 1.0
            if fill <= 1e-4:
                ctype = FS_GAS
            elif fill >= 1.0 - 1e-4:
                ctype = FS_LIQUID
            else:
                ctype = FS_INTERFACE
            if lattice_open[k, j, i][0] == 0:
                fill = 0.0
                ctype = FS_GAS
        free_surface_fill[k, j, i] = fill
        free_surface_type_prev[k, j, i] = old_type
        free_surface_type[k, j, i] = ctype


@ti.func
def interval_overlap_fraction(cell_min: ti.f32, cell_max: ti.f32, box_min: ti.f32, box_max: ti.f32) -> ti.f32:
    lo = cell_min
    if box_min > lo:
        lo = box_min
    hi = cell_max
    if box_max < hi:
        hi = box_max
    overlap = hi - lo
    if overlap < 0.0:
        overlap = 0.0
    return overlap / (cell_max - cell_min)


@ti.kernel
def apply_vof_initial_disconnected_shape():
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof" and vof_initial_shape == "block"):
        x_min = vof_block_center_x_m - 0.5 * vof_block_size_x_m
        x_max = vof_block_center_x_m + 0.5 * vof_block_size_x_m
        y_min = vof_block_center_y_m - 0.5 * vof_block_size_y_m
        y_max = vof_block_center_y_m + 0.5 * vof_block_size_y_m
        z_min = vof_block_center_z_m - 0.5 * vof_block_size_z_m
        z_max = vof_block_center_z_m + 0.5 * vof_block_size_z_m

        for k, j, i in free_surface_fill:
            if lattice_open[k, j, i][0] == 1:
                cell_x0 = ti.cast(i, ti.f32) * dx
                cell_x1 = cell_x0 + dx
                cell_y0 = ti.cast(j, ti.f32) * dx
                cell_y1 = cell_y0 + dx
                cell_z0 = ti.cast(k, ti.f32) * dx
                cell_z1 = cell_z0 + dx

                fx = interval_overlap_fraction(cell_x0, cell_x1, x_min, x_max)
                fy = interval_overlap_fraction(cell_y0, cell_y1, y_min, y_max)
                fz = interval_overlap_fraction(cell_z0, cell_z1, z_min, z_max)
                block_fill = fx * fy * fz

                if block_fill > 1.0e-4:
                    old_type = free_surface_type[k, j, i]
                    fill = free_surface_fill[k, j, i]
                    if block_fill > fill:
                        fill = block_fill
                    ctype = FS_INTERFACE
                    if fill >= 1.0 - 1.0e-4:
                        fill = 1.0
                        ctype = FS_LIQUID
                    free_surface_fill[k, j, i] = fill
                    free_surface_type_prev[k, j, i] = old_type
                    free_surface_type[k, j, i] = ctype

@ti.kernel
def initialize_vof_mass_from_fill():
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        for k, j, i in free_surface_mass:
            if lattice_open[k, j, i][0] == 1 and free_surface_type[k, j, i] != FS_GAS:
                free_surface_mass[k, j, i] = free_surface_fill[k, j, i] * rho[k, j, i]
            else:
                free_surface_mass[k, j, i] = 0.0
                free_surface_fill[k, j, i] = 0.0
                free_surface_type[k, j, i] = FS_GAS
            free_surface_type_prev[k, j, i] = free_surface_type[k, j, i]
            free_surface_mass_delta[k, j, i] = 0.0
            free_surface_mass_excess[k, j, i] = 0.0
            free_surface_mass_excess_dir[k, j, i] = 0


@ti.kernel
def compute_vof_mass_exchange():
    # Interface-cell mass exchange for the single-fluid free-surface LBM.
    # Bulk liquid cells remain full. Interface cells exchange tracked mass with
    # liquid/interface neighbors using streamed populations; interface-interface
    # links are weighted by mean fill fraction.
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        for k, j, i in free_surface_mass_delta:
            free_surface_mass_delta[k, j, i] = 0.0
            free_surface_mass_excess[k, j, i] = 0.0
            free_surface_mass_excess_dir[k, j, i] = 0
        free_surface_vof_excess_count[None] = 0

        for k, j, i in free_surface_mass:
            if lattice_open[k, j, i][0] == 1 and free_surface_type[k, j, i] == FS_INTERFACE:
                dm_cell = 0.0
                for q in ti.static(range(1, 19)):
                    ni_raw = i + cx[q]
                    ni = ni_raw
                    valid = True
                    if ti.static(x_boundary_mode == "periodic"):
                        ni = (ni_raw + Nx) % Nx
                    else:
                        if ni_raw < 0 or ni_raw >= Nx:
                            valid = False
                    nj = (j + cy[q] + Ny) % Ny
                    nk = k + cz[q]
                    if nk < 0 or nk >= Nz:
                        valid = False

                    if valid:
                        if lattice_open[nk, nj, ni][0] == 1 and free_surface_type[nk, nj, ni] != FS_GAS:
                            link_weight = 1.0
                            if free_surface_type[nk, nj, ni] == FS_INTERFACE:
                                link_weight = 0.5 * (
                                    free_surface_fill[k, j, i]
                                    + free_surface_fill[nk, nj, ni]
                                )

                            if link_weight > 0.0:
                                # Incoming mass from the neighbor after streaming
                                # minus outgoing mass that streamed to that neighbor.
                                dm_cell += link_weight * (
                                    f[k, j, i][opp[q]] - f[nk, nj, ni][q]
                                )
                free_surface_mass_delta[k, j, i] = dm_cell


@ti.kernel
def apply_vof_mass_update_and_preclassify():
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        for k, j, i in free_surface_mass:
            old_type = free_surface_type[k, j, i]
            free_surface_type_prev[k, j, i] = old_type
            free_surface_mass_excess[k, j, i] = 0.0
            free_surface_mass_excess_dir[k, j, i] = 0

            if lattice_open[k, j, i][0] == 0:
                free_surface_mass[k, j, i] = 0.0
                free_surface_fill[k, j, i] = 0.0
                free_surface_type[k, j, i] = FS_GAS
            else:
                rho_ref = rho[k, j, i]
                if rho_ref < 1.0e-8:
                    rho_ref = rho0
                m = free_surface_mass[k, j, i]
                if old_type == FS_INTERFACE:
                    m += free_surface_mass_delta[k, j, i]
                elif old_type == FS_LIQUID:
                    m = rho_ref
                else:
                    m = 0.0

                excess = 0.0
                excess_dir = 0
                if m < 0.0:
                    excess = m
                    excess_dir = -1
                    m = 0.0
                elif m > rho_ref:
                    excess = m - rho_ref
                    excess_dir = 1
                    m = rho_ref

                eps = m / rho_ref
                fill = eps
                ctype = FS_INTERFACE
                if fill > 1.0e-4 and fill < vof_empty_fill_threshold:
                    # A sub-resolution wet sliver should not remain an active
                    # interface cell. Empty it and conservatively move its
                    # residual water mass toward the liquid side of the
                    # reconstructed local interface.
                    excess = m
                    excess_dir = -1
                    m = 0.0
                    fill = 0.0
                    ctype = FS_GAS
                elif fill <= 1.0e-4:
                    fill = 0.0
                    ctype = FS_GAS
                elif fill >= 1.0 - 1.0e-4:
                    fill = 1.0
                    ctype = FS_LIQUID
                free_surface_mass[k, j, i] = m
                free_surface_mass_excess[k, j, i] = excess
                free_surface_mass_excess_dir[k, j, i] = excess_dir
                if ti.abs(excess) > 1.0e-7:
                    ti.atomic_add(free_surface_vof_excess_count[None], 1)
                free_surface_fill[k, j, i] = fill
                free_surface_type[k, j, i] = ctype


@ti.kernel
def redistribute_vof_mass_excess():
    # Conservative boundedness/state-change repair. The signed excess is the
    # amount of liquid mass to add/remove, while excess_dir tells which side of
    # the reconstructed interface receives it: +1 toward gas, -1 toward liquid.
    # This distinction matters for an under-resolved wet sliver that empties:
    # its remaining positive liquid mass must move back toward wet cells.
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        for k, j, i in free_surface_mass_excess:
            excess = free_surface_mass_excess[k, j, i]
            if ti.abs(excess) > 1.0e-12:
                excess_dir = free_surface_mass_excess_dir[k, j, i]
                if excess_dir == 0:
                    excess_dir = 1
                    if excess < 0.0:
                        excess_dir = -1

                ip = i + 1
                im = i - 1
                if ti.static(x_boundary_mode == "periodic"):
                    ip = (i + 1) % Nx
                    im = (i - 1 + Nx) % Nx
                else:
                    if ip >= Nx:
                        ip = Nx - 1
                    if im < 0:
                        im = 0
                jp = (j + 1) % Ny
                jm = (j - 1 + Ny) % Ny
                kp = k + 1
                km = k - 1
                if kp >= Nz:
                    kp = Nz - 1
                if km < 0:
                    km = 0

                # Fill is one in liquid and zero in gas, so -grad(fill) points
                # from liquid toward gas.
                nx = -(free_surface_fill[k, j, ip] - free_surface_fill[k, j, im])
                ny = -(free_surface_fill[k, jp, i] - free_surface_fill[k, jm, i])
                nz = -(free_surface_fill[kp, j, i] - free_surface_fill[km, j, i])
                nmag = ti.sqrt(nx * nx + ny * ny + nz * nz)
                if nmag > 1.0e-8:
                    inv_n = 1.0 / nmag
                    nx *= inv_n
                    ny *= inv_n
                    nz *= inv_n

                weight_sum = 0.0
                capacity_sum = 0.0
                for qn in ti.static(range(1, 19)):
                    ni_raw = i + cx[qn]
                    ni = ni_raw
                    valid = True
                    if ti.static(x_boundary_mode == "periodic"):
                        ni = (ni_raw + Nx) % Nx
                    else:
                        if ni_raw < 0 or ni_raw >= Nx:
                            valid = False
                    nj = (j + cy[qn] + Ny) % Ny
                    nk = k + cz[qn]
                    if nk < 0 or nk >= Nz:
                        valid = False
                    if valid and lattice_open[nk, nj, ni][0] == 1:
                        normal_dot = (
                            ti.cast(cx[qn], ti.f32) * nx
                            + ti.cast(cy[qn], ti.f32) * ny
                            + ti.cast(cz[qn], ti.f32) * nz
                        )
                        if excess_dir > 0:
                            if free_surface_type[nk, nj, ni] != FS_LIQUID:
                                room = 0.0
                                if excess > 0.0:
                                    rho_room = rho[nk, nj, ni]
                                    if rho_room < 1.0e-8:
                                        rho_room = rho0
                                    room = rho_room - free_surface_mass[nk, nj, ni]
                                else:
                                    room = free_surface_mass[nk, nj, ni]
                                if room > 1.0e-12:
                                    capacity_sum += room
                                    weight = normal_dot
                                    if weight > 0.0:
                                        weight_sum += weight * room
                        else:
                            if free_surface_type[nk, nj, ni] != FS_GAS:
                                room = 0.0
                                if excess > 0.0:
                                    rho_room = rho[nk, nj, ni]
                                    if rho_room < 1.0e-8:
                                        rho_room = rho0
                                    room = rho_room - free_surface_mass[nk, nj, ni]
                                else:
                                    room = free_surface_mass[nk, nj, ni]
                                if room > 1.0e-12:
                                    capacity_sum += room
                                    weight = -normal_dot
                                    if weight > 0.0:
                                        weight_sum += weight * room

                if capacity_sum > 1.0e-12:
                    for qn in ti.static(range(1, 19)):
                        ni_raw = i + cx[qn]
                        ni = ni_raw
                        valid = True
                        if ti.static(x_boundary_mode == "periodic"):
                            ni = (ni_raw + Nx) % Nx
                        else:
                            if ni_raw < 0 or ni_raw >= Nx:
                                valid = False
                        nj = (j + cy[qn] + Ny) % Ny
                        nk = k + cz[qn]
                        if nk < 0 or nk >= Nz:
                            valid = False
                        if valid and lattice_open[nk, nj, ni][0] == 1:
                            take = False
                            weight = 0.0
                            room = 0.0
                            normal_dot = (
                                ti.cast(cx[qn], ti.f32) * nx
                                + ti.cast(cy[qn], ti.f32) * ny
                                + ti.cast(cz[qn], ti.f32) * nz
                            )
                            if excess_dir > 0:
                                if free_surface_type[nk, nj, ni] != FS_LIQUID:
                                    weight = normal_dot
                                    if excess > 0.0:
                                        rho_room = rho[nk, nj, ni]
                                        if rho_room < 1.0e-8:
                                            rho_room = rho0
                                        room = rho_room - free_surface_mass[nk, nj, ni]
                                    else:
                                        room = free_surface_mass[nk, nj, ni]
                                    if room > 1.0e-12:
                                        take = True
                            else:
                                if free_surface_type[nk, nj, ni] != FS_GAS:
                                    weight = -normal_dot
                                    if excess > 0.0:
                                        rho_room = rho[nk, nj, ni]
                                        if rho_room < 1.0e-8:
                                            rho_room = rho0
                                        room = rho_room - free_surface_mass[nk, nj, ni]
                                    else:
                                        room = free_surface_mass[nk, nj, ni]
                                    if room > 1.0e-12:
                                        take = True
                            if take:
                                share = excess * room / capacity_sum
                                if weight_sum > 1.0e-12 and weight > 0.0:
                                    share = excess * weight * room / weight_sum
                                elif weight_sum > 1.0e-12:
                                    share = 0.0
                                ti.atomic_add(free_surface_mass[nk, nj, ni], share)
                else:
                    ti.atomic_add(free_surface_mass[k, j, i], excess)
                free_surface_mass_excess[k, j, i] = 0.0
                free_surface_mass_excess_dir[k, j, i] = 0


@ti.kernel
def finalize_vof_classification_from_mass():
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        for k, j, i in free_surface_mass:
            free_surface_mass_excess[k, j, i] = 0.0
            free_surface_mass_excess_dir[k, j, i] = 0
            if lattice_open[k, j, i][0] == 0:
                free_surface_mass[k, j, i] = 0.0
                free_surface_fill[k, j, i] = 0.0
                free_surface_type[k, j, i] = FS_GAS
            else:
                rho_ref = rho[k, j, i]
                if rho_ref < 1.0e-8:
                    rho_ref = rho0
                m = free_surface_mass[k, j, i]
                excess = 0.0
                excess_dir = 0
                if m < 0.0:
                    excess = m
                    excess_dir = -1
                    m = 0.0
                    ti.atomic_add(free_surface_vof_excess_count[None], 1)
                if m > rho_ref:
                    excess = m - rho_ref
                    excess_dir = 1
                    m = rho_ref
                    ti.atomic_add(free_surface_vof_excess_count[None], 1)

                fill = m / rho_ref
                ctype = FS_INTERFACE
                if fill > 1.0e-4 and fill < vof_empty_fill_threshold:
                    excess = m
                    excess_dir = -1
                    m = 0.0
                    fill = 0.0
                    ctype = FS_GAS
                    ti.atomic_add(free_surface_vof_excess_count[None], 1)
                elif fill <= 1.0e-4:
                    fill = 0.0
                    ctype = FS_GAS
                elif fill >= 1.0 - 1.0e-4:
                    fill = 1.0
                    ctype = FS_LIQUID
                free_surface_mass[k, j, i] = m
                free_surface_mass_excess[k, j, i] = excess
                free_surface_mass_excess_dir[k, j, i] = excess_dir
                free_surface_fill[k, j, i] = fill
                free_surface_type[k, j, i] = ctype


@ti.kernel
def enforce_vof_interface_layer():
    # Maintain the FSLBM topology invariant that cells adjacent to gas are
    # treated as interface cells. A full liquid cell can be an interface cell
    # with fill=1 and mass=rho; this changes only the boundary interpretation,
    # not the water volume or physical mass.
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        for k, j, i in free_surface_type:
            if lattice_open[k, j, i][0] == 1 and free_surface_type[k, j, i] == FS_LIQUID:
                touches_gas = False
                for qn in ti.static(range(1, 19)):
                    ni_raw = i + cx[qn]
                    ni = ni_raw
                    valid = True
                    if ti.static(x_boundary_mode == "periodic"):
                        ni = (ni_raw + Nx) % Nx
                    else:
                        if ni_raw < 0 or ni_raw >= Nx:
                            valid = False
                    nj = (j + cy[qn] + Ny) % Ny
                    nk = k + cz[qn]
                    if nk < 0 or nk >= Nz:
                        valid = False
                    if valid:
                        if lattice_open[nk, nj, ni][0] == 1 and free_surface_type[nk, nj, ni] == FS_GAS:
                            touches_gas = True
                if touches_gas:
                    free_surface_type[k, j, i] = FS_INTERFACE
                    free_surface_fill[k, j, i] = 1.0
                    free_surface_mass[k, j, i] = rho[k, j, i]


@ti.kernel
def remove_orphan_vof_interface_cells():
    # Established FSLBM cleanup for "lonely" interface artifacts: a low-fill
    # interface cell with no wet D3Q19 neighbor is not a resolved free-surface
    # feature. Empty it only when its mass can be conservatively returned to
    # nearby resolved wet cells with available VOF capacity.
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        free_surface_vof_orphan_count[None] = 0
        for k, j, i in free_surface_type:
            if (
                lattice_open[k, j, i][0] == 1
                and free_surface_type[k, j, i] != FS_GAS
                and free_surface_fill[k, j, i] > vof_empty_fill_threshold
            ):
                wet_neighbors = 0
                resolved_neighbors = 0
                for qn in ti.static(range(1, 19)):
                    ni_raw = i + cx[qn]
                    ni = ni_raw
                    valid = True
                    if ti.static(x_boundary_mode == "periodic"):
                        ni = (ni_raw + Nx) % Nx
                    else:
                        if ni_raw < 0 or ni_raw >= Nx:
                            valid = False
                    nj = (j + cy[qn] + Ny) % Ny
                    nk = k + cz[qn]
                    if nk < 0 or nk >= Nz:
                        valid = False
                    if valid and lattice_open[nk, nj, ni][0] == 1:
                        if free_surface_type[nk, nj, ni] != FS_GAS:
                            wet_neighbors += 1
                            if free_surface_fill[nk, nj, ni] >= vof_orphan_empty_threshold:
                                resolved_neighbors += 1

                if resolved_neighbors == 0 and (
                    free_surface_fill[k, j, i] < vof_orphan_empty_threshold
                    or wet_neighbors <= vof_orphan_max_weak_neighbors
                ):
                    m = free_surface_mass[k, j, i]
                    capacity_weight_sum = 0.0
                    topology_weight_sum = 0.0
                    for qn in ti.static(range(1, 19)):
                        for dist in range(2, vof_orphan_search_radius + 1):
                            di = dist * cx[qn]
                            dj = dist * cy[qn]
                            dk = dist * cz[qn]
                            ni_raw = i + di
                            ni = ni_raw
                            valid = True
                            if ti.static(x_boundary_mode == "periodic"):
                                ni = (ni_raw + Nx) % Nx
                            else:
                                if ni_raw < 0 or ni_raw >= Nx:
                                    valid = False
                            nj = (j + dj + Ny) % Ny
                            nk = k + dk
                            if nk < 0 or nk >= Nz:
                                valid = False
                            if valid and lattice_open[nk, nj, ni][0] == 1:
                                if (
                                    free_surface_type[nk, nj, ni] != FS_GAS
                                    and free_surface_fill[nk, nj, ni] >= vof_orphan_empty_threshold
                                ):
                                    rho_room = rho[nk, nj, ni]
                                    if rho_room < 1.0e-8:
                                        rho_room = rho0
                                    room = rho_room - free_surface_mass[nk, nj, ni]
                                    dist2 = ti.cast(di * di + dj * dj + dk * dk, ti.f32)
                                    topology_weight_sum += free_surface_fill[nk, nj, ni] / dist2
                                    if room > 1.0e-12:
                                        capacity_weight_sum += room / dist2

                    if m > 1.0e-12 and (capacity_weight_sum > 1.0e-12 or topology_weight_sum > 1.0e-12):
                        free_surface_mass[k, j, i] = 0.0
                        free_surface_fill[k, j, i] = 0.0
                        free_surface_type[k, j, i] = FS_GAS
                        ti.atomic_add(free_surface_vof_orphan_count[None], 1)
                        for qn in ti.static(range(1, 19)):
                            for dist in range(2, vof_orphan_search_radius + 1):
                                di = dist * cx[qn]
                                dj = dist * cy[qn]
                                dk = dist * cz[qn]
                                ni_raw = i + di
                                ni = ni_raw
                                valid = True
                                if ti.static(x_boundary_mode == "periodic"):
                                    ni = (ni_raw + Nx) % Nx
                                else:
                                    if ni_raw < 0 or ni_raw >= Nx:
                                        valid = False
                                nj = (j + dj + Ny) % Ny
                                nk = k + dk
                                if nk < 0 or nk >= Nz:
                                    valid = False
                                if valid and lattice_open[nk, nj, ni][0] == 1:
                                    if (
                                        free_surface_type[nk, nj, ni] != FS_GAS
                                        and free_surface_fill[nk, nj, ni] >= vof_orphan_empty_threshold
                                    ):
                                        rho_room = rho[nk, nj, ni]
                                        if rho_room < 1.0e-8:
                                            rho_room = rho0
                                        room = rho_room - free_surface_mass[nk, nj, ni]
                                        dist2 = ti.cast(di * di + dj * dj + dk * dk, ti.f32)
                                        weight = 0.0
                                        denom = capacity_weight_sum
                                        if capacity_weight_sum > 1.0e-12:
                                            if room > 1.0e-12:
                                                weight = room / dist2
                                        else:
                                            weight = free_surface_fill[nk, nj, ni] / dist2
                                            denom = topology_weight_sum
                                        if denom > 1.0e-12 and weight > 0.0:
                                            ti.atomic_add(
                                                free_surface_mass[nk, nj, ni],
                                                m * weight / denom,
                                            )


@ti.func
def vof_unresolved_detached_cell(k: ti.i32, j: ti.i32, i: ti.i32) -> ti.i32:
    detached = 0
    if (
        lattice_open[k, j, i][0] == 1
        and free_surface_type[k, j, i] != FS_GAS
        and free_surface_fill[k, j, i] > 1.0e-4
    ):
        wet_neighbors = 0
        resolved_neighbors = 0
        for qn in ti.static(range(1, 19)):
            ni_raw = i + cx[qn]
            ni = ni_raw
            valid = True
            if ti.static(x_boundary_mode == "periodic"):
                ni = (ni_raw + Nx) % Nx
            else:
                if ni_raw < 0 or ni_raw >= Nx:
                    valid = False
            nj = (j + cy[qn] + Ny) % Ny
            nk = k + cz[qn]
            if nk < 0 or nk >= Nz:
                valid = False
            if valid and lattice_open[nk, nj, ni][0] == 1:
                if free_surface_type[nk, nj, ni] != FS_GAS:
                    wet_neighbors += 1
                    if free_surface_fill[nk, nj, ni] >= vof_orphan_empty_threshold:
                        resolved_neighbors += 1

        if (
            wet_neighbors <= vof_detached_max_wet_neighbors
            and resolved_neighbors <= vof_detached_max_resolved_neighbors
        ):
            detached = 1
    return detached


@ti.func
def vof_gas_neighbor_rate(k: ti.i32, j: ti.i32, i: ti.i32, di: ti.i32, dj: ti.i32, dk: ti.i32, rate: ti.f32) -> ti.f32:
    out_rate = 0.0
    if rate > 0.0:
        ni_raw = i + di
        ni = ni_raw
        valid = True
        if ti.static(x_boundary_mode == "periodic"):
            ni = (ni_raw + Nx) % Nx
        else:
            if ni_raw < 0 or ni_raw >= Nx:
                valid = False
        nj = (j + dj + Ny) % Ny
        nk = k + dk
        if nk < 0 or nk >= Nz:
            valid = False
        if valid and lattice_open[nk, nj, ni][0] == 1 and free_surface_type[nk, nj, ni] == FS_GAS:
            out_rate = rate
    return out_rate


@ti.func
def add_vof_detached_flux(k: ti.i32, j: ti.i32, i: ti.i32, di: ti.i32, dj: ti.i32, dk: ti.i32, rate: ti.f32, m: ti.f32, scale: ti.f32) -> ti.f32:
    moved = 0.0
    if rate > 0.0:
        ni_raw = i + di
        ni = ni_raw
        valid = True
        if ti.static(x_boundary_mode == "periodic"):
            ni = (ni_raw + Nx) % Nx
        else:
            if ni_raw < 0 or ni_raw >= Nx:
                valid = False
        nj = (j + dj + Ny) % Ny
        nk = k + dk
        if nk < 0 or nk >= Nz:
            valid = False
        if valid and lattice_open[nk, nj, ni][0] == 1 and free_surface_type[nk, nj, ni] == FS_GAS:
            moved = m * rate * scale
            if moved > 1.0e-12:
                ti.atomic_add(free_surface_detached_mass_delta[k, j, i], -moved)
                ti.atomic_add(free_surface_detached_mass_delta[nk, nj, ni], moved)
    return moved


@ti.kernel
def advect_detached_vof_cells():
    # Conservative kinematic transport for unresolved detached VOF remnants.
    # Standard FSLBM mass exchange has no wet neighbor for a lone interface cell,
    # so a tiny spray cell can have gravity/velocity but no geometric route to
    # move through inactive gas. This adds only upwind VOF fluxes from unresolved
    # detached wet cells into adjacent gas cells; connected/free-surface cells
    # continue to use the ordinary population-based mass exchange.
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        free_surface_vof_detached_advect_count[None] = 0
        free_surface_vof_detached_advect_mass[None] = 0.0
        for k, j, i in free_surface_detached_mass_delta:
            free_surface_detached_mass_delta[k, j, i] = 0.0

        if ti.static(vof_detached_advection_enabled):
            for k, j, i in free_surface_mass:
                if vof_unresolved_detached_cell(k, j, i) == 1:
                    m = free_surface_mass[k, j, i]
                    if m > 1.0e-12:
                        ux_loc = ux[k, j, i]
                        uy_loc = uy[k, j, i]
                        uz_loc = uz[k, j, i]

                        rxp = vof_gas_neighbor_rate(k, j, i, 1, 0, 0, ux_loc)
                        rxm = vof_gas_neighbor_rate(k, j, i, -1, 0, 0, -ux_loc)
                        ryp = vof_gas_neighbor_rate(k, j, i, 0, 1, 0, uy_loc)
                        rym = vof_gas_neighbor_rate(k, j, i, 0, -1, 0, -uy_loc)
                        rzp = vof_gas_neighbor_rate(k, j, i, 0, 0, 1, uz_loc)
                        rzm = vof_gas_neighbor_rate(k, j, i, 0, 0, -1, -uz_loc)

                        rate_sum = rxp + rxm + ryp + rym + rzp + rzm
                        if rate_sum > 1.0e-12:
                            # Positivity-preserving VOF CFL limiter. It only
                            # rescales transfer if the local velocity would move
                            # nearly all sub-cell liquid in one lattice step.
                            scale = 1.0
                            if rate_sum > 0.95:
                                scale = 0.95 / rate_sum

                            moved = 0.0
                            moved += add_vof_detached_flux(k, j, i, 1, 0, 0, rxp, m, scale)
                            moved += add_vof_detached_flux(k, j, i, -1, 0, 0, rxm, m, scale)
                            moved += add_vof_detached_flux(k, j, i, 0, 1, 0, ryp, m, scale)
                            moved += add_vof_detached_flux(k, j, i, 0, -1, 0, rym, m, scale)
                            moved += add_vof_detached_flux(k, j, i, 0, 0, 1, rzp, m, scale)
                            moved += add_vof_detached_flux(k, j, i, 0, 0, -1, rzm, m, scale)
                            if moved > 1.0e-12:
                                ti.atomic_add(free_surface_vof_detached_advect_count[None], 1)
                                ti.atomic_add(free_surface_vof_detached_advect_mass[None], moved)

            for k, j, i in free_surface_mass:
                dm = free_surface_detached_mass_delta[k, j, i]
                if ti.abs(dm) > 1.0e-12:
                    ti.atomic_add(free_surface_mass[k, j, i], dm)


@ti.kernel
def compress_detached_vof_residuals():
    # Interface-compression cleanup for the numerical tail left by unresolved
    # detached VOF cells. This is a geometry remap, not dissipation: the residual
    # liquid mass is moved into the neighboring gas cell selected by the local
    # velocity direction and remains in the VOF mass budget.
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        free_surface_vof_detached_compress_count[None] = 0
        free_surface_vof_detached_compress_mass[None] = 0.0
        for k, j, i in free_surface_detached_mass_delta:
            free_surface_detached_mass_delta[k, j, i] = 0.0

        if ti.static(vof_detached_advection_enabled and vof_detached_residual_fill_threshold > 0.0):
            for k, j, i in free_surface_mass:
                if vof_unresolved_detached_cell(k, j, i) == 1:
                    rho_ref = rho[k, j, i]
                    if rho_ref < 1.0e-8:
                        rho_ref = rho0
                    m = free_surface_mass[k, j, i]
                    fill = m / rho_ref
                    if fill > 1.0e-4 and fill < vof_detached_residual_fill_threshold:
                        ux_loc = ux[k, j, i]
                        uy_loc = uy[k, j, i]
                        uz_loc = uz[k, j, i]

                        best_di = ti.cast(0, ti.i32)
                        best_dj = ti.cast(0, ti.i32)
                        best_dk = ti.cast(0, ti.i32)
                        best_rate = 0.0

                        rxp = vof_gas_neighbor_rate(k, j, i, 1, 0, 0, ux_loc)
                        if rxp > best_rate:
                            best_rate = rxp
                            best_di = 1
                            best_dj = 0
                            best_dk = 0
                        rxm = vof_gas_neighbor_rate(k, j, i, -1, 0, 0, -ux_loc)
                        if rxm > best_rate:
                            best_rate = rxm
                            best_di = -1
                            best_dj = 0
                            best_dk = 0
                        ryp = vof_gas_neighbor_rate(k, j, i, 0, 1, 0, uy_loc)
                        if ryp > best_rate:
                            best_rate = ryp
                            best_di = 0
                            best_dj = 1
                            best_dk = 0
                        rym = vof_gas_neighbor_rate(k, j, i, 0, -1, 0, -uy_loc)
                        if rym > best_rate:
                            best_rate = rym
                            best_di = 0
                            best_dj = -1
                            best_dk = 0
                        rzp = vof_gas_neighbor_rate(k, j, i, 0, 0, 1, uz_loc)
                        if rzp > best_rate:
                            best_rate = rzp
                            best_di = 0
                            best_dj = 0
                            best_dk = 1
                        rzm = vof_gas_neighbor_rate(k, j, i, 0, 0, -1, -uz_loc)
                        if rzm > best_rate:
                            best_rate = rzm
                            best_di = 0
                            best_dj = 0
                            best_dk = -1

                        if best_rate > 1.0e-12:
                            moved = add_vof_detached_flux(k, j, i, best_di, best_dj, best_dk, 1.0, m, 1.0)
                            if moved > 1.0e-12:
                                ti.atomic_add(free_surface_vof_detached_compress_count[None], 1)
                                ti.atomic_add(free_surface_vof_detached_compress_mass[None], moved)

            for k, j, i in free_surface_mass:
                dm = free_surface_detached_mass_delta[k, j, i]
                if ti.abs(dm) > 1.0e-12:
                    ti.atomic_add(free_surface_mass[k, j, i], dm)


@ti.func
def vof_gas_fraction_at(k: ti.i32, j: ti.i32, i: ti.i32) -> ti.f32:
    gas_frac = 0.0
    if lattice_open[k, j, i][0] == 1:
        if free_surface_type[k, j, i] == FS_GAS:
            gas_frac = 1.0
        elif free_surface_fill[k, j, i] < 1.0:
            gas_frac = 1.0 - free_surface_fill[k, j, i]
            if gas_frac < 0.0:
                gas_frac = 0.0
            if gas_frac > 1.0:
                gas_frac = 1.0
    return gas_frac


@ti.kernel
def clear_vof_collapse_diagnostics():
    free_surface_vof_collapse_cell_count[None] = 0
    free_surface_vof_collapse_candidate_volume_m3[None] = 0.0
    free_surface_vof_collapse_applied_volume_m3[None] = 0.0
    free_surface_vof_collapse_removal_scale[None] = 0.0
    free_surface_vof_collapse_candidate_mass[None] = 0.0
    free_surface_vof_collapse_exterior_capacity[None] = 0.0


@ti.kernel
def initialize_vof_exterior_gas_marks():
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        for k, j, i in vof_exterior_gas:
            gas_frac = vof_gas_fraction_at(k, j, i)
            mark = 0
            if gas_frac > 1.0e-6 and k == Nz - 1:
                mark = 1
            vof_exterior_gas[k, j, i] = mark


@ti.kernel
def propagate_vof_exterior_gas_marks():
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        for k, j, i in vof_exterior_gas:
            if vof_exterior_gas[k, j, i] == 0 and vof_gas_fraction_at(k, j, i) > 1.0e-6:
                connected = False
                for q in ti.static(range(1, 7)):
                    ni_raw = i + cx[q]
                    ni = ni_raw
                    valid = True
                    if ti.static(x_boundary_mode == "periodic"):
                        ni = (ni_raw + Nx) % Nx
                    else:
                        if ni_raw < 0 or ni_raw >= Nx:
                            valid = False
                    nj = (j + cy[q] + Ny) % Ny
                    nk = k + cz[q]
                    if nk < 0 or nk >= Nz:
                        valid = False
                    if valid and vof_exterior_gas[nk, nj, ni] == 1:
                        connected = True
                if connected:
                    vof_exterior_gas[k, j, i] = 1


@ti.kernel
def summarize_vof_trapped_gas_collapse():
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        free_surface_vof_collapse_candidate_mass[None] = 0.0
        free_surface_vof_collapse_candidate_volume_m3[None] = 0.0
        free_surface_vof_collapse_exterior_capacity[None] = 0.0
        for k, j, i in free_surface_fill:
            gas_frac = vof_gas_fraction_at(k, j, i)
            if gas_frac > 1.0e-6:
                if vof_exterior_gas[k, j, i] == 0:
                    rho_ref = rho[k, j, i]
                    if rho_ref < 1.0e-8:
                        rho_ref = rho0
                    ti.atomic_add(free_surface_vof_collapse_candidate_mass[None], gas_frac * rho_ref)
                    ti.atomic_add(free_surface_vof_collapse_candidate_volume_m3[None], gas_frac * dx * dx * dx)
                elif free_surface_type[k, j, i] != FS_GAS:
                    m = free_surface_mass[k, j, i]
                    if m > 0.0:
                        ti.atomic_add(free_surface_vof_collapse_exterior_capacity[None], m)

        free_surface_vof_collapse_applied_volume_m3[None] = 0.0
        free_surface_vof_collapse_removal_scale[None] = 0.0
        free_surface_vof_collapse_cell_count[None] = 0
        candidate_mass = free_surface_vof_collapse_candidate_mass[None]
        candidate_volume = free_surface_vof_collapse_candidate_volume_m3[None]
        exterior_removal_capacity = free_surface_vof_collapse_exterior_capacity[None]
        if (
            candidate_volume > 0.0
            and candidate_volume <= vof_collapse_max_volume_m3
            and exterior_removal_capacity >= candidate_mass
            and exterior_removal_capacity > 1.0e-12
        ):
            free_surface_vof_collapse_applied_volume_m3[None] = candidate_volume
            free_surface_vof_collapse_removal_scale[None] = candidate_mass / exterior_removal_capacity


@ti.kernel
def apply_vof_trapped_gas_collapse():
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        scale = free_surface_vof_collapse_removal_scale[None]
        for k, j, i in free_surface_mass:
            if scale > 0.0:
                gas_frac = vof_gas_fraction_at(k, j, i)
                if gas_frac > 1.0e-6:
                    rho_ref = rho[k, j, i]
                    if rho_ref < 1.0e-8:
                        rho_ref = rho0

                    if vof_exterior_gas[k, j, i] == 0:
                        free_surface_mass[k, j, i] = rho_ref
                        free_surface_fill[k, j, i] = 1.0
                        free_surface_type[k, j, i] = FS_LIQUID
                        ti.atomic_add(free_surface_vof_collapse_cell_count[None], 1)
                    elif free_surface_type[k, j, i] != FS_GAS:
                        m = free_surface_mass[k, j, i] * (1.0 - scale)
                        if m < 0.0:
                            m = 0.0
                        fill = m / rho_ref
                        ctype = FS_INTERFACE
                        if fill <= 1.0e-4:
                            fill = 0.0
                            m = 0.0
                            ctype = FS_GAS
                        elif fill >= 1.0 - 1.0e-4:
                            fill = 1.0
                            m = rho_ref
                            ctype = FS_LIQUID
                        free_surface_mass[k, j, i] = m
                        free_surface_fill[k, j, i] = fill
                        free_surface_type[k, j, i] = ctype


def collapse_trapped_gas_if_due(step: int):
    clear_vof_collapse_diagnostics()
    if (
        not vof_collapse_trapped_gas_enabled
        or water_free_surface_enabled != 1
        or free_surface_tracking_mode != "vof"
        or step % vof_collapse_interval != 0
    ):
        return

    initialize_vof_exterior_gas_marks()
    for _ in range(vof_collapse_flood_sweeps):
        propagate_vof_exterior_gas_marks()
    summarize_vof_trapped_gas_collapse()
    apply_vof_trapped_gas_collapse()


@ti.kernel
def update_free_surface_height_from_vof():
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        for j, i in free_surface_h:
            h_old = free_surface_h[j, i]
            column_depth = 0.0
            for k in range(Nz):
                if lattice_open[k, j, i][0] == 1:
                    column_depth += free_surface_fill[k, j, i] * dx
            free_surface_h_last[j, i] = h_old
            free_surface_h[j, i] = water_h[j, i] + column_depth

@ti.kernel
def refill_new_free_surface_cells():
    if ti.static(water_free_surface_enabled == 1):
        free_surface_refill_count[None] = 0
        for k, j, i in f:
            if free_surface_type_prev[k, j, i] == FS_GAS and free_surface_type[k, j, i] != FS_GAS and lattice_open[k, j, i][0] == 1:
                weight_sum = 0.0
                fallback_weight_sum = 0.0
                rho_acc = 0.0
                ux_acc = 0.0
                uy_acc = 0.0
                uz_acc = 0.0
                rho_fallback_acc = 0.0
                ux_fallback_acc = 0.0
                uy_fallback_acc = 0.0
                uz_fallback_acc = 0.0
                fneq_acc = ti.Vector.zero(ti.f32, 19)
                fneq_fallback_acc = ti.Vector.zero(ti.f32, 19)

                for qn in ti.static(range(1, 19)):
                    ni_raw = i + cx[qn]
                    nj = (j + cy[qn] + Ny) % Ny
                    nk = k + cz[qn]
                    ni = ni_raw
                    valid = True
                    if ti.static(x_boundary_mode == "periodic"):
                        ni = (ni_raw + Nx) % Nx
                    else:
                        if ni_raw < 0 or ni_raw >= Nx:
                            valid = False
                    if nk < 0 or nk >= Nz:
                        valid = False

                    if valid:
                        if lattice_open[nk, nj, ni][0] == 1 and free_surface_type_prev[nk, nj, ni] != FS_GAS and free_surface_type[nk, nj, ni] != FS_GAS:
                            wn = w[qn]
                            fill_n = free_surface_fill[nk, nj, ni]
                            rho_n = rho[nk, nj, ni]
                            ux_n = ux[nk, nj, ni]
                            uy_n = uy[nk, nj, ni]
                            uz_n = uz[nk, nj, ni]
                            fallback_weight_sum += wn
                            rho_fallback_acc += wn * rho_n
                            ux_fallback_acc += wn * ux_n
                            uy_fallback_acc += wn * uy_n
                            uz_fallback_acc += wn * uz_n
                            liquid_weight = wn * fill_n
                            weight_sum += liquid_weight
                            rho_acc += liquid_weight * rho_n
                            ux_acc += liquid_weight * ux_n
                            uy_acc += liquid_weight * uy_n
                            uz_acc += liquid_weight * uz_n
                            if ti.static(free_surface_refill_mode == "noneq"):
                                for qq in ti.static(range(19)):
                                    fneq_n = (
                                        f[nk, nj, ni][qq]
                                        - calculate_feq(rho_n, ux_n, uy_n, uz_n, qq)
                                    )
                                    fneq_acc[qq] += liquid_weight * fneq_n
                                    fneq_fallback_acc[qq] += wn * fneq_n

                rho_refill = rho0
                if ti.static(free_surface_tracking_mode != "vof"):
                    rho_refill = free_surface_hydrostatic_density_at_cell(k, j, i)
                ux_refill = u_current_lb
                uy_refill = 0.0
                uz_refill = 0.0

                if weight_sum > 0.0:
                    inv_w = 1.0 / weight_sum
                    if ti.static(free_surface_tracking_mode == "vof"):
                        rho_refill = rho_acc * inv_w
                    ux_refill = ux_acc * inv_w
                    uy_refill = uy_acc * inv_w
                    uz_refill = uz_acc * inv_w
                elif fallback_weight_sum > 0.0:
                    inv_w = 1.0 / fallback_weight_sum
                    if ti.static(free_surface_tracking_mode == "vof"):
                        rho_refill = rho_fallback_acc * inv_w
                    ux_refill = ux_fallback_acc * inv_w
                    uy_refill = uy_fallback_acc * inv_w
                    uz_refill = uz_fallback_acc * inv_w
                elif ti.static(free_surface_tracking_mode == "vof"):
                    rho_refill = rho0

                f_refill_vec = ti.Vector.zero(ti.f32, 19)
                rho_from_pop = 0.0
                mx_from_pop = 0.0
                my_from_pop = 0.0
                mz_from_pop = 0.0
                for q in ti.static(range(19)):
                    f_refill = calculate_feq(rho_refill, ux_refill, uy_refill, uz_refill, q)
                    if ti.static(free_surface_refill_mode == "noneq"):
                        if weight_sum > 0.0:
                            f_refill += fneq_acc[q] / weight_sum
                        elif fallback_weight_sum > 0.0:
                            f_refill += fneq_fallback_acc[q] / fallback_weight_sum
                    elif ti.static(free_surface_refill_mode == "directional"):
                        src_i_raw = i - cx[q]
                        src_i = src_i_raw
                        src_j = (j - cy[q] + Ny) % Ny
                        src_k = k - cz[q]
                        src_valid = True
                        if ti.static(x_boundary_mode == "periodic"):
                            src_i = (src_i_raw + Nx) % Nx
                        else:
                            if src_i_raw < 0 or src_i_raw >= Nx:
                                src_valid = False
                        if src_k < 0 or src_k >= Nz:
                            src_valid = False

                        if src_valid:
                            if (
                                lattice_open[src_k, src_j, src_i][0] == 1
                                and free_surface_type_prev[src_k, src_j, src_i] != FS_GAS
                                and free_surface_type[src_k, src_j, src_i] != FS_GAS
                            ):
                                f_refill = f[src_k, src_j, src_i][q]

                    f_refill_vec[q] = f_refill
                    rho_from_pop += f_refill

                if ti.static(free_surface_refill_mode == "directional"):
                    scale = 1.0
                    if rho_from_pop > 1.0e-8:
                        scale = rho_refill / rho_from_pop
                    rho_from_pop = 0.0
                    for q in ti.static(range(19)):
                        f_val = f_refill_vec[q] * scale
                        f[k, j, i][q] = f_val
                        rho_from_pop += f_val
                        mx_from_pop += f_val * ti.cast(cx[q], ti.f32)
                        my_from_pop += f_val * ti.cast(cy[q], ti.f32)
                        mz_from_pop += f_val * ti.cast(cz[q], ti.f32)
                    rho[k, j, i] = rho_from_pop
                    if rho_from_pop > 1.0e-8:
                        ux[k, j, i] = mx_from_pop / rho_from_pop
                        uy[k, j, i] = my_from_pop / rho_from_pop
                        uz[k, j, i] = mz_from_pop / rho_from_pop
                    else:
                        ux[k, j, i] = ux_refill
                        uy[k, j, i] = uy_refill
                        uz[k, j, i] = uz_refill
                else:
                    rho[k, j, i] = rho_refill
                    ux[k, j, i] = ux_refill
                    uy[k, j, i] = uy_refill
                    uz[k, j, i] = uz_refill
                    for q in ti.static(range(19)):
                        f[k, j, i][q] = f_refill_vec[q]
                ti.atomic_add(free_surface_refill_count[None], 1)


@ti.kernel
def repair_vof_coalescence_links():
    # When a VOF gas gap closes, populations that were reconstructed by the
    # atmospheric free-surface boundary on the previous step become ordinary
    # liquid-link populations. Replace only those stale gas-facing populations
    # with the newly connected wet donor along the same characteristic.
    if ti.static(water_free_surface_enabled == 1 and free_surface_tracking_mode == "vof"):
        free_surface_coalescence_repair_count[None] = 0
        for k, j, i in f:
            if (
                lattice_open[k, j, i][0] == 1
                and free_surface_type[k, j, i] != FS_GAS
                and free_surface_type_prev[k, j, i] != FS_GAS
            ):
                repaired = False
                for q in ti.static(range(1, 19)):
                    src_i_raw = i - cx[q]
                    src_i = src_i_raw
                    src_j = (j - cy[q] + Ny) % Ny
                    src_k = k - cz[q]
                    src_valid = True
                    if ti.static(x_boundary_mode == "periodic"):
                        src_i = (src_i_raw + Nx) % Nx
                    else:
                        if src_i_raw < 0 or src_i_raw >= Nx:
                            src_valid = False
                    if src_k < 0 or src_k >= Nz:
                        src_valid = False

                    if src_valid:
                        if (
                            lattice_open[src_k, src_j, src_i][0] == 1
                            and free_surface_type_prev[src_k, src_j, src_i] == FS_GAS
                            and free_surface_type[src_k, src_j, src_i] != FS_GAS
                        ):
                            f[k, j, i][q] = f[src_k, src_j, src_i][q]
                            repaired = True
                            ti.atomic_add(free_surface_coalescence_repair_count[None], 1)

                if repaired:
                    rho_new = 0.0
                    mx_new = 0.0
                    my_new = 0.0
                    mz_new = 0.0
                    for q in ti.static(range(19)):
                        fq = f[k, j, i][q]
                        rho_new += fq
                        mx_new += fq * ti.cast(cx[q], ti.f32)
                        my_new += fq * ti.cast(cy[q], ti.f32)
                        mz_new += fq * ti.cast(cz[q], ti.f32)
                    rho[k, j, i] = rho_new
                    if rho_new > 1.0e-8:
                        ux[k, j, i] = mx_new / rho_new
                        uy[k, j, i] = my_new / rho_new
                        uz[k, j, i] = mz_new / rho_new


@ti.kernel
def compute_free_surface_column_flux():
    if ti.static(water_free_surface_enabled == 1):
        for j, i in free_surface_flux_x:
            qx = 0.0
            qy = 0.0
            for k in range(Nz):
                if lattice_open[k, j, i][0] == 1 and free_surface_type[k, j, i] != FS_GAS:
                    fill = free_surface_fill[k, j, i]
                    qx += fill * ux[k, j, i] * vel_scale * dx
                    qy += fill * uy[k, j, i] * vel_scale * dx
            free_surface_flux_x[j, i] = qx
            free_surface_flux_y[j, i] = qy


@ti.kernel
def advect_free_surface_height(t: ti.f32):
    if ti.static(water_free_surface_enabled == 1):
        Lz = ti.cast(Nz, ti.f32) * dx
        for j, i in free_surface_h:
            h_old = free_surface_h[j, i]
            h_new = h_old
            if ti.static(free_surface_mode == "prescribed_solitary"):
                x = (ti.cast(i, ti.f32) + 0.5) * dx
                h_new = solitary_surface_height_at_x(x, t)
            elif ti.static(free_surface_mode == "height_kinematic"):
                dq_dx = 0.0
                if ti.static(x_boundary_mode == "solid"):
                    flux_l = 0.0
                    flux_r = 0.0
                    if i > 0:
                        flux_l = 0.5 * (free_surface_flux_x[j, i - 1] + free_surface_flux_x[j, i])
                    if i < Nx - 1:
                        flux_r = 0.5 * (free_surface_flux_x[j, i] + free_surface_flux_x[j, i + 1])
                    dq_dx = (flux_r - flux_l) / dx
                else:
                    ip = (i + 1) % Nx
                    im = (i - 1 + Nx) % Nx
                    dq_dx = (free_surface_flux_x[j, ip] - free_surface_flux_x[j, im]) * (0.5 / dx)
                jp = (j + 1) % Ny
                jm = (j - 1 + Ny) % Ny
                dq_dy = (free_surface_flux_y[jp, i] - free_surface_flux_y[jm, i]) * (0.5 / dx)
                dh_dt = -(dq_dx + dq_dy)
                h_new = h_old + dt_phys * dh_dt

                min_h = water_h[j, i] + 0.5 * dx
                max_h = Lz - 0.5 * dx
                if h_new < min_h:
                    h_new = min_h
                if h_new > max_h:
                    h_new = max_h

            free_surface_h_last[j, i] = h_old
            free_surface_h[j, i] = h_new


@ti.kernel # compute signed distance function with slope correction
def compute_phi_slope_corrected():
    # -----------------------------------------------------------------------------
    # Signed-distance (level-set-like) field construction for the immersed boundary
    #
    # phi(k,j,i) is positive in air and negative inside the “water/solid” region.
    # A slope correction converts vertical distance to approximate distance along
    # the local surface normal, improving interpolation when the interface is
    # inclined relative to the grid.
    # In current code, phi is only used by the wall model in the LES closure
    #
    # Steps:
    # 1) Compute dh/dx and dh/dy using periodic central differences on water_h.
    # 2) Compute denom = sqrt(1 + (dh/dx)^2 + (dh/dy)^2), proportional to |n|^{-1}.
    # 3) For each vertical index k, compute the signed normal distance:
    #       zc = (k+0.5)*dx is the cell-center height
    #       phi = (zc - h_loc) / denom
    # -----------------------------------------------------------------------------
    for j, i in water_h:  # Parallel loop over index space (Taichi SPMD)
        # Periodic finite differences in x and y
        ip = (i + 1) % Nx  # Periodic +x neighbor index (i+1 mod Nx)
        im = (i - 1 + Nx) % Nx  # Periodic -x neighbor index (i-1 mod Nx)
        jp = (j + 1) % Ny  # Periodic +y neighbor index (j+1 mod Ny)
        jm = (j - 1 + Ny) % Ny  # Periodic -y neighbor index (j-1 mod Ny)

        # central differences of surface height h(x,y)
        # h = offset + eta, so dh/dx == deta/dx
        dh_dx = (water_h[j, ip] - water_h[j, im]) * (0.5 / dx)  # Central-difference ∂h/∂x slope of prescribed interface
        if ti.static(x_boundary_mode == "solid"):
            if i == 0:
                dh_dx = (water_h[j, i + 1] - water_h[j, i]) / dx
            elif i == Nx - 1:
                dh_dx = (water_h[j, i] - water_h[j, i - 1]) / dx
        dh_dy = (water_h[jp, i] - water_h[jm, i]) * (0.5 / dx)  # Central-difference ∂h/∂y slope of prescribed interface
        # 2D normal-distance denominator
        denom = ti.sqrt(1.0 + dh_dx * dh_dx + dh_dy * dh_dy)  # Normalization sqrt(1+|∇h|^2) for slope-corrected normal distance

        # avoid pathological tiny denom (shouldn't happen, but safe)
        if denom < 1e-6:  # Guard against degenerate slope norm (avoid divide-by-zero)
            denom = 1.0  # Normalization sqrt(1+|∇h|^2) for slope-corrected normal distance

        # signed distance along the local normal
        h_loc = water_h[j, i]  # Local prescribed interface height h(x,y,t) (m)
        for k in range(Nz):  # Parallel loop over index space (Taichi SPMD)
            zc = (ti.cast(k, ti.f32) + 0.5) * dx  # Cell-center vertical coordinate (m) at level k
            xc = (ti.cast(i, ti.f32) + 0.5) * dx
            yc = (ti.cast(j, ti.f32) + 0.5) * dx
            phi_val = 1.0e6
            if ti.static(include_bed_geometry == 1):
                phi_val = (zc - h_loc) / denom
            if ti.static(include_cube_geometry == 1):
                cube_phi = cube_sdf_at_point(xc, yc, zc)
                if cube_phi < phi_val:
                    phi_val = cube_phi
            if ti.static(x_boundary_mode == "solid"):
                x_wall_phi = ti.min(xc, ti.cast(Nx, ti.f32) * dx - xc)
                if x_wall_phi < phi_val:
                    phi_val = x_wall_phi
            phi[k, j, i] = phi_val  # Scalar 'phi' for this kernel block

@ti.kernel
def build_lattice_open_from_free_surface_periodic():
    # -----------------------------------------------------------------------------
    # Build per-direction open/blocked flags and open fractions from the free surface η(x,y,t),
    # using periodic bilinear sampling of the height field water_h(y,x).
    #
    # For each cell center (xc,yc,zc) and each D3Q19 direction q:
    # 1) Evaluate the link half-step endpoint:
    #       (xq,yq,zq) = (xc,yc,zc) + 0.5*dx*(cx[q],cy[q],cz[q])
    # 2) Wrap (xq,yq) periodically into [0,Lx)×[0,Ly), convert to continuous indices (u,v),
    #    and bilinearly interpolate η = water_h at (xq,yq).
    # 3) Form continuous openness from the vertical gap:
    #       gap = zq - η
    #       frac = clamp(0.5 + gap/dx, 0, 1)    -> lattice_open_frac[...,q]
    # 4) Form a binary open flag (your “Option B” hard test):
    #       lo[q] = 1 if (η <= zq) else 0       -> lattice_open[...,q]
    #
    # Diagnostics / cell-wise indicators:
    # - lf[0] stores the same frac mapping at the cell center (q=0 diagnostic).
    # - lo[0] is set to 1 if any lo[q]==1 (cell has at least one open link); else 0.
    # - near_obstacle[k,j,i] = 1 if any lo[q]==0 (cell touches the interface/blocked link).
    # - If lo[0]==0, optionally force lf[q]=0 for all q to prevent porous “solid” cells.
    #
    # Outputs: lattice_open, lattice_open_frac, near_obstacle.
    # -----------------------------------------------------------------------------


    Lx = ti.cast(Nx, ti.f32) * dx  # Type cast for Taichi kernel arithmetic / indexing
    Ly = ti.cast(Ny, ti.f32) * dx  # Type cast for Taichi kernel arithmetic / indexing
    Nx_f = ti.cast(Nx, ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
    Ny_f = ti.cast(Ny, ti.f32)  # Type cast for Taichi kernel arithmetic / indexing

    for k, j, i in ti.ndrange(Nz, Ny, Nx):  # Parallel sweep over full 3D domain via ti.ndrange (SPMD)

        # Cell-center coordinates
        xc = (ti.cast(i, ti.f32) + 0.5) * dx  # Type cast for Taichi kernel arithmetic / indexing
        yc = (ti.cast(j, ti.f32) + 0.5) * dx  # Type cast for Taichi kernel arithmetic / indexing
        zc = (ti.cast(k, ti.f32) + 0.5) * dx  # Cell-center vertical coordinate (m) at level k

        lo = ti.Vector.zero(ti.i32, 19)  # Binary open/blocked flags per direction q for this cell (1=open link)
        lf = ti.Vector.zero(ti.f32, 19)  # Continuous open fraction per direction q in [0,1] (1=open, 0=blocked)
        wt = ti.Vector.zero(ti.i32, 19)  # Nearest wall source per direction q

        # q=0: you can define a cell "fluid fraction" at the center (useful diagnostic)
        # sample eta at (xc,yc)
        xq = xc  # Intermediate scalar 'xq' for this kernel block
        yq = yc  # Intermediate scalar 'yq' for this kernel block

        xw = xq - ti.floor(xq / Lx) * Lx  # Integer/fractional decomposition for periodic bilinear sampling
        yw = yq - ti.floor(yq / Ly) * Ly  # Integer/fractional decomposition for periodic bilinear sampling

        u = xw / dx - 0.5  # Intermediate scalar 'u' for this kernel block
        v = yw / dx - 0.5  # Intermediate scalar 'v' for this kernel block

        u = u - ti.floor(u / Nx_f) * Nx_f  # Integer/fractional decomposition for periodic bilinear sampling
        v = v - ti.floor(v / Ny_f) * Ny_f  # Integer/fractional decomposition for periodic bilinear sampling

        i0 = ti.floor(u, ti.i32)  # Integer/fractional decomposition for periodic bilinear sampling
        j0 = ti.floor(v, ti.i32)  # Integer/fractional decomposition for periodic bilinear sampling
        i1 = (i0 + 1) % Nx  # Intermediate scalar 'i1' for this kernel block
        j1 = (j0 + 1) % Ny  # Intermediate scalar 'j1' for this kernel block

        fx = u - ti.cast(i0, ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
        fy = v - ti.cast(j0, ti.f32)  # Type cast for Taichi kernel arithmetic / indexing

        h00 = water_h[j0, i0]  # Intermediate scalar 'h00' for this kernel block
        h10 = water_h[j0, i1]  # Intermediate scalar 'h10' for this kernel block
        h01 = water_h[j1, i0]  # Intermediate scalar 'h01' for this kernel block
        h11 = water_h[j1, i1]  # Intermediate scalar 'h11' for this kernel block

        hx0 = h00 * (1.0 - fx) + h10 * fx  # Intermediate scalar 'hx0' for this kernel block
        hx1 = h01 * (1.0 - fx) + h11 * fx  # Intermediate scalar 'hx1' for this kernel block
        eta_c = hx0 * (1.0 - fy) + hx1 * fy  # Intermediate scalar 'eta_c' for this kernel block

        gap_c = zc - eta_c  # Vertical gap between cell center z_c and interface η(x_c,y_c,t) (m)
        geometry_gap_c = 1.0e6
        wall_type_c = WALL_NONE
        if ti.static(include_bed_geometry == 1):
            geometry_gap_c = gap_c
            wall_type_c = WALL_BED
        if ti.static(include_cube_geometry == 1):
            cube_gap_c = cube_sdf_at_point(xc, yc, zc)
            if cube_gap_c < geometry_gap_c:
                geometry_gap_c = cube_gap_c
                wall_type_c = WALL_CUBE
        signed_gap_c = geometry_gap_c
        if ti.static(use_phi_open_fraction == 1):
            signed_gap_c = phi[k, j, i]
        frac_c = 0.5 + signed_gap_c / dx  # Cell-center open fraction in [0,1] (diagnostic / q=0)
        if frac_c < 0.0:  # Branch for boundary/stability logic
            frac_c = 0.0  # Cell-center open fraction in [0,1] (diagnostic / q=0)
        if frac_c > 1.0:  # Branch for boundary/stability logic
            frac_c = 1.0  # Cell-center open fraction in [0,1] (diagnostic / q=0)

        lf[0] = frac_c  # Continuous open fraction per direction q in [0,1] (1=open, 0=blocked)
        wt[0] = wall_type_c

        # directions
        for q in ti.static(range(1, 19)):  # Unrolled loop over lattice directions q (compile-time static)

            # Endpoint at face midpoint (half-step) - unchanged
            xq = xc + 0.5 * dx * ti.cast(cx[q], ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
            yq = yc + 0.5 * dx * ti.cast(cy[q], ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
            zq = zc + 0.5 * dx * ti.cast(cz[q], ti.f32)  # Type cast for Taichi kernel arithmetic / indexing

            # --- periodic wrap of physical coordinates into [0, L) ---
            xw = xq - ti.floor(xq / Lx) * Lx  # Integer/fractional decomposition for periodic bilinear sampling
            yw = yq - ti.floor(yq / Ly) * Ly  # Integer/fractional decomposition for periodic bilinear sampling

            # --- convert to continuous index space for center-sampled water_h ---
            u = xw / dx - 0.5  # Intermediate scalar 'u' for this kernel block
            v = yw / dx - 0.5  # Intermediate scalar 'v' for this kernel block

            u = u - ti.floor(u / Nx_f) * Nx_f  # Integer/fractional decomposition for periodic bilinear sampling
            v = v - ti.floor(v / Ny_f) * Ny_f  # Integer/fractional decomposition for periodic bilinear sampling

            i0 = ti.floor(u, ti.i32)  # Integer/fractional decomposition for periodic bilinear sampling
            j0 = ti.floor(v, ti.i32)  # Integer/fractional decomposition for periodic bilinear sampling
            i1 = (i0 + 1) % Nx  # Intermediate scalar 'i1' for this kernel block
            j1 = (j0 + 1) % Ny  # Intermediate scalar 'j1' for this kernel block

            fx = u - ti.cast(i0, ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
            fy = v - ti.cast(j0, ti.f32)  # Type cast for Taichi kernel arithmetic / indexing

            # --- bilinear interpolation of η ---
            h00 = water_h[j0, i0]  # Intermediate scalar 'h00' for this kernel block
            h10 = water_h[j0, i1]  # Intermediate scalar 'h10' for this kernel block
            h01 = water_h[j1, i0]  # Intermediate scalar 'h01' for this kernel block
            h11 = water_h[j1, i1]  # Intermediate scalar 'h11' for this kernel block

            hx0 = h00 * (1.0 - fx) + h10 * fx  # Intermediate scalar 'hx0' for this kernel block
            hx1 = h01 * (1.0 - fx) + h11 * fx  # Intermediate scalar 'hx1' for this kernel block
            eta = hx0 * (1.0 - fy) + hx1 * fy  # Interface elevation perturbation η(x,y,t) (m)

            # --- open fraction from endpoint gap (continuous) ---
            gap = zq - eta  # Vertical gap between link endpoint z_q and interface η(x_q,y_q,t) (m)
            geometry_gap = 1.0e6
            wall_type_q = WALL_NONE
            if ti.static(include_bed_geometry == 1):
                geometry_gap = gap
                wall_type_q = WALL_BED
            if ti.static(include_cube_geometry == 1):
                cube_gap = cube_sdf_at_point(xq, yq, zq)
                if cube_gap < geometry_gap:
                    geometry_gap = cube_gap
                    wall_type_q = WALL_CUBE
            signed_gap = geometry_gap
            if ti.static(use_phi_open_fraction == 1):
                signed_gap = sample_phi_periodic(xq, yq, zq)
            frac = 0.5 + signed_gap / dx  # Open-link fraction in [0,1] from endpoint gap vs. interface height
            if frac < 0.0:  # Clamp open fraction to [0,1] for stability
                frac = 0.0  # Open-link fraction in [0,1] from endpoint gap vs. interface height
            if frac > 1.0:  # Clamp open fraction to [0,1] for stability
                frac = 1.0  # Open-link fraction in [0,1] from endpoint gap vs. interface height

            lf[q] = frac  # Continuous open fraction per direction q in [0,1] (1=open, 0=blocked)
            wt[q] = wall_type_q

            # --- binary open/blocked ---
            open_flag_gap = geometry_gap
            if ti.static(use_phi_open_flags == 1):
                open_flag_gap = signed_gap
            lo[q] = 0 if (open_flag_gap < 0.0) else 1  # Binary open/blocked flags per direction q for this cell (1=open link)

            if ti.static(x_boundary_mode == "solid"):
                if i == 0 and cx[q] > 0:
                    lo[q] = 0
                    lf[q] = 0.0
                    wt[q] = WALL_XMIN
                elif i == Nx - 1 and cx[q] < 0:
                    lo[q] = 0
                    lf[q] = 0.0
                    wt[q] = WALL_XMAX

        # set solid cell if fully closed in all directions
        lo[0] = 0   # Binary open/blocked flags per direction q for this cell (1=open link)
        for q in ti.static(range(1, 19)):  # Unrolled loop over lattice directions q (compile-time static)
            if lo[q] == 1:  # Branch for boundary/stability logic
                lo[0] = 1  # at least one open direction

        # determine if near or under the boundary
        near_obstacle[k, j, i] = 0  # Intermediate scalar 'near_obstacle' for this kernel block
        for q in ti.static(range(1, 19)):  # Unrolled loop over lattice directions q (compile-time static)
            if lo[q] == 0:  # if a direction is blocked, not far from boundary
                near_obstacle[k, j, i] = 1   # Intermediate scalar 'near_obstacle' for this kernel block

        # Consistency: if cell is "solid", force all links closed
        # (helps eliminate porous solids at corners)
        if lo[0] == 0:  # Branch for boundary/stability logic
            for q in ti.static(range(1, 19)):  # Unrolled loop over lattice directions q (compile-time static)
                lf[q] = 0.0  # Continuous open fraction per direction q in [0,1] (1=open, 0=blocked)

        lattice_open[k, j, i] = lo  # Intermediate scalar 'lattice_open' for this kernel block
        lattice_open_frac[k, j, i] = lf  # Intermediate scalar 'lattice_open_frac' for this kernel block
        lattice_wall_type[k, j, i] = wt  # Nearest solid source for each boundary link

@ti.kernel
def init_fields():
    # -----------------------------------------------------------------------------
    # Initialize simulation fields at t=0 (masking, inlet profile, velocity, and populations).
    #
    # Steps:
    # 1) Bed/solid mask: mark obstacle=1 wherever the cell-center link (q=0) is closed
    #    according to lattice_open[...,0] (derived from the free-surface / geometry kernel).
    # 2) Build a log-law mean velocity profile u_inlet_profile[k] in lattice units:
    #       u^+ = (1/kappa) ln(y^+) + B,  with y^+ = y*u_tau/nu
    #    then normalize the profile so the top value equals u_top_lb.
    # 3) Initialize the full 3D velocity field:
    #    - ux set from the mean profile with a small random perturbation (turbulence seeding),
    #      uy=uz=0 initially.
    #    - For bed (closed) cells, overwrite velocity with prescribed moving-wall values
    #      (u_wave, v_wave, w_wave) to enforce no-slip / wall motion.
    # 4) Initialize the distribution function f to equilibrium (D3Q19 low-Mach isothermal)
    #    using the initialized (rho,u) fields.
    #
    # Outputs: obstacle, u_inlet_profile, u_inlet_fluct, (ux,uy,uz), and f (equilibrium).
    # -----------------------------------------------------------------------------

    # Build initial bed mask (t=0)
    # The bed is represented by obstacle=1 cells at the bottom, up to some height defined by a Gaussian function.
    for k, j, i in obstacle:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        obstacle[k, j, i] = 1 if lattice_open[k, j, i][0] == 0 else 0  # Intermediate scalar 'obstacle' for this kernel block

    # Log-law initial velocity profile (lattice units)
    # We construct a theoretical turbulent profile so the simulation starts 
    # closer to a developed state, reducing transient run-up time.
    # Law: u = (u_*/k) * ln(y*u_*/nu) + B
    B = 5.0  # log-law intercept (Smooth wall constant approx)
    u_tau = 0.1 * u_top_lb  # rough guess for friction velocity
    nu_lbm = (tau0 - 0.5) / 3.0  # kinematic viscosity in lattice units

    # compute u_profile(j) then normalize so that top = u_top_lb
    # First pass: compute tmp at each j for a fictitious x=0
    for k in range(Nz):  # Parallel loop over index space (Taichi SPMD)
        y = float(k)  # Intermediate scalar 'y' for this kernel block
        y_plus = y * u_tau / nu_lbm  # Intermediate scalar 'y_plus' for this kernel block
        if y_plus <= 1.0:  # Branch for boundary/stability logic
            y_plus = 1.0  # Intermediate scalar 'y_plus' for this kernel block
        tmp[k] = (1.0 / kappa) * ti.log(y_plus) + B # buffer for log-law profile

    top_val = tmp[Nz - 1]  # Intermediate scalar 'top_val' for this kernel block
    if top_val == 0.0: # avoid division by zero
        top_val = 1.0  # Intermediate scalar 'top_val' for this kernel block
    for k in range(Nz): # normalize so that top of the profile matches u_top_lb
        tmp[k] = tmp[k] / top_val * u_top_lb  # Intermediate scalar 'tmp' for this kernel block

    # Store the profile for the Inlet BC
    for k in range(Nz):  # Parallel loop over index space (Taichi SPMD)
        u_inlet_profile[k] = tmp[k]  # Prescribed inlet profile in lattice units
        for j in range(Ny):  # Parallel loop over index space (Taichi SPMD)
                   u_inlet_fluct[k, j] = 0.0 # no initial fluctuations; can be set to small random values if desired

    # Assign initial velocity field based on the log-law profile, and set uz to zero.
    for k, j, i in ux:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        # Generate random kick (-0.5 to 0.5)
        rand_kick = (ti.random() - 0.5) * 2.0        # Uniform random variate in [0,1) for synthetic turbulence seeding
        ux[k, j, i] = u_inlet_profile[k] * (1.0 + rand_kick * TI)  # Add scaled random fluctuation to the mean profile
        uy[k, j, i] = 0.0  # Intermediate scalar 'uy' for this kernel block
        uz[k, j, i] = 0.0  # Intermediate scalar 'uz' for this kernel block
        if ti.static(water_free_surface_enabled == 1):
            if free_surface_type[k, j, i] == FS_GAS:
                rho[k, j, i] = rho0
                ux[k, j, i] = 0.0
                uy[k, j, i] = 0.0
                uz[k, j, i] = 0.0
            else:
                rho[k, j, i] = rho_initial[k, j, i]
                ux[k, j, i] = u_current_lb
                if ti.static(
                    free_surface_initial_condition == "solitary"
                    and solitary_initial_velocity_mode in ("shallow_water", "shallow_water_divfree")
                ):
                    eta_fs = free_surface_h[j, i] - free_surface_base_m
                    if eta_fs < 0.0:
                        eta_fs = 0.0
                    u_sol_phys = (
                        solitary_direction
                        * solitary_celerity_phys
                        * eta_fs
                        / (free_surface_depth_m + eta_fs)
                    )
                    ux[k, j, i] += u_sol_phys / vel_scale

                    if ti.static(solitary_initial_velocity_mode == "shallow_water_divfree"):
                        xc = (ti.cast(i, ti.f32) + 0.5) * dx
                        Lx_sol = ti.cast(Nx, ti.f32) * dx
                        xrel = xc - solitary_x0_m
                        if ti.static(x_boundary_mode == "periodic"):
                            if xrel > 0.5 * Lx_sol:
                                xrel -= Lx_sol
                            if xrel < -0.5 * Lx_sol:
                                xrel += Lx_sol
                        arg = solitary_kappa_m_inv * xrel
                        exp_p = ti.exp(arg)
                        exp_m = ti.exp(-arg)
                        denom = exp_p + exp_m
                        sech = 2.0 / denom
                        tanh_arg = (exp_p - exp_m) / denom
                        eta_x = -2.0 * solitary_amp_m * solitary_kappa_m_inv * sech * sech * tanh_arg
                        du_dx_phys = (
                            solitary_direction
                            * solitary_celerity_phys
                            * free_surface_depth_m
                            * eta_x
                            / ((free_surface_depth_m + eta_fs) * (free_surface_depth_m + eta_fs))
                        )
                        zc = (ti.cast(k, ti.f32) + 0.5) * dx
                        z_rel = zc - water_h[j, i]
                        if z_rel < 0.0:
                            z_rel = 0.0
                        uz[k, j, i] += (-z_rel * du_dx_phys) / vel_scale
        if lattice_open[k, j, i][0] == 0: # enforce no-slip for the bed
            wall_type = lattice_wall_type[k, j, i][0]
            ux[k, j, i] = 0.0
            uy[k, j, i] = 0.0
            uz[k, j, i] = 0.0
            rho[k, j, i] = rho0
            if wall_type == WALL_BED:
                ux[k, j, i] = u_wave[j, i]  # Intermediate scalar 'ux' for this kernel block
                uy[k, j, i] = v_wave[j, i]  # Intermediate scalar 'uy' for this kernel block
                uz[k, j, i] = w_wave[j, i]  # Intermediate scalar 'uz' for this kernel block

    # Initialize equilibrium f
    for k, j, i in ux:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        ux_eq = ux[k, j, i]
        uy_eq = uy[k, j, i]
        uz_eq = uz[k, j, i]
        if ti.static(water_free_surface_enabled == 1):
            if lattice_open[k, j, i][0] == 1 and free_surface_type[k, j, i] != FS_GAS:
                uz_eq += 0.5 * local_g_body_lb(k, j, i)
        u2 = ux_eq * ux_eq + uy_eq * uy_eq + uz_eq * uz_eq  # Squared speed |u|^2 for equilibrium evaluation
        for q in range(19):  # Parallel loop over index space (Taichi SPMD)
            cu = 3.0 * (cx[q] * ux_eq + cy[q] * uy_eq + cz[q] * uz_eq)  # Scaled dot product for equilibrium evaluation
            f[k, j, i][q] = rho[k, j, i] * w[q] * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)


@ti.kernel
def initialize_free_surface_boundary_populations():
    # Make the t=0 interface populations consistent with the same gas-pressure
    # missing-population closure used during pull streaming. This is an
    # initialization projection only; it does not add damping or change material
    # parameters.
    for k, j, i in f:
        f_new[k, j, i] = f[k, j, i]

    if ti.static(water_free_surface_enabled == 1 and initialize_free_surface_populations):
        for k, j, i in f:
            if lattice_open[k, j, i][0] == 1 and free_surface_type[k, j, i] != FS_GAS:
                for q in ti.static(range(1, 19)):
                    src_i_raw = i - cx[q]
                    src_i = src_i_raw
                    src_i_is_fluid = True
                    if ti.static(x_boundary_mode == "periodic"):
                        src_i = (src_i_raw + Nx) % Nx
                    else:
                        if src_i_raw < 0:
                            src_i = 0
                            src_i_is_fluid = False
                        elif src_i_raw >= Nx:
                            src_i = Nx - 1
                            src_i_is_fluid = False

                    src_j = (j - cy[q] + Ny) % Ny
                    src_k = k - cz[q]
                    src_is_gas = False
                    if not src_i_is_fluid:
                        src_is_gas = False
                    elif src_k >= Nz:
                        src_is_gas = True
                    elif src_k >= 0:
                        if free_surface_type[src_k, src_j, src_i] == FS_GAS and lattice_open[src_k, src_j, src_i][0] == 1:
                            src_is_gas = True

                    if src_is_gas:
                        qbar = opp[q]
                        rho_fs = rho0
                        ux_fs = ux[k, j, i]
                        uy_fs = uy[k, j, i]
                        uz_fs = uz[k, j, i]
                        if ti.static(use_hydrostatic_balanced_pressure == 1 or free_surface_boundary_mode == "hydrostatic"):
                            rho_fs = free_surface_pressure_density_at_link(k, j, i, q)
                        f_eq_q = calculate_feq(rho_fs, ux_fs, uy_fs, uz_fs, q)
                        f_eq_bar = calculate_feq(rho_fs, ux_fs, uy_fs, uz_fs, qbar)
                        f_new[k, j, i][q] = f_eq_q + f_eq_bar - f[k, j, i][qbar]


@ti.kernel
def update_bed_populations_and_reinit(step: ti.i32):
    # -----------------------------------------------------------------------------
    # Reapply moving-bed boundary populations using lattice_open / lattice_open_frac gating.
    #
    # Purpose:
    # - Maintain the original continuous moving-wall treatment on links/cells classified
    #   as closed by the geometry (lattice_open), while allowing a smooth transition
    #   across partially open links via lattice_open_frac.
    #
    # This intentionally avoids cell-state refill pulses. The moving geometry is
    # represented by continuous link fractions instead of a hard solid->fluid transition.
    # -----------------------------------------------------------------------------

    boundary_refill_count[None] = 0

    for k, j, i in obstacle:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        obstacle_prev[k, j, i] = obstacle[k, j, i]
        obstacle[k, j, i] = 1 if lattice_open[k, j, i][0] == 0 else 0

    for k, j, i in rho:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        rho_loc = rho0  # Local density accumulator / boundary density used for feq
        ux_loc = u_wave[j, i]  # Intermediate scalar 'ux_loc' for this kernel block
        uy_loc = v_wave[j, i]  # Intermediate scalar 'uy_loc' for this kernel block
        uz_loc = w_wave[j, i]  # Intermediate scalar 'uz_loc' for this kernel block
        for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
            if lattice_open[k, j, i][q] == 0:  # Apply logic only on solid/blocked links or cells
                frac = lattice_open_frac[k, j, i][q]  # Open-link fraction in [0,1] from endpoint gap vs. interface height
                wall_type = lattice_wall_type[k, j, i][q]
                if wall_type == WALL_BED and bed_is_moving == 1:
                    ux_q = ux_loc
                    uy_q = uy_loc
                    uz_q = uz_loc
                    if ti.static(use_link_wall_velocity == 1):
                        xc = (ti.cast(i, ti.f32) + 0.5) * dx
                        yc = (ti.cast(j, ti.f32) + 0.5) * dx
                        xq = xc + 0.5 * dx * ti.cast(cx[q], ti.f32)
                        yq = yc + 0.5 * dx * ti.cast(cy[q], ti.f32)
                        u_wall_q = sample_wall_velocity_lb(xq, yq)
                        ux_q = u_wall_q[0]
                        uy_q = u_wall_q[1]
                        uz_q = u_wall_q[2]
                    feq = calculate_feq(rho_loc, ux_q, uy_q, uz_q, q)  # Intermediate scalar 'feq' for this kernel block
                    f[k, j, i][q] = frac * f[k, j, i][q] + (1.0 - frac) * feq
                elif ti.static(physics_mode == "air"):
                    ux_q = 0.0
                    uy_q = 0.0
                    uz_q = 0.0
                    if wall_type == WALL_BED:
                        ux_q = ux_loc
                        uy_q = uy_loc
                        uz_q = uz_loc
                        if ti.static(use_link_wall_velocity == 1):
                            xc = (ti.cast(i, ti.f32) + 0.5) * dx
                            yc = (ti.cast(j, ti.f32) + 0.5) * dx
                            xq = xc + 0.5 * dx * ti.cast(cx[q], ti.f32)
                            yq = yc + 0.5 * dx * ti.cast(cy[q], ti.f32)
                            u_wall_q = sample_wall_velocity_lb(xq, yq)
                            ux_q = u_wall_q[0]
                            uy_q = u_wall_q[1]
                            uz_q = u_wall_q[2]
                    feq = calculate_feq(rho_loc, ux_q, uy_q, uz_q, q)  # Intermediate scalar 'feq' for this kernel block
                    f[k, j, i][q] = frac * f[k, j, i][q] + (1.0 - frac) * feq


@ti.kernel
def macro_step(step: ti.i32):
    # -----------------------------------------------------------------------------
    # Macroscopic update: compute rho and u from current populations f
    #
    # LBM moment relations:
    #   rho = Σ_q f_q
    #   j   = Σ_q f_q * c_q   (momentum density)
    #   u   = j / rho
    #
    # Boundary enforcement at the macro level:
    # - Top layer: enforce a prescribed free-stream velocity (u_top_lb).
    # - Solid (bed) cells: enforce interface kinematics (u_wave, v_wave, w_wave).
    #
    # Note:
    # - Even though the immersed boundary is applied at the population level (IBB),
    #   keeping macros consistent in solid/interface cells helps stability and avoids
    #   invalid rho divisions.
    # -----------------------------------------------------------------------------
    # compute rho, ux, uz from f; enforce macro BCs (top & bed)
    # The zeroth moment of f is density. The first moment is momentum density.
    for k, j, i in rho:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        # density
        rho_loc = 0.0  # local density
        for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
            rho_loc += f[k, j, i][q]  # sum of distribution functions

        rho[k, j, i] = rho_loc  # Intermediate scalar 'rho' for this kernel block

    # velocity
    for k, j, i in ux:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        # x,y,z-momentum: contributions from all x-directed lattice velocities
        jx = 0.0  # Momentum density component Σ f_q c_{qx}
        jy = 0.0  # Momentum density component Σ f_q c_{qy}
        jz = 0.0  # Momentum density component Σ f_q c_{qz}
        for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
            fval = f[k, j, i][q]  # Population value f_q at this cell (used for momentum sums)
            jx += fval * cx[q]
            jy += fval * cy[q]
            jz += fval * cz[q]

        if ti.static(water_free_surface_enabled == 1):
            if free_surface_type[k, j, i] == FS_GAS:
                rho[k, j, i] = rho0
                ux[k, j, i] = 0.0
                uy[k, j, i] = 0.0
                uz[k, j, i] = 0.0
            else:
                force_z = -rho[k, j, i] * local_g_body_lb(k, j, i)
                inv_rho = 1.0 / rho[k, j, i]  # 1/rho for converting momentum density to velocity
                ux[k, j, i] = jx * inv_rho  # Intermediate scalar 'ux' for this kernel block
                uy[k, j, i] = jy * inv_rho  # Intermediate scalar 'uy' for this kernel block
                uz[k, j, i] = (jz + 0.5 * force_z) * inv_rho  # Guo forcing half-step velocity
        else:
            inv_rho = 1.0 / rho[k, j, i]  # 1/rho for converting momentum density to velocity
            ux[k, j, i] = jx * inv_rho  # Intermediate scalar 'ux' for this kernel block
            uy[k, j, i] = jy * inv_rho  # Intermediate scalar 'uy' for this kernel block
            uz[k, j, i] = jz * inv_rho  # Intermediate scalar 'uz' for this kernel block

    # enforce top boundary: free-stream velocity u_top_lb at the top row
    if ti.static(enable_top_drive == 1):
        for j in range(Ny):  # Parallel loop over index space (Taichi SPMD)
            for i in range(Nx):  # Parallel loop over index space (Taichi SPMD)
                ux[Nz - 1, j, i] = u_top_lb  # Intermediate scalar 'ux' for this kernel block
                uy[Nz - 1, j, i] = 0.0  # Intermediate scalar 'uy' for this kernel block
                uz[Nz - 1, j, i] = 0.0  # Intermediate scalar 'uz' for this kernel block


@ti.kernel
def compute_LES():
    # -----------------------------------------------------------------------------
    # LES + wall model: compute local omegaLoc (relaxation parameter) from strain
    #
    # 1) Compute velocity gradients (central differences) and strain tensor S_ij.
    # 2) Compute |S| = sqrt(2 S_ij S_ij) and Smagorinsky eddy viscosity:
    #       nu_t = (Cs*Delta)^2 * |S|
    #    with a clamp nu_t <= nu_t_max for numerical robustness.
    # 3) Near-wall adjustment:
    #    - Identify near-wall fluid cells by checking for solid directly below.
    #    - Use a log-law estimate to infer u_tau and wall shear tau_w.
    #    - Convert to an equivalent viscosity increment to augment nu_t.
    # 4) Convert total viscosity nu_eff = nu0 + nu_t into tau_eff and omegaLoc=1/tau_eff.
    #
    # Output fields:
    # - nu_t_field, Smag_stress_field for diagnostics
    # - omegaLoc used directly in the collision kernel
    # -----------------------------------------------------------------------------
    # Compute local strain rate magnitude and apply Smagorinsky LES & wall model
    # We use a basic Smagorinsky eddy viscosity:
    #   nu_t = (Cs * Delta)^2 * |S|
    # where |S| is the magnitude of the strain rate tensor based on velocity gradients.
    #
    # For the wall model, near the bottom, we enforce a relationship between wall shear and local velocity
    # using the log-law of the wall. This is done by adjusting the local eddy viscosity nu_t to match
    # the desired wall shear stress.

    for k, j, i in ux:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        is_active_fluid = lattice_open[k, j, i][0] == 1
        if ti.static(water_free_surface_enabled == 1):
            if free_surface_type[k, j, i] == FS_GAS:
                is_active_fluid = False
        if is_active_fluid:  # at least partially fluid cell
            # left/right/top/bottom indexing for gradient calculation
            # x-direction for periodic:
            ip = (i + 1) % Nx  # Periodic +x neighbor index (i+1 mod Nx)
            im = (i + Nx - 1) % Nx  # Periodic -x neighbor index (i-1 mod Nx)

            # y-direction for spanwise periodic:
            jp = (j + 1) % Ny  # Periodic +y neighbor index (j+1 mod Ny)
            jm = (j + Ny - 1) % Ny  # Periodic -y neighbor index (j-1 mod Ny)

            # z-direction (walls top/bottom, no wrap)
            kp = k + 1  # +z neighbor index (clamped at top wall)
            km = k - 1  # -z neighbor index (clamped at bottom wall)
            if kp >= Nz:  # Clamp z-neighbor index at domain boundary (no periodic wrap)
                kp = Nz - 1  # +z neighbor index (clamped at top wall)
            if km < 0:  # Clamp z-neighbor index at domain boundary (no periodic wrap)
                km = 0  # -z neighbor index (clamped at bottom wall)

            # compute velocity gradients using central differences
            du_dx = 0.5 * (ux[k, j, ip] - ux[k, j, im])  # Intermediate scalar 'du_dx' for this kernel block
            du_dy = 0.5 * (ux[k, jp, i] - ux[k, jm, i])  # Intermediate scalar 'du_dy' for this kernel block
            du_dz = 0.5 * (ux[kp, j, i] - ux[km, j, i])  # Intermediate scalar 'du_dz' for this kernel block

            dv_dx = 0.5 * (uy[k, j, ip] - uy[k, j, im])  # Intermediate scalar 'dv_dx' for this kernel block
            dv_dy = 0.5 * (uy[k, jp, i] - uy[k, jm, i])  # Intermediate scalar 'dv_dy' for this kernel block
            dv_dz = 0.5 * (uy[kp, j, i] - uy[km, j, i])  # Intermediate scalar 'dv_dz' for this kernel block

            dw_dx = 0.5 * (uz[k, j, ip] - uz[k, j, im])  # Intermediate scalar 'dw_dx' for this kernel block
            dw_dy = 0.5 * (uz[k, jp, i] - uz[k, jm, i])  # Intermediate scalar 'dw_dy' for this kernel block
            dw_dz = 0.5 * (uz[kp, j, i] - uz[km, j, i])  # Intermediate scalar 'dw_dz' for this kernel block

            # Strain-rate tensor components S_ij = 0.5(∂u_i/∂x_j + ∂u_j/∂x_i)
            Sxx = du_dx  # Intermediate scalar 'Sxx' for this kernel block
            Syy = dv_dy  # Intermediate scalar 'Syy' for this kernel block
            Szz = dw_dz  # Intermediate scalar 'Szz' for this kernel block

            Sxy = 0.5 * (du_dy + dv_dx)  # Intermediate scalar 'Sxy' for this kernel block
            Sxz = 0.5 * (du_dz + dw_dx)  # Intermediate scalar 'Sxz' for this kernel block
            Syz = 0.5 * (dv_dz + dw_dy)  # Intermediate scalar 'Syz' for this kernel block

            # |S| = sqrt(2 S_ij S_ij)
            S_contr = (Sxx*Sxx + Syy*Syy + Szz*Szz  # Intermediate scalar 'S_contr' for this kernel block
                       + 2.0 * (Sxy*Sxy + Sxz*Sxz + Syz*Syz))
            S_mag = ti.sqrt(2.0 * S_contr + 1e-6)  # Compute magnitude / normalization term

            # Smagorinsky eddy viscosity
            nu_t = (Cs * Delta)**2 * S_mag  # Intermediate scalar 'nu_t' for this kernel block
            if nu_t > nu_t_max:  # Branch for boundary/stability logic
                nu_t = nu_t_max  # Intermediate scalar 'nu_t' for this kernel block

            # Wall model near the bottom: override nu_t close to bed based on log-law shear
            is_near_wall = False  # Intermediate scalar 'is_near_wall' for this kernel block
            if k > 0 and lattice_open[k - 1, j, i][0] == 0:  # Apply logic only on solid/blocked links or cells
                is_near_wall = True  # Intermediate scalar 'is_near_wall' for this kernel block

            if is_near_wall:  # Branch for boundary/stability logic
                # Get distance to wall (ensure positive and prevent log(0))
                dist_to_wall_m = max(phi[k, j, i], 0.001 * dx)  # Length scale in meters (SI)
                log_val = ti.log(dist_to_wall_m / z0_phys)  # Log-law evaluation for inlet/wall modeling
                if log_val < 2.0:  # Branch for boundary/stability logic
                    log_val = 2.0  # Intermediate scalar 'log_val' for this kernel block

                wall_type_below = lattice_wall_type[k - 1, j, i][0]
                u_wall_x = 0.0
                u_wall_y = 0.0
                if wall_type_below == WALL_BED:
                    u_wall_x = u_wave[j, i]
                    u_wall_y = v_wave[j, i]

                # tangential velocity relative to the nearest vertical wall source
                u_rel_x = ux[k, j, i] - u_wall_x  # Intermediate scalar 'u_rel_x' for this kernel block
                u_rel_y = uy[k, j, i] - u_wall_y  # Intermediate scalar 'u_rel_y' for this kernel block
                speed_rel = ti.sqrt(u_rel_x * u_rel_x + u_rel_y * u_rel_y)  # Compute magnitude / normalization term

                if speed_rel > 1e-6:  # Branch for boundary/stability logic
                    u_tau = speed_rel * kappa / log_val  # Intermediate scalar 'u_tau' for this kernel block
                    tau_w = u_tau * u_tau  # Intermediate scalar 'tau_w' for this kernel block
                    strain_approx = speed_rel / 0.5  # same heuristic as before
                    nu_wm = tau_w / strain_approx  # Intermediate scalar 'nu_wm' for this kernel block
                    nu_t_wall = nu_wm - nu0  # Intermediate scalar 'nu_t_wall' for this kernel block
                    if nu_t_wall < 0:  # Branch for boundary/stability logic
                        nu_t_wall = 0  # Intermediate scalar 'nu_t_wall' for this kernel block
                        
                    # if the resolved LES strain is huge (e.g., a massive
                    # separation event hitting the wall), respect it, but never
                    # drop below what the log-law demands.
                    nu_t = max(nu_t, nu_t_wall)  # Intermediate scalar 'nu_t' for this kernel block

            # effective viscosity and relaxation time
            nu_eff = nu0 + nu_t  # Intermediate scalar 'nu_eff' for this kernel block
            tau_eff = 0.5 + 3.0 * nu_eff  # Intermediate scalar 'tau_eff' for this kernel block
            if tau_eff > 2.0:  # Branch for boundary/stability logic
                tau_eff = 2.0  # Intermediate scalar 'tau_eff' for this kernel block

            nu_t_field[k, j, i] = nu_t  # Diagnostic field for post-processing / visualization
            Smag_stress_field[k, j, i] = S_mag  # Diagnostic field for post-processing / visualization
            # Local relaxation rate ω=1/τ: encodes ν_eff=ν0+ν_t so collision adapts to resolved shear (LES).
            omegaLoc[k, j, i] = 1.0 / tau_eff

@ti.kernel
def collide_KBC():
    # -----------------------------------------------------------------------------
    # Collision operator: BGK relaxation with a KBC-style positivity limiter
    #
    # Steps per cell:
    # 1) Build equilibrium populations feq from local (rho,u).
    # 2) Compute standard BGK post-collision f_star = f - omega*(f - feq).
    # 3) Apply an “entropic-like” limiter (alpha in [0,1]) to prevent negative f:
    #       f_post = feq + alpha*(f_star - feq)
    #    This is a pragmatic positivity safeguard in high-Re, coarse-grid LES.
    # 4) Optional backscatter (currently gated by C_backscatter):
    #    - Computes a deviatoric non-equilibrium stress tensor from (f_post - feq).
    #    - Injects a random signed fraction back into second-order moments.
    #    - Applies another positivity limiter (gamma) for safety.
    #
    # Sponge logic:
    # - Left sponge is currently disabled by condition (i < -3).
    # - Right sponge region can be used to suppress reflections (not fully active here).
    # -----------------------------------------------------------------------------
    
    # Perform collision step using entropic-like KBC with local omegaLoc
    c_s2 = 1.0 / 3.0  # Intermediate scalar 'c_s2' for this kernel block
    inv_2cs4 = 1.0 / (2.0 * c_s2 * c_s2)   # = 1 / (2*(1/9)) = 4.5
    eps = 1e-6  # Intermediate scalar 'eps' for this kernel block

    sponge_thickness = 3  # Intermediate scalar 'sponge_thickness' for this kernel block
    tau_sponge_min = 0.95  # Intermediate scalar 'tau_sponge_min' for this kernel block
    om_sponge_left = 1.0 / tau_sponge_min  # Intermediate scalar 'om_sponge_left' for this kernel block

    for k, j, i in ux:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        rho_loc = rho[k, j, i]  # Local density accumulator / boundary density used for feq
        ux_loc = ux[k, j, i]  # Intermediate scalar 'ux_loc' for this kernel block
        uy_loc = uy[k, j, i]  # Intermediate scalar 'uy_loc' for this kernel block
        uz_loc = uz[k, j, i]  # Intermediate scalar 'uz_loc' for this kernel block

        is_solid = (lattice_open[k, j, i][0] == 0)  # Intermediate scalar 'is_solid' for this kernel block
        is_gas = False
        if ti.static(water_free_surface_enabled == 1):
            if free_surface_type[k, j, i] == FS_GAS:
                is_gas = True
                rho_loc = rho0
                ux_loc = 0.0
                uy_loc = 0.0
                uz_loc = 0.0

        u2 = ux_loc * ux_loc + uy_loc * uy_loc + uz_loc * uz_loc  # Squared speed |u|^2 for equilibrium evaluation

        is_sponge  = (enable_top_sponge == 1) and (k > Nz - sponge_thickness)  # sponge along top only 

        if is_sponge:  # Branch for boundary/stability logic
            # strong relaxation toward equilibrium to damp perturbations
            for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                cu = 3.0 * (cx[q] * ux_loc + cy[q] * uy_loc + cz[q] * uz_loc)  # Scaled dot product 3(c_q·u) for equilibrium evaluation
                feq_q = rho_loc * w[q] * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)  # Intermediate scalar 'feq_q' for this kernel block
                f_val = f[k, j, i][q]  # Intermediate scalar 'f_val' for this kernel block
                f[k, j, i][q] = f_val - om_sponge_left * (f_val - feq_q)

        else:
            # interior: use LES-adjusted omegaLoc
            om = omegaLoc[k, j, i]  # Intermediate scalar 'om' for this kernel block

            # --- build feq
            feq = ti.Vector.zero(ti.f32, 19)  # Initialize vector accumulator (all zeros)
            for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                cu = 3.0 * (cx[q] * ux_loc + cy[q] * uy_loc + cz[q] * uz_loc)  # Scaled dot product 3(c_q·u) for equilibrium evaluation
                feq[q] = rho_loc * w[q] * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)  # Intermediate scalar 'feq' for this kernel block

            # --- BGK/KBC-style relaxation
            f_star = ti.Vector.zero(ti.f32, 19)  # Initialize vector accumulator (all zeros)
            for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                f_val = f[k, j, i][q]  # Intermediate scalar 'f_val' for this kernel block
                force_i = 0.0
                if ti.static(water_free_surface_enabled == 1):
                    if not is_gas:
                        force_z = -rho_loc * local_g_body_lb(k, j, i)
                        cxi = ti.cast(cx[q], ti.f32)
                        cyi = ti.cast(cy[q], ti.f32)
                        czi = ti.cast(cz[q], ti.f32)
                        cdu = cxi * ux_loc + cyi * uy_loc + czi * uz_loc
                        force_i = (
                            w[q] * (1.0 - 0.5 * om)
                            * (((czi - uz_loc) / c_s2) + (cdu * czi / (c_s2 * c_s2)))
                            * force_z
                        )
                f_star[q] = f_val - om * (f_val - feq[q]) + force_i  # Intermediate scalar 'f_star' for this kernel block

            # --- positivity limiter on alpha: f_post = feq + alpha*(f_star-feq)
            alpha = 1.0  # Intermediate scalar 'alpha' for this kernel block
            for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                if f_star[q] < eps:  # Branch for boundary/stability logic
                    denom = f_star[q] - feq[q]  # Normalization sqrt(1+|∇h|^2) for slope-corrected normal distance
                    if denom < 0.0:  # Guard against degenerate slope norm (avoid divide-by-zero)
                        candidate = (eps - feq[q]) / denom  # Intermediate scalar 'candidate' for this kernel block
                        if candidate < alpha:  # Branch for boundary/stability logic
                            alpha = candidate  # Intermediate scalar 'alpha' for this kernel block

            if alpha < 0.0:  # Branch for boundary/stability logic
                alpha = 0.0  # Intermediate scalar 'alpha' for this kernel block
            if alpha > 1.0:  # Branch for boundary/stability logic
                alpha = 1.0  # Intermediate scalar 'alpha' for this kernel block

            f_post = ti.Vector.zero(ti.f32, 19)  # Initialize vector accumulator (all zeros)
            for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                f_post[q] = feq[q] + alpha * (f_star[q] - feq[q])  # Intermediate scalar 'f_post' for this kernel block

            # ============================================================
            # BACKSCATTER (stochastic deviatoric-stress injection)
            # - injects energy into shear modes only (no net mass/momentum)
            # - uses local LES fields to scale amplitude
            # - gated off in solids and (optionally) outlet sponge
            # ============================================================
            # --- Backscatter based on local non-equilibrium stress (recommended) ---
            if (C_backscatter > 0.0) and (not is_solid):  # Branch for boundary/stability logic

                # 1) compute non-equilibrium stress Pi_neq from f_post - feq
                Pxx = 0.0; Pyy = 0.0; Pzz = 0.0  # Intermediate scalar 'Pxx' for this kernel block
                Pxy = 0.0; Pxz = 0.0; Pyz = 0.0  # Intermediate scalar 'Pxy' for this kernel block

                for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                    fneq = f_post[q] - feq[q]  # Intermediate scalar 'fneq' for this kernel block
                    cxi = ti.cast(cx[q], ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
                    cyi = ti.cast(cy[q], ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
                    czi = ti.cast(cz[q], ti.f32)  # Type cast for Taichi kernel arithmetic / indexing

                    Pxx += fneq * cxi * cxi
                    Pyy += fneq * cyi * cyi
                    Pzz += fneq * czi * czi
                    Pxy += fneq * cxi * cyi
                    Pxz += fneq * cxi * czi
                    Pyz += fneq * cyi * czi

                # deviatoric part (trace-free)
                tr = (Pxx + Pyy + Pzz) / 3.0  # Intermediate scalar 'tr' for this kernel block
                Pxx -= tr; Pyy -= tr; Pzz -= tr

                # 2) random signed strength (mean ~0, but correlated with existing shear structure)
                xi = 2.0 * ti.random(ti.f32) - 1.0  # in [-1,1]
                # If you want net injection (not zero-mean), use abs(xi) instead.
                scale = C_backscatter * xi  # Intermediate scalar 'scale' for this kernel block

                # 3) map Pi_bs to population increments (same Hermite mapping)
                inv_2cs4 = 1.0 / (2.0 * c_s2 * c_s2)  # Intermediate scalar 'inv_2cs4' for this kernel block
                delta = ti.Vector.zero(ti.f32, 19)  # Initialize vector accumulator (all zeros)

                for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                    cxi = ti.cast(cx[q], ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
                    cyi = ti.cast(cy[q], ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
                    czi = ti.cast(cz[q], ti.f32)  # Type cast for Taichi kernel arithmetic / indexing

                    Qxx = cxi * cxi - c_s2  # Intermediate scalar 'Qxx' for this kernel block
                    Qyy = cyi * cyi - c_s2  # Intermediate scalar 'Qyy' for this kernel block
                    Qzz = czi * czi - c_s2  # Intermediate scalar 'Qzz' for this kernel block
                    Qxy = cxi * cyi  # Intermediate scalar 'Qxy' for this kernel block
                    Qxz = cxi * czi  # Intermediate scalar 'Qxz' for this kernel block
                    Qyz = cyi * czi  # Intermediate scalar 'Qyz' for this kernel block

                    contr = (Qxx * (scale * Pxx) + Qyy * (scale * Pyy) + Qzz * (scale * Pzz)  # Intermediate scalar 'contr' for this kernel block
                            + 2.0 * (Qxy * (scale * Pxy) + Qxz * (scale * Pxz) + Qyz * (scale * Pyz)))

                    delta[q] = w[q] * inv_2cs4 * contr  # Intermediate scalar 'delta' for this kernel block

                # 4) positivity limiter on gamma
                gamma = 1.0  # Intermediate scalar 'gamma' for this kernel block
                for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                    if delta[q] < 0.0:  # Branch for boundary/stability logic
                        cand = (eps - f_post[q]) / delta[q]  # Intermediate scalar 'cand' for this kernel block
                        if cand < gamma:  # Branch for boundary/stability logic
                            gamma = cand  # Intermediate scalar 'gamma' for this kernel block
                if gamma < 0.0: gamma = 0.0  # Branch for boundary/stability logic
                if gamma > 1.0: gamma = 1.0  # Branch for boundary/stability logic

                for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                    f_post[q] = f_post[q] + gamma * delta[q]  # Intermediate scalar 'f_post' for this kernel block


            # write back
            for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                f[k, j, i][q] = f_post[q]


@ti.kernel
def collide_regularized():
    c_s2 = 1.0 / 3.0
    inv_2cs4 = 1.0 / (2.0 * c_s2 * c_s2)
    eps = 1e-6

    for k, j, i in ux:
        rho_loc = rho[k, j, i]
        ux_loc = ux[k, j, i]
        uy_loc = uy[k, j, i]
        uz_loc = uz[k, j, i]

        is_gas = False
        if ti.static(water_free_surface_enabled == 1):
            if free_surface_type[k, j, i] == FS_GAS:
                is_gas = True
                rho_loc = rho0
                ux_loc = 0.0
                uy_loc = 0.0
                uz_loc = 0.0

        om = omegaLoc[k, j, i]
        u2 = ux_loc * ux_loc + uy_loc * uy_loc + uz_loc * uz_loc

        feq = ti.Vector.zero(ti.f32, 19)
        for q in ti.static(range(19)):
            cu = 3.0 * (cx[q] * ux_loc + cy[q] * uy_loc + cz[q] * uz_loc)
            feq[q] = rho_loc * w[q] * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)

        if is_gas:
            for q in ti.static(range(19)):
                f[k, j, i][q] = feq[q]
        else:
            Pxx = 0.0
            Pyy = 0.0
            Pzz = 0.0
            Pxy = 0.0
            Pxz = 0.0
            Pyz = 0.0
            for q in ti.static(range(19)):
                fneq = f[k, j, i][q] - feq[q]
                cxi = ti.cast(cx[q], ti.f32)
                cyi = ti.cast(cy[q], ti.f32)
                czi = ti.cast(cz[q], ti.f32)
                Pxx += fneq * cxi * cxi
                Pyy += fneq * cyi * cyi
                Pzz += fneq * czi * czi
                Pxy += fneq * cxi * cyi
                Pxz += fneq * cxi * czi
                Pyz += fneq * cyi * czi

            f_star = ti.Vector.zero(ti.f32, 19)
            for q in ti.static(range(19)):
                cxi = ti.cast(cx[q], ti.f32)
                cyi = ti.cast(cy[q], ti.f32)
                czi = ti.cast(cz[q], ti.f32)
                Qxx = cxi * cxi - c_s2
                Qyy = cyi * cyi - c_s2
                Qzz = czi * czi - c_s2
                Qxy = cxi * cyi
                Qxz = cxi * czi
                Qyz = cyi * czi
                fneq_reg = w[q] * inv_2cs4 * (
                    Qxx * Pxx + Qyy * Pyy + Qzz * Pzz
                    + 2.0 * (Qxy * Pxy + Qxz * Pxz + Qyz * Pyz)
                )

                force_i = 0.0
                if ti.static(water_free_surface_enabled == 1):
                    force_z = -rho_loc * local_g_body_lb(k, j, i)
                    cdu = cxi * ux_loc + cyi * uy_loc + czi * uz_loc
                    force_i = (
                        w[q] * (1.0 - 0.5 * om)
                        * (((czi - uz_loc) / c_s2) + (cdu * czi / (c_s2 * c_s2)))
                        * force_z
                    )
                f_star[q] = feq[q] + (1.0 - om) * fneq_reg + force_i

            alpha = 1.0
            for q in ti.static(range(19)):
                if f_star[q] < eps:
                    denom = f_star[q] - feq[q]
                    if denom < 0.0:
                        candidate = (eps - feq[q]) / denom
                        if candidate < alpha:
                            alpha = candidate
            if alpha < 0.0:
                alpha = 0.0
            if alpha > 1.0:
                alpha = 1.0

            for q in ti.static(range(19)):
                f[k, j, i][q] = feq[q] + alpha * (f_star[q] - feq[q])

@ti.kernel
def stream():
    # -----------------------------------------------------------------------------
    # Streaming step (pull scheme) with free-surface / cut-link gating.
    #
    # For each destination cell and direction q>0, the open fraction blends ordinary
    # pull streaming from the upstream cell with local bounce-back from the opposite
    # population. This preserves the original continuous boundary path and avoids
    # hard cell-center fluid/solid switching.
    # -----------------------------------------------------------------------------

    boundary_cut_link_count[None] = 0
    free_surface_cut_link_count[None] = 0
    free_surface_thin_gap_bridge_count[None] = 0
    wall_momentum_x[None] = 0.0
    wall_momentum_y[None] = 0.0
    wall_momentum_z[None] = 0.0

    for k, j, i in f_new:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        # Rest population: always local
        if ti.static(water_free_surface_enabled == 1):
            if free_surface_type[k, j, i] == FS_GAS and lattice_open[k, j, i][0] == 1:
                f_new[k, j, i][0] = calculate_feq(rho0, 0.0, 0.0, 0.0, 0)
            else:
                f_new[k, j, i][0] = f[k, j, i][0]
        else:
            f_new[k, j, i][0] = f[k, j, i][0]
        for q in ti.static(range(1, 19)):  # Unrolled loop over lattice directions q (compile-time static)
            src_i_raw = i - cx[q]
            src_i = src_i_raw
            src_i_is_fluid = True
            if ti.static(x_boundary_mode == "periodic"):
                src_i = (src_i_raw + Nx) % Nx
            else:
                if src_i_raw < 0:
                    src_i = 0
                    src_i_is_fluid = False
                elif src_i_raw >= Nx:
                    src_i = Nx - 1
                    src_i_is_fluid = False
            src_j = (j - cy[q]) % Ny  # Intermediate scalar 'src_j' for this kernel block
            src_k = k - cz[q]  # Intermediate scalar 'src_k' for this kernel block

            if ti.static(water_free_surface_enabled == 1):
                dest_is_real_gas = free_surface_type[k, j, i] == FS_GAS and lattice_open[k, j, i][0] == 1
                src_is_real_gas = False
                if src_i_is_fluid and 0 <= src_k < Nz:
                    src_is_real_gas = free_surface_type[src_k, src_j, src_i] == FS_GAS and lattice_open[src_k, src_j, src_i][0] == 1
                elif src_k >= Nz:
                    src_is_real_gas = True
                if dest_is_real_gas:
                    f_new[k, j, i][q] = calculate_feq(rho0, 0.0, 0.0, 0.0, q)
                elif not src_i_is_fluid:
                    f_new[k, j, i][q] = f[k, j, i][opp[q]]
                    ti.atomic_add(boundary_cut_link_count[None], 1)
                elif 0 <= src_k < Nz and not src_is_real_gas:
                    frac = lattice_open_frac[k, j, i][q]  # Open-link fraction for solid geometry only
                    f_new[k, j, i][q] = frac * f[src_k, src_j, src_i][q] + (1.0 - frac) * f[k, j, i][opp[q]]
                    if frac < 1.0:
                        ti.atomic_add(boundary_cut_link_count[None], 1)
                elif src_k < 0:
                    f_new[k, j, i][q] = f[k, j, i][opp[q]]
                    ti.atomic_add(boundary_cut_link_count[None], 1)
                else:
                    qbar = opp[q]
                    rho_fs = rho0
                    ux_fs = ux[k, j, i]
                    uy_fs = uy[k, j, i]
                    uz_fs = uz[k, j, i]
                    if ti.static(use_hydrostatic_balanced_pressure == 1 or free_surface_boundary_mode == "hydrostatic"):
                        rho_fs = free_surface_pressure_density_at_link(k, j, i, q)
                    f_eq_q = calculate_feq(rho_fs, ux_fs, uy_fs, uz_fs, q)
                    f_eq_bar = calculate_feq(rho_fs, ux_fs, uy_fs, uz_fs, qbar)
                    f_fs = f_eq_q + f_eq_bar - f[k, j, i][qbar]
                    f_out = f_fs
                    if ti.static(free_surface_tracking_mode == "vof" and vof_thin_gap_bridge_enabled):
                        if 0 <= src_k < Nz and src_is_real_gas:
                            bridge_i_raw = src_i - cx[q]
                            bridge_i = bridge_i_raw
                            bridge_j = (src_j - cy[q] + Ny) % Ny
                            bridge_k = src_k - cz[q]
                            bridge_valid = True
                            if ti.static(x_boundary_mode == "periodic"):
                                bridge_i = (bridge_i_raw + Nx) % Nx
                            else:
                                if bridge_i_raw < 0 or bridge_i_raw >= Nx:
                                    bridge_valid = False
                            if bridge_k < 0 or bridge_k >= Nz:
                                bridge_valid = False

                            if bridge_valid:
                                if (
                                    lattice_open[bridge_k, bridge_j, bridge_i][0] == 1
                                    and free_surface_type[bridge_k, bridge_j, bridge_i] != FS_GAS
                                ):
                                    bridge_weight = free_surface_fill[k, j, i]
                                    if free_surface_fill[bridge_k, bridge_j, bridge_i] < bridge_weight:
                                        bridge_weight = free_surface_fill[bridge_k, bridge_j, bridge_i]
                                    bridge_weight *= vof_thin_gap_bridge_strength
                                    if bridge_weight > 0.0:
                                        if bridge_weight > 1.0:
                                            bridge_weight = 1.0
                                        f_out = (1.0 - bridge_weight) * f_fs + bridge_weight * f[bridge_k, bridge_j, bridge_i][q]
                                        ti.atomic_add(free_surface_thin_gap_bridge_count[None], 1)
                    f_new[k, j, i][q] = f_out
                    ti.atomic_add(free_surface_cut_link_count[None], 1)
            elif not src_i_is_fluid:
                f_new[k, j, i][q] = f[k, j, i][opp[q]]
                ti.atomic_add(boundary_cut_link_count[None], 1)
            elif 0 <= src_k < Nz:  # Branch for boundary/stability logic
                frac = lattice_open_frac[k, j, i][q]  # Open-link fraction in [0,1] from endpoint gap vs. interface height
                f_new[k, j, i][q] = frac * f[src_k, src_j, src_i][q] + (1.0 - frac) * f[k, j, i][opp[q]]
                if frac < 1.0:
                    ti.atomic_add(boundary_cut_link_count[None], 1)
            else:
                # Out of domain in z: keep local value for later boundary handling
                f_new[k, j, i][q] = f[k, j, i][q]


@ti.kernel
def apply_x_solid_wall_bounceback():
    # Standard stationary solid-wall LBM boundary on the x-min/x-max faces.
    #
    # Physical boundary conditions:
    # - no normal flux through the left/right walls
    # - no-slip wall velocity u_wall = 0
    #
    # Numerical implementation:
    # - after pull streaming, reconstruct only populations that would have
    #   streamed from outside the domain by local opposite-direction bounce-back.
    # - the wall is stationary, so no moving-wall momentum correction is added.
    if ti.static(x_boundary_mode == "solid"):
        for k, j in ti.ndrange(Nz, Ny):
            i_left = 0
            i_right = Nx - 1
            left_active = True
            right_active = True
            if ti.static(water_free_surface_enabled == 1):
                left_active = not (free_surface_type[k, j, i_left] == FS_GAS and lattice_open[k, j, i_left][0] == 1)
                right_active = not (free_surface_type[k, j, i_right] == FS_GAS and lattice_open[k, j, i_right][0] == 1)

            for q in ti.static(range(1, 19)):
                if cx[q] > 0 and left_active:
                    f_new[k, j, i_left][q] = f[k, j, i_left][opp[q]]
                if cx[q] < 0 and right_active:
                    f_new[k, j, i_right][q] = f[k, j, i_right][opp[q]]

@ti.kernel
def apply_open_boundary_conditions():
    # ============================================================
    # Apply “open” / driving boundary conditions after streaming by
    # selectively relaxing f_new toward target equilibria.
    #
    # (A) X-FRINGE / HYBRID INLET FORCING (periodic-x domains)
    #     In i = 0..inlet_width-1 (upper half of the domain), relax
    #     f_new toward a target inflow feq(rho0, u_inlet_profile[k]
    #     + u_inlet_fluct[k,j], 0, 0). This continuously injects
    #     momentum/turbulence so a periodic domain does not spin down
    #     under bottom drag. Relaxation uses a smooth spatial weight
    #     sigma = inlet_strength * smoothstep01(s), not a hard overwrite.
    #
    # (B) TOP BOUNDARY (k = Nz-1)
    #     Impose a simple “open/drive” condition by setting populations
    #     to equilibrium corresponding to (rho0, u_top_lb, 0, 0) on open
    #     cells. (As written, this overwrites all q; a less dissipative
    #     variant would reconstruct only incoming populations.)
    #
    # (C) BOTTOM BED IBB (near-interface moving wall)
    #     For cells flagged near_obstacle==1, apply a link-wise moving-wall
    #     bounce-back / IBB-style correction blended by lattice_open_frac:
    #       f_new(q) <- frac_open * f_new(q) + (1-frac_open) * f_wall
    #     where f_wall is the reflected opposite population with a standard
    #     momentum correction proportional to (c_q̄ · u_wall). This enforces
    #     no-slip wall motion smoothly on partially blocked links.
    # ============================================================

    # ============================================================
    # (A) X-FRINGE / HYBRID INLET FORCING (for periodic-x domains)
    #     Relax f_new in i = 0..inlet_width-1 toward a target
    #     inflow profile u_inlet_profile[k] (+ fluctuations).
    #
    #     This acts like an "inlet" in an otherwise periodic domain
    #     and continuously re-injects momentum/turbulence so the
    #     domain does not spin down under bottom drag.
    # ============================================================
    inlet_width = 5        # tune 12–24
    inlet_strength = 0.01   # tune 0.10–0.40 (too large -> over-constraint)

    # Guard for inlet_width=1 to avoid division by zero
    denom_i = ti.cast(inlet_width - 1, ti.f32)  # Type cast for Taichi kernel arithmetic / indexing
    if denom_i < 1.0:  # Branch for boundary/stability logic
        denom_i = 1.0  # Intermediate scalar 'denom_i' for this kernel block

    if ti.static(enable_x_fringe == 1):
        for k, j, i in f_new:  # Parallel sweep over field indices (SPMD; Taichi schedules)
            if i < inlet_width and lattice_open[k, j, i][0] == 1 and k > Nz // 2:  # Apply logic only on fluid/open cells
                # Weight is strongest at i=0, fades to 0 at i=inlet_width-1
                s = 1.0 - ti.cast(i, ti.f32) / denom_i      # Type cast for Taichi kernel arithmetic / indexing
                inlet_strength = 0.05 * (ti.pow(ti.cast(k, ti.f32) / ti.cast(Nz - 1, ti.f32), 2) - 0.25)  # Type cast for Taichi kernel arithmetic / indexing
                sigma = inlet_strength * smoothstep01(s)  # Blending factor for fringe/inlet relaxation toward feq

                # Target inflow macros
                rho_t = rho0  # Intermediate scalar 'rho_t' for this kernel block
                ux_t  = u_inlet_profile[k] + u_inlet_fluct[k, j]  # Intermediate scalar 'ux_t' for this kernel block
                uy_t  = 0.0  # Intermediate scalar 'uy_t' for this kernel block
                uz_t  = 0.0  # Intermediate scalar 'uz_t' for this kernel block

                # Relax populations toward target equilibrium (hybrid, not hard overwrite)
                for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                    feq_t = calculate_feq(rho_t, ux_t, uy_t, uz_t, q)  # Intermediate scalar 'feq_t' for this kernel block
                    f_new[k, j, i][q] = (1.0 - sigma) * f_new[k, j, i][q] + sigma * feq_t

    # ============================================================
    # (B) TOP BOUNDARY
    #     note - to be less of a turbulence sink,
    #     only reconstruct incoming populations (cz[q] == -1) instead.
    # ============================================================
    if ti.static(enable_top_drive == 1):
        k_top = Nz - 1  # Intermediate scalar 'k_top' for this kernel block
        for j in range(Ny):  # Parallel loop over index space (Taichi SPMD)
            for i in range(Nx):  # Parallel loop over index space (Taichi SPMD)
                if lattice_open[k_top, j, i][0] == 1:  # Apply logic only on fluid/open cells
                    rho_top = rho0  # Intermediate scalar 'rho_top' for this kernel block
                    ux_top  = u_top_lb  # Intermediate scalar 'ux_top' for this kernel block
                    uy_top  = 0.0  # Intermediate scalar 'uy_top' for this kernel block
                    uz_top  = 0.0  # Intermediate scalar 'uz_top' for this kernel block
                    for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
                        # Equilibrium (feq) evaluation: 2nd-order low-Mach Hermite expansion (isothermal BGK LBM).
                        f_new[k_top, j, i][q] = calculate_feq(rho_top, ux_top, uy_top, uz_top, q)

    # ============================================================
    # (C) BOTTOM BED IBB (moving-wall bounce-back with open-fraction blending)
    #     For near_obstacle cells, build a moving-wall reflected population using
    #     the opposite direction q̄ and wall velocity (u_wave,v_wave,w_wave), then
    #     blend it with the streamed value using lattice_open_frac:
    #         f_wall = f_pre[q̄] - 6*w[q̄]*rho0*(c_q̄ · u_wall)
    #         f_new[q] = frac_open*f_new[q] + (1-frac_open)*f_wall
    #     so frac_open=1 is fully open, frac_open=0 fully enforces the wall.
    # ============================================================
    for k, j, i in f_new:  # Parallel sweep over field indices (SPMD; Taichi schedules)

        if near_obstacle[k, j, i] == 1: # fluid cell near the obstacle (fluid cell with at least direction blocked)

            # wave info at q, inclduing cx, cy shifts
            u_wx = u_wave[j, i]  # Intermediate scalar 'u_wx' for this kernel block
            u_wy = v_wave[j, i]  # Intermediate scalar 'u_wy' for this kernel block
            u_wz = w_wave[j, i]  # Intermediate scalar 'u_wz' for this kernel block

            for q in ti.static(range(1, 19)):  # Unrolled loop over lattice directions q (compile-time static)
                qbar = opp[q]  # Opposite lattice direction index q̄

                frac_open = lattice_open_frac[k, j, i][q]   # Open fraction for this link used to blend stream vs. bounce-back

                wall_type = lattice_wall_type[k, j, i][q]
                u_wx = 0.0
                u_wy = 0.0
                u_wz = 0.0
                if wall_type == WALL_BED:
                    u_wx = u_wave[j, i]
                    u_wy = v_wave[j, i]
                    u_wz = w_wave[j, i]
                    if ti.static(use_link_wall_velocity == 1):
                        xc = (ti.cast(i, ti.f32) + 0.5) * dx
                        yc = (ti.cast(j, ti.f32) + 0.5) * dx
                        xq = xc + 0.5 * dx * ti.cast(cx[q], ti.f32)
                        yq = yc + 0.5 * dx * ti.cast(cy[q], ti.f32)
                        u_wall_q = sample_wall_velocity_lb(xq, yq)
                        u_wx = u_wall_q[0]
                        u_wy = u_wall_q[1]
                        u_wz = u_wall_q[2]

                cuw = (ti.cast(cx[qbar], ti.f32) * u_wx +  # Dot product (c_q̄ · u_wall) for moving-wall momentum correction
                       ti.cast(cy[qbar], ti.f32) * u_wy +
                       ti.cast(cz[qbar], ti.f32) * u_wz)

                # Reflect the outgoing population (from pre-stream) + moving-wall correction
                f_wall = f[k, j, i][qbar] - 6.0 * w[qbar] * rho0 * cuw  # Moving-wall corrected reflected population for IBB
                f_new[k, j, i][q] = f_new[k, j, i][q] * frac_open + f_wall * (1.0 - frac_open)

@ti.kernel
def copy_post_and_swap():
    # -----------------------------------------------------------------------------
    # Copy/swap step
    #
    # This implementation uses explicit copy:
    #   f <- f_new
    # rather than pointer swapping. This is straightforward but can be more memory
    # intensive. For performance tuning, a ping-pong swap of references can be used,
    # but explicit copy is often clearer and avoids aliasing mistakes.
    # -----------------------------------------------------------------------------
    # Swap for next iteration
    for k, j, i in f:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        f[k, j, i] = f_new[k, j, i]  # Intermediate scalar 'f' for this kernel block


def visualize(step, time_phys, Re_phys_air, Re_lb_base):
    raise RuntimeError(
        "Visualization has not been configured. Use run_ab_case.py or run "
        "this solver as a script so direct-run visualization can be bound."
    )


def configure_direct_visualization():
    if __name__ != "__main__":
        return None

    import csv
    import sys
    from pathlib import Path

    from run_ab_case import METRIC_FIELDNAMES, make_visualize

    label = os.environ.get("LBM_RUN_LABEL", "direct")
    output_dir = Path(os.environ.get("LBM_OUTPUT_DIR", "frames")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = (output_dir / f"{label}_metrics.csv").open("w", newline="")
    writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDNAMES)
    writer.writeheader()
    globals()["visualize"] = make_visualize(sys.modules[__name__], label, output_dir, writer)
    return metrics_file


def main():
    # -----------------------------------------------------------------------------
    # Main driver / program entry point.
    #
    # Initialization:
    # - Set constants/scales, initialize moving bed kinematics, build interface geometry
    #   (phi, lattice_open), then initialize macroscopic fields and equilibrium populations.
    #
    # Per-timestep sequence:
    # 1) Update moving bed + prescribed wall velocity (u_wave,v_wave,w_wave)
    # 2) Recompute interface geometry (phi) and per-link openness (lattice_open / frac)
    # 3) Enforce bed populations via open-fraction blending (reinit near boundary)
    # 4) Reconstruct macros (rho,u), apply LES closure (optional), collide, stream
    # 5) Apply post-stream boundary conditions, then commit f_new -> f
    #
    # Monitoring:
    # - Periodic visualization and max-velocity reporting for stability diagnostics.
    # -----------------------------------------------------------------------------
    direct_metrics_file = configure_direct_visualization()

    init_constants()  # set physical/lattice scales, solver parameters, and constants
    update_wave_bed_and_velocities(0.0)  # initialize moving bed geometry + wall velocity at t=0
    compute_phi_slope_corrected()  # build signed-distance-like phi for interface/wall model
    build_lattice_open_from_free_surface_periodic()  # compute per-link open flags/fractions from water_h (periodic x/y)
    initialize_free_surface_height(0.0)
    classify_free_surface_cells()
    apply_vof_initial_disconnected_shape()
    update_pressure_reference_fields()
    prepare_initial_density_field()
    init_fields()  # initialize obstacle mask, inlet profile, velocity field, and f=feq
    initialize_free_surface_boundary_populations()
    copy_post_and_swap()
    initialize_vof_mass_from_fill()
    enforce_vof_interface_layer()
    update_free_surface_height_from_vof()

    print("Grid Size (Nx, Ny, Nz): ", Nx, Ny, Nz)  # runtime configuration summary
    print(f"Physics mode: {physics_mode} ({fluid_name})")
    print(f"Free-surface mode: {free_surface_mode}, initial={free_surface_initial_condition}, tracking={free_surface_tracking_mode}")
    if water_free_surface_enabled == 1:
        print(
            f"Free surface: base={free_surface_base_m:.3f} m, depth={free_surface_depth_m:.3f} m, "
            f"solitary_amp={solitary_amp_m:.3f} m, H/h={solitary_height_depth_ratio:.3f}, "
            f"solitary_c={solitary_celerity_phys:.3f} m/s, "
            f"solitary_dir={solitary_direction:+.0f}, solitary_1/kappa={solitary_length_scale_m:.3f} m, "
            f"solitary_u0={solitary_initial_velocity_mode}, "
            f"boundary={free_surface_boundary_mode}, refill={free_surface_refill_mode}, "
            f"initial_pressure={initial_pressure_mode}, pressure_formulation={pressure_formulation}"
        )
        if pressure_formulation_auto_corrected:
            print(
                "Pressure formulation auto-corrected from total_pressure to "
                "vof_component_balanced for disconnected VOF. Set "
                "LBM_ALLOW_TOTAL_PRESSURE_VOF_DIAGNOSTIC=1 only for a known-bad "
                "full-pressure diagnostic."
            )
        if free_surface_tracking_mode == "vof":
            print(
                f"VOF state change: empty_fill_threshold={vof_empty_fill_threshold:.4f}, "
                f"orphan_empty_threshold={vof_orphan_empty_threshold:.4f}, "
                f"orphan_search_radius={vof_orphan_search_radius}, "
                f"orphan_max_weak_neighbors={vof_orphan_max_weak_neighbors}, "
                f"detached_advection={vof_detached_advection_enabled}, "
                f"detached_max_wet_neighbors={vof_detached_max_wet_neighbors}, "
                f"detached_max_resolved_neighbors={vof_detached_max_resolved_neighbors}, "
                f"detached_residual_fill_threshold={vof_detached_residual_fill_threshold:.4f}, "
                f"thin_gap_bridge={vof_thin_gap_bridge_enabled}, "
                f"bridge_strength={vof_thin_gap_bridge_strength:.3f}, "
                f"collapse_trapped_gas={vof_collapse_trapped_gas_enabled}, "
                f"collapse_interval={vof_collapse_interval}, "
                f"collapse_max_volume={vof_collapse_max_volume_m3:.3f} m^3, "
                f"collapse_flood_sweeps={vof_collapse_flood_sweeps}"
            )
            if vof_initial_shape == "block":
                print(
                    f"VOF initial block: center=({vof_block_center_x_m:.3f}, "
                    f"{vof_block_center_y_m:.3f}, {vof_block_center_z_m:.3f}) m, "
                    f"size=({vof_block_size_x_m:.3f}, {vof_block_size_y_m:.3f}, "
                    f"{vof_block_size_z_m:.3f}) m"
                )
        if free_surface_initial_condition == "gaussian":
            print(
                f"Gaussian hump: amp={gaussian_amp_m:.3f} m, center_x={gaussian_center_x_m:.3f} m, "
                f"sigma={gaussian_sigma_m:.3f} m, zero initial velocity"
            )
    print(f"Bed profile: {bed_profile}, bed_level={offset_m:.3f} m")
    print(f"Obstacle mode: {obstacle_mode}")
    if include_cube_geometry == 1:
        print(
            f"Cube size/center/yaw/pitch: size=({cube_size_x_m:.3f}, "
            f"{cube_size_y_m:.3f}, {cube_size_z_m:.3f}) m, "
            f"center=({cube_center_x_m:.3f}, {cube_center_y_m:.3f}, {cube_center_z_m:.3f}) m, "
            f"yaw={cube_yaw_deg:.3f} deg, pitch={cube_pitch_deg:.3f} deg"
        )
    print(f"Boundary geometry mode: {boundary_geometry_mode}")  # active wall-geometry interpretation
    print(f"X boundary mode: {x_boundary_mode}")
    print(f"Wall velocity sampling: {wall_velocity_sampling}")  # active moving-wall velocity interpolation
    print(f"Initial free-surface population projection: {initialize_free_surface_populations}")
    print(f"Collision model: {collision_model}")
    print(f"Fluid rho={rho_phys:.3f} kg/m^3, nu={nu_phys:.6e} m^2/s, gravity={gravity_phys:.3f} m/s^2")
    print(f"Reference speed={U_ref_phys:.3f} m/s, dt={dt_phys:.6e} s, g_lb={g_lb:.6e}, g_body_lb={g_body_lb:.6e}")
    print(f"Physical Re ({fluid_name}): {Re_phys_air:.3e}")  # diagnostic Reynolds number in physical units
    print(f"Lattice Re (base): {Re_lb_base:.2f}")  # diagnostic Reynolds number in lattice units

    for step in range(1, steps + 1):  # main time-marching loop over integer timesteps
        time_phys = (step + 1) * dt_phys  # physical time corresponding to this step (SI units)

        update_wave_bed_and_velocities(time_phys)  # update bed position and prescribed wall velocity at current time
        compute_phi_slope_corrected()  # refresh phi after bed/interface motion
        build_lattice_open_from_free_surface_periodic()  # refresh per-link openness after interface update
        if free_surface_tracking_mode == "height":
            classify_free_surface_cells()
        update_pressure_reference_fields()
        update_bed_populations_and_reinit(step)  # moving-geometry mask update and transition-only refill
        macro_step(step)  # reconstruct rho and u from f (zeroth/first moments)

        if use_LES:  # optional LES closure
            compute_LES()  # compute eddy viscosity / effective relaxation parameters from local strain rate

        if collision_model == "regularized":
            collide_regularized()  # collision/relaxation with regularized non-equilibrium projection
        else:
            collide_KBC()  # collision/relaxation (with limiter/backscatter as implemented)
        stream()  # pull-stream populations into f_new
        apply_x_solid_wall_bounceback()  # standard stationary solid walls at x-min/x-max when enabled
        apply_open_boundary_conditions()  # fringe inlet forcing and top boundary handling
        copy_post_and_swap()  # commit f_new -> f for the next timestep
        if water_free_surface_enabled == 1:
            macro_step(step)
            if free_surface_tracking_mode == "vof":
                compute_vof_mass_exchange()
                apply_vof_mass_update_and_preclassify()
                redistribute_vof_mass_excess()
                finalize_vof_classification_from_mass()
                redistribute_vof_mass_excess()
                finalize_vof_classification_from_mass()
                redistribute_vof_mass_excess()
                finalize_vof_classification_from_mass()
                enforce_vof_interface_layer()
                remove_orphan_vof_interface_cells()
                advect_detached_vof_cells()
                finalize_vof_classification_from_mass()
                compress_detached_vof_residuals()
                finalize_vof_classification_from_mass()
                redistribute_vof_mass_excess()
                finalize_vof_classification_from_mass()
                enforce_vof_interface_layer()
                collapse_trapped_gas_if_due(step)
                enforce_vof_interface_layer()
                update_free_surface_height_from_vof()
                refill_new_free_surface_cells()
                repair_vof_coalescence_links()
            else:
                compute_free_surface_column_flux()
                advect_free_surface_height(time_phys)
                classify_free_surface_cells()
                refill_new_free_surface_cells()

        if step % plot_freq == 0 and step > plot_step_start:  # periodic plotting window
            visualize(step, time_phys, Re_phys_air, Re_lb_base)  # visualization / output

        if step % 100 == 0:  # periodic stability/health report
            ux_np = ux.to_numpy()  # copy ux from Taichi field to NumPy for reduction
            uy_np = uy.to_numpy()
            uz_np = uz.to_numpy()
            obstacle_np = obstacle.to_numpy()  # copy obstacle mask to NumPy for filtering
            active_np = obstacle_np == 0
            if water_free_surface_enabled == 1:
                fs_np = free_surface_type.to_numpy()
                active_np = active_np & (fs_np != FS_GAS)
            speed_np = np.sqrt(ux_np**2 + uy_np**2 + uz_np**2)
            max_u = np.nanmax(speed_np[active_np])  # max speed over active fluid cells only
            print(  # console diagnostic: max speed in lattice and approximate physical units
                f"Time Step {step}, Max |U| (LB units) = {max_u:.4f}, "
                f"Max U (phys) ~ {max_u * vel_scale:.2f} m/s, "
                f"cut_links={boundary_cut_link_count[None]}, refill={boundary_refill_count[None]}, "
                f"fs_refill={free_surface_refill_count[None]}, fs_coalesce={free_surface_coalescence_repair_count[None]}, "
                f"fs_bridge={free_surface_thin_gap_bridge_count[None]}, "
                f"vof_excess={free_surface_vof_excess_count[None]}, "
                f"vof_orphans={free_surface_vof_orphan_count[None]}, "
                f"vof_detached={free_surface_vof_detached_advect_count[None]} "
                f"({free_surface_vof_detached_advect_mass[None]:.3e}), "
                f"vof_compact={free_surface_vof_detached_compress_count[None]} "
                f"({free_surface_vof_detached_compress_mass[None]:.3e}), "
                f"vof_collapse_cells={free_surface_vof_collapse_cell_count[None]}, "
                f"vof_collapse_vol={free_surface_vof_collapse_applied_volume_m3[None]:.3f} m^3"
            )

    print("Simulation Complete.")  # end-of-run marker
    if direct_metrics_file is not None:
        direct_metrics_file.close()


if __name__ == "__main__":
    main()  # execute driver when run as a script
