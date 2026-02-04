"""
Lattice Boltzmann Simulation: Open-Channel Airflow Over Moving Ripples with LES
Model: D3Q19 BGK with Smagorinsky LES
Boundary Conditions: Inlet/Outlet (Open Channel) + Moving Bottom + Slip Top

Flow is driven by a fixed inlet velocity profile (Log-law). The physical free-stream 
velocity (m/s) is specified by the user as U_top_phys. The corresponding 
lattice reference speed u_lid is fixed for stability, and a simple scaling 
converts lattice velocities to physical units.

-------------------------------------------------------------------------
HIGH-LEVEL THEORETICAL OVERVIEW
-------------------------------------------------------------------------
This code solves an incompressible, isothermal flow using the 
Lattice Boltzmann Method (LBM) in 3D with a D3Q19 lattice:

 - Space is discretized into a regular Cartesian grid.
 - At each node, 19 distribution functions f_i represent the probability 
   of finding a "particle" moving along discrete velocities c_i.
 - The algorithm alternates between:
      (1) Collision: f_i relax towards an equilibrium distribution feq_i 
                     (BGK model with relaxation time tau),
      (2) Streaming: f_i are shifted to neighboring nodes along c_i.
 - Macroscopic quantities are obtained by taking moments:
      rho = sum_i f_i          (density, in lattice units)
      u   = (1/rho)*sum_i f_i c_i  (velocity, in lattice units)

The equilibrium feq_i is a low-Mach-number expansion of the Maxwell–
Boltzmann distribution, consistent with weakly compressible Navier–Stokes.

Viscosity in LBM:
  - In standard BGK LBM, the kinematic viscosity in lattice units is 
        nu_lbm = (tau - 0.5)/3
  - Here, we use a space-dependent effective viscosity nu_eff (via LES), 
    and hence a space-dependent relaxation time tau_eff.

LES (Large Eddy Simulation) with Smagorinsky model:
  - The small, unresolved turbulent scales are modeled by adding a 
    turbulent viscosity nu_t to the molecular/base viscosity nu0:
        nu_eff = nu0 + nu_t
  - nu_t is computed from the local strain-rate magnitude |S|:
        nu_t = (C_s * Delta)^2 * |S|
    where C_s is the Smagorinsky constant, and Delta is the filter scale 
    (taken here ~ grid spacing in lattice units).

Boundary Conditions:
  - Inlet (Left): Dirichlet condition. A fixed log-law velocity profile 
    is enforced by setting distributions to equilibrium (f = f_eq).
  - Outlet (Right): Neumann condition (Zero Gradient). Distributions are 
    copied from the neighbor (i-1) to allow structures to exit freely.
  - Top: Free-Slip / Specular Reflection. Vertical velocity is mirrored, 
    acting as a symmetry plane or frictionless ceiling.
  - Bottom: Moving Rippled Bed. We use an immersed boundary approach, where
    the stream in each direction is partially bounced back based on the local
    distance to the moving boundary. For near-wall cells, a momentum correction
    is applied to impose the local bed velocity (from the wave model).
    This simulates the friction/drag of the bed moving relative to the fluid.

Physical <-> Lattice mapping:
  - We fix a reference lattice speed u_lid (must remain < ~0.1 for low Mach).
  - The user specifies U_top_phys in m/s.
  - We define a velocity scaling:
        vel_scale = U_top_phys / u_lid  [m/s per lattice unit]
  - Thus, velocities in lattice units are converted to physical m/s by 
        u_phys = u_lbm * vel_scale
  - Time in physical units is inferred from advective scaling:
        t_phys ≈ (t_lbm * u_lid * dx) / U_top_phys

The code also computes a pressure-like field:
  - Static / thermodynamic pressure difference from a reference density:
        delta_p_lb = c_s^2 * (rho - rho0),   c_s^2 = 1/3 for D3Q19
    which is mapped to physical Pa via a simple scaling.
-------------------------------------------------------------------------
"""
# -----------------------------------------------------------------------------
# Imports
# - Standard library: file paths and OS utilities for output management.
# - Taichi: GPU-accelerated compute kernels for the LBM.
# - NumPy/Matplotlib: host-side diagnostics and plotting.
# - skimage.marching_cubes: optional 3D isosurface extraction for vorticity.
# -----------------------------------------------------------------------------
import os
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

# for 3D iso surface plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

# -----------------------------------------------------------------------------
# Taichi runtime initialization
# - arch=ti.gpu selects GPU backend (CUDA/Metal/Vulkan depending on platform).
# - If you need deterministic behavior for debugging, consider setting ti.init(..., random_seed=...).
# -----------------------------------------------------------------------------
ti.init(arch=ti.gpu)  # use GPU if available. Change to "cpu" if you want to force use of CPU over GPU

# -----------------------------------------------------------------------------
# Configuration overview
# This script is self-contained: it defines physics, numerical parameters,
# allocates Taichi fields, then runs an LBM time loop with periodic visualization.
# -----------------------------------------------------------------------------
# ==============================
# 0. User Input / Physical Params
# ==============================
# Physical scale parameters. These define the "Real World" problem we are solving.
U_top_phys = 30.      # [m/s] lid / free stream speed (typical for atmospheric boundary layer wind)
Lx_m = 300.0            # [m] domain length
Ly_m = 10.0            # [m] spanwise width (new y-direction "slice" extent)
Lz_m = 20.0            # [m] domain height

# Spatial Resolution
# dx determines the fidelity of the simulation. 
# At dx=~1m, we are in the VLES (Very Large Eddy Simulation) regime.
# We resolve large wake structures, but the boundary layer is mostly sub-grid.
dx = 0.5              # [m] grid spacing in x and y
nu_air = 1.516e-5      # [m^2/s] kinematic viscosity of air (Standard Atmosphere)
rho_air_phys = 1.2    # [kg/m^3] standard density of air

# Lattice resolution
Nx = int(Lx_m / dx)  # Lattice grid resolution along x: derived from physical domain size / dx
Ny = int(Ly_m / dx)  # Lattice grid resolution along y: derived from physical domain size / dx
Nz = int(Lz_m / dx)  # Lattice grid resolution along z: derived from physical domain size / dx

# Timesteps & plotting
steps = 400000  # number of LBM "time" steps to run
plot_freq = 100  # create a plot every this many steps
plot_step_start = 0  # start plotting after this maNz steps (plotting can be slow, so we skip the initial transient)
write_image = False # set to True to save images of each frame

# parameters for the moving bed (sine wave)
bed_amp_m = 0.0 # amplitude [m] of the sine wave
depth_swl_m = 10.0  # mean water depth [m] above the bed (not used directly here, but for reference)
wave_speed = np.sqrt(9.81 * depth_swl_m)  # wave celerity [m/s] for shallow water wave
bed_wavelength_m = 30.0  # wavelength [m] of the sine wave
wave_period = bed_wavelength_m / wave_speed  # period [s] of the sine wave (not used directly here)

offset_m = 2.0*bed_amp_m + 3.0*dx  # vertical offset [m] of the bed from the bottom of the domain
U_wave_phys = wave_speed # max horizontal velocity of the boundary [m/s]

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
U_ref_phys = max(abs(U_top_phys), abs(U_wave_phys), 1e-6)   # [m/s] mapping velocity scale
vel_scale = U_ref_phys / u_lid          # [m/s] per lattice unit
dt_phys   = u_lid * dx / U_ref_phys     # [s] physical time per LBM step

# Top / Bottom boundary velocity in lattice units (can be 0.0 cleanly)
u_top_lb = U_top_phys / vel_scale
u_bed_lb = wave_speed / vel_scale 

Re_phys_air = U_ref_phys * Lz_m / nu_air  # Physical Reynolds number based on domain height
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
use_LES = True # set to False to disable LES (laminar simulation)

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
C_backscatter = 0.02 # tuning parameter for backscatter amplitude; higher means more energy injected into small scales

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
ux = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # x-velocity field
uy = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # y-velocity field
uz = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # z-velocity field  

u_wave = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # x-dir velocity along wave surface
v_wave = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # y-dir velocity along wave surface
w_wave = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # z-dir velocity along wave surface

water_h = ti.field(dtype=ti.f32, shape=(Ny, Nx))  # instantaneous water surface height (meters)
water_h_last = ti.field(dtype=ti.f32, shape=(Ny, Nx)) # previous water surface height (meters)
phi = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx)) # signed distance (positive = air)

# main distribution functions (ping-pong)
# 'f' holds the distribution functions at the current step
# 'f_new' holds the post-collision/post-streaming functions for the next step
f = ti.Vector.field(19, dtype=ti.f32, shape=(Nz, Ny, Nx))  # main distribution functions
f_new = ti.Vector.field(19, dtype=ti.f32, shape=(Nz, Ny, Nx))  # post-collision distribution functions
lattice_open = ti.Vector.field(19, dtype=ti.i32, shape=(Nz, Ny, Nx))  # boolean mark for open-channel cells (1=open,0=solid)
lattice_open_frac = ti.Vector.field(19, dtype=ti.f32, shape=(Nz, Ny, Nx))  # δ for each q (only meaningful when blocked)

# LES fields
omegaLoc = ti.field(dtype=ti.f32, shape=(Nz, Ny, Nx))  # local relaxation parameter for LES (1/tau_effective)

# obstacle masks (Integer fields: 0 = Fluid, 1 = Solid)
near_obstacle = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx)) # near-obstacle flag
obstacle = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx))        # current obstacle mask
obstacle_prev = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx))    # previous obstacle mask

# 3D object masks
near_object3D = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx)) # near-3D-object flag
object3D = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx))        # current 3D-object mask
object3D_prev = ti.field(dtype=ti.i32, shape=(Nz, Ny, Nx))    # previous 3D-object mask
lattice_open_3D = ti.Vector.field(19, dtype=ti.i32, shape=(Nz, Ny, Nx))  # boolean mark for open-channel cells (1=open,0=solid) for 3D object
lattice_open_frac_3D = ti.Vector.field(19, dtype=ti.f32, shape=(Nz, Ny, Nx))  # δ for each q (only meaningful when blocked) for 3D object

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
def smoothstep01(x: ti.f32) -> ti.f32:
    # -----------------------------------------------------------------------------
    # Smooth ramp on [0,1] with C1 continuity
    return x * x * (3.0 - 2.0 * x)


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
    w_vals = ti.static([  # D3Q19 quadrature weights (w0=1/3, w_axis=1/18, w_diag=1/36)
        1.0/3.0,         # 0: rest
        1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,  # axes
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,                      # diagonals
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
    ])

    cx_vals = ti.static([  # D3Q19 discrete velocity x-components (integer c_q)
        0,
        1, -1,  0,  0,  0,  0,
        1, -1, -1,  1,
        1, -1, -1,  1,
        0,  0,  0,  0
    ])

    cy_vals = ti.static([  # D3Q19 discrete velocity y-components (integer c_q)
        0,
        0,  0,  1, -1,  0,  0,
        1,  1, -1, -1,
        0,  0,  0,  0,
        1, -1, -1,  1
    ])

    cz_vals = ti.static([  # D3Q19 discrete velocity z-components (integer c_q)
        0,
        0,  0,  0,  0,  1, -1,
        0,  0,  0,  0,
        1,  1, -1, -1,
        1,  1, -1, -1
    ])

    # Opposites: for each q, opp[q] has (cx,cy,cz) = -(cx,cy,cz)
    opp_vals = ti.static([  # Opposite-direction map q̄ for bounce-back/specular reflection
        0,  # 0 ↔ 0
        2,  # 1 ↔ 2
        1,
        4,  # 3 ↔ 4
        3,
        6,  # 5 ↔ 6
        5,
        9,  # 7 ↔ 9
        10, # 8 ↔ 10
        7,
        8,
        13, # 11 ↔ 13
        14, # 12 ↔ 14
        11,
        12,
        17, # 15 ↔ 17
        18, # 16 ↔ 18
        15,
        16
    ])

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
        near_object3D[k, j, i] = 0  # Intermediate scalar 'near_object3D' for this kernel block
        object3D[k, j, i] = 0  # Intermediate scalar 'object
        object3D_prev[k, j, i] = 0  # Intermediate scalar 'object3D_prev' for this kernel block
        f[k, j, i] = ti.Vector.zero(ti.f32, 19) # initialize to zero; will be set to equilibrium later
        f_new[k, j, i] = ti.Vector.zero(ti.f32, 19)  # Initialize vector accumulator (all zeros)
        # Local relaxation rate ω=1/τ: encodes ν_eff=ν0+ν_t so collision adapts to resolved shear (LES).
        omegaLoc[k, j, i] = 1.0 / tau0

    for j, i in u_wave:  # Parallel loop over index space (Taichi SPMD)
        u_wave[j, i] = 0.0  # Intermediate scalar 'u_wave' for this kernel block
        v_wave[j, i] = 0.0  # Intermediate scalar 'v_wave' for this kernel block
        w_wave[j, i] = 0.0  # Intermediate scalar 'w_wave' for this kernel block
        water_h[j, i] = offset_m  # Intermediate scalar 'water_h' for this kernel block
        water_h_last[j, i] = offset_m  # Intermediate scalar 'water_h_last' for this kernel block

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
    bed_amp_now = bed_amp_m * smoothstep01(t / bed_time_ramp) if t < bed_time_ramp else bed_amp_m  # Ramp-up amplitude for prescribed bed motion (m)
    for j in range(Ny):  # Parallel loop over index space (Taichi SPMD)
        for i in range(Nx):  # Parallel loop over index space (Taichi SPMD)
            x0 = ti.cast(i, ti.f32) * dx            # physical x (m)
            y0 = ti.cast(j, ti.f32) * dx  # Type cast for Taichi kernel arithmetic / indexing

            # store last water height for time derivative
            water_h_last[j, i] = water_h[j, i]  # Intermediate scalar 'water_h_last' for this kernel block

            # Traveling wave: eta(x,t) = a sin(k(x - c t))
            x_phase = x0 - c * t  # Intermediate scalar 'x_phase' for this kernel block
            x_phase2 = x0 - 0.5 * c * t  # second harmonic at half speed
            phase   = 0.5*k * x_phase  # Phase argument for first harmonic (rad)
            phase2  = k * x_phase2  # Phase argument for second harmonic (rad)
            eta1 = bed_amp_now * ti.sin(phase)  # First harmonic elevation component (m)
            eta2 = bed_amp_now * ti.sin(phase2)  # Second harmonic elevation component (m)
            eta     = eta1 + eta2  # m

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


# create similar kernal as above to 

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
            phi[k, j, i] = (zc - h_loc) / denom  # Scalar 'phi' for this kernel block

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
        frac_c = 0.5 + gap_c / dx  # Cell-center open fraction in [0,1] (diagnostic / q=0)
        if frac_c < 0.0:  # Branch for boundary/stability logic
            frac_c = 0.0  # Cell-center open fraction in [0,1] (diagnostic / q=0)
        if frac_c > 1.0:  # Branch for boundary/stability logic
            frac_c = 1.0  # Cell-center open fraction in [0,1] (diagnostic / q=0)

        lf[0] = frac_c  # Continuous open fraction per direction q in [0,1] (1=open, 0=blocked)

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
            frac = 0.5 + gap / dx  # Open-link fraction in [0,1] from endpoint gap vs. interface height
            if frac < 0.0:  # Clamp open fraction to [0,1] for stability
                frac = 0.0  # Open-link fraction in [0,1] from endpoint gap vs. interface height
            if frac > 1.0:  # Clamp open fraction to [0,1] for stability
                frac = 1.0  # Open-link fraction in [0,1] from endpoint gap vs. interface height

            lf[q] = frac  # Continuous open fraction per direction q in [0,1] (1=open, 0=blocked)

            # --- binary open/blocked ---
            lo[q] = 0 if (eta > zq) else 1  # Binary open/blocked flags per direction q for this cell (1=open link)

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
        if lattice_open[k, j, i][0] == 0: # enforce no-slip for the bed
            ux[k, j, i] = u_wave[j, i]  # Intermediate scalar 'ux' for this kernel block
            uy[k, j, i] = v_wave[j, i]  # Intermediate scalar 'uy' for this kernel block
            uz[k, j, i] = w_wave[j, i]  # Intermediate scalar 'uz' for this kernel block

    # Initialize equilibrium f
    for k, j, i in ux:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        u2 = ux[k, j, i] * ux[k, j, i] + uy[k, j, i] * uy[k, j, i] + uz[k, j, i] * uz[k, j, i]  # Squared speed |u|^2 for equilibrium evaluation
        for q in range(19):  # Parallel loop over index space (Taichi SPMD)
            cu = 3.0 * (cx[q] * ux[k, j, i] + cy[q] * uy[k, j, i] + cz[q] * uz[k, j, i])  # Scaled dot product 3(c_q·u) for equilibrium evaluation
            f[k, j, i][q] = rho[k, j, i] * w[q] * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)

@ti.kernel
def update_bed_populations_and_reinit(step: ti.i32):
    # -----------------------------------------------------------------------------
    # Reapply moving-bed boundary populations using lattice_open / lattice_open_frac gating.
    #
    # Purpose:
    # - Maintain a stable no-slip (moving wall) condition on links/cells classified as closed
    #   by the free-surface-derived geometry (lattice_open), while allowing a smooth transition
    #   across partially open links via lattice_open_frac (reduces binary wet/dry chatter).
    #
    # Steps:
    # 1) Update `obstacle` from lattice_open[...,0] (diagnostic / plotting mask only).
    # 2) For each cell and each direction q:
    #    - If the link is closed (lattice_open[...,q]==0), blend the current population toward
    #      the moving-wall equilibrium feq(rho0, u_wave, v_wave, w_wave) using the openness
    #      fraction:
    #          f <- frac * f + (1-frac) * feq
    #      where frac in [0,1] acts as a continuous gate (0 = fully enforce wall, 1 = leave unchanged).
    #
    # Notes:
    # - rho_loc is pinned to rho0 here (Dirichlet-like wall density); adjust only if your formulation
    #   requires pressure coupling at the wall.
    # - u_wave/v_wave/w_wave provide the prescribed bed velocity (moving bottom / wavemaker).
    # - `step` is available for time-dependent logic (e.g., ramping), though not used in this snippet.
    # -----------------------------------------------------------------------------

    
    # set obstacle mask from lattice_open, only used for plotting
    for k, j, i in obstacle:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        obstacle[k, j, i] = 1 if lattice_open[k, j, i][0] == 0 else 0  # Intermediate scalar 'obstacle' for this kernel block

    # set all closed cells / directions to boundary populations
    for k, j, i in rho:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        rho_loc = rho0  # Local density accumulator / boundary density used for feq
        ux_loc = u_wave[j, i]  # Intermediate scalar 'ux_loc' for this kernel block
        uy_loc = v_wave[j, i]  # Intermediate scalar 'uy_loc' for this kernel block
        uz_loc = w_wave[j, i]  # Intermediate scalar 'uz_loc' for this kernel block
        for q in ti.static(range(19)):  # Unrolled loop over lattice directions q (compile-time static)
            if lattice_open[k, j, i][q] == 0:  # Apply logic only on solid/blocked links or cells
                frac = lattice_open_frac[k, j, i][q]  # Open-link fraction in [0,1] from endpoint gap vs. interface height
                feq = calculate_feq(rho_loc, ux_loc, uy_loc, uz_loc, q)  # Intermediate scalar 'feq' for this kernel block
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

        inv_rho = 1.0 / rho[k, j, i]  # 1/rho for converting momentum density to velocity
        ux[k, j, i] = jx * inv_rho  # Intermediate scalar 'ux' for this kernel block
        uy[k, j, i] = jy * inv_rho  # Intermediate scalar 'uy' for this kernel block
        uz[k, j, i] = jz * inv_rho  # Intermediate scalar 'uz' for this kernel block

    # enforce top boundary: free-stream velocity u_top_lb at the top row
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
        if lattice_open[k, j, i][0] == 1:  # at least partially fluid cell
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

                # tangential velocity relative to moving bed (bed moves only in x)
                u_rel_x = ux[k, j, i] - u_wave[j, i]  # Intermediate scalar 'u_rel_x' for this kernel block
                u_rel_y = uy[k, j, i] - v_wave[j, i]  # Intermediate scalar 'u_rel_y' for this kernel block
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

        u2 = ux_loc * ux_loc + uy_loc * uy_loc + uz_loc * uz_loc  # Squared speed |u|^2 for equilibrium evaluation

        is_sponge  = (k > Nz - sponge_thickness)  # sponge along top only 

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
                f_star[q] = f_val - om * (f_val - feq[q])  # Intermediate scalar 'f_star' for this kernel block

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
def stream():
    # -----------------------------------------------------------------------------
    # Streaming step (pull scheme) with free-surface / cut-link gating.
    #
    # Purpose:
    # - Advect (stream) lattice populations along discrete velocities using a pull formulation:
    #     f_new(x, q) <- f(x - c_q, q)
    # - Incorporate a per-link openness fraction (lattice_open_frac) to smoothly blend across
    #   partially blocked links near the free surface / moving boundary, reducing binary
    #   switching artifacts.
    #
    # Method:
    # - For each destination cell (k,j,i) and direction q>0:
    #     src = (i-cx[q], j-cy[q], k-cz[q]) with periodic wrap in x/y.
    # - If src_k is inside [0,Nz):
    #     frac = lattice_open_frac[k,j,i][q]
    #     f_new[q] = frac * f(src,q) + (1-frac) * f(dest, opp[q])
    #   i.e., open portion streams from upstream; closed portion reflects using opposite population
    #   (bounce-back-like fallback for the blocked fraction).
    # - If src_k is outside the vertical domain, keep local population for later boundary handling.
    #
    # Notes:
    # - q=0 (rest) does not stream and is copied locally.
    # - This kernel assumes periodicity in x/y via modulo indexing; z is bounded.
    # - The reflection term uses f(dest, opp[q]) at the pre-stream state, consistent with a simple
    #   on-site fallback when the link is not fully open.
    # -----------------------------------------------------------------------------

    for k, j, i in f_new:  # Parallel sweep over field indices (SPMD; Taichi schedules)
        # Rest population: always local
        f_new[k, j, i][0] = f[k, j, i][0]
        for q in ti.static(range(1, 19)):  # Unrolled loop over lattice directions q (compile-time static)
            src_i = (i - cx[q]) % Nx  # Intermediate scalar 'src_i' for this kernel block
            src_j = (j - cy[q]) % Ny  # Intermediate scalar 'src_j' for this kernel block
            src_k = k - cz[q]  # Intermediate scalar 'src_k' for this kernel block

            if 0 <= src_k < Nz:  # Branch for boundary/stability logic
                 # Open fraction for the link from src -> (k,j,i) in direction q
                # (matches your previous binary check lattice_open[src][q])
                frac = lattice_open_frac[k, j, i][q]  # Open-link fraction in [0,1] from endpoint gap vs. interface height
                f_new[k, j, i][q] = frac * f[src_k, src_j, src_i][q] + (1.0 - frac) * f[k, j, i][opp[q]]
            else:
                # Out of domain in z: keep local value for later BC handling
                f_new[k, j, i][q] = f[k, j, i][q]
                
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
    # -----------------------------------------------------------------------------
    # Visualization / diagnostics (host-side)
    #
    # This function pulls Taichi fields back to NumPy arrays and generates plots:
    # - Mid-span x–z slice: vorticity-like quantity and speed.
    # - A representative x–y slice at a chosen height: spanwise vorticity proxy.
    # - Optional 3D isosurface extraction of |ω| using marching_cubes.
    #
    # Notes:
    # - Pulling large 3D fields to host and plotting can dominate runtime; plot_freq
    #   and plot_step_start are intended to manage this cost.
    # - The vorticity estimates use simple centered finite differences in lattice
    #   units and are intended primarily for qualitative visualization.
    # -----------------------------------------------------------------------------
    # Host-side: gather data and plot.
    rho_np = rho.to_numpy()       # (Nz, Ny, Nx)
    ux_np = ux.to_numpy()  # Intermediate scalar 'ux_np' for this kernel block
    uy_np = uy.to_numpy()  # Intermediate scalar 'uy_np' for this kernel block
    uz_np = uz.to_numpy()  # Intermediate scalar 'uz_np' for this kernel block
    obs_np = obstacle.to_numpy().astype(bool)  # Intermediate scalar 'obs_np' for this kernel block

    speed = np.sqrt(ux_np**2 + uz_np**2)  # Intermediate scalar 'speed' for this kernel block
    speed_phys = speed * vel_scale  # Physical-units parameter (SI)
    speed_phys[obs_np] = np.nan  # Physical-units parameter (SI)
    max_speed_phys = np.nanmax(speed_phys)  # Physical-units parameter (SI)

    # Mid-span slice in y (j ~ Ny/2)
    mid_j = Ny // 2  # Intermediate scalar 'mid_j' for this kernel block
    rho_mid = rho_np[:, mid_j, :]   # (Nz, Nx)
    ux_mid = ux_np[:, mid_j, :]  # Intermediate scalar 'ux_mid' for this kernel block
    uz_mid = uz_np[:, mid_j, :]  # Intermediate scalar 'uz_mid' for this kernel block
    obs_mid = obs_np[:, mid_j, :]  # Intermediate scalar 'obs_mid' for this kernel block

    # Approximate vorticity in the x-z plane
    dv_dx = (np.roll(uz_mid, -1, axis=1) - np.roll(uz_mid, 1, axis=1)) / 2.0  # Intermediate scalar 'dv_dx' for this kernel block
    du_dz = (np.roll(ux_mid, -1, axis=0) - np.roll(ux_mid, 1, axis=0)) / 2.0  # Intermediate scalar 'du_dz' for this kernel block
    vort = dv_dx - du_dz  # Intermediate scalar 'vort' for this kernel block
    vort[obs_mid] = np.nan  # Intermediate scalar 'vort' for this kernel block

    # speed on x-z plane - same size as vort, with nan in obstacle
    speed_mid = np.sqrt(ux_mid**2 + uz_mid**2)  # Intermediate scalar 'speed_mid' for this kernel block
    speed_phys_xz = speed_mid * vel_scale  # Intermediate scalar 'speed_phys_xz' for this kernel block
    speed_phys_xz[obs_mid] = np.nan  # Intermediate scalar 'speed_phys_xz' for this kernel block

    c_s2 = 1.0 / 3.0  # Intermediate scalar 'c_s2' for this kernel block

    delta_p_lb = c_s2 * (rho_mid - rho0)  # Lattice-units parameter (LBM nondimensional)
    delta_p_static_phys = rho_air_phys * (vel_scale**2) * delta_p_lb  # Physical-units parameter (SI)
    delta_p_static_phys[obs_mid] = np.nan  # Physical-units parameter (SI)

    # Height slice in z (k ~ Nz/2)
    mid_k = int((bed_amp_m + offset_m) / dx) - 2  # Intermediate scalar 'mid_k' for this kernel block
    rho_mid = rho_np[mid_k, :, :]   # (Ny, Nx)
    ux_mid = ux_np[mid_k, :, :]  # Intermediate scalar 'ux_mid' for this kernel block
    uy_mid = uy_np[mid_k, :, :]  # Intermediate scalar 'uy_mid' for this kernel block
    obs_mid = obs_np[mid_k, :, :]  # Intermediate scalar 'obs_mid' for this kernel block

    # Approximate vorticity in the x-z plane
    dv_dx = (np.roll(uy_mid, -1, axis=1) - np.roll(uy_mid, 1, axis=1)) / 2.0  # Intermediate scalar 'dv_dx' for this kernel block
    du_dy = (np.roll(ux_mid, -1, axis=0) - np.roll(ux_mid, 1, axis=0)) / 2.0  # Intermediate scalar 'du_dy' for this kernel block
    vort_z = dv_dx - du_dy  # Intermediate scalar 'vort_z' for this kernel block
    vort_z[obs_mid] = np.nan  # Intermediate scalar 'vort_z' for this kernel block

    # 2D diagnostic plots on mid-span slice
    plt.figure(1, figsize=(10, 6))
    plt.clf()

    # Plot x-z mid-span vorticity slice
    ax1 = plt.subplot(3, 1, 1)  # Intermediate scalar 'ax1' for this kernel block
    X_slice = X_phys[:, mid_j, :]  # Intermediate scalar 'X_slice' for this kernel block
    Z_slice = Z_phys[:, mid_j, :]  # Intermediate scalar 'Z_slice' for this kernel block
    im1 = ax1.pcolormesh(X_slice, Z_slice, vort, shading='auto')
    plt.colorbar(im1, ax=ax1, label='Vorticity (lattice units)')
    #ax1.set_aspect('equal', 'box')
    ax1.set_xlim(0, Lx_m)
    ax1.set_ylim(0, Lz_m)
    #ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')
    im1.set_clim(-0.005, 0.005)
    ax1.set_title(
        f"Mid-span slice (y ~ {(mid_j + 0.5)*dx:.1f} m) | "
        f"t = {time_phys:.3f} s | U_max ≈ {max_speed_phys:.2f} m/s | "
        f"Re = {Re_phys_air:.2e}"
    )

    # Plot speed at x-z mid-span slice
    ax1b = plt.subplot(3, 1, 2)  # Intermediate scalar 'ax1b' for this kernel block
    im1b = ax1b.pcolormesh(X_slice, Z_slice, speed_phys_xz, shading='auto')
    plt.colorbar(im1b, ax=ax1b, label='Speed (m/s)')
    #ax1b.set_aspect('equal', 'box')
    ax1b.set_xlim(0, Lx_m)
    ax1b.set_ylim(0, Lz_m)
    #ax1b.set_xlabel('x (m)')
    ax1b.set_ylabel('z (m)')    
    im1b.set_clim(0, 1.2*U_ref_phys)
    ax1b.set_title("Speed (m/s) at mid-span slice")


    # Plot pressure at x-z mid-span slice
    ax1c = plt.subplot(3, 1, 3)  # Intermediate scalar 'ax1c' for this kernel block
    im1c = ax1c.pcolormesh(X_slice, Z_slice, delta_p_static_phys, shading='auto')
    plt.colorbar(im1c, ax=ax1c, label='Pressure (Pa)')
    #ax1b.set_aspect('equal', 'box')
    ax1c.set_xlim(0, Lx_m)
    ax1c.set_ylim(0, Lz_m)
    ax1c.set_xlabel('x (m)')
    ax1c.set_ylabel('z (m)')    
    pressure_scale = rho_air_phys*U_ref_phys*U_ref_phys  # Intermediate scalar 'pressure_scale' for this kernel block
    im1c.set_clim(-0.15*pressure_scale, 0.15*pressure_scale)
    ax1c.set_title("Pressure Delta (Pa) at mid-span slice")

    # Plot x-y vorticity slice at mid-height
    #ax2 = plt.subplot(3, 1, 3)
    #X_slice = X_phys[mid_k, :, :]
    #Y_slice = Y_phys[mid_k, :, :]
    #im2 = ax2.pcolormesh(X_slice, Y_slice, vort_z, shading='auto')
    #plt.colorbar(im2, ax=ax2, label='Vorticity (lattice units)')
    ##ax2.set_aspect('equal', 'box')
    #ax2.set_xlim(0, Lx_m)
    #ax2.set_ylim(0, Ly_m)
    #ax2.set_xlabel('x (m)')
    #ax2.set_ylabel('y (m)')
    #im2.set_clim(-0.005, 0.005)
    #ax2.set_title(f"Height slice (z ~ {(mid_k + 0.5)*dx:.2f} m)")

    if write_image:  # Branch for boundary/stability logic
        output_dir = "frames"
        os.makedirs(output_dir, exist_ok=True)
        out_fname = os.path.join(output_dir, f"wave_v6_lbm_output_{step:05d}.jpg")
        plt.tight_layout()
        plt.savefig(out_fname, dpi=600)
        print("Saved:", out_fname, flush=True)
    plt.pause(0.1)


    # 3D isosurface of vorticity magnitude, colored by local speed
    plot_isosurfaces = 0  # Intermediate scalar 'plot_isosurfaces' for this kernel block
    if Ny > 1 and plot_isosurfaces == 1:  # Branch for boundary/stability logic

        # ---- build 3D vorticity field ω_y(k,j,i) on full domain ----
        # (Nz, Ny, Nx) = (k, j, i)
        # same discrete formula as vort in the mid-plane:
        # ω_y = ∂w/∂x - ∂u/∂z  (here uy is vertical / "w")
        dv_dx_3d = (np.roll(uz_np, -1, axis=2) - np.roll(uz_np, 1, axis=2)) / 2.0  # Intermediate scalar 'dv_dx_3d' for this kernel block
        du_dz_3d = (np.roll(ux_np, -1, axis=0) - np.roll(ux_np, 1, axis=0)) / 2.0  # Intermediate scalar 'du_dz_3d' for this kernel block
        omega_3d = dv_dx_3d - du_dz_3d   # lattice units, consistent with vort

        # zero out inside obstacles and near top so we don't get surfaces in solids
        omega_abs = np.abs(omega_3d)  # Intermediate scalar 'omega_abs' for this kernel block
        omega_abs = np.where(obs_np, 0.0, omega_abs)  # Intermediate scalar 'omega_abs' for this kernel block
        top_pad = 10  # number of lattice layers to zero
        omega_abs[Nz - top_pad : Nz, :, :] = 0.0  # Intermediate scalar 'omega_abs' for this kernel block

        # choose isosurface level in lattice units
        omega_iso = 0.005  # typical small value; tweak as you like

        # marching_cubes expects (axis0, axis1, axis2) = (Nz, Ny, Nx)
        # spacing sets physical distance between grid points
        verts, faces, normals, values = marching_cubes(
            omega_abs, level=omega_iso, spacing=(dx, dx, dx)
        )
        # verts are (z, y, x) in physical units because of spacing
        # reorder to (x, y, z) for plotting
        verts_xyz = np.zeros_like(verts)  # Intermediate scalar 'verts_xyz' for this kernel block
        verts_xyz[:, 0] = verts[:, 2]  # x
        verts_xyz[:, 1] = verts[:, 1]  # y
        verts_xyz[:, 2] = verts[:, 0]  # z

        # ---- color by local speed |u| at the isosurface ----
        # sample speed_phys at nearest grid node for each vertex
        k_idx = np.clip((verts[:, 0] / dx).astype(int), 0, Nz - 1)  # z index
        j_idx = np.clip((verts[:, 1] / dx).astype(int), 0, Ny - 1)  # y index
        i_idx = np.clip((verts[:, 2] / dx).astype(int), 0, Nx - 1)  # x index
        u_vert = speed_phys[k_idx, j_idx, i_idx]  # Intermediate scalar 'u_vert' for this kernel block

        # get a scalar per face (average of its vertices)
        u_face = u_vert[faces].mean(axis=1)  # Intermediate scalar 'u_face' for this kernel block

        u_min = 0.0  # Intermediate scalar 'u_min' for this kernel block
        u_max = 2*U_top_phys  # Intermediate scalar 'u_max' for this kernel block
        u_norm = (u_face - u_min) / (u_max - u_min + 1e-12)  # Intermediate scalar 'u_norm' for this kernel block
        u_norm = np.clip(u_norm, 0.0, 1.0)  # Intermediate scalar 'u_norm' for this kernel block

        cmap = plt.get_cmap('jet')  # classic LES colorbar vibe
        face_colors = cmap(u_norm)  # Intermediate scalar 'face_colors' for this kernel block

        # ---- build Poly3DCollection and plot ----
        plt.figure(2, figsize=(10, 6))
        plt.clf()

        # Full-window 3D axes (no shrinking)
        ax = plt.axes([0.0, 0.0, 1.0, 1.0], projection='3d')

        mesh = Poly3DCollection(verts_xyz[faces], alpha=0.7)  # Intermediate scalar 'mesh' for this kernel block
        mesh.set_edgecolor('none')
        mesh.set_facecolor(face_colors)
        ax.add_collection3d(mesh)

        # Limits and labels
        ax.set_xlim(0, Lx_m)
        ax.set_ylim(0, Ny * dx)
        ax.set_zlim(0, Lz_m)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.set_title(
            f'Isosurface |ω| = {omega_iso:.3g} (lattice), colored by |u| (m/s)',
            pad=20  # Intermediate scalar 'pad' for this kernel block
        )

        # Physical aspect ratio
        #ax.set_box_aspect((Lx_m, Ly_m, Lz_m))
        ax.set_box_aspect((2, 1, 1))

        # ---- colorbar in independent axes ----
        cax = plt.axes([0.88, 0.15, 0.03, 0.7])  # right side
        mappable = plt.cm.ScalarMappable(cmap=cmap)  # Intermediate scalar 'mappable' for this kernel block
        mappable.set_array([u_min, u_max])
        cbar = plt.colorbar(mappable, cax=cax)  # Intermediate scalar 'cbar' for this kernel block
        cbar.set_label('|u| (m/s)')


        plt.tight_layout()
        plt.pause(0.1)

        
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
    init_constants()  # set physical/lattice scales, solver parameters, and constants
    update_wave_bed_and_velocities(0.0)  # initialize moving bed geometry + wall velocity at t=0
    compute_phi_slope_corrected()  # build signed-distance-like phi for interface/wall model
    build_lattice_open_from_free_surface_periodic()  # compute per-link open flags/fractions from water_h (periodic x/y)
    init_fields()  # initialize obstacle mask, inlet profile, velocity field, and f=feq

    print("Grid Size (Nx, Ny, Nz): ", Nx, Ny, Nz)  # runtime configuration summary
    print(f"Physical Re (air): {Re_phys_air:.3e}")  # diagnostic Reynolds number in physical units
    print(f"Lattice Re (base): {Re_lb_base:.2f}")  # diagnostic Reynolds number in lattice units

    for step in range(1, steps + 1):  # main time-marching loop over integer timesteps
        time_phys = (step + 1) * dt_phys  # physical time corresponding to this step (SI units)

        update_wave_bed_and_velocities(time_phys)  # update bed position and prescribed wall velocity at current time
        compute_phi_slope_corrected()  # refresh phi after bed/interface motion
        build_lattice_open_from_free_surface_periodic()  # refresh per-link openness after interface update
        update_bed_populations_and_reinit(step)  # blend closed links toward moving-wall equilibrium populations
        macro_step(step)  # reconstruct rho and u from f (zeroth/first moments)

        if use_LES:  # optional LES closure
            compute_LES()  # compute eddy viscosity / effective relaxation parameters from local strain rate

        collide_KBC()  # collision/relaxation (with limiter/backscatter as implemented)
        stream()  # pull-stream populations into f_new
        apply_open_boundary_conditions()  # fringe inlet forcing, top boundary, and near-bed IBB handling
        copy_post_and_swap()  # commit f_new -> f for the next timestep

        if step % plot_freq == 0 and step > plot_step_start:  # periodic plotting window
            visualize(step, time_phys, Re_phys_air, Re_lb_base)  # visualization / output

        if step % 100 == 0:  # periodic stability/health report
            ux_np = ux.to_numpy()  # copy ux from Taichi field to NumPy for reduction
            obstacle_np = obstacle.to_numpy()  # copy obstacle mask to NumPy for filtering
            max_u = np.max(ux_np[obstacle_np == 0])  # max ux over fluid cells only (exclude obstacle)
            print(  # console diagnostic: max speed in lattice and approximate physical units
                f"Time Step {step}, Max U (LB units) = {max_u:.4f}, "
                f"Max U (phys) ~ {max_u * vel_scale:.2f} m/s"
            )

    print("Simulation Complete.")  # end-of-run marker


if __name__ == "__main__":
    main()  # execute driver when run as a script
