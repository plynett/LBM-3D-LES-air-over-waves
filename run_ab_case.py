import argparse
import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_FIELDNAMES = [
    "label", "step", "time_phys", "max_speed_phys", "max_speed_x",
    "max_speed_y", "max_speed_z", "max_speed_fs_type",
    "max_speed_fs_fill", "max_speed_wet_neighbors",
    "max_speed_full_neighbors", "max_speed_gas_neighbors", "p_rms",
    "p_hp_rms", "p_total_rms", "p_total_hp_rms",
    "p_linear_resid_rms", "p_linear_resid_hp_rms", "vort_rms",
    "vort_hp_rms", "fs_p_rms", "fs_p_hp_rms", "fs_p_total_rms",
    "fs_p_total_hp_rms", "fs_p_linear_resid_rms",
    "fs_p_linear_resid_hp_rms", "fs_vort_rms", "fs_vort_hp_rms", "cut_links",
    "fs_cut_links", "refill", "fs_refill", "fs_coalesce", "fs_bridge",
    "vof_detached_count", "vof_detached_mass",
    "vof_compact_count", "vof_compact_mass",
    "surface_min", "surface_max",
    "water_volume", "vof_mass_total", "vof_excess_count",
    "vof_orphan_count", "vof_collapse_cells",
    "vof_collapse_candidate_volume", "vof_collapse_applied_volume",
    "vof_min_interface_fill", "vof_tiny_interface_count",
    "vof_pending_excess_mass", "vof_pending_excess_count",
    "surface_symmetry_l2", "surface_eta_hp_rms", "surface_left_max",
    "surface_right_max",
]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def finite_depth_linear_pressure(module, eta_line, z_slice, bed_line, rho_phys):
    """Linear finite-depth pressure implied by the instantaneous surface shape.

    This is a diagnostic reference for flat-bottom Gaussian-wave tests. It is
    not used to modify the simulation state.
    """
    if not np.allclose(bed_line, bed_line[0], atol=1.0e-12, rtol=0.0):
        return None

    x = module.x_phys.astype(np.float64)
    z = z_slice.astype(np.float64)
    bed = float(bed_line[0])
    depth = max(float(module.free_surface_base_m) - bed, float(module.dx))
    eta = eta_line.astype(np.float64)
    gravity = float(getattr(module, "gravity_phys", 0.0))

    if getattr(module, "x_boundary_mode", "periodic") == "solid":
        length = float(module.Nx) * float(module.dx)
        n = np.arange(module.Nx, dtype=np.float64)
        kx = n * np.pi / length
        cos_matrix = np.cos(np.outer(kx, x))
        coeff = (2.0 / module.Nx) * (cos_matrix @ eta)
        coeff[0] = np.mean(eta)

        eta_pressure = np.zeros_like(z, dtype=np.float64)
        for kk in range(module.Nz):
            z_rel = np.clip(z[kk, 0] - bed, 0.0, depth)
            vertical = np.ones_like(kx)
            active = kx > 0.0
            kh = kx[active] * depth
            kz = kx[active] * z_rel
            vertical[active] = (
                np.exp(kz - kh) + np.exp(-kz - kh)
            ) / (1.0 + np.exp(-2.0 * kh))
            eta_pressure[kk, :] = coeff @ (vertical[:, None] * cos_matrix)
    else:
        modes = np.fft.rfftfreq(module.Nx, d=float(module.dx)) * 2.0 * np.pi
        eta_hat = np.fft.rfft(eta)
        eta_pressure = np.zeros_like(z, dtype=np.float64)
        for kk in range(module.Nz):
            z_rel = np.clip(z[kk, 0] - bed, 0.0, depth)
            vertical = np.ones_like(modes)
            active = modes > 0.0
            kh = modes[active] * depth
            kz = modes[active] * z_rel
            vertical[active] = (
                np.exp(kz - kh) + np.exp(-kz - kh)
            ) / (1.0 + np.exp(-2.0 * kh))
            eta_pressure[kk, :] = np.fft.irfft(eta_hat * vertical, n=module.Nx)

    if getattr(module, "use_hydrostatic_balanced_pressure", 0) == 1:
        pressure_head = eta_pressure
    else:
        base_head = np.maximum(float(module.free_surface_base_m) - z, 0.0)
        pressure_head = base_head + eta_pressure
    return rho_phys * gravity * pressure_head


def high_pass_xz(field: np.ndarray, x_periodic: bool) -> np.ndarray:
    """Grid-scale residual without wrapping the vertical direction.

    The old diagnostic used np.roll in z, which makes a linear hydrostatic
    pressure column look noisy by wrapping the bottom row against the top row.
    Linear ghost values make the residual zero for a linear pressure profile.
    """
    z_plus = np.empty_like(field)
    z_minus = np.empty_like(field)
    x_plus = np.empty_like(field)
    x_minus = np.empty_like(field)

    z_plus[:-1, :] = field[1:, :]
    z_plus[-1, :] = 2.0 * field[-1, :] - field[-2, :]
    z_minus[1:, :] = field[:-1, :]
    z_minus[0, :] = 2.0 * field[0, :] - field[1, :]

    if x_periodic:
        x_plus[:, :] = np.roll(field, -1, axis=1)
        x_minus[:, :] = np.roll(field, 1, axis=1)
    else:
        x_plus[:, :-1] = field[:, 1:]
        x_plus[:, -1] = 2.0 * field[:, -1] - field[:, -2]
        x_minus[:, 1:] = field[:, :-1]
        x_minus[:, 0] = 2.0 * field[:, 0] - field[:, 1]

    return field - 0.25 * (z_plus + z_minus + x_plus + x_minus)


def save_render_snapshot(
    module,
    *,
    label: str,
    step: int,
    time_phys: float,
    output_dir: Path,
    free_surface_enabled: bool,
    tracking_mode: str,
    fs_fill_np: np.ndarray | None,
    free_surface_h_np: np.ndarray | None,
    water_h_np: np.ndarray,
    obs_np: np.ndarray,
):
    if not env_bool("LBM_SAVE_RENDER_SNAPSHOT", False):
        return None

    snapshot_path = output_dir / f"{label}_render_{step:05d}.npz"
    data = {
        "time_phys": np.array(time_phys, dtype=np.float64),
        "dx": np.array(module.dx, dtype=np.float64),
        "lx_m": np.array(module.Lx_m, dtype=np.float64),
        "ly_m": np.array(module.Ly_m, dtype=np.float64),
        "lz_m": np.array(module.Lz_m, dtype=np.float64),
        "x_phys": module.x_phys.astype(np.float32),
        "y_phys": module.y_phys.astype(np.float32),
        "z_phys": module.z_phys.astype(np.float32),
        "water_h": water_h_np.astype(np.float32),
        "obstacle": obs_np.astype(np.uint8),
        "free_surface_enabled": np.array(int(free_surface_enabled), dtype=np.int8),
        "tracking_mode": np.array(str(tracking_mode)),
        "include_cube_geometry": np.array(int(getattr(module, "include_cube_geometry", 0)), dtype=np.int8),
        "cube_size_m": np.array(
            [
                float(getattr(module, "cube_size_x_m", getattr(module, "cube_side_m", 0.0))),
                float(getattr(module, "cube_size_y_m", getattr(module, "cube_side_m", 0.0))),
                float(getattr(module, "cube_size_z_m", getattr(module, "cube_side_m", 0.0))),
            ],
            dtype=np.float32,
        ),
        "cube_center_m": np.array(
            [
                float(getattr(module, "cube_center_x_m", 0.0)),
                float(getattr(module, "cube_center_y_m", 0.0)),
                float(getattr(module, "cube_center_z_m", 0.0)),
            ],
            dtype=np.float32,
        ),
        "cube_yaw_pitch_deg": np.array(
            [
                float(getattr(module, "cube_yaw_deg", 0.0)),
                float(getattr(module, "cube_pitch_deg", 0.0)),
            ],
            dtype=np.float32,
        ),
    }
    if free_surface_h_np is not None:
        data["free_surface_h"] = free_surface_h_np.astype(np.float32)
    if free_surface_enabled and fs_fill_np is not None:
        data["free_surface_fill"] = fs_fill_np.astype(np.float32)

    np.savez_compressed(snapshot_path, **data)
    return snapshot_path


RENDER_CLI_ENV_ARGS = (
    ("--size", "LBM_RENDER_SIZE", "2560,1440"),
    ("--camera", "LBM_RENDER_CAMERA_M", "155,-30,28"),
    ("--target", "LBM_RENDER_TARGET_M", "75,10,11.5"),
    ("--zoom", "LBM_RENDER_ZOOM", "1.1"),
    ("--view-angle", "LBM_RENDER_VIEW_ANGLE_DEG", "26"),
    ("--stride", "LBM_RENDER_STRIDE", "1"),
    ("--upsample", "LBM_RENDER_UPSAMPLE", "2"),
    ("--interp-order", "LBM_RENDER_INTERP_ORDER", "1"),
    ("--iso-level", "LBM_RENDER_ISO_LEVEL", "0.5"),
    ("--smooth-iter", "LBM_RENDER_SMOOTH_ITER", "30"),
    ("--smooth-pass-band", "LBM_RENDER_SMOOTH_PASS_BAND", "0.05"),
    ("--water-opacity", "LBM_RENDER_WATER_OPACITY", "0.5"),
    ("--water-color", "LBM_RENDER_WATER_COLOR", "#36b7e6"),
    ("--obstacle-color", "LBM_RENDER_OBSTACLE_COLOR", "#2c2c30"),
    ("--bed-color", "LBM_RENDER_BED_COLOR", "#4b4741"),
    ("--obstacle-source", "LBM_RENDER_OBSTACLE_SOURCE", "auto"),
)


def render_snapshot_inline(snapshot_path: Path):
    if snapshot_path is None or not env_bool("LBM_RENDER_SNAPSHOT_INLINE", False):
        return None

    inline_every = max(1, env_int("LBM_RENDER_SNAPSHOT_INLINE_EVERY", 1))
    plot_freq = max(1, env_int("LBM_RENDER_PLOT_FREQ", env_int("LBM_PLOT_FREQ", 1)))
    try:
        step = int(snapshot_path.stem.rsplit("_", 1)[-1])
    except ValueError:
        step = 0
    if step > 0 and (step // plot_freq) % inline_every != 0:
        return None

    render_script = Path(__file__).resolve().parent / "render_lbm_snapshot.py"
    render_output = snapshot_path.with_name(snapshot_path.stem + "_pyvista.png")
    cmd = [
        sys.executable,
        str(render_script),
        str(snapshot_path),
        "--out",
        str(render_output),
    ]
    for arg_name, env_name, default in RENDER_CLI_ENV_ARGS:
        cmd.extend((arg_name, os.environ.get(env_name, default)))
    if env_bool("LBM_RENDER_NO_BED", False):
        cmd.append("--no-bed")
    if env_bool("LBM_RENDER_SHOW_WINDOW", False):
        cmd.append("--show")

    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
    if result.returncode != 0:
        print(f"PyVista render failed for {snapshot_path} with code {result.returncode}", flush=True)
        return None
    return render_output


def load_solver(path: Path, label: str, random_seed: int | None):
    if random_seed is not None:
        import taichi as ti

        original_init = ti.init

        def init_with_seed(*args, **kwargs):
            kwargs.setdefault("random_seed", random_seed)
            return original_init(*args, **kwargs)

        ti.init = init_with_seed

    spec = importlib.util.spec_from_file_location(f"lbm_ab_{label}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load solver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_visualize(module, label: str, output_dir: Path, metrics_writer):
    output_dir.mkdir(parents=True, exist_ok=True)

    def visualize(step, time_phys, Re_phys_air, Re_lb_base):
        rho_np = module.rho.to_numpy()
        ux_np = module.ux.to_numpy()
        uy_np = module.uy.to_numpy()
        uz_np = module.uz.to_numpy()
        obs_np = module.obstacle.to_numpy().astype(bool)
        phi_np = module.phi.to_numpy()
        water_h_np = module.water_h.to_numpy()
        free_surface_enabled = getattr(module, "water_free_surface_enabled", 0) == 1
        tracking_mode = getattr(module, "free_surface_tracking_mode", "height")
        inactive_np = obs_np.copy()
        if free_surface_enabled:
            fs_type_np = module.free_surface_type.to_numpy()
            fs_fill_np = module.free_surface_fill.to_numpy()
            inactive_np = inactive_np | ((fs_type_np == module.FS_GAS) & (~obs_np))
            free_surface_h_np = module.free_surface_h.to_numpy()
            surface_min = float(np.nanmin(free_surface_h_np))
            surface_max = float(np.nanmax(free_surface_h_np))
            water_volume = float(np.nansum(np.where(obs_np, 0.0, fs_fill_np)) * module.dx**3)
            vof_mass = getattr(module, "free_surface_mass", None)
            if vof_mass is not None and tracking_mode == "vof":
                vof_mass_np = vof_mass.to_numpy()
                vof_mass_total = float(np.nansum(np.where(obs_np, 0.0, vof_mass_np)))
            else:
                vof_mass_total = np.nan
            vof_excess = getattr(module, "free_surface_vof_excess_count", None)
            vof_excess_count = int(vof_excess[None]) if vof_excess is not None and tracking_mode == "vof" else -1
            vof_orphans = getattr(module, "free_surface_vof_orphan_count", None)
            vof_orphan_count = int(vof_orphans[None]) if vof_orphans is not None and tracking_mode == "vof" else -1
            vof_collapse_cells_field = getattr(module, "free_surface_vof_collapse_cell_count", None)
            vof_collapse_candidate_field = getattr(module, "free_surface_vof_collapse_candidate_volume_m3", None)
            vof_collapse_applied_field = getattr(module, "free_surface_vof_collapse_applied_volume_m3", None)
            vof_collapse_cells = int(vof_collapse_cells_field[None]) if vof_collapse_cells_field is not None and tracking_mode == "vof" else -1
            vof_collapse_candidate_volume = float(vof_collapse_candidate_field[None]) if vof_collapse_candidate_field is not None and tracking_mode == "vof" else np.nan
            vof_collapse_applied_volume = float(vof_collapse_applied_field[None]) if vof_collapse_applied_field is not None and tracking_mode == "vof" else np.nan
            if tracking_mode == "vof":
                interface_mask = (fs_type_np == module.FS_INTERFACE) & (~obs_np)
                interface_fills = fs_fill_np[interface_mask]
                vof_min_interface_fill = float(np.nanmin(interface_fills)) if interface_fills.size else np.nan
                vof_empty_threshold = float(getattr(module, "vof_empty_fill_threshold", 0.0))
                vof_tiny_interface_count = int(np.count_nonzero(
                    interface_mask
                    & (fs_fill_np > 0.0)
                    & (fs_fill_np < vof_empty_threshold)
                ))
                vof_excess_field = getattr(module, "free_surface_mass_excess", None)
                if vof_excess_field is not None:
                    vof_excess_np = vof_excess_field.to_numpy()
                    vof_pending_excess_mass = float(np.nansum(np.where(obs_np, 0.0, vof_excess_np)))
                    vof_pending_excess_count = int(np.count_nonzero(
                        np.abs(np.where(obs_np, 0.0, vof_excess_np)) > 1.0e-12
                    ))
                else:
                    vof_pending_excess_mass = np.nan
                    vof_pending_excess_count = -1
            else:
                vof_min_interface_fill = np.nan
                vof_tiny_interface_count = -1
                vof_pending_excess_mass = np.nan
                vof_pending_excess_count = -1
                vof_orphan_count = -1
            eta_line = free_surface_h_np.mean(axis=0) - module.free_surface_base_m
            eta_ref = eta_line[::-1]
            surface_symmetry_l2 = float(np.sqrt(np.nanmean((eta_line - eta_ref) ** 2)))
            x_line_centers = module.x_phys
            left_mask = x_line_centers < 0.5 * module.Lx_m
            right_mask = ~left_mask
            surface_left_max = float(np.nanmax(eta_line[left_mask]))
            surface_right_max = float(np.nanmax(eta_line[right_mask]))
            if getattr(module, "x_boundary_mode", "periodic") == "solid":
                eta_hp = eta_line.copy()
                eta_hp[1:-1] = eta_line[1:-1] - 0.5 * (eta_line[:-2] + eta_line[2:])
                eta_hp[0] = eta_line[0] - eta_line[1]
                eta_hp[-1] = eta_line[-1] - eta_line[-2]
            else:
                eta_hp = eta_line - 0.5 * (np.roll(eta_line, 1) + np.roll(eta_line, -1))
            surface_eta_hp_rms = float(np.sqrt(np.nanmean(eta_hp**2)))
        else:
            free_surface_h_np = water_h_np
            surface_min = np.nan
            surface_max = np.nan
            water_volume = np.nan
            vof_mass_total = np.nan
            vof_excess_count = -1
            vof_orphan_count = -1
            vof_collapse_cells = -1
            vof_collapse_candidate_volume = np.nan
            vof_collapse_applied_volume = np.nan
            vof_min_interface_fill = np.nan
            vof_tiny_interface_count = -1
            vof_pending_excess_mass = np.nan
            vof_pending_excess_count = -1
            surface_symmetry_l2 = np.nan
            surface_left_max = np.nan
            surface_right_max = np.nan
            surface_eta_hp_rms = np.nan

        speed = np.sqrt(ux_np**2 + uy_np**2 + uz_np**2)
        speed_phys = speed * module.vel_scale
        speed_phys[inactive_np] = np.nan
        max_speed_phys = np.nanmax(speed_phys)
        max_speed_index = np.nanargmax(speed_phys)
        max_k, max_j, max_i = np.unravel_index(max_speed_index, speed_phys.shape)
        max_x = float(module.x_phys[max_i])
        max_y = float(module.y_phys[max_j])
        max_z = float(module.z_phys[max_k])
        if free_surface_enabled:
            max_fs_type = int(fs_type_np[max_k, max_j, max_i])
            max_fs_fill = float(fs_fill_np[max_k, max_j, max_i])
            cx_np = module.cx.to_numpy()
            cy_np = module.cy.to_numpy()
            cz_np = module.cz.to_numpy()
            max_wet_neighbors = 0
            max_full_neighbors = 0
            max_gas_neighbors = 0
            for qn in range(1, 19):
                ni_raw = max_i + int(cx_np[qn])
                valid = True
                if getattr(module, "x_boundary_mode", "periodic") == "periodic":
                    ni = (ni_raw + module.Nx) % module.Nx
                else:
                    ni = ni_raw
                    if ni_raw < 0 or ni_raw >= module.Nx:
                        valid = False
                nj = (max_j + int(cy_np[qn])) % module.Ny
                nk = max_k + int(cz_np[qn])
                if nk < 0 or nk >= module.Nz:
                    valid = False
                if valid and not obs_np[nk, nj, ni]:
                    if fs_type_np[nk, nj, ni] == module.FS_GAS:
                        max_gas_neighbors += 1
                    else:
                        max_wet_neighbors += 1
                        if fs_fill_np[nk, nj, ni] > 0.5:
                            max_full_neighbors += 1
        else:
            max_fs_type = -1
            max_fs_fill = np.nan
            max_wet_neighbors = -1
            max_full_neighbors = -1
            max_gas_neighbors = -1

        mid_j = module.Ny // 2
        rho_xz = rho_np[:, mid_j, :]
        ux_xz = ux_np[:, mid_j, :]
        uy_xz = uy_np[:, mid_j, :]
        uz_xz = uz_np[:, mid_j, :]
        obs_xz = inactive_np[:, mid_j, :]
        z_slice = module.Z_phys[:, mid_j, :]
        bed_line_mid = water_h_np[mid_j, :]
        surface_line_mid = free_surface_h_np[mid_j, :]
        vof_fill_slice = None
        if free_surface_enabled and tracking_mode == "vof":
            vof_fill_slice = fs_fill_np[:, mid_j, :]

        if getattr(module, "x_boundary_mode", "periodic") == "solid":
            dv_dx = np.empty_like(uz_xz)
            dv_dx[:, 1:-1] = (uz_xz[:, 2:] - uz_xz[:, :-2]) / 2.0
            dv_dx[:, 0] = uz_xz[:, 1] - uz_xz[:, 0]
            dv_dx[:, -1] = uz_xz[:, -1] - uz_xz[:, -2]
        else:
            dv_dx = (np.roll(uz_xz, -1, axis=1) - np.roll(uz_xz, 1, axis=1)) / 2.0
        du_dz = (np.roll(ux_xz, -1, axis=0) - np.roll(ux_xz, 1, axis=0)) / 2.0
        vort_raw = dv_dx - du_dz
        x_periodic = getattr(module, "x_boundary_mode", "periodic") == "periodic"
        vort_hp = high_pass_xz(vort_raw, x_periodic)
        vort = vort_raw.copy()
        vort[obs_xz] = np.nan

        speed_xz_phys = np.sqrt(ux_xz**2 + uy_xz**2 + uz_xz**2) * module.vel_scale
        speed_xz_phys[obs_xz] = np.nan

        delta_p = (1.0 / 3.0) * (rho_xz - module.rho0)
        rho_phys = getattr(module, "rho_phys", module.rho_air_phys)
        pressure_lbm_phys = rho_phys * (module.vel_scale**2) * delta_p
        balanced_pressure = getattr(module, "use_hydrostatic_balanced_pressure", 0) == 1
        component_balanced_pressure = getattr(module, "use_component_balanced_pressure", 0) == 1
        if free_surface_enabled:
            gravity = getattr(module, "gravity_phys", 0.0)
            use_column_pressure_diagnostic = tracking_mode != "vof"
            if component_balanced_pressure:
                ref_head_field = getattr(module, "pressure_reference_head", None)
                if ref_head_field is None:
                    reference_head_xz = np.zeros_like(pressure_lbm_phys)
                else:
                    reference_head_xz = ref_head_field.to_numpy()[:, mid_j, :]
                pressure_total_phys = pressure_lbm_phys + rho_phys * gravity * reference_head_xz
                surface_z = None
                hydrostatic_phys = np.zeros_like(pressure_total_phys)
                pressure_phys = pressure_total_phys
                pressure_compare_phys = pressure_total_phys
            elif balanced_pressure:
                base_z = float(module.free_surface_base_m)
                base_hydrostatic_phys = rho_phys * gravity * (base_z - z_slice)
                pressure_total_phys = pressure_lbm_phys + base_hydrostatic_phys
                if use_column_pressure_diagnostic:
                    surface_z = free_surface_h_np[mid_j, :][None, :]
                    hydrostatic_phys = rho_phys * gravity * np.maximum(surface_z - z_slice, 0.0)
                    pressure_phys = pressure_lbm_phys
                    pressure_compare_phys = pressure_lbm_phys
                else:
                    surface_z = None
                    hydrostatic_phys = np.zeros_like(pressure_total_phys)
                    pressure_phys = pressure_total_phys
                    pressure_compare_phys = pressure_total_phys
            else:
                pressure_total_phys = pressure_lbm_phys
                if use_column_pressure_diagnostic:
                    surface_z = free_surface_h_np[mid_j, :][None, :]
                    hydrostatic_phys = rho_phys * gravity * np.maximum(surface_z - z_slice, 0.0)
                    pressure_phys = pressure_total_phys - hydrostatic_phys
                else:
                    surface_z = None
                    hydrostatic_phys = np.zeros_like(pressure_total_phys)
                    pressure_phys = pressure_total_phys
                pressure_compare_phys = pressure_total_phys
        else:
            surface_z = None
            pressure_total_phys = pressure_lbm_phys
            hydrostatic_phys = np.zeros_like(pressure_total_phys)
            pressure_phys = pressure_total_phys
            pressure_compare_phys = pressure_total_phys
        pressure_hp = high_pass_xz(pressure_phys, x_periodic)
        pressure_total_hp = high_pass_xz(pressure_total_phys, x_periodic)
        pressure_linear_expected = None
        pressure_linear_residual = np.full_like(pressure_phys, np.nan)
        pressure_linear_residual_hp = np.full_like(pressure_phys, np.nan)
        if free_surface_enabled and tracking_mode != "vof":
            pressure_linear_expected = finite_depth_linear_pressure(
                module,
                surface_line_mid - module.free_surface_base_m,
                z_slice,
                bed_line_mid,
                rho_phys,
            )
            if pressure_linear_expected is not None:
                pressure_linear_residual = pressure_compare_phys - pressure_linear_expected
                pressure_linear_residual_hp = high_pass_xz(pressure_linear_residual, x_periodic)
        pressure_plot = pressure_phys.copy()
        pressure_plot[obs_xz] = np.nan

        phi_xz = phi_np[:, mid_j, :]
        band = (phi_xz > 0.0) & (phi_xz <= 10.0 * module.dx) & (~obs_xz)
        p_rms = np.nan
        p_hp_rms = np.nan
        p_total_rms = np.nan
        p_total_hp_rms = np.nan
        p_linear_resid_rms = np.nan
        p_linear_resid_hp_rms = np.nan
        vort_rms = np.nan
        vort_hp_rms = np.nan
        fs_p_rms = np.nan
        fs_p_hp_rms = np.nan
        fs_p_total_rms = np.nan
        fs_p_total_hp_rms = np.nan
        fs_p_linear_resid_rms = np.nan
        fs_p_linear_resid_hp_rms = np.nan
        fs_vort_rms = np.nan
        fs_vort_hp_rms = np.nan
        if np.any(band):
            p_vals = pressure_phys[band]
            p_hp_vals = pressure_hp[band]
            p_total_vals = pressure_total_phys[band]
            p_total_hp_vals = pressure_total_hp[band]
            p_linear_resid_vals = pressure_linear_residual[band]
            p_linear_resid_hp_vals = pressure_linear_residual_hp[band]
            v_vals = vort_raw[band]
            v_hp_vals = vort_hp[band]
            p_rms = float(np.sqrt(np.nanmean((p_vals - np.nanmean(p_vals)) ** 2)))
            p_hp_rms = float(np.sqrt(np.nanmean(p_hp_vals**2)))
            p_total_rms = float(np.sqrt(np.nanmean((p_total_vals - np.nanmean(p_total_vals)) ** 2)))
            p_total_hp_rms = float(np.sqrt(np.nanmean(p_total_hp_vals**2)))
            if np.any(np.isfinite(p_linear_resid_vals)):
                p_linear_resid_rms = float(np.sqrt(np.nanmean((p_linear_resid_vals - np.nanmean(p_linear_resid_vals)) ** 2)))
                p_linear_resid_hp_rms = float(np.sqrt(np.nanmean(p_linear_resid_hp_vals**2)))
            vort_rms = float(np.sqrt(np.nanmean((v_vals - np.nanmean(v_vals)) ** 2)))
            vort_hp_rms = float(np.sqrt(np.nanmean(v_hp_vals**2)))

        if free_surface_enabled:
            if tracking_mode == "vof" and vof_fill_slice is not None:
                interface_seed = (
                    (vof_fill_slice > 0.0)
                    & (vof_fill_slice < 1.0)
                    & (~obs_xz)
                )
                fs_band = interface_seed.copy()
                fs_band[1:, :] |= interface_seed[:-1, :]
                fs_band[:-1, :] |= interface_seed[1:, :]
                fs_band[:, 1:] |= interface_seed[:, :-1]
                fs_band[:, :-1] |= interface_seed[:, 1:]
                fs_band &= ~obs_xz
            else:
                fs_band = (
                    (z_slice <= surface_z + 0.5 * module.dx)
                    & (z_slice >= surface_z - 3.0 * module.dx)
                    & (~obs_xz)
                )
            if np.any(fs_band):
                fs_p_vals = pressure_phys[fs_band]
                fs_p_hp_vals = pressure_hp[fs_band]
                fs_p_total_vals = pressure_total_phys[fs_band]
                fs_p_total_hp_vals = pressure_total_hp[fs_band]
                fs_p_linear_resid_vals = pressure_linear_residual[fs_band]
                fs_p_linear_resid_hp_vals = pressure_linear_residual_hp[fs_band]
                fs_v_vals = vort_raw[fs_band]
                fs_v_hp_vals = vort_hp[fs_band]
                fs_p_rms = float(np.sqrt(np.nanmean((fs_p_vals - np.nanmean(fs_p_vals)) ** 2)))
                fs_p_hp_rms = float(np.sqrt(np.nanmean(fs_p_hp_vals**2)))
                fs_p_total_rms = float(np.sqrt(np.nanmean((fs_p_total_vals - np.nanmean(fs_p_total_vals)) ** 2)))
                fs_p_total_hp_rms = float(np.sqrt(np.nanmean(fs_p_total_hp_vals**2)))
                if np.any(np.isfinite(fs_p_linear_resid_vals)):
                    fs_p_linear_resid_rms = float(np.sqrt(np.nanmean((fs_p_linear_resid_vals - np.nanmean(fs_p_linear_resid_vals)) ** 2)))
                    fs_p_linear_resid_hp_rms = float(np.sqrt(np.nanmean(fs_p_linear_resid_hp_vals**2)))
                fs_vort_rms = float(np.sqrt(np.nanmean((fs_v_vals - np.nanmean(fs_v_vals)) ** 2)))
                fs_vort_hp_rms = float(np.sqrt(np.nanmean(fs_v_hp_vals**2)))

        cut_links = getattr(module, "boundary_cut_link_count", None)
        fs_cut_links = getattr(module, "free_surface_cut_link_count", None)
        refill = getattr(module, "boundary_refill_count", None)
        fs_refill = getattr(module, "free_surface_refill_count", None)
        fs_coalesce = getattr(module, "free_surface_coalescence_repair_count", None)
        fs_bridge = getattr(module, "free_surface_thin_gap_bridge_count", None)
        vof_detached_count_field = getattr(module, "free_surface_vof_detached_advect_count", None)
        vof_detached_mass_field = getattr(module, "free_surface_vof_detached_advect_mass", None)
        vof_compact_count_field = getattr(module, "free_surface_vof_detached_compress_count", None)
        vof_compact_mass_field = getattr(module, "free_surface_vof_detached_compress_mass", None)
        cut_links_val = int(cut_links[None]) if cut_links is not None else -1
        fs_cut_links_val = int(fs_cut_links[None]) if fs_cut_links is not None else -1
        refill_val = int(refill[None]) if refill is not None else -1
        fs_refill_val = int(fs_refill[None]) if fs_refill is not None else -1
        fs_coalesce_val = int(fs_coalesce[None]) if fs_coalesce is not None else -1
        fs_bridge_val = int(fs_bridge[None]) if fs_bridge is not None else -1
        vof_detached_count = int(vof_detached_count_field[None]) if vof_detached_count_field is not None else -1
        vof_detached_mass = float(vof_detached_mass_field[None]) if vof_detached_mass_field is not None else np.nan
        vof_compact_count = int(vof_compact_count_field[None]) if vof_compact_count_field is not None else -1
        vof_compact_mass = float(vof_compact_mass_field[None]) if vof_compact_mass_field is not None else np.nan

        metrics_writer.writerow({
            "label": label,
            "step": step,
            "time_phys": time_phys,
            "max_speed_phys": max_speed_phys,
            "max_speed_x": max_x,
            "max_speed_y": max_y,
            "max_speed_z": max_z,
            "max_speed_fs_type": max_fs_type,
            "max_speed_fs_fill": max_fs_fill,
            "max_speed_wet_neighbors": max_wet_neighbors,
            "max_speed_full_neighbors": max_full_neighbors,
            "max_speed_gas_neighbors": max_gas_neighbors,
            "p_rms": p_rms,
            "p_hp_rms": p_hp_rms,
            "p_total_rms": p_total_rms,
            "p_total_hp_rms": p_total_hp_rms,
            "p_linear_resid_rms": p_linear_resid_rms,
            "p_linear_resid_hp_rms": p_linear_resid_hp_rms,
            "vort_rms": vort_rms,
            "vort_hp_rms": vort_hp_rms,
            "fs_p_rms": fs_p_rms,
            "fs_p_hp_rms": fs_p_hp_rms,
            "fs_p_total_rms": fs_p_total_rms,
            "fs_p_total_hp_rms": fs_p_total_hp_rms,
            "fs_p_linear_resid_rms": fs_p_linear_resid_rms,
            "fs_p_linear_resid_hp_rms": fs_p_linear_resid_hp_rms,
            "fs_vort_rms": fs_vort_rms,
            "fs_vort_hp_rms": fs_vort_hp_rms,
            "cut_links": cut_links_val,
            "fs_cut_links": fs_cut_links_val,
            "refill": refill_val,
            "fs_refill": fs_refill_val,
            "fs_coalesce": fs_coalesce_val,
            "fs_bridge": fs_bridge_val,
            "vof_detached_count": vof_detached_count,
            "vof_detached_mass": vof_detached_mass,
            "vof_compact_count": vof_compact_count,
            "vof_compact_mass": vof_compact_mass,
            "surface_min": surface_min,
            "surface_max": surface_max,
            "water_volume": water_volume,
            "vof_mass_total": vof_mass_total,
            "vof_excess_count": vof_excess_count,
            "vof_orphan_count": vof_orphan_count,
            "vof_collapse_cells": vof_collapse_cells,
            "vof_collapse_candidate_volume": vof_collapse_candidate_volume,
            "vof_collapse_applied_volume": vof_collapse_applied_volume,
            "vof_min_interface_fill": vof_min_interface_fill,
            "vof_tiny_interface_count": vof_tiny_interface_count,
            "vof_pending_excess_mass": vof_pending_excess_mass,
            "vof_pending_excess_count": vof_pending_excess_count,
            "surface_symmetry_l2": surface_symmetry_l2,
            "surface_eta_hp_rms": surface_eta_hp_rms,
            "surface_left_max": surface_left_max,
            "surface_right_max": surface_right_max,
        })

        plt.figure(1, figsize=(10, 6))
        plt.clf()
        x_slice = module.X_phys[:, mid_j, :]
        x_line = module.x_phys
        bed_line = bed_line_mid
        surface_line = surface_line_mid
        band_line = bed_line + 10.0 * module.dx
        cube_sdf_slice = None
        if getattr(module, "include_cube_geometry", 0) == 1:
            half_x = float(getattr(module, "cube_half_x_m", 0.5 * module.cube_side_m))
            half_y = float(getattr(module, "cube_half_y_m", 0.5 * module.cube_side_m))
            half_z = float(getattr(module, "cube_half_z_m", 0.5 * module.cube_side_m))
            plane_y = (mid_j + 0.5) * module.dx
            dxp = x_slice - module.cube_center_x_m
            dyp = plane_y - module.cube_center_y_m
            if getattr(module, "x_boundary_mode", "periodic") == "periodic":
                dxp = ((dxp + 0.5 * module.Lx_m) % module.Lx_m) - 0.5 * module.Lx_m
            dyp = ((dyp + 0.5 * module.Ly_m) % module.Ly_m) - 0.5 * module.Ly_m

            x_local = module.cube_yaw_cos * dxp + module.cube_yaw_sin * dyp
            y_local = -module.cube_yaw_sin * dxp + module.cube_yaw_cos * dyp
            z1 = z_slice - module.cube_center_z_m
            x_pitch = module.cube_pitch_cos * x_local - module.cube_pitch_sin * z1
            z_local = module.cube_pitch_sin * x_local + module.cube_pitch_cos * z1

            qx = np.abs(x_pitch) - half_x
            qy = np.abs(y_local) - half_y
            qz = np.abs(z_local) - half_z
            outside = np.sqrt(
                np.maximum(qx, 0.0)**2
                + np.maximum(qy, 0.0)**2
                + np.maximum(qz, 0.0)**2
            )
            inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
            cube_sdf_slice = outside + inside

        def draw_free_surface(ax):
            if free_surface_enabled:
                if (
                    vof_fill_slice is not None
                    and np.nanmin(vof_fill_slice) < 0.5 < np.nanmax(vof_fill_slice)
                ):
                    ax.contour(
                        x_slice,
                        z_slice,
                        vof_fill_slice,
                        levels=[0.5],
                        colors="cyan",
                        linewidths=1.1,
                    )
                else:
                    ax.plot(x_line, surface_line, color="cyan", linewidth=1.1)

        def draw_cube_outline(ax):
            if cube_sdf_slice is not None and np.nanmin(cube_sdf_slice) <= 0.0 <= np.nanmax(cube_sdf_slice):
                ax.contour(x_slice, z_slice, cube_sdf_slice, levels=[0.0], colors="black", linewidths=1.2)

        ax1 = plt.subplot(3, 1, 1)
        im1 = ax1.pcolormesh(x_slice, z_slice, vort, shading="auto")
        ax1.plot(x_line, bed_line, color="black", linewidth=1.0)
        ax1.plot(x_line, band_line, color="black", linewidth=0.8, linestyle="--")
        draw_free_surface(ax1)
        draw_cube_outline(ax1)
        plt.colorbar(im1, ax=ax1, label="Vorticity (LB)")
        ax1.set_xlim(0, module.Lx_m)
        ax1.set_ylim(0, module.Lz_m)
        ax1.set_ylabel("z (m)")
        im1.set_clim(-0.005, 0.005)
        ax1.set_title(
            f"{label}: y={(mid_j + 0.5) * module.dx:.1f} m | "
            f"t={time_phys:.3f} s | Umax={max_speed_phys:.2f} m/s | "
            f"Re={Re_phys_air:.2e}"
        )

        ax2 = plt.subplot(3, 1, 2)
        im2 = ax2.pcolormesh(x_slice, z_slice, speed_xz_phys, shading="auto")
        ax2.plot(x_line, bed_line, color="black", linewidth=1.0)
        ax2.plot(x_line, band_line, color="black", linewidth=0.8, linestyle="--")
        draw_free_surface(ax2)
        draw_cube_outline(ax2)
        plt.colorbar(im2, ax=ax2, label="Speed (m/s)")
        ax2.set_xlim(0, module.Lx_m)
        ax2.set_ylim(0, module.Lz_m)
        ax2.set_ylabel("z (m)")
        im2.set_clim(0, 1.2 * module.U_ref_phys)
        ax2.set_title("Speed")

        ax3 = plt.subplot(3, 1, 3)
        pressure_diag_mode = os.environ.get("LBM_PRESSURE_DIAGNOSTIC", "total_pressure").lower()
        if pressure_diag_mode == "linear_residual" and pressure_linear_expected is not None:
            pressure_plot = pressure_linear_residual.copy()
            pressure_plot[obs_xz] = np.nan
            pressure_label = "Pressure residual vs linear wave (Pa)"
            pressure_title = (
                f"p - p_linear: rms={p_linear_resid_rms:.2e} Pa, "
                f"hp={p_linear_resid_hp_rms:.2e} Pa | "
                f"FS hp={fs_p_linear_resid_hp_rms:.2e} Pa"
            )
        elif pressure_diag_mode in ("dynamic", "perturbation", "pressure_perturbation"):
            if free_surface_enabled and tracking_mode == "vof":
                pressure_label = "Total gauge pressure (Pa)"
                pressure_title = (
                    "Total pressure (VOF; no column hydrostatic subtraction) | "
                    f"hp={p_total_hp_rms:.2e} Pa | FS hp={fs_p_total_hp_rms:.2e} Pa"
                )
            else:
                pressure_label = "Pressure perturbation (Pa)" if balanced_pressure else "Dynamic pressure (Pa)"
                pressure_title = (
                    f"p': rms={p_rms:.2e} Pa, hp={p_hp_rms:.2e} Pa | "
                    f"vort_rms={vort_rms:.2e}, vort_hp={vort_hp_rms:.2e} | "
                    f"FS hp: p'={fs_p_hp_rms:.2e} Pa, p_total={fs_p_total_hp_rms:.2e} Pa, "
                    f"p_lin_resid={fs_p_linear_resid_hp_rms:.2e} Pa, vort={fs_vort_hp_rms:.2e}"
                )
        else:
            pressure_plot = pressure_total_phys.copy()
            pressure_plot[obs_xz] = np.nan
            pressure_label = "Total gauge pressure (Pa)"
            if component_balanced_pressure:
                pressure_title = (
                    f"Total pressure | hp={p_total_hp_rms:.2e} Pa | "
                    f"FS hp={fs_p_total_hp_rms:.2e} Pa"
                )
            else:
                pressure_title = (
                    f"Total pressure | hp={p_total_hp_rms:.2e} Pa | "
                    f"FS hp={fs_p_total_hp_rms:.2e} Pa | p'_hp={p_hp_rms:.2e} Pa"
                )
        im3 = ax3.pcolormesh(x_slice, z_slice, pressure_plot, shading="auto")
        ax3.plot(x_line, bed_line, color="black", linewidth=1.0)
        ax3.plot(x_line, band_line, color="black", linewidth=0.8, linestyle="--")
        draw_free_surface(ax3)
        draw_cube_outline(ax3)
        plt.colorbar(im3, ax=ax3, label=pressure_label)
        ax3.set_xlim(0, module.Lx_m)
        ax3.set_ylim(0, module.Lz_m)
        ax3.set_xlabel("x (m)")
        ax3.set_ylabel("z (m)")
        if pressure_diag_mode in ("total", "total_pressure", "pressure_total"):
            finite_pressure = pressure_plot[np.isfinite(pressure_plot)]
            if finite_pressure.size:
                p_hi = max(float(np.nanpercentile(finite_pressure, 99.5)), 1.0)
                p_lo = min(0.0, float(np.nanpercentile(finite_pressure, 0.5)))
                im3.set_clim(p_lo, p_hi)
        else:
            pressure_scale = rho_phys * module.U_ref_phys * module.U_ref_phys
            im3.set_clim(-0.15 * pressure_scale, 0.15 * pressure_scale)
        ax3.set_title(pressure_title)

        plt.tight_layout()
        out_file = output_dir / f"{label}_{step:05d}.png"
        plt.savefig(out_file, dpi=180)
        snapshot_file = save_render_snapshot(
            module,
            label=label,
            step=step,
            time_phys=time_phys,
            output_dir=output_dir,
            free_surface_enabled=free_surface_enabled,
            tracking_mode=tracking_mode,
            fs_fill_np=fs_fill_np if free_surface_enabled else None,
            free_surface_h_np=free_surface_h_np if free_surface_enabled else None,
            water_h_np=water_h_np,
            obs_np=obs_np,
        )
        render_file = render_snapshot_inline(snapshot_file) if snapshot_file is not None else None
        frame_note = ""
        if snapshot_file is not None:
            frame_note += f", render_snapshot={snapshot_file}"
        if render_file is not None:
            frame_note += f", render_frame={render_file}"
        print(
            f"[{label}] Boundary band <=10 cells: p_dyn_rms={p_rms:.3e} Pa, "
            f"p_dyn_hp_rms={p_hp_rms:.3e} Pa, p_total_hp_rms={p_total_hp_rms:.3e} Pa, "
            f"p_lin_resid_hp={p_linear_resid_hp_rms:.3e} Pa, "
            f"vort_rms={vort_rms:.3e}, "
            f"vort_hp_rms={vort_hp_rms:.3e}, cut_links={cut_links_val}, "
            f"fs_cut_links={fs_cut_links_val}, refill={refill_val}, fs_refill={fs_refill_val}, "
            f"fs_coalesce={fs_coalesce_val}, fs_bridge={fs_bridge_val}, "
            f"{'column_surface' if tracking_mode == 'vof' else 'surface'}=[{surface_min:.3f},{surface_max:.3f}] m, "
            f"volume={water_volume:.3f} m^3, vof_mass={vof_mass_total:.3f}, "
            f"vof_excess={vof_excess_count}, vof_orphans={vof_orphan_count}, "
            f"vof_detached={vof_detached_count} ({vof_detached_mass:.3e}), "
            f"vof_compact={vof_compact_count} ({vof_compact_mass:.3e}), "
            f"vof_collapse={vof_collapse_applied_volume:.3f}/{vof_collapse_candidate_volume:.3f} m^3 "
            f"({vof_collapse_cells} cells), "
            f"vof_min_fill={vof_min_interface_fill:.3f}, "
            f"vof_tiny={vof_tiny_interface_count}, vof_pending={vof_pending_excess_mass:.3e} "
            f"({vof_pending_excess_count}), "
            f"sym_l2={surface_symmetry_l2:.3e} m, "
            f"eta_hp={surface_eta_hp_rms:.3e} m, "
            f"fs_p_dyn_hp={fs_p_hp_rms:.3e} Pa, fs_p_total_hp={fs_p_total_hp_rms:.3e} Pa, "
            f"fs_p_lin_resid_hp={fs_p_linear_resid_hp_rms:.3e} Pa, "
            f"fs_vort_hp={fs_vort_hp_rms:.3e}, "
            f"Umax_loc=({max_x:.2f},{max_y:.2f},{max_z:.2f}) m, "
            f"Umax_type={max_fs_type}, Umax_fill={max_fs_fill:.3f}, "
            f"Umax_nbrs(wet/full/gas)={max_wet_neighbors}/{max_full_neighbors}/{max_gas_neighbors}, "
            f"L/R max=[{surface_left_max:.3f},{surface_right_max:.3f}] m, frame={out_file}{frame_note}",
            flush=True,
        )
        plt.pause(0.1)

    return visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--plot-freq", type=int, default=500)
    parser.add_argument("--plot-start", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-seed", type=int, default=12345)
    args = parser.parse_args()

    module_path = Path(args.module).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{args.label}_metrics.csv"

    module = load_solver(module_path, args.label, args.random_seed)
    module.steps = args.steps
    module.plot_freq = args.plot_freq
    module.plot_step_start = args.plot_start

    with metrics_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        module.visualize = make_visualize(module, args.label, output_dir, writer)
        print(
            f"[{args.label}] Starting {args.steps} steps from {module_path.name}; "
            f"frames/metrics -> {output_dir}",
            flush=True,
        )
        module.main()


if __name__ == "__main__":
    main()
