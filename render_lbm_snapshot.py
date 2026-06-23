"""Render an LBM free-surface snapshot with PyVista/VTK.

The input snapshot is written by run_ab_case.py when
LBM_SAVE_RENDER_SNAPSHOT=1. This script is visualization-only; it does not
modify solver state, VOF fill, pressure, velocity, or boundary fields.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy import ndimage
from skimage import measure


def parse_xyz(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return (parts[0], parts[1], parts[2])


def pv_faces(faces: np.ndarray, width: int) -> np.ndarray:
    counts = np.full((faces.shape[0], 1), width, dtype=np.int64)
    return np.hstack((counts, faces.astype(np.int64))).ravel()


def make_polydata(points: np.ndarray, faces: np.ndarray, width: int = 3) -> pv.PolyData:
    return pv.PolyData(points.astype(np.float32), pv_faces(faces, width))


def inpaint_solid_fill(volume: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    obs = np.asarray(obstacle, dtype=bool)
    if obs.any() and (~obs).any():
        nearest = ndimage.distance_transform_edt(
            obs,
            return_distances=False,
            return_indices=True,
        )
        volume = volume.copy()
        volume[obs] = volume[tuple(axis_index[obs] for axis_index in nearest)]
    return volume


def extract_vof_surface(
    snapshot: np.lib.npyio.NpzFile,
    *,
    stride: int,
    upsample: int,
    interp_order: int,
    iso_level: float,
) -> pv.PolyData | None:
    if "free_surface_fill" not in snapshot:
        return None

    fill = snapshot["free_surface_fill"].astype(np.float32, copy=False)
    obstacle = snapshot["obstacle"].astype(bool, copy=False)
    dx = float(snapshot["dx"])
    stride = max(1, stride)
    upsample = max(1, upsample)

    volume = fill[::stride, ::stride, ::stride].astype(np.float32, copy=True)
    obs_ds = obstacle[::stride, ::stride, ::stride]
    volume = inpaint_solid_fill(volume, obs_ds)
    spacing = dx * stride

    if upsample > 1:
        volume = ndimage.zoom(volume, upsample, order=interp_order)
        volume = np.clip(volume, 0.0, 1.0).astype(np.float32, copy=False)
        spacing /= upsample

    if not (float(np.nanmin(volume)) <= iso_level <= float(np.nanmax(volume))):
        return None

    verts, faces, _normals, _values = measure.marching_cubes(
        volume,
        level=iso_level,
        spacing=(spacing, spacing, spacing),
    )
    points = np.column_stack(
        (
            verts[:, 2] + 0.5 * dx,
            verts[:, 1] + 0.5 * dx,
            verts[:, 0] + 0.5 * dx,
        )
    )
    return make_polydata(points, faces, 3).clean().triangulate()


def extract_height_surface(snapshot: np.lib.npyio.NpzFile, *, stride: int) -> pv.PolyData | None:
    if "free_surface_h" not in snapshot:
        return None
    stride = max(1, stride)
    x = snapshot["x_phys"][::stride].astype(np.float32)
    y = snapshot["y_phys"][::stride].astype(np.float32)
    z = snapshot["free_surface_h"][::stride, ::stride].astype(np.float32)
    xx, yy = np.meshgrid(x, y)
    return pv.StructuredGrid(xx, yy, z).extract_surface(algorithm="dataset_surface").triangulate()


def bed_surface(snapshot: np.lib.npyio.NpzFile, *, stride: int) -> pv.PolyData:
    stride = max(1, stride)
    x = snapshot["x_phys"][::stride].astype(np.float32)
    y = snapshot["y_phys"][::stride].astype(np.float32)
    z = snapshot["water_h"][::stride, ::stride].astype(np.float32)
    xx, yy = np.meshgrid(x, y)
    return pv.StructuredGrid(xx, yy, z).extract_surface(algorithm="dataset_surface").triangulate()


def prism_mesh(snapshot: np.lib.npyio.NpzFile) -> pv.PolyData | None:
    if int(snapshot["include_cube_geometry"]) != 1:
        return None

    sx, sy, sz = snapshot["cube_size_m"].astype(float)
    cx, cy, cz = snapshot["cube_center_m"].astype(float)
    yaw_deg, pitch_deg = snapshot["cube_yaw_pitch_deg"].astype(float)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    yaw_cos, yaw_sin = math.cos(yaw), math.sin(yaw)
    pitch_cos, pitch_sin = math.cos(pitch), math.sin(pitch)

    points = []
    for x_loc in (-0.5 * sx, 0.5 * sx):
        for y_loc in (-0.5 * sy, 0.5 * sy):
            for z_loc in (-0.5 * sz, 0.5 * sz):
                x_after_pitch = pitch_cos * x_loc + pitch_sin * z_loc
                z_world_rel = -pitch_sin * x_loc + pitch_cos * z_loc
                x_world_rel = yaw_cos * x_after_pitch - yaw_sin * y_loc
                y_world_rel = yaw_sin * x_after_pitch + yaw_cos * y_loc
                points.append((cx + x_world_rel, cy + y_world_rel, cz + z_world_rel))

    faces = np.array(
        [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
        ],
        dtype=np.int64,
    )
    return make_polydata(np.array(points), faces, 4).triangulate()


def obstacle_mask_mesh(snapshot: np.lib.npyio.NpzFile, *, stride: int) -> pv.PolyData | None:
    obstacle = snapshot["obstacle"].astype(np.float32, copy=False)
    stride = max(1, stride)
    volume = obstacle[::stride, ::stride, ::stride]
    if not (float(volume.min()) <= 0.5 <= float(volume.max())):
        return None
    dx = float(snapshot["dx"])
    spacing = dx * stride
    verts, faces, _normals, _values = measure.marching_cubes(
        volume,
        level=0.5,
        spacing=(spacing, spacing, spacing),
    )
    points = np.column_stack((verts[:, 2] + 0.5 * dx, verts[:, 1] + 0.5 * dx, verts[:, 0] + 0.5 * dx))
    return make_polydata(points, faces, 3).clean().triangulate()


def default_target(snapshot: np.lib.npyio.NpzFile) -> tuple[float, float, float]:
    if int(snapshot["include_cube_geometry"]) == 1:
        size = snapshot["cube_size_m"].astype(float)
        center = snapshot["cube_center_m"].astype(float)
        return (float(center[0] - 0.5 * size[0]), float(center[1]), float(center[2]))
    return (0.5 * float(snapshot["lx_m"]), 0.5 * float(snapshot["ly_m"]), 0.5 * float(snapshot["lz_m"]))


def default_camera(snapshot: np.lib.npyio.NpzFile) -> tuple[float, float, float]:
    return (float(snapshot["lx_m"]), 0.5 * float(snapshot["ly_m"]), float(snapshot["lz_m"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an LBM snapshot using PyVista/VTK.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--size", default="1920,1080", help="image width,height in pixels")
    parser.add_argument("--camera", type=parse_xyz, default=None, help="camera position x,y,z")
    parser.add_argument("--target", type=parse_xyz, default=None, help="camera target x,y,z")
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--view-angle", type=float, default=26.0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--upsample", type=int, default=2)
    parser.add_argument("--interp-order", type=int, default=1)
    parser.add_argument("--iso-level", type=float, default=0.5)
    parser.add_argument("--smooth-iter", type=int, default=20)
    parser.add_argument("--smooth-pass-band", type=float, default=0.08)
    parser.add_argument("--water-opacity", type=float, default=0.54)
    parser.add_argument("--water-color", default="#36b7e6")
    parser.add_argument("--obstacle-color", default="#2c2c30")
    parser.add_argument("--bed-color", default="#4b4741")
    parser.add_argument("--obstacle-source", choices=("auto", "prism", "mask", "none"), default="auto")
    parser.add_argument("--no-bed", action="store_true")
    parser.add_argument("--show", action="store_true", help="open an interactive VTK window")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    snapshot_path = args.snapshot.resolve()
    out_path = args.out.resolve() if args.out else snapshot_path.with_name(snapshot_path.stem + "_render.png")
    width, height = [int(v.strip()) for v in args.size.split(",", 1)]

    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        water = extract_vof_surface(
            snapshot,
            stride=args.stride,
            upsample=args.upsample,
            interp_order=int(np.clip(args.interp_order, 0, 3)),
            iso_level=args.iso_level,
        )
        if water is None:
            water = extract_height_surface(snapshot, stride=args.stride)
        if water is None:
            raise RuntimeError("snapshot does not contain a renderable free surface")

        if args.smooth_iter > 0:
            water = water.smooth_taubin(
                n_iter=args.smooth_iter,
                pass_band=args.smooth_pass_band,
                boundary_smoothing=False,
                feature_smoothing=False,
            )
            water.compute_normals(cell_normals=False, point_normals=True, inplace=True)

        obstacle = None
        if args.obstacle_source in ("auto", "prism"):
            obstacle = prism_mesh(snapshot)
        if obstacle is None and args.obstacle_source in ("auto", "mask"):
            obstacle = obstacle_mask_mesh(snapshot, stride=max(args.stride, 2))
        bed = None if args.no_bed else bed_surface(snapshot, stride=max(args.stride, 2))

        camera = args.camera or default_camera(snapshot)
        target = args.target or default_target(snapshot)

    pv.global_theme.window_size = [width, height]
    plotter = pv.Plotter(off_screen=not args.show, window_size=(width, height))
    plotter.set_background("#eef7fb", top="#d4e8f1")
    plotter.enable_anti_aliasing("ssaa")
    plotter.camera_position = [camera, target, (0.0, 0.0, 1.0)]
    plotter.camera.view_angle = args.view_angle
    plotter.camera.zoom(args.zoom)

    sun = pv.Light(
        position=(camera[0], camera[1] - 80.0, camera[2] + 60.0),
        focal_point=target,
        color="white",
        intensity=1.1,
    )
    fill = pv.Light(
        position=(target[0] - 60.0, target[1] + 60.0, target[2] + 25.0),
        focal_point=target,
        color="#9ed8ff",
        intensity=0.35,
    )
    plotter.add_light(sun)
    plotter.add_light(fill)

    if bed is not None:
        plotter.add_mesh(
            bed,
            color=args.bed_color,
            opacity=1.0,
            smooth_shading=True,
            pbr=True,
            roughness=0.75,
            metallic=0.0,
        )
    if obstacle is not None and args.obstacle_source != "none":
        plotter.add_mesh(
            obstacle,
            color=args.obstacle_color,
            opacity=1.0,
            smooth_shading=False,
            pbr=True,
            roughness=0.62,
            metallic=0.0,
            specular=0.18,
        )

    plotter.add_mesh(
        water,
        color=args.water_color,
        opacity=args.water_opacity,
        smooth_shading=True,
        pbr=True,
        roughness=0.03,
        metallic=0.0,
        specular=0.9,
        specular_power=80,
    )

    plotter.hide_axes()
    plotter.screenshot(str(out_path), transparent_background=False)
    if args.show:
        plotter.show()
    else:
        plotter.close()
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
