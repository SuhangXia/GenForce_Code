#!/usr/bin/env python3
"""Generate a large multiscale tactile dataset with cheap marker diversification.

Core idea (Simulation Economics):
- Expensive stage: Taichi MPM deformation (.npz -> .stl) for each (episode, scale).
- Cheap stage: Blender re-renders same deformed .stl with ALL marker textures.

Output layout:
    adapter_dataset_ultimate/
      manifest.json
      episode_000000/
        metadata.json
        scale_15mm/
          marker_Array1.jpg
          ...
          patch_coords_16x16.json
        scale_16mm/
          ...

This script is parallel-friendly:
- Physics workers: capped (default 7) to avoid CUDA OOM.
- Render workers: CPU-oriented process pool.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml


DEFAULT_SCALES_MM = list(range(15, 26))
DEFAULT_X_RANGE = (-3.0, 3.0)
DEFAULT_Y_RANGE = (-3.0, 3.0)
DEFAULT_DEPTH_RANGE = (0.4, 2.2)
DEFAULT_FOV_DEG = 40.0
DEFAULT_REFERENCE_SCALE_MM = 25.0
DEFAULT_DISTANCE_SAFETY = 0.98
DEFAULT_IMAGE_RES = (640, 480)
TEXTURE_EXTS = {".jpg", ".jpeg", ".png"}


OPEN3D_TEMP_SCRIPT = r'''
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def parse_args():
    p = argparse.ArgumentParser(description="Convert deformation npz to STL")
    p.add_argument("--input-npz", required=True)
    p.add_argument("--output-stl", required=True)
    return p.parse_args()


def load_surface_points(npz_path: Path):
    data = np.load(npz_path)
    for key in ("p_xpos_list", "p_ypos_list", "p_zpos_list"):
        if key not in data:
            raise KeyError(f"Missing key {key} in {npz_path}")

    x = np.asarray(data["p_xpos_list"])
    y = np.asarray(data["p_ypos_list"])
    z = np.asarray(data["p_zpos_list"])

    x_last = x[-1] if x.ndim > 1 else x
    y_last = y[-1] if y.ndim > 1 else y
    z_last = z[-1] if z.ndim > 1 else z

    if not (x_last.shape == y_last.shape == z_last.shape):
        raise ValueError("x/y/z shape mismatch in deformation npz")

    n = int(x_last.shape[0])
    grid_n = int(round(np.sqrt(n)))
    regular_grid = (grid_n * grid_n == n)

    # mm -> m for Blender physical scale consistency.
    points = np.stack([x_last, y_last, z_last], axis=1).astype(np.float64) / 1000.0

    finite = np.isfinite(points).all(axis=1)
    if not np.all(finite):
        points = points[finite]
        regular_grid = False

    if points.shape[0] < 100:
        raise ValueError(f"Too few valid points in {npz_path}: {points.shape[0]}")

    return points, regular_grid, grid_n


def mesh_from_regular_grid(points: np.ndarray, grid_n: int):
    tris = []
    for j in range(grid_n - 1):
        row = j * grid_n
        row_next = (j + 1) * grid_n
        for i in range(grid_n - 1):
            v00 = row + i
            v10 = row + i + 1
            v01 = row_next + i
            v11 = row_next + i + 1
            tris.append([v00, v01, v10])
            tris.append([v10, v01, v11])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


def mesh_from_point_cloud(points: np.ndarray):
    # Filter z outliers conservatively before reconstruction.
    z = points[:, 2]
    q1, q99 = np.quantile(z, [0.01, 0.99])
    zmask = (z >= q1 - 0.001) & (z <= q99 + 0.001)
    if np.sum(zmask) >= max(100, int(0.8 * points.shape[0])):
        points = points[zmask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    if len(pcd.points) < 100:
        raise RuntimeError("Point cloud too sparse after outlier filtering")

    span = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    diag = float(np.linalg.norm(span))
    radius = max(diag * 0.03, 1e-4)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60)
    )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    if len(mesh.triangles) == 0:
        raise RuntimeError("Poisson reconstruction produced empty mesh")

    densities = np.asarray(densities)
    if densities.size:
        th = float(np.quantile(densities, 0.02))
        mesh.remove_vertices_by_mask(densities < th)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def main():
    args = parse_args()
    npz_path = Path(args.input_npz)
    stl_path = Path(args.output_stl)
    stl_path.parent.mkdir(parents=True, exist_ok=True)

    points, regular_grid, grid_n = load_surface_points(npz_path)

    if regular_grid and points.shape[0] == grid_n * grid_n:
        mesh = mesh_from_regular_grid(points, grid_n)
    else:
        mesh = mesh_from_point_cloud(points)

    ok = o3d.io.write_triangle_mesh(str(stl_path), mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write STL to {stl_path}")

    print(f"Converted {npz_path} -> {stl_path} with {len(mesh.triangles)} triangles")


if __name__ == "__main__":
    main()
'''


BLENDER_TEMP_SCRIPT = r'''
import argparse
import math
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    import sys

    user_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Render one STL with all marker textures")
    p.add_argument("--stl", required=True)
    p.add_argument("--textures-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scale-mm", type=float, required=True)
    p.add_argument(
        "--camera-mode",
        choices=["fixed_distance_variable_fov", "fixed_fov_variable_distance"],
        default="fixed_distance_variable_fov",
    )
    p.add_argument("--base-fov-deg", type=float, default=40.0)
    p.add_argument("--fixed-distance-m", type=float, required=True)
    p.add_argument("--distance-safety", type=float, default=0.98)
    p.add_argument(
        "--uv-mode",
        choices=["unwrap_genforce", "physical_math"],
        default="unwrap_genforce",
    )
    p.add_argument("--uv-inset-ratio", type=float, default=0.01)
    return p.parse_args(user_argv)


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)


def import_stl(path: str):
    try:
        bpy.ops.import_mesh.stl(filepath=path)
    except Exception:
        bpy.ops.wm.stl_import(filepath=path)

    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Failed to import STL mesh")
    return obj


def delete_bottom_faces(obj, cutoff_offset=0.002):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)

    if not bm.verts:
        bm.free()
        raise RuntimeError("Mesh has no vertices")

    # If the mesh is open (has boundary edges), it is already a surface mesh.
    # Do not try to delete bottom faces, otherwise deep indent regions can be removed.
    if any(e.is_boundary for e in bm.edges):
        bm.free()
        return

    z_bottom = min(v.co.z for v in bm.verts)
    z_top = max(v.co.z for v in bm.verts)
    z_span = z_top - z_bottom
    if z_span <= 0.0:
        bm.free()
        return

    # Robust threshold: target only the lowest few faces of closed Poisson shells.
    face_z = []
    for f in bm.faces:
        zc = sum(v.co.z for v in f.verts) / float(len(f.verts))
        face_z.append((f, zc))
    z_values = np.array([z for _, z in face_z], dtype=np.float64)
    low_q = float(np.quantile(z_values, 0.04))
    hard_cap = z_bottom + max(cutoff_offset, 0.06 * z_span)
    threshold = min(low_q, hard_cap)

    to_delete = [f for f, zc in face_z if zc <= threshold and f.normal.z < -0.25]
    if to_delete and len(to_delete) < int(0.25 * max(len(bm.faces), 1)):
        bmesh.ops.delete(bm, geom=to_delete, context="FACES")

    bm.to_mesh(me)
    bm.free()
    me.update()


def compute_bbox_world(obj):
    verts = np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices], dtype=np.float64)
    if verts.shape[0] < 3:
        raise RuntimeError("Mesh has too few vertices after cleanup")

    min_x, min_y, min_z = verts.min(axis=0)
    max_x, max_y, max_z = verts.max(axis=0)

    cx = float((min_x + max_x) * 0.5)
    cy = float((min_y + max_y) * 0.5)
    z_top = float(max_z)
    return cx, cy, z_top


def apply_uv_math(obj, cx: float, cy: float, scale_m: float, uv_inset_ratio: float):
    if scale_m <= 0:
        raise ValueError("scale_m must be > 0")
    if uv_inset_ratio < 0.0 or uv_inset_ratio >= 0.49:
        raise ValueError("uv_inset_ratio must be in [0.0, 0.49)")

    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)

    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        uv_layer = bm.loops.layers.uv.new("UVMap")

    for face in bm.faces:
        for loop in face.loops:
            w = obj.matrix_world @ loop.vert.co
            u = (w.x - cx) / scale_m + 0.5
            v = (w.y - cy) / scale_m + 0.5
            if uv_inset_ratio > 0.0:
                u = u * (1.0 - 2.0 * uv_inset_ratio) + uv_inset_ratio
                v = v * (1.0 - 2.0 * uv_inset_ratio) + uv_inset_ratio
            loop[uv_layer].uv = (u, v)

    bm.to_mesh(me)
    bm.free()
    me.update()


def check_and_rotate_uvs_genforce(obj):
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()

    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xmin = min(corner.x for corner in bbox_corners)
    xmax = max(corner.x for corner in bbox_corners)
    ymin = min(corner.y for corner in bbox_corners)
    ymax = max(corner.y for corner in bbox_corners)

    for _ in range(4):
        uv_at_max = None
        uv_at_min = None
        for face in bm.faces:
            if not face.select:
                continue
            for loop in face.loops:
                vert = loop.vert
                uv_coords = loop[uv_layer]
                if abs(vert.co.x - xmax) < 0.001 and abs(vert.co.y - ymax) < 0.001:
                    uv_at_max = Vector((uv_coords.uv.x, uv_coords.uv.y))
                if abs(vert.co.x - xmin) < 0.001 and abs(vert.co.y - ymin) < 0.001:
                    uv_at_min = Vector((uv_coords.uv.x, uv_coords.uv.y))

        if uv_at_max and uv_at_min:
            if (
                abs(uv_at_max.x - 1.0) < 0.1
                and abs(uv_at_max.y - 0.0) < 0.1
                and abs(uv_at_min.x - 0.0) < 0.1
                and abs(uv_at_min.y - 1.0) < 0.1
            ):
                break

            for face in bm.faces:
                if not face.select:
                    continue
                for loop in face.loops:
                    uv_coords = loop[uv_layer]
                    old_x, old_y = uv_coords.uv.x, uv_coords.uv.y
                    uv_coords.uv.x = old_y
                    uv_coords.uv.y = 1.0 - old_x
            bmesh.update_edit_mesh(me)

    bmesh.update_edit_mesh(me)


def apply_uv_unwrap_genforce(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)

    for face in bm.faces:
        face.select = (face.normal.z < 0)
    bmesh.update_edit_mesh(obj.data)

    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
    check_and_rotate_uvs_genforce(obj)
    bpy.ops.object.mode_set(mode="OBJECT")


def setup_material():
    mat = bpy.data.materials.new("MarkerMaterial")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    uv = nodes.new("ShaderNodeTexCoord")

    tex.extension = "CLIP"
    tex.interpolation = "Cubic"

    bsdf.inputs["Specular"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.45

    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, tex


def setup_world_light():
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    bg = nodes.get("Background")
    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 60.0


def add_black_backdrop(cx: float, cy: float, z_top: float, scale_m: float):
    # Put a black plane behind the gel so any uncovered pixels remain black.
    plane_z = z_top + max(scale_m * 0.08, 0.002)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(cx, cy, plane_z))
    plane = bpy.context.active_object
    plane.name = "BlackBackdrop"
    plane.scale = (scale_m * 4.0, scale_m * 4.0, 1.0)

    mat = bpy.data.materials.new("BlackBackdropMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Specular"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0

    plane.data.materials.clear()
    plane.data.materials.append(mat)


def setup_camera(
    cx: float,
    cy: float,
    z_top: float,
    scale_m: float,
    camera_mode: str,
    base_fov_deg: float,
    fixed_distance_m: float,
    distance_safety: float,
):
    scene = bpy.context.scene

    cam_data = bpy.data.cameras.new("ScaleCam")
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.0001
    cam_data.clip_end = 100.0

    if scale_m <= 0:
        raise ValueError("scale_m must be > 0")
    if fixed_distance_m <= 0:
        raise ValueError("fixed_distance_m must be > 0")

    if camera_mode == "fixed_distance_variable_fov":
        distance = fixed_distance_m
        fov = 2.0 * math.atan((scale_m / 2.0) / distance)
    else:
        fov = math.radians(base_fov_deg)
        if fov <= 1e-6 or fov >= math.pi - 1e-6:
            raise ValueError("base_fov_deg produces invalid camera angle")
        distance = (scale_m / 2.0) / math.tan(fov / 2.0)
        distance *= distance_safety

    cam_data.angle = fov

    cam = bpy.data.objects.new("ScaleCam", cam_data)
    cam.location = (cx, cy, z_top - distance)
    cam.rotation_euler = (math.pi, 0.0, 0.0)

    scene.collection.objects.link(cam)
    scene.camera = cam
    print(
        f"Camera setup | mode={camera_mode} scale_mm={scale_m*1000.0:.1f} "
        f"fov_deg={math.degrees(fov):.4f} distance_m={distance:.6f}"
    )


def configure_render(out_dir: Path):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 64
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.max_bounces = 4
        scene.cycles.device = "CPU"

    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 100
    scene.render.film_transparent = False

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    out_dir.mkdir(parents=True, exist_ok=True)


def list_textures(textures_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in textures_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not paths:
        raise FileNotFoundError(f"No texture files found in {textures_dir}")
    return paths


def main():
    args = parse_args()
    stl = Path(args.stl)
    textures_dir = Path(args.textures_dir)
    output_dir = Path(args.output_dir)

    clean_scene()
    obj = import_stl(str(stl))
    delete_bottom_faces(obj, cutoff_offset=0.002)

    scale_m = float(args.scale_mm) / 1000.0
    cx, cy, z_top = compute_bbox_world(obj)
    if args.uv_mode == "unwrap_genforce":
        apply_uv_unwrap_genforce(obj)
    else:
        apply_uv_math(obj, cx, cy, scale_m, args.uv_inset_ratio)
    add_black_backdrop(cx, cy, z_top, scale_m)

    mat, tex_node = setup_material()
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    setup_world_light()
    setup_camera(
        cx,
        cy,
        z_top,
        scale_m,
        args.camera_mode,
        args.base_fov_deg,
        args.fixed_distance_m,
        args.distance_safety,
    )
    configure_render(output_dir)

    textures = list_textures(textures_dir)
    for tex_path in textures:
        img = bpy.data.images.load(str(tex_path), check_existing=True)
        img.colorspace_settings.name = "sRGB"
        tex_node.image = img

        out_file = output_dir / f"marker_{tex_path.stem}.jpg"
        bpy.context.scene.render.filepath = str(out_file)
        bpy.ops.render.render(write_still=True)
        print(f"Rendered: {out_file}")


if __name__ == "__main__":
    main()
'''


@dataclass
class EpisodeState:
    episode_id: int
    indenter: str
    x_mm: float
    y_mm: float
    depth_mm: float
    episode_dir: Path
    scales: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base / value)


def format_coord_suffix(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def expected_npz_path(npz_root: Path, indenter: str, x_mm: float, y_mm: float, depth_mm: float) -> Path:
    suffix = f"{format_coord_suffix(x_mm)}_{format_coord_suffix(y_mm)}_{round(depth_mm, 1)}.npz"
    return npz_root / indenter / suffix


def npz_deformation_stats(npz_path: Path) -> Dict[str, float]:
    data = np.load(npz_path)
    z = np.asarray(data["p_zpos_list"], dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2:
        return {
            "surface_max_down_mm": 0.0,
            "surface_mean_down_mm": 0.0,
            "surface_min_z_start_mm": float(np.min(z)) if z.size else 0.0,
            "surface_min_z_end_mm": float(np.min(z)) if z.size else 0.0,
        }

    start = z[0]
    end = z[-1]
    down = start - end
    return {
        "surface_max_down_mm": float(np.max(down)),
        "surface_mean_down_mm": float(np.mean(down)),
        "surface_min_z_start_mm": float(np.min(start)),
        "surface_min_z_end_mm": float(np.min(end)),
    }


def patch_coords_16x16(scale_mm: int) -> List[List[List[float]]]:
    axis = np.linspace(-scale_mm / 2.0, scale_mm / 2.0, 16, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coords = np.stack([xx, yy], axis=-1)
    return np.round(coords, 6).tolist()


def run_cmd_checked(cmd: Sequence[str], cwd: Path, timeout_sec: int, stage: str) -> None:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    blender_python_error = "Error: Python:" in combined

    # Blender can return exit code 0 even when its embedded Python script crashed.
    if proc.returncode != 0 or blender_python_error:
        stderr_tail = (proc.stderr or "")[-2000:]
        stdout_tail = (proc.stdout or "")[-1200:]
        raise RuntimeError(
            f"{stage} failed (exit={proc.returncode}, blender_python_error={blender_python_error})\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR(tail):\n{stderr_tail}\n"
            f"STDOUT(tail):\n{stdout_tail}"
        )


def write_scaled_config(base_cfg: Path, out_cfg: Path, scale_mm: int) -> None:
    with open(base_cfg, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        data["elastomer"]["size"]["l"] = int(scale_mm)
        data["elastomer"]["size"]["w"] = int(scale_mm)
    except Exception as exc:
        raise KeyError("Missing elastomer.size.l/w in parameters yaml") from exc

    with open(out_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_physics_task(task: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = int(task["episode_id"])
    scale_mm = int(task["scale_mm"])
    indenter = str(task["indenter"])
    x_mm = float(task["x_mm"])
    y_mm = float(task["y_mm"])
    depth_mm = float(task["depth_mm"])

    repo_root = Path(task["repo_root"])
    temp_scale_dir = Path(task["temp_scale_dir"])
    temp_scale_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = Path(task["base_parameters"])
    temp_cfg = temp_scale_dir / "parameters_scaled.yml"
    npz_root = temp_scale_dir / "npz"
    npz_root.mkdir(parents=True, exist_ok=True)

    try:
        write_scaled_config(base_cfg, temp_cfg, scale_mm)

        gel_cmd = [
            str(task["python_cmd"]),
            "sim/deformation/gel_press.py",
            "--config",
            str(temp_cfg),
            "--particle",
            str(task["particle"]),
            "--dir_output",
            str(npz_root),
            "--dataset",
            "sim/assets/indenters/input",
            "--object",
            indenter,
            "--x",
            f"{x_mm:.4f}",
            "--y",
            f"{y_mm:.4f}",
            "--depth",
            f"{depth_mm:.1f}",
        ]
        run_cmd_checked(gel_cmd, cwd=repo_root, timeout_sec=int(task["physics_timeout_sec"]), stage="gel_press")

        npz_path = expected_npz_path(npz_root, indenter, x_mm, y_mm, depth_mm)
        if not npz_path.exists():
            x_tag = format_coord_suffix(x_mm)
            y_tag = format_coord_suffix(y_mm)
            candidates = sorted((npz_root / indenter).glob(f"{x_tag}_{y_tag}_*.npz"))
            if not candidates:
                raise FileNotFoundError(f"No deformation npz found for {indenter} at {npz_root}")
            npz_path = candidates[-1]

        stl_path = temp_scale_dir / f"{indenter}_ep{episode_id:06d}_s{scale_mm}.stl"
        mesh_cmd = [
            str(task["python_cmd"]),
            str(task["open3d_script"]),
            "--input-npz",
            str(npz_path),
            "--output-stl",
            str(stl_path),
        ]
        run_cmd_checked(mesh_cmd, cwd=repo_root, timeout_sec=int(task["physics_timeout_sec"]), stage="npz_to_stl")

        deform = npz_deformation_stats(npz_path)

        return {
            "status": "ok",
            "episode_id": episode_id,
            "scale_mm": scale_mm,
            "indenter": indenter,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "depth_mm": depth_mm,
            "temp_scale_dir": str(temp_scale_dir),
            "stl_path": str(stl_path),
            "deformation_stats": deform,
        }
    except Exception as exc:
        return {
            "status": "error",
            "episode_id": episode_id,
            "scale_mm": scale_mm,
            "indenter": indenter,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "depth_mm": depth_mm,
            "temp_scale_dir": str(temp_scale_dir),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_render_task(task: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = int(task["episode_id"])
    scale_mm = int(task["scale_mm"])
    repo_root = Path(task["repo_root"])
    scale_dir = Path(task["scale_dir"])
    scale_dir.mkdir(parents=True, exist_ok=True)

    stl_path = Path(task["stl_path"])
    temp_scale_dir = Path(task["temp_scale_dir"])
    keep_intermediates = bool(task["keep_intermediates"])

    try:
        render_cmd = [
            str(task["blender_cmd"]),
            "-b",
            "--python",
            str(task["blender_script"]),
            "--",
            "--stl",
            str(stl_path),
            "--textures-dir",
            str(task["textures_dir"]),
            "--output-dir",
            str(scale_dir),
            "--scale-mm",
            str(scale_mm),
            "--camera-mode",
            str(task["camera_mode"]),
            "--base-fov-deg",
            str(task["base_fov_deg"]),
            "--fixed-distance-m",
            str(task["fixed_distance_m"]),
            "--distance-safety",
            str(task["distance_safety"]),
            "--uv-mode",
            str(task["uv_mode"]),
            "--uv-inset-ratio",
            str(task["uv_inset_ratio"]),
        ]
        run_cmd_checked(
            render_cmd,
            cwd=repo_root,
            timeout_sec=int(task["render_timeout_sec"]),
            stage="blender_render",
        )

        patch_path = scale_dir / "patch_coords_16x16.json"
        with open(patch_path, "w", encoding="utf-8") as f:
            json.dump(patch_coords_16x16(scale_mm), f, indent=2)

        rendered = sorted(p.name for p in scale_dir.glob("marker_*.jpg"))
        if not rendered:
            raise RuntimeError(f"No marker renders produced in {scale_dir}")

        return {
            "status": "ok",
            "episode_id": episode_id,
            "scale_mm": scale_mm,
            "scale_dir": str(scale_dir),
            "patch_path": str(patch_path),
            "rendered_markers": rendered,
            "temp_scale_dir": str(temp_scale_dir),
        }
    except Exception as exc:
        return {
            "status": "error",
            "episode_id": episode_id,
            "scale_mm": scale_mm,
            "scale_dir": str(scale_dir),
            "temp_scale_dir": str(temp_scale_dir),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if not keep_intermediates:
            shutil.rmtree(temp_scale_dir, ignore_errors=True)


def discover_indenter_names(indenter_dir: Path, subset: Sequence[str] | None) -> List[str]:
    names = sorted(p.stem for p in indenter_dir.glob("*.npy") if p.is_file())
    if not names:
        raise FileNotFoundError(f"No indenter npy files found in {indenter_dir}")

    if subset:
        subset_set = set(subset)
        filtered = [n for n in names if n in subset_set]
        if not filtered:
            raise ValueError(f"Requested objects not found. Available: {names}")
        return filtered
    return names


def discover_textures(texture_dir: Path) -> List[Path]:
    textures = sorted(
        p for p in texture_dir.iterdir() if p.is_file() and p.suffix.lower() in TEXTURE_EXTS
    )
    if not textures:
        raise FileNotFoundError(f"No marker textures found in {texture_dir}")
    return textures


def create_backup(parameters_path: Path) -> Path:
    backup_path = parameters_path.with_suffix(parameters_path.suffix + ".bak")
    if backup_path.exists():
        # Recover from previous interrupted run.
        shutil.copy2(backup_path, parameters_path)
        backup_path.unlink()

    shutil.copy2(parameters_path, backup_path)
    return backup_path


def restore_backup(parameters_path: Path, backup_path: Path) -> None:
    if backup_path.exists():
        shutil.copy2(backup_path, parameters_path)
        backup_path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate large multiscale tactile dataset.")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--dataset-root", type=Path, default=Path("adapter_dataset_ultimate"))

    p.add_argument("--particle", type=str, default="100000")
    p.add_argument("--episodes-per-indenter", type=int, default=1)
    p.add_argument("--scales-mm", type=int, nargs="+", default=DEFAULT_SCALES_MM)
    p.add_argument("--objects", nargs="*", default=None, help="Optional subset of indenter names")

    p.add_argument("--x-min", type=float, default=DEFAULT_X_RANGE[0])
    p.add_argument("--x-max", type=float, default=DEFAULT_X_RANGE[1])
    p.add_argument("--y-min", type=float, default=DEFAULT_Y_RANGE[0])
    p.add_argument("--y-max", type=float, default=DEFAULT_Y_RANGE[1])
    p.add_argument("--depth-min", type=float, default=DEFAULT_DEPTH_RANGE[0])
    p.add_argument("--depth-max", type=float, default=DEFAULT_DEPTH_RANGE[1])

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--camera-mode",
        choices=["fixed_distance_variable_fov", "fixed_fov_variable_distance"],
        default="fixed_distance_variable_fov",
        help="Camera policy for scale adaptation.",
    )
    p.add_argument(
        "--fov-deg",
        type=float,
        default=DEFAULT_FOV_DEG,
        help="Base FOV (deg). Used directly in fixed_fov mode; used as reference in fixed_distance mode.",
    )
    p.add_argument(
        "--reference-scale-mm",
        type=float,
        default=DEFAULT_REFERENCE_SCALE_MM,
        help="Reference scale used to derive fixed distance when --camera-distance-m is not provided.",
    )
    p.add_argument(
        "--camera-distance-m",
        type=float,
        default=None,
        help="Fixed camera distance in meters. If omitted, derived from reference-scale and fov.",
    )
    p.add_argument(
        "--distance-safety",
        type=float,
        default=DEFAULT_DISTANCE_SAFETY,
        help="Safety multiplier used when deriving camera distance (and in legacy fixed_fov mode).",
    )
    p.add_argument(
        "--uv-inset-ratio",
        type=float,
        default=0.01,
        help="Inset UVs from [0,1] boundary to avoid edge artifacts on texture borders.",
    )
    p.add_argument(
        "--uv-mode",
        choices=["unwrap_genforce", "physical_math"],
        default="unwrap_genforce",
        help="UV strategy for marker mapping.",
    )

    p.add_argument("--max-physics-workers", type=int, default=7)
    p.add_argument("--max-render-workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    p.add_argument("--physics-timeout-sec", type=int, default=240)
    p.add_argument("--render-timeout-sec", type=int, default=300)

    p.add_argument("--python-cmd", type=str, default=sys.executable)
    p.add_argument("--blender-cmd", type=str, default="blender")

    p.add_argument("--clean-output", action="store_true")
    p.add_argument("--keep-intermediates", action="store_true")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    repo_root = args.repo_root.resolve()
    dataset_root = resolve_path(repo_root, args.dataset_root)

    if args.episodes_per_indenter <= 0:
        raise ValueError("--episodes-per-indenter must be > 0")
    if args.max_physics_workers <= 0:
        raise ValueError("--max-physics-workers must be > 0")
    if args.max_render_workers <= 0:
        raise ValueError("--max-render-workers must be > 0")
    if args.depth_min <= 0 or args.depth_max <= 0 or args.depth_min > args.depth_max:
        raise ValueError("Invalid depth range")
    if args.reference_scale_mm <= 0:
        raise ValueError("--reference-scale-mm must be > 0")
    if args.distance_safety <= 0:
        raise ValueError("--distance-safety must be > 0")
    if args.fov_deg <= 0 or args.fov_deg >= 179.0:
        raise ValueError("--fov-deg must be in (0, 179)")
    if args.uv_inset_ratio < 0.0 or args.uv_inset_ratio >= 0.49:
        raise ValueError("--uv-inset-ratio must be in [0.0, 0.49)")

    if args.camera_distance_m is not None:
        if args.camera_distance_m <= 0:
            raise ValueError("--camera-distance-m must be > 0")
        fixed_camera_distance_m = float(args.camera_distance_m)
    else:
        ref_scale_m = float(args.reference_scale_mm) / 1000.0
        ref_fov_rad = math.radians(float(args.fov_deg))
        fixed_camera_distance_m = (ref_scale_m / 2.0) / math.tan(ref_fov_rad / 2.0)
        fixed_camera_distance_m *= float(args.distance_safety)

    parameters_path = repo_root / "sim" / "parameters.yml"
    texture_dir = repo_root / "sim" / "marker" / "marker_pattern"
    indenter_dir = repo_root / "sim" / "assets" / "indenters" / "input" / f"npy_{args.particle}"

    if not parameters_path.exists():
        raise FileNotFoundError(f"Missing {parameters_path}")
    if not indenter_dir.exists():
        raise FileNotFoundError(f"Missing indenter directory: {indenter_dir}")
    if not texture_dir.exists():
        raise FileNotFoundError(f"Missing marker texture directory: {texture_dir}")

    indenter_names = discover_indenter_names(indenter_dir, args.objects)
    textures = discover_textures(texture_dir)

    if args.clean_output and dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    backup_path = create_backup(parameters_path)
    logging.info("Backup created: %s", backup_path)

    script_tmp_dir = Path(tempfile.mkdtemp(prefix="multiscale_scripts_"))
    work_tmp_root = Path(tempfile.mkdtemp(prefix="multiscale_work_"))

    open3d_script = script_tmp_dir / "tmp_npz2stl.py"
    blender_script = script_tmp_dir / "tmp_blender_multirender.py"
    open3d_script.write_text(OPEN3D_TEMP_SCRIPT, encoding="utf-8")
    blender_script.write_text(BLENDER_TEMP_SCRIPT, encoding="utf-8")

    rng = random.Random(args.seed)

    episode_states: Dict[int, EpisodeState] = {}
    physics_tasks: List[Dict[str, Any]] = []

    episode_id = 0
    for indenter in indenter_names:
        for _ in range(args.episodes_per_indenter):
            x_mm = round(rng.uniform(args.x_min, args.x_max), 4)
            y_mm = round(rng.uniform(args.y_min, args.y_max), 4)
            depth_mm = round(rng.uniform(args.depth_min, args.depth_max), 1)

            episode_dir = dataset_root / f"episode_{episode_id:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            episode_states[episode_id] = EpisodeState(
                episode_id=episode_id,
                indenter=indenter,
                x_mm=x_mm,
                y_mm=y_mm,
                depth_mm=depth_mm,
                episode_dir=episode_dir,
            )

            for scale_mm in args.scales_mm:
                physics_tasks.append(
                    {
                        "episode_id": episode_id,
                        "scale_mm": int(scale_mm),
                        "indenter": indenter,
                        "x_mm": x_mm,
                        "y_mm": y_mm,
                        "depth_mm": depth_mm,
                        "repo_root": str(repo_root),
                        "particle": str(args.particle),
                        "python_cmd": str(args.python_cmd),
                        "open3d_script": str(open3d_script),
                        "base_parameters": str(parameters_path),
                        "temp_scale_dir": str(
                            work_tmp_root / f"episode_{episode_id:06d}" / f"scale_{int(scale_mm)}mm"
                        ),
                        "physics_timeout_sec": int(args.physics_timeout_sec),
                    }
                )

            logging.info(
                "Prepared episode_%06d | indenter=%s | contact=(x=%.4f, y=%.4f, d=%.1f)",
                episode_id,
                indenter,
                x_mm,
                y_mm,
                depth_mm,
            )
            episode_id += 1

    total_episodes = len(episode_states)
    total_scale_tasks = len(physics_tasks)

    logging.info("=" * 72)
    logging.info("Dataset root: %s", dataset_root)
    logging.info("Indenters: %d", len(indenter_names))
    logging.info("Episodes: %d", total_episodes)
    logging.info("Scales: %s", args.scales_mm)
    logging.info("Total (episode,scale) physics tasks: %d", total_scale_tasks)
    logging.info("Marker textures: %d (%s)", len(textures), texture_dir)
    logging.info("Workers: physics=%d (GPU), render=%d (CPU)", args.max_physics_workers, args.max_render_workers)
    logging.info(
        "Camera: mode=%s fixed_distance=%.6fm base_fov=%.3fdeg ref_scale=%.2fmm uv_mode=%s uv_inset=%.3f",
        args.camera_mode,
        fixed_camera_distance_m,
        float(args.fov_deg),
        float(args.reference_scale_mm),
        str(args.uv_mode),
        float(args.uv_inset_ratio),
    )
    logging.info("=" * 72)

    physics_ok = 0
    render_ok = 0

    try:
        with cf.ProcessPoolExecutor(max_workers=args.max_physics_workers) as physics_pool, cf.ProcessPoolExecutor(
            max_workers=args.max_render_workers
        ) as render_pool:
            physics_futures: Dict[cf.Future, Dict[str, Any]] = {}
            for task in physics_tasks:
                f = physics_pool.submit(run_physics_task, task)
                physics_futures[f] = task

            render_futures: Dict[cf.Future, Dict[str, Any]] = {}

            for f in cf.as_completed(physics_futures):
                task = physics_futures[f]
                episode_id = int(task["episode_id"])
                scale_mm = int(task["scale_mm"])
                ep_state = episode_states[episode_id]

                try:
                    result = f.result()
                except Exception as exc:
                    msg = f"physics future crashed: ep={episode_id} scale={scale_mm} err={exc}"
                    ep_state.errors.append(msg)
                    logging.error(msg)
                    continue

                if result.get("status") != "ok":
                    msg = (
                        f"physics failed: ep={episode_id} scale={scale_mm} err={result.get('error')}\n"
                        f"{result.get('traceback', '')}"
                    )
                    ep_state.errors.append(msg)
                    logging.error("Physics failed | ep=%06d scale=%d", episode_id, scale_mm)
                    continue

                physics_ok += 1
                logging.info(
                    "Physics OK | ep=%06d scale=%d | max_down=%.4fmm mean_down=%.4fmm",
                    episode_id,
                    scale_mm,
                    result["deformation_stats"]["surface_max_down_mm"],
                    result["deformation_stats"]["surface_mean_down_mm"],
                )

                scale_dir = ep_state.episode_dir / f"scale_{scale_mm}mm"
                scale_dir.mkdir(parents=True, exist_ok=True)

                render_task = {
                    "episode_id": episode_id,
                    "scale_mm": scale_mm,
                    "repo_root": str(repo_root),
                    "blender_cmd": str(args.blender_cmd),
                    "blender_script": str(blender_script),
                    "textures_dir": str(texture_dir),
                    "scale_dir": str(scale_dir),
                    "stl_path": result["stl_path"],
                    "temp_scale_dir": result["temp_scale_dir"],
                    "camera_mode": str(args.camera_mode),
                    "base_fov_deg": float(args.fov_deg),
                    "fixed_distance_m": float(fixed_camera_distance_m),
                    "distance_safety": float(args.distance_safety),
                    "uv_mode": str(args.uv_mode),
                    "uv_inset_ratio": float(args.uv_inset_ratio),
                    "render_timeout_sec": int(args.render_timeout_sec),
                    "keep_intermediates": bool(args.keep_intermediates),
                }

                rf = render_pool.submit(run_render_task, render_task)
                render_futures[rf] = {
                    "physics_result": result,
                    "render_task": render_task,
                }

            for rf in cf.as_completed(render_futures):
                payload = render_futures[rf]
                physics_result = payload["physics_result"]
                episode_id = int(physics_result["episode_id"])
                scale_mm = int(physics_result["scale_mm"])
                ep_state = episode_states[episode_id]

                try:
                    rr = rf.result()
                except Exception as exc:
                    msg = f"render future crashed: ep={episode_id} scale={scale_mm} err={exc}"
                    ep_state.errors.append(msg)
                    logging.error(msg)
                    continue

                if rr.get("status") != "ok":
                    msg = (
                        f"render failed: ep={episode_id} scale={scale_mm} err={rr.get('error')}\n"
                        f"{rr.get('traceback', '')}"
                    )
                    ep_state.errors.append(msg)
                    logging.error("Render failed | ep=%06d scale=%d", episode_id, scale_mm)
                    continue

                render_ok += 1
                scale_key = f"scale_{scale_mm}mm"
                scale_dir = Path(rr["scale_dir"])
                if args.camera_mode == "fixed_distance_variable_fov":
                    render_fov_deg = math.degrees(
                        2.0 * math.atan((float(scale_mm) / 1000.0 / 2.0) / float(fixed_camera_distance_m))
                    )
                    render_distance_m = float(fixed_camera_distance_m)
                else:
                    render_fov_deg = float(args.fov_deg)
                    render_distance_m = (
                        (float(scale_mm) / 1000.0 / 2.0) / math.tan(math.radians(float(args.fov_deg)) / 2.0)
                    )
                    render_distance_m *= float(args.distance_safety)

                ep_state.scales[scale_key] = {
                    "scale_mm": scale_mm,
                    "contact_x_mm": float(physics_result["x_mm"]),
                    "contact_y_mm": float(physics_result["y_mm"]),
                    "contact_depth_mm": float(physics_result["depth_mm"]),
                    "deformation_stats": physics_result["deformation_stats"],
                    "camera_mode": str(args.camera_mode),
                    "camera_distance_m": float(render_distance_m),
                    "camera_fov_deg": float(render_fov_deg),
                    "rendered_markers": rr["rendered_markers"],
                    "patch_coords_16x16": str((scale_dir / "patch_coords_16x16.json").relative_to(ep_state.episode_dir)),
                }

                logging.info(
                    "Render OK | ep=%06d scale=%d | markers=%d",
                    episode_id,
                    scale_mm,
                    len(rr["rendered_markers"]),
                )

        manifest_episodes: List[Dict[str, Any]] = []
        failed_episodes: List[Dict[str, Any]] = []

        for episode_id in sorted(episode_states):
            state = episode_states[episode_id]
            expected_scales = len(args.scales_mm)
            is_ok = (len(state.errors) == 0) and (len(state.scales) == expected_scales)

            if not is_ok:
                failed_episodes.append(
                    {
                        "episode_id": state.episode_id,
                        "indenter": state.indenter,
                        "errors": state.errors,
                    }
                )
                if state.episode_dir.exists():
                    shutil.rmtree(state.episode_dir, ignore_errors=True)
                continue

            metadata = {
                "episode_id": state.episode_id,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "indenter": state.indenter,
                "particle": str(args.particle),
                "contact": {
                    "x_mm": state.x_mm,
                    "y_mm": state.y_mm,
                    "depth_mm": state.depth_mm,
                },
                "image_resolution": {"width": DEFAULT_IMAGE_RES[0], "height": DEFAULT_IMAGE_RES[1]},
                "scales": state.scales,
            }
            with open(state.episode_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            manifest_episodes.append(
                {
                    "episode_id": state.episode_id,
                    "path": state.episode_dir.name,
                    "indenter": state.indenter,
                    "contact": {
                        "x_mm": state.x_mm,
                        "y_mm": state.y_mm,
                        "depth_mm": state.depth_mm,
                    },
                }
            )

        manifest = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_root": str(dataset_root),
            "particle": str(args.particle),
            "scales_mm": [int(s) for s in args.scales_mm],
            "episodes_per_indenter": int(args.episodes_per_indenter),
            "indenter_count": len(indenter_names),
            "texture_count": len(textures),
            "total_episodes_planned": total_episodes,
            "total_scale_tasks_planned": total_scale_tasks,
            "physics_tasks_succeeded": physics_ok,
            "render_tasks_succeeded": render_ok,
            "successful_episodes": len(manifest_episodes),
            "failed_episodes": failed_episodes,
            "episodes": manifest_episodes,
        }

        with open(dataset_root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logging.info("=" * 72)
        logging.info("DONE | successful episodes: %d | failed episodes: %d", len(manifest_episodes), len(failed_episodes))
        logging.info("Manifest: %s", dataset_root / "manifest.json")
        logging.info("=" * 72)

    finally:
        restore_backup(parameters_path, backup_path)
        logging.info("Restored %s from backup", parameters_path)

        if script_tmp_dir.exists():
            shutil.rmtree(script_tmp_dir, ignore_errors=True)
        if work_tmp_root.exists() and not args.keep_intermediates:
            shutil.rmtree(work_tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
