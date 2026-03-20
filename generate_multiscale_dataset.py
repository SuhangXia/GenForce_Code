#!/usr/bin/env python3
"""Generate a large multiscale tactile dataset with cheap marker diversification.

Core idea (Simulation Economics):
- Stage 1 (GPU): Taichi MPM deformation, one .npz per (episode, scale).
- Stage 2 (CPU): Open3D meshing, one .stl per sampled frame.
- Stage 3 (CPU): Blender re-renders each frame .stl with ALL marker textures.

Output layout:
    datasets/usa_static_v1/
      manifest.json
      episode_000000/
        metadata.json
        scale_15mm/
          adapter_coord_map.npy
          frame_000/
            marker_Array1.jpg
            ...
          frame_001/
            ...
        scale_16mm/
          ...

This script is parallel-friendly:
- Physics workers: capped (default 7) to avoid CUDA OOM.
- Meshing workers: modest CPU pool for Open3D.
- Render workers: CPU-oriented process pool.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import yaml

try:
    from pytransform3d import rotations as pytransform3d_rotations
except ImportError:
    pytransform3d_rotations = None


DEFAULT_SCALES_MM = list(range(15, 26))
DEFAULT_REFERENCE_SCALE_MM = 25.0
DEFAULT_X_RANGE = (-DEFAULT_REFERENCE_SCALE_MM / 2.0, DEFAULT_REFERENCE_SCALE_MM / 2.0)
DEFAULT_Y_RANGE = (-DEFAULT_REFERENCE_SCALE_MM / 2.0, DEFAULT_REFERENCE_SCALE_MM / 2.0)
DEFAULT_DEPTH_RANGE = (0.4, 2.2)
DEFAULT_FOV_DEG = 40.0
DEFAULT_DISTANCE_SAFETY = 0.98
DEFAULT_PATCH_GRID = "14x14"
DEFAULT_POSITION_SAMPLE_RATIOS = {
    "clean": 0.65,
    "near_boundary": 0.25,
    "partial_crop": 0.10,
}
DEFAULT_INDENTER_BBOX_BOTTOM_HEIGHT_MM = 5.0
DEFAULT_NEAR_BOUNDARY_MAX_MARGIN_MM = 0.75
DEFAULT_PARTIAL_CROP_MIN_OVERHANG_MM = 0.10
DEFAULT_PARTIAL_CROP_MAX_OVERHANG_MM = 1.00
DEFAULT_FRAME_FRACTIONS = [0.4, 0.6, 0.8, 1.0]
DEFAULT_FRAME_SAMPLING_MODE = "depth_random"
DEFAULT_FRAME_COUNT = 4
DEFAULT_FRAME_DEPTH_START_MM = 0.4
DEFAULT_FRAME_DEPTH_END_MM = 2.2
DEFAULT_IMAGE_RES = (568, 568)
TEXTURE_EXTS = {".jpg", ".jpeg", ".png"}
IMAGE_INDEX_COLUMNS = [
    "dataset_root",
    "dataset_variant",
    "episode_id",
    "episode_dir",
    "indenter_name",
    "particle",
    "command_x_mm",
    "command_y_mm",
    "command_depth_mm",
    "position_sample_type_requested",
    "position_sample_type_actual",
    "contact_margin_min_scale_mm",
    "contact_margin_reference_scale_mm",
    "scale_key",
    "scale_mm",
    "frame_name",
    "frame_index",
    "frame_fraction",
    "frame_fraction_requested",
    "frame_target_max_down_mm",
    "frame_actual_max_down_mm",
    "trajectory_length",
    "frame_sampling_mode",
    "surface_max_down_mm",
    "surface_mean_down_mm",
    "surface_min_z_start_mm",
    "surface_min_z_frame_mm",
    "camera_mode",
    "camera_distance_m",
    "camera_fov_deg",
    "patch_grid_h",
    "patch_grid_w",
    "coordinate_convention",
    "image_width",
    "image_height",
    "marker_name",
    "image_relpath",
    "image_abspath",
    "adapter_coord_map_relpath",
    "adapter_coord_map_abspath",
    "split",
    "is_unseen_indenter",
]


def compute_full_sensor_fit_scale_m(scale_m: float, res_x: int, res_y: int) -> float:
    if scale_m <= 0:
        raise ValueError("scale_m must be > 0")
    if res_x <= 0 or res_y <= 0:
        raise ValueError("render resolution must be > 0")

    # The gel is square, but the render is 4:3. Expand the fitted span by the
    # limiting image dimension so the full square stays inside the frame.
    aspect = float(res_x) / float(res_y)
    return float(scale_m) * max(1.0, aspect)


OPEN3D_TEMP_SCRIPT = r'''
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def parse_args():
    p = argparse.ArgumentParser(description="Convert deformation npz to STL")
    p.add_argument("--input-npz", required=True)
    p.add_argument("--output-stl", required=True)
    p.add_argument("--frame-index", type=int, default=-1, help="Trajectory frame index. -1 means final frame.")
    return p.parse_args()


def _pick_frame(arr: np.ndarray, frame_index: int) -> np.ndarray:
    if arr.ndim == 1:
        # Single static frame.
        return arr
    if arr.ndim >= 2:
        t = int(arr.shape[0])
        idx = int(frame_index)
        if idx < 0:
            idx += t
        idx = max(0, min(t - 1, idx))
        return np.asarray(arr[idx]).reshape(-1)
    raise ValueError(f"Unsupported trajectory array rank: {arr.ndim}")


def load_surface_points(npz_path: Path, frame_index: int):
    data = np.load(npz_path)
    for key in ("p_xpos_list", "p_ypos_list", "p_zpos_list"):
        if key not in data:
            raise KeyError(f"Missing key {key} in {npz_path}")

    x = np.asarray(data["p_xpos_list"])
    y = np.asarray(data["p_ypos_list"])
    z = np.asarray(data["p_zpos_list"])

    x_frame = _pick_frame(x, frame_index)
    y_frame = _pick_frame(y, frame_index)
    z_frame = _pick_frame(z, frame_index)

    if not (x_frame.shape == y_frame.shape == z_frame.shape):
        raise ValueError("x/y/z shape mismatch in deformation npz")

    n = int(x_frame.shape[0])
    grid_n = int(round(np.sqrt(n)))
    regular_grid = (grid_n * grid_n == n)

    # mm -> m for Blender physical scale consistency.
    points = np.stack([x_frame, y_frame, z_frame], axis=1).astype(np.float64) / 1000.0

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

    points, regular_grid, grid_n = load_surface_points(npz_path, int(args.frame_index))

    if regular_grid and points.shape[0] == grid_n * grid_n:
        mesh = mesh_from_regular_grid(points, grid_n)
    else:
        mesh = mesh_from_point_cloud(points)

    ok = o3d.io.write_triangle_mesh(str(stl_path), mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write STL to {stl_path}")

    print(
        f"Converted {npz_path} frame={int(args.frame_index)} -> {stl_path} "
        f"with {len(mesh.triangles)} triangles"
    )


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

DEFAULT_IMAGE_RES = (568, 568)


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
    p.add_argument("--render-samples", type=int, default=32)
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


def compute_full_sensor_fit_scale_m(scale_m: float, res_x: int, res_y: int) -> float:
    if scale_m <= 0:
        raise ValueError("scale_m must be > 0")
    if res_x <= 0 or res_y <= 0:
        raise ValueError("render resolution must be > 0")

    # The gel is square, but the render is 4:3. With a horizontal-fit camera,
    # matching the square width would crop the square vertically. Inflate the
    # fitted span by the render aspect so the full square stays visible.
    aspect = float(res_x) / float(res_y)
    return float(scale_m) * max(1.0, aspect)


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

    # Only unwrap true top-contact faces. Including near-vertical side walls causes bright borders.
    top_face_min_normal_z = -0.2
    for face in bm.faces:
        face.select = (face.normal.z < top_face_min_normal_z)
    bmesh.update_edit_mesh(obj.data)

    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
    check_and_rotate_uvs_genforce(obj)

    # Non-contact side/bottom faces are not part of the visible gel marker plane.
    # Force their UVs outside [0,1] so texture CLIP outputs black instead of white seams.
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()
    for face in bm.faces:
        if face.select:
            continue
        for loop in face.loops:
            loop[uv_layer].uv = (-1.0, -1.0)
    bmesh.update_edit_mesh(obj.data)
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

    fit_scale_m = compute_full_sensor_fit_scale_m(
        scale_m,
        DEFAULT_IMAGE_RES[0],
        DEFAULT_IMAGE_RES[1],
    )

    if camera_mode == "fixed_distance_variable_fov":
        distance = fixed_distance_m
        fov = 2.0 * math.atan((fit_scale_m / 2.0) / distance)
    else:
        fov = math.radians(base_fov_deg)
        if fov <= 1e-6 or fov >= math.pi - 1e-6:
            raise ValueError("base_fov_deg produces invalid camera angle")
        distance = (fit_scale_m / 2.0) / math.tan(fov / 2.0)
        distance *= distance_safety

    cam_data.angle = fov

    cam = bpy.data.objects.new("ScaleCam", cam_data)
    cam.location = (cx, cy, z_top - distance)
    cam.rotation_euler = (math.pi, 0.0, 0.0)

    scene.collection.objects.link(cam)
    scene.camera = cam
    print(
        f"Camera setup | mode={camera_mode} scale_mm={scale_m*1000.0:.1f} "
        f"fit_scale_mm={fit_scale_m*1000.0:.4f} "
        f"fov_deg={math.degrees(fov):.4f} distance_m={distance:.6f}"
    )


def configure_render(out_dir: Path, render_samples: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if hasattr(scene, "cycles"):
        scene.cycles.samples = max(1, int(render_samples))
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.max_bounces = 4
        scene.cycles.device = "CPU"

    scene.render.resolution_x = DEFAULT_IMAGE_RES[0]
    scene.render.resolution_y = DEFAULT_IMAGE_RES[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = 100
    # Keep world lighting for photometric realism, but hide world in render layer.
    # We'll explicitly composite over black to avoid white-side borders in JPEG output.
    scene.render.film_transparent = True

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # Force transparent background to be composited over black.
    scene.use_nodes = True
    nt = scene.node_tree
    nodes = nt.nodes
    links = nt.links
    for n in list(nodes):
        nodes.remove(n)

    n_render = nodes.new("CompositorNodeRLayers")
    n_rgb = nodes.new("CompositorNodeRGB")
    n_rgb.outputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    n_alpha_over = nodes.new("CompositorNodeAlphaOver")
    n_alpha_over.premul = 1.0
    n_comp = nodes.new("CompositorNodeComposite")

    # AlphaOver expects: top/background from input[1], bottom/foreground from input[2].
    links.new(n_rgb.outputs[0], n_alpha_over.inputs[1])
    links.new(n_render.outputs[0], n_alpha_over.inputs[2])
    links.new(n_alpha_over.outputs["Image"], n_comp.inputs["Image"])

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
    configure_render(output_dir, int(args.render_samples))

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
    x_norm: float
    y_norm: float
    position_sample_type_requested: str
    position_sample_type_actual: str
    contact_margin_min_scale_mm: float
    contact_margin_reference_scale_mm: float
    reference_scale_mm: float
    depth_mm: float
    episode_dir: Path
    scales: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class ScaleState:
    episode_id: int
    scale_mm: int
    scale_key: str
    scale_dir: Path
    temp_scale_dir: Path
    contact_x_mm: float
    contact_y_mm: float
    contact_depth_mm: float
    trajectory_length: int
    deformation_stats_final: Dict[str, float]
    frame_sampling_mode: str
    frame_depth_targets_mm: List[float] | None
    sampled_frames: List[Dict[str, Any]]
    camera_mode: str
    camera_distance_m: float
    camera_fov_deg: float
    adapter_coord_map_path: Path
    adapter_coord_map_shape: List[int]
    frames: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending_mesh_count: int = 0
    pending_render_count: int = 0
    frames_reused: int = 0
    frames_rendered: int = 0
    failed: bool = False
    sealed: bool = False


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def make_scale_progress_entry(
    episode_id: int,
    scale_mm: int,
    indenter: str,
    x_mm: float,
    y_mm: float,
    x_norm: float,
    y_norm: float,
    position_sample_type_requested: str,
    position_sample_type_actual: str,
    depth_mm: float,
    frame_sampling_mode: str,
    frame_depth_targets_mm: Sequence[float] | None,
) -> Dict[str, Any]:
    return {
        "episode_id": int(episode_id),
        "episode_dir": f"episode_{int(episode_id):06d}",
        "scale_key": f"scale_{int(scale_mm)}mm",
        "scale_mm": int(scale_mm),
        "indenter": str(indenter),
        "contact": {
            "x_mm": float(x_mm),
            "y_mm": float(y_mm),
            "x_norm": float(x_norm),
            "y_norm": float(y_norm),
            "depth_mm": float(depth_mm),
        },
        "position_sample_type_requested": str(position_sample_type_requested),
        "position_sample_type_actual": str(position_sample_type_actual),
        "frame_sampling_mode": str(frame_sampling_mode),
        "frame_depth_targets_mm": [float(v) for v in frame_depth_targets_mm] if frame_depth_targets_mm else [],
        "physics": {
            "status": "pending",
            "trajectory_length": None,
            "npz_path": "",
            "reused": False,
        },
        "meshing": {
            "status": "pending",
            "planned_frames": 0,
            "completed_frames": 0,
            "failed_frames": 0,
        },
        "render": {
            "status": "pending",
            "planned_frames": 0,
            "completed_frames": 0,
            "reused_frames": 0,
            "failed_frames": 0,
        },
        "scale_complete": False,
        "restored_from_metadata": False,
        "last_error": "",
    }


def make_scale_progress_key(episode_id: int, scale_mm: int) -> str:
    return f"episode_{int(episode_id):06d}/scale_{int(scale_mm)}mm"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def _progress_eta_text(stage_start_ts: float, done: int, total: int) -> str:
    elapsed = max(0.0, time.time() - float(stage_start_ts))
    if total <= 0:
        return f"progress=0/0 elapsed={_format_duration(elapsed)} eta=-- eta_at=--"

    done = max(0, min(int(done), int(total)))
    if done == 0:
        return (
            f"progress=0/{total} elapsed={_format_duration(elapsed)} "
            f"eta=-- eta_at=--"
        )

    avg_sec = elapsed / float(done)
    remaining = max(int(total) - done, 0)
    eta_sec = avg_sec * float(remaining)
    eta_at = (dt.datetime.now() + dt.timedelta(seconds=eta_sec)).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"progress={done}/{total} elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta_sec)} eta_at={eta_at}"
    )


def _throughput_text(stage_start_ts: float, done: int) -> str:
    elapsed = max(1e-6, time.time() - float(stage_start_ts))
    rate = float(done) / elapsed
    return f"throughput={rate:.3f}/s"


def resolve_path(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base / value)


def format_coord_suffix(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def normalized_contact_from_reference_mm(x_mm: float, y_mm: float, reference_scale_mm: float) -> Tuple[float, float]:
    if reference_scale_mm <= 0:
        raise ValueError(f"reference_scale_mm must be > 0, got {reference_scale_mm}")
    half_scale = float(reference_scale_mm) / 2.0
    return float(x_mm) / half_scale, float(y_mm) / half_scale


def contact_mm_from_normalized(x_norm: float, y_norm: float, scale_mm: float) -> Tuple[float, float]:
    if scale_mm <= 0:
        raise ValueError(f"scale_mm must be > 0, got {scale_mm}")
    half_scale = float(scale_mm) / 2.0
    return float(x_norm) * half_scale, float(y_norm) * half_scale


def assign_position_sample_types(
    episodes_per_indenter: int,
    clean_ratio: float,
    near_boundary_ratio: float,
    partial_crop_ratio: float,
    rng: random.Random,
) -> List[str]:
    if episodes_per_indenter <= 0:
        raise ValueError(f"episodes_per_indenter must be > 0, got {episodes_per_indenter}")

    ratios = {
        "clean": float(clean_ratio),
        "near_boundary": float(near_boundary_ratio),
        "partial_crop": float(partial_crop_ratio),
    }
    if any(v < 0 for v in ratios.values()):
        raise ValueError(f"Position sample ratios must be >= 0, got {ratios}")

    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("At least one position sample ratio must be > 0")

    scaled = {
        key: float(episodes_per_indenter) * (value / total_ratio)
        for key, value in ratios.items()
    }
    counts = {key: int(math.floor(value)) for key, value in scaled.items()}
    remainder = int(episodes_per_indenter - sum(counts.values()))
    if remainder > 0:
        order = sorted(
            scaled.keys(),
            key=lambda key: (scaled[key] - counts[key], scaled[key]),
            reverse=True,
        )
        for idx in range(remainder):
            counts[order[idx % len(order)]] += 1

    sample_types = (
        ["clean"] * counts["clean"]
        + ["near_boundary"] * counts["near_boundary"]
        + ["partial_crop"] * counts["partial_crop"]
    )
    if len(sample_types) != episodes_per_indenter:
        raise RuntimeError(
            f"Position sample count mismatch: expected {episodes_per_indenter}, got {len(sample_types)}"
        )
    rng.shuffle(sample_types)
    return sample_types


def rotation_matrix_from_pose(pose: Dict[str, float], degrees: bool = False) -> np.ndarray:
    if pytransform3d_rotations is None:
        roll = float(pose.get("R", 0.0))
        pitch = float(pose.get("P", -np.pi))
        yaw = float(pose.get("Y", 0.0))
        if degrees:
            roll = np.radians(roll)
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return rz @ ry @ rx
    return pytransform3d_rotations.matrix_from_euler(
        (
            float(pose.get("R", 0.0)),
            float(pose.get("P", -np.pi)),
            float(pose.get("Y", 0.0)),
        ),
        0,
        1,
        2,
        degrees,
    )


def compute_indenter_contact_bbox_mm(
    indenter_npy_path: Path,
    rotation_matrix: np.ndarray,
    bottom_height_mm: float = DEFAULT_INDENTER_BBOX_BOTTOM_HEIGHT_MM,
) -> Dict[str, float]:
    if bottom_height_mm <= 0:
        raise ValueError(f"bottom_height_mm must be > 0, got {bottom_height_mm}")
    points = np.load(indenter_npy_path).astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N,3) array in {indenter_npy_path}, got {points.shape}")
    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid):
        raise ValueError(f"No finite points in {indenter_npy_path}")
    points = points[valid]
    points_rot = (rotation_matrix @ points.T).T
    z = points_rot[:, 2]
    z_min = float(np.min(z))
    z_max_slice = z_min + float(bottom_height_mm)
    footprint_xy = points_rot[(z >= z_min) & (z <= z_max_slice), :2]
    if footprint_xy.shape[0] < 3:
        raise ValueError(
            f"Too few bottom-slice points in {indenter_npy_path}: {footprint_xy.shape[0]}"
        )
    x_min = float(np.min(footprint_xy[:, 0]))
    x_max = float(np.max(footprint_xy[:, 0]))
    y_min = float(np.min(footprint_xy[:, 1]))
    y_max = float(np.max(footprint_xy[:, 1]))
    return {
        "x_min_mm": x_min,
        "x_max_mm": x_max,
        "y_min_mm": y_min,
        "y_max_mm": y_max,
        "width_mm": x_max - x_min,
        "height_mm": y_max - y_min,
        "bbox_mode": "bottom_slice",
        "bottom_height_mm": float(bottom_height_mm),
    }


def compute_margin_constrained_normalized_sampling_bounds(
    bbox_mm: Dict[str, float],
    scale_mm: float,
    reference_scale_mm: float,
    requested_x_min_mm: float,
    requested_x_max_mm: float,
    requested_y_min_mm: float,
    requested_y_max_mm: float,
    min_margin_mm: float = 0.0,
) -> Dict[str, float]:
    if scale_mm <= 0 or reference_scale_mm <= 0:
        raise ValueError("scale_mm and reference_scale_mm must be > 0")
    if min_margin_mm < 0:
        raise ValueError(f"min_margin_mm must be >= 0, got {min_margin_mm}")

    half_scale = float(scale_mm) / 2.0
    ref_half = float(reference_scale_mm) / 2.0

    feasible_x_min_norm = (-half_scale - float(bbox_mm["x_min_mm"]) + float(min_margin_mm)) / half_scale
    feasible_x_max_norm = (half_scale - float(bbox_mm["x_max_mm"]) - float(min_margin_mm)) / half_scale
    feasible_y_min_norm = (-half_scale - float(bbox_mm["y_min_mm"]) + float(min_margin_mm)) / half_scale
    feasible_y_max_norm = (half_scale - float(bbox_mm["y_max_mm"]) - float(min_margin_mm)) / half_scale

    requested_x_min_norm = float(requested_x_min_mm) / ref_half
    requested_x_max_norm = float(requested_x_max_mm) / ref_half
    requested_y_min_norm = float(requested_y_min_mm) / ref_half
    requested_y_max_norm = float(requested_y_max_mm) / ref_half

    x_min_norm = max(feasible_x_min_norm, requested_x_min_norm)
    x_max_norm = min(feasible_x_max_norm, requested_x_max_norm)
    y_min_norm = max(feasible_y_min_norm, requested_y_min_norm)
    y_max_norm = min(feasible_y_max_norm, requested_y_max_norm)

    return {
        "x_min_norm": x_min_norm,
        "x_max_norm": x_max_norm,
        "y_min_norm": y_min_norm,
        "y_max_norm": y_max_norm,
        "feasible_x_min_norm": feasible_x_min_norm,
        "feasible_x_max_norm": feasible_x_max_norm,
        "feasible_y_min_norm": feasible_y_min_norm,
        "feasible_y_max_norm": feasible_y_max_norm,
        "requested_x_min_norm": requested_x_min_norm,
        "requested_x_max_norm": requested_x_max_norm,
        "requested_y_min_norm": requested_y_min_norm,
        "requested_y_max_norm": requested_y_max_norm,
        "min_margin_mm": float(min_margin_mm),
    }


def compute_fully_inside_normalized_sampling_bounds(
    bbox_mm: Dict[str, float],
    min_scale_mm: float,
    reference_scale_mm: float,
    requested_x_min_mm: float,
    requested_x_max_mm: float,
    requested_y_min_mm: float,
    requested_y_max_mm: float,
) -> Dict[str, float]:
    return compute_margin_constrained_normalized_sampling_bounds(
        bbox_mm=bbox_mm,
        scale_mm=min_scale_mm,
        reference_scale_mm=reference_scale_mm,
        requested_x_min_mm=requested_x_min_mm,
        requested_x_max_mm=requested_x_max_mm,
        requested_y_min_mm=requested_y_min_mm,
        requested_y_max_mm=requested_y_max_mm,
        min_margin_mm=0.0,
    )


def sample_normalized_contact_from_bounds(
    rng: random.Random,
    bounds: Dict[str, float],
) -> Tuple[float, float]:
    if bounds["x_min_norm"] > bounds["x_max_norm"] or bounds["y_min_norm"] > bounds["y_max_norm"]:
        raise ValueError(f"Invalid sampling bounds: {bounds}")
    return (
        round(rng.uniform(bounds["x_min_norm"], bounds["x_max_norm"]), 6),
        round(rng.uniform(bounds["y_min_norm"], bounds["y_max_norm"]), 6),
    )


def compute_contact_margins_mm(
    bbox_mm: Dict[str, float],
    x_norm: float,
    y_norm: float,
    scale_mm: float,
) -> Dict[str, float]:
    half_scale = float(scale_mm) / 2.0
    x_mm, y_mm = contact_mm_from_normalized(x_norm=x_norm, y_norm=y_norm, scale_mm=scale_mm)
    left = x_mm + float(bbox_mm["x_min_mm"]) + half_scale
    right = half_scale - (x_mm + float(bbox_mm["x_max_mm"]))
    bottom = y_mm + float(bbox_mm["y_min_mm"]) + half_scale
    top = half_scale - (y_mm + float(bbox_mm["y_max_mm"]))
    return {
        "left_mm": float(left),
        "right_mm": float(right),
        "bottom_mm": float(bottom),
        "top_mm": float(top),
        "min_margin_mm": float(min(left, right, bottom, top)),
    }


def sample_stratified_contact_position(
    rng: random.Random,
    requested_sample_type: str,
    bbox_mm: Dict[str, float],
    clean_bounds: Dict[str, float],
    inside_bounds_min_scale: Dict[str, float],
    inside_bounds_reference_scale: Dict[str, float],
    min_scale_mm: float,
    reference_scale_mm: float,
    near_boundary_max_margin_mm: float,
    partial_crop_min_overhang_mm: float,
    partial_crop_max_overhang_mm: float,
    max_attempts: int = 512,
) -> Dict[str, float | str]:
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be > 0, got {max_attempts}")

    def _bounds_valid(bounds: Dict[str, float]) -> bool:
        return bounds["x_min_norm"] <= bounds["x_max_norm"] and bounds["y_min_norm"] <= bounds["y_max_norm"]

    def _sample_clean() -> Tuple[float, float] | None:
        if not _bounds_valid(clean_bounds):
            return None
        return sample_normalized_contact_from_bounds(rng, clean_bounds)

    def _sample_near_boundary() -> Tuple[float, float] | None:
        if not _bounds_valid(inside_bounds_min_scale):
            return None
        for _ in range(max_attempts):
            x_norm, y_norm = sample_normalized_contact_from_bounds(rng, inside_bounds_min_scale)
            margin = compute_contact_margins_mm(bbox_mm, x_norm, y_norm, min_scale_mm)["min_margin_mm"]
            if 0.0 <= margin <= float(near_boundary_max_margin_mm):
                return x_norm, y_norm
        return None

    def _sample_partial_crop() -> Tuple[float, float] | None:
        if not _bounds_valid(inside_bounds_reference_scale):
            return None
        for _ in range(max_attempts):
            x_norm, y_norm = sample_normalized_contact_from_bounds(rng, inside_bounds_reference_scale)
            min_scale_margin = compute_contact_margins_mm(bbox_mm, x_norm, y_norm, min_scale_mm)["min_margin_mm"]
            if -float(partial_crop_max_overhang_mm) <= min_scale_margin <= -float(partial_crop_min_overhang_mm):
                return x_norm, y_norm
        return None

    def _sample_reference_safe() -> Tuple[float, float] | None:
        if not _bounds_valid(inside_bounds_reference_scale):
            return None
        return sample_normalized_contact_from_bounds(rng, inside_bounds_reference_scale)

    strategy_order = {
        "clean": ("clean", "near_boundary", "partial_crop", "reference_safe_fallback"),
        "near_boundary": ("near_boundary", "clean", "partial_crop", "reference_safe_fallback"),
        "partial_crop": ("partial_crop", "near_boundary", "clean", "reference_safe_fallback"),
    }
    if requested_sample_type not in strategy_order:
        raise ValueError(f"Unknown requested_sample_type: {requested_sample_type}")

    samplers = {
        "clean": _sample_clean,
        "near_boundary": _sample_near_boundary,
        "partial_crop": _sample_partial_crop,
        "reference_safe_fallback": _sample_reference_safe,
    }

    chosen: Tuple[float, float] | None = None
    actual_sample_type = requested_sample_type
    for actual_sample_type in strategy_order[requested_sample_type]:
        chosen = samplers[actual_sample_type]()
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError(
            f"Failed to sample position for {requested_sample_type} after trying {strategy_order[requested_sample_type]}"
        )

    x_norm, y_norm = chosen
    min_scale_margin = compute_contact_margins_mm(bbox_mm, x_norm, y_norm, min_scale_mm)["min_margin_mm"]
    reference_scale_margin = compute_contact_margins_mm(
        bbox_mm, x_norm, y_norm, reference_scale_mm
    )["min_margin_mm"]
    return {
        "x_norm": float(x_norm),
        "y_norm": float(y_norm),
        "requested_sample_type": str(requested_sample_type),
        "actual_sample_type": str(actual_sample_type),
        "min_scale_margin_mm": float(min_scale_margin),
        "reference_scale_margin_mm": float(reference_scale_margin),
    }


def expected_npz_path(npz_root: Path, indenter: str, x_mm: float, y_mm: float, depth_mm: float) -> Path:
    suffix = f"{format_coord_suffix(x_mm)}_{format_coord_suffix(y_mm)}_{round(depth_mm, 1)}.npz"
    return npz_root / indenter / suffix


def _load_z_trajectory(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "p_zpos_list" not in data:
        raise KeyError(f"Missing key p_zpos_list in {npz_path}")

    z = np.asarray(data["p_zpos_list"], dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    elif z.ndim >= 2:
        z = z.reshape(z.shape[0], -1)
    else:
        raise ValueError(f"Unsupported p_zpos_list shape in {npz_path}: {z.shape}")

    if z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError(f"Empty z trajectory in {npz_path}: {z.shape}")
    return z


def deformation_stats_for_frame(z_trajectory: np.ndarray, frame_index: int) -> Dict[str, float]:
    if z_trajectory.ndim != 2:
        raise ValueError(f"z_trajectory must be 2D [T,N], got {z_trajectory.shape}")
    t = int(z_trajectory.shape[0])
    idx = int(frame_index)
    if idx < 0:
        idx += t
    idx = max(0, min(t - 1, idx))

    start = z_trajectory[0]
    frame = z_trajectory[idx]
    down = start - frame
    return {
        "surface_max_down_mm": float(np.max(down)),
        "surface_mean_down_mm": float(np.mean(down)),
        "surface_min_z_start_mm": float(np.min(start)),
        "surface_min_z_frame_mm": float(np.min(frame)),
    }


def make_stratified_random_frame_depth_targets_mm(
    frame_count: int,
    depth_start_mm: float,
    depth_end_mm: float,
    rng: random.Random,
) -> List[float]:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}")
    if depth_start_mm <= 0:
        raise ValueError(f"depth_start_mm must be > 0, got {depth_start_mm}")
    if depth_end_mm <= depth_start_mm:
        raise ValueError(
            f"depth_end_mm must be > depth_start_mm, got {depth_end_mm} <= {depth_start_mm}"
        )

    # Sample one depth target per consecutive depth stage for shallow->deep coverage.
    bin_edges = np.linspace(float(depth_start_mm), float(depth_end_mm), frame_count + 1, dtype=np.float64)
    targets: List[float] = []
    for bin_idx in range(frame_count):
        low = float(bin_edges[bin_idx])
        high = float(bin_edges[bin_idx + 1])
        target = float(rng.uniform(low, high))
        targets.append(round(target, 4))
    targets.sort()
    return [round(v, 4) for v in targets]


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _csv_cell(value: Any) -> Any:
    return "" if value is None else value


def build_image_index_csv(dataset_root: Path) -> Tuple[Path, int]:
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "manifest.json"
    csv_path = dataset_root / "image_index.csv"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    patch_grid = manifest.get("patch_grid")
    patch_grid_h = ""
    patch_grid_w = ""
    if isinstance(patch_grid, (list, tuple)) and len(patch_grid) >= 2:
        patch_grid_h = int(patch_grid[0])
        patch_grid_w = int(patch_grid[1])

    coordinate_convention = str(manifest.get("coordinate_convention", ""))
    dataset_variant = str(manifest.get("dataset_variant", ""))
    rows_written = 0

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=IMAGE_INDEX_COLUMNS)
        writer.writeheader()

        for episode_entry in manifest.get("episodes", []):
            if not isinstance(episode_entry, dict):
                continue

            episode_dir_name = str(
                episode_entry.get(
                    "path",
                    f"episode_{int(episode_entry.get('episode_id', 0)):06d}",
                )
            )
            episode_dir = dataset_root / episode_dir_name
            metadata_path = episode_dir / "metadata.json"
            if not metadata_path.exists():
                logging.warning("Skipping CSV indexing for missing metadata: %s", metadata_path)
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as ep_f:
                    episode_meta = json.load(ep_f)
            except Exception as exc:
                logging.warning("Skipping CSV indexing for unreadable metadata (%s): %s", metadata_path, exc)
                continue

            contact = episode_meta.get("contact", {})
            image_resolution = episode_meta.get("image_resolution", {})
            image_width = DEFAULT_IMAGE_RES[0]
            image_height = DEFAULT_IMAGE_RES[1]
            if isinstance(image_resolution, dict):
                image_width = int(image_resolution.get("width", DEFAULT_IMAGE_RES[0]))
                image_height = int(image_resolution.get("height", DEFAULT_IMAGE_RES[1]))

            scales = episode_meta.get("scales", {})
            if not isinstance(scales, dict):
                logging.warning("Skipping CSV indexing for invalid scales block: %s", metadata_path)
                continue

            for scale_key, scale_meta in sorted(scales.items()):
                if not isinstance(scale_meta, dict):
                    logging.warning("Skipping invalid scale metadata in %s: %s", metadata_path, scale_key)
                    continue

                scale_mm = scale_meta.get("scale_mm", "")
                adapter_rel = str(scale_meta.get("adapter_coord_map", "") or "")
                adapter_abs = str((episode_dir / adapter_rel).resolve()) if adapter_rel else ""
                frames = scale_meta.get("frames", {})
                if not isinstance(frames, dict):
                    logging.warning("Skipping invalid frames block in %s: %s", metadata_path, scale_key)
                    continue

                for frame_name, frame_meta in sorted(frames.items()):
                    if not isinstance(frame_meta, dict):
                        logging.warning(
                            "Skipping invalid frame metadata in %s: %s/%s",
                            metadata_path,
                            scale_key,
                            frame_name,
                        )
                        continue

                    rendered_markers = frame_meta.get("rendered_markers", [])
                    if not isinstance(rendered_markers, list):
                        logging.warning(
                            "Skipping invalid rendered_markers in %s: %s/%s",
                            metadata_path,
                            scale_key,
                            frame_name,
                        )
                        continue

                    deformation_stats = frame_meta.get("deformation_stats", {})
                    if not isinstance(deformation_stats, dict):
                        deformation_stats = {}

                    for marker_name in rendered_markers:
                        marker_str = str(marker_name)
                        image_path = episode_dir / str(scale_key) / str(frame_name) / marker_str
                        if not image_path.exists():
                            logging.warning("Skipping missing image during CSV rebuild: %s", image_path)
                            continue

                        row = {
                            "dataset_root": str(dataset_root),
                            "dataset_variant": dataset_variant,
                            "episode_id": episode_meta.get("episode_id", episode_entry.get("episode_id", "")),
                            "episode_dir": episode_dir_name,
                            "indenter_name": episode_meta.get("indenter", episode_entry.get("indenter", "")),
                            "particle": episode_meta.get("particle", manifest.get("particle", "")),
                            "command_x_mm": scale_meta.get("contact_x_mm", contact.get("x_mm", "")),
                            "command_y_mm": scale_meta.get("contact_y_mm", contact.get("y_mm", "")),
                            "command_depth_mm": contact.get("depth_mm", ""),
                            "position_sample_type_requested": contact.get(
                                "position_sample_type_requested",
                                scale_meta.get("position_sample_type_requested", ""),
                            ),
                            "position_sample_type_actual": contact.get(
                                "position_sample_type_actual",
                                scale_meta.get("position_sample_type_actual", ""),
                            ),
                            "contact_margin_min_scale_mm": contact.get("min_scale_margin_mm", ""),
                            "contact_margin_reference_scale_mm": contact.get("reference_scale_margin_mm", ""),
                            "scale_key": scale_key,
                            "scale_mm": scale_mm,
                            "frame_name": frame_name,
                            "frame_index": frame_meta.get("frame_index", ""),
                            "frame_fraction": frame_meta.get("frame_fraction", ""),
                            "frame_fraction_requested": frame_meta.get("frame_fraction_requested", ""),
                            "frame_target_max_down_mm": frame_meta.get("frame_target_max_down_mm", ""),
                            "frame_actual_max_down_mm": frame_meta.get("frame_actual_max_down_mm", ""),
                            "trajectory_length": frame_meta.get("trajectory_length", scale_meta.get("trajectory_length", "")),
                            "frame_sampling_mode": scale_meta.get("frame_sampling_mode", manifest.get("frame_sampling_mode", "")),
                            "surface_max_down_mm": deformation_stats.get("surface_max_down_mm", ""),
                            "surface_mean_down_mm": deformation_stats.get("surface_mean_down_mm", ""),
                            "surface_min_z_start_mm": deformation_stats.get("surface_min_z_start_mm", ""),
                            "surface_min_z_frame_mm": deformation_stats.get("surface_min_z_frame_mm", ""),
                            "camera_mode": scale_meta.get("camera_mode", ""),
                            "camera_distance_m": scale_meta.get("camera_distance_m", ""),
                            "camera_fov_deg": scale_meta.get("camera_fov_deg", ""),
                            "patch_grid_h": patch_grid_h,
                            "patch_grid_w": patch_grid_w,
                            "coordinate_convention": coordinate_convention,
                            "image_width": image_width,
                            "image_height": image_height,
                            "marker_name": marker_str,
                            "image_relpath": _safe_relpath(image_path, dataset_root),
                            "image_abspath": str(image_path.resolve()),
                            "adapter_coord_map_relpath": adapter_rel,
                            "adapter_coord_map_abspath": adapter_abs,
                            "split": episode_meta.get("split", manifest.get("split", "")),
                            "is_unseen_indenter": episode_meta.get("is_unseen_indenter", ""),
                        }
                        writer.writerow({key: _csv_cell(row.get(key)) for key in IMAGE_INDEX_COLUMNS})
                        rows_written += 1

    return csv_path, rows_written


def max_down_trajectory_mm(z_trajectory: np.ndarray) -> np.ndarray:
    if z_trajectory.ndim != 2:
        raise ValueError(f"z_trajectory must be 2D [T,N], got {z_trajectory.shape}")
    start = z_trajectory[0:1, :]
    return np.max(start - z_trajectory, axis=1)


def resolve_sampled_frames_by_depth(
    z_trajectory: np.ndarray,
    frame_depth_targets_mm: Sequence[float],
    deduplicate: bool,
) -> List[Dict[str, float | int]]:
    if z_trajectory.ndim != 2:
        raise ValueError(f"z_trajectory must be 2D [T,N], got {z_trajectory.shape}")
    if not frame_depth_targets_mm:
        raise ValueError("frame_depth_targets_mm cannot be empty in depth sampling mode")

    max_down = max_down_trajectory_mm(z_trajectory)
    t = int(max_down.shape[0])
    t_last = max(t - 1, 0)
    samples: List[Dict[str, float | int]] = []

    for depth_target in frame_depth_targets_mm:
        target = float(depth_target)
        if target < 0:
            raise ValueError(f"frame depth target must be >= 0, got {target}")

        idx = int(np.argmin(np.abs(max_down - target)))

        idx = max(0, min(t_last, idx))
        frac = 1.0 if t_last == 0 else float(idx) / float(t_last)
        samples.append(
            {
                "frame_index": idx,
                "frame_fraction": frac,
                "frame_fraction_requested": frac,
                "frame_target_max_down_mm": target,
                "frame_actual_max_down_mm": float(max_down[idx]),
            }
        )

    samples.sort(key=lambda x: int(x["frame_index"]))
    if deduplicate:
        unique: Dict[int, Dict[str, float | int]] = {}
        for s in samples:
            idx = int(s["frame_index"])
            if idx not in unique:
                unique[idx] = s
        samples = [unique[idx] for idx in sorted(unique.keys())]
    return samples


def resolve_sampled_frames(
    trajectory_length: int,
    frame_indices: Sequence[int] | None,
    frame_fractions: Sequence[float] | None,
    deduplicate: bool,
) -> List[Dict[str, float | int]]:
    if trajectory_length <= 0:
        raise ValueError(f"trajectory_length must be > 0, got {trajectory_length}")

    samples: List[Dict[str, float | int]] = []
    t_last = max(trajectory_length - 1, 0)

    if frame_indices is not None and len(frame_indices) > 0:
        for raw_idx in frame_indices:
            idx = int(raw_idx)
            idx = max(0, min(t_last, idx))
            frac = 1.0 if t_last == 0 else float(idx) / float(t_last)
            samples.append(
                {
                    "frame_index": idx,
                    "frame_fraction": frac,
                    "frame_fraction_requested": frac,
                }
            )
    else:
        use_fractions = list(frame_fractions) if frame_fractions else [1.0]
        for f in use_fractions:
            if not (0.0 < float(f) <= 1.0):
                raise ValueError(f"frame fraction must be in (0,1], got {f}")
            idx = int(round(float(f) * t_last))
            idx = max(0, min(t_last, idx))
            frac_actual = 1.0 if t_last == 0 else float(idx) / float(t_last)
            samples.append(
                {
                    "frame_index": idx,
                    "frame_fraction": frac_actual,
                    "frame_fraction_requested": float(f),
                }
            )

    samples.sort(key=lambda x: int(x["frame_index"]))
    if deduplicate:
        unique: Dict[int, Dict[str, float | int]] = {}
        for s in samples:
            idx = int(s["frame_index"])
            if idx not in unique:
                unique[idx] = s
        samples = [unique[idx] for idx in sorted(unique.keys())]

    if not samples:
        samples = [{"frame_index": t_last, "frame_fraction": 1.0, "frame_fraction_requested": 1.0}]
    return samples


def make_frame_metadata(sample: Dict[str, Any], rendered_markers: Sequence[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "frame_index": int(sample["frame_index"]),
        "frame_fraction": float(sample.get("frame_fraction", 0.0)),
        "frame_fraction_requested": float(
            sample.get("frame_fraction_requested", sample.get("frame_fraction", 0.0))
        ),
        "trajectory_length": int(sample["trajectory_length"]),
        "rendered_markers": [str(v) for v in rendered_markers],
        "deformation_stats": sample["deformation_stats"],
    }
    if "frame_target_max_down_mm" in sample:
        meta["frame_target_max_down_mm"] = float(sample["frame_target_max_down_mm"])
    if "frame_actual_max_down_mm" in sample:
        meta["frame_actual_max_down_mm"] = float(sample["frame_actual_max_down_mm"])
    return meta


def parse_patch_grid(spec: str) -> Tuple[int, int]:
    value = str(spec).strip().lower()
    if "x" not in value:
        raise ValueError(f"Invalid patch grid '{spec}', expected format HxW (for example 16x16)")
    parts = value.split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid patch grid '{spec}', expected format HxW")
    try:
        patch_h = int(parts[0])
        patch_w = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid patch grid '{spec}', H and W must be integers") from exc
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"Invalid patch grid '{spec}', H and W must be > 0")
    return patch_h, patch_w


def make_adapter_coord_map(scale_mm: float, patch_h: int, patch_w: int) -> np.ndarray:
    if scale_mm <= 0:
        raise ValueError("scale_mm must be > 0")
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_h and patch_w must be > 0")

    # Patch-center coordinates with image row-major convention:
    # i (row) goes top->bottom, j (col) goes left->right.
    # X increases to the right, Y increases downward.
    step_x = float(scale_mm) / float(patch_w)
    step_y = float(scale_mm) / float(patch_h)
    x_axis = np.linspace(
        -float(scale_mm) / 2.0 + step_x / 2.0,
        float(scale_mm) / 2.0 - step_x / 2.0,
        patch_w,
        dtype=np.float64,
    )
    y_axis = np.linspace(
        -float(scale_mm) / 2.0 + step_y / 2.0,
        float(scale_mm) / 2.0 - step_y / 2.0,
        patch_h,
        dtype=np.float64,
    )
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    coord_map = np.stack([xx, yy], axis=-1).astype(np.float64, copy=False)
    return coord_map


def frame_outputs_complete(frame_dir: Path, expected_marker_files: Sequence[str]) -> bool:
    if not frame_dir.exists():
        return False
    for name in expected_marker_files:
        if not (frame_dir / name).exists():
            return False
    return True


def validate_existing_scale_metadata(
    episode_dir: Path,
    scale_key: str,
    scale_meta: Dict[str, Any],
    expected_marker_files: Sequence[str] | None = None,
    expected_frame_sampling_mode: str | None = None,
    expected_frame_depth_targets_mm: Sequence[float] | None = None,
    expected_contact_x_mm: float | None = None,
    expected_contact_y_mm: float | None = None,
) -> bool:
    if not isinstance(scale_meta, dict):
        return False

    rel = scale_meta.get("adapter_coord_map")
    if not rel:
        return False
    coord_map_path = episode_dir / rel
    if not coord_map_path.exists():
        return False

    if expected_frame_sampling_mode is not None:
        got_mode = str(scale_meta.get("frame_sampling_mode", "fraction"))
        if got_mode != str(expected_frame_sampling_mode):
            return False

    if expected_frame_depth_targets_mm is not None:
        got_targets = scale_meta.get("frame_depth_targets_mm")
        if not isinstance(got_targets, list):
            return False
        if len(got_targets) != len(expected_frame_depth_targets_mm):
            return False
        for a, b in zip(got_targets, expected_frame_depth_targets_mm):
            if abs(float(a) - float(b)) > 1e-3:
                return False

    if expected_contact_x_mm is not None:
        got_x = scale_meta.get("contact_x_mm")
        if got_x is None or abs(float(got_x) - float(expected_contact_x_mm)) > 1e-4:
            return False
    if expected_contact_y_mm is not None:
        got_y = scale_meta.get("contact_y_mm")
        if got_y is None or abs(float(got_y) - float(expected_contact_y_mm)) > 1e-4:
            return False

    frames = scale_meta.get("frames")
    if not isinstance(frames, dict) or len(frames) == 0:
        return False

    expected = set(expected_marker_files or [])
    for frame_name, frame_meta in frames.items():
        if not isinstance(frame_meta, dict):
            return False
        rendered = frame_meta.get("rendered_markers")
        if not isinstance(rendered, list) or len(rendered) == 0:
            return False
        rendered_set = set(str(v) for v in rendered)
        if expected and (not expected.issubset(rendered_set)):
            return False
        frame_dir = episode_dir / scale_key / str(frame_name)
        check_names = expected if expected else rendered_set
        for marker_name in check_names:
            if not (frame_dir / str(marker_name)).exists():
                return False

    return True


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
    resume_mode = bool(task.get("resume", False))

    try:
        npz_path = expected_npz_path(npz_root, indenter, x_mm, y_mm, depth_mm)
        physics_reused = False
        if resume_mode and npz_path.exists():
            physics_reused = True
        else:
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

        if not npz_path.exists():
            x_tag = format_coord_suffix(x_mm)
            y_tag = format_coord_suffix(y_mm)
            candidates = sorted((npz_root / indenter).glob(f"{x_tag}_{y_tag}_*.npz"))
            if not candidates:
                raise FileNotFoundError(f"No deformation npz found for {indenter} at {npz_root}")
            npz_path = candidates[-1]
            if resume_mode:
                physics_reused = True

        z_trajectory = _load_z_trajectory(npz_path)
        trajectory_length = int(z_trajectory.shape[0])
        deform_final = deformation_stats_for_frame(z_trajectory, trajectory_length - 1)
        frame_sampling_mode = str(task.get("frame_sampling_mode", "fraction"))
        if frame_sampling_mode == "depth_random":
            frame_depth_targets_mm = [float(v) for v in task.get("frame_depth_targets_mm", [])]
            samples = resolve_sampled_frames_by_depth(
                z_trajectory=z_trajectory,
                frame_depth_targets_mm=frame_depth_targets_mm,
                deduplicate=bool(task.get("deduplicate_frame_indices", True)),
            )
        else:
            frame_depth_targets_mm = None
            samples = resolve_sampled_frames(
                trajectory_length=trajectory_length,
                frame_indices=task.get("frame_indices"),
                frame_fractions=task.get("frame_fractions"),
                deduplicate=bool(task.get("deduplicate_frame_indices", True)),
            )

        sampled_frames: List[Dict[str, Any]] = []
        for sample_ord, sample in enumerate(samples):
            frame_index = int(sample["frame_index"])
            frame_meta: Dict[str, Any] = {
                "frame_name": f"frame_{sample_ord:03d}",
                "frame_index": frame_index,
                "frame_fraction": float(sample.get("frame_fraction", 0.0)),
                "frame_fraction_requested": float(
                    sample.get("frame_fraction_requested", sample.get("frame_fraction", 0.0))
                ),
                "trajectory_length": trajectory_length,
                "deformation_stats": deformation_stats_for_frame(z_trajectory, frame_index),
            }
            if "frame_target_max_down_mm" in sample:
                frame_meta["frame_target_max_down_mm"] = float(sample["frame_target_max_down_mm"])
            if "frame_actual_max_down_mm" in sample:
                frame_meta["frame_actual_max_down_mm"] = float(sample["frame_actual_max_down_mm"])
            sampled_frames.append(frame_meta)

        return {
            "status": "ok",
            "episode_id": episode_id,
            "scale_mm": scale_mm,
            "indenter": indenter,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "depth_mm": depth_mm,
            "temp_scale_dir": str(temp_scale_dir),
            "npz_path": str(npz_path),
            "trajectory_length": trajectory_length,
            "deformation_stats_final": deform_final,
            "physics_reused": physics_reused,
            "frame_sampling_mode": frame_sampling_mode,
            "frame_depth_targets_mm": frame_depth_targets_mm,
            "sampled_frames": sampled_frames,
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


def run_meshing_task(task: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = int(task["episode_id"])
    scale_mm = int(task["scale_mm"])
    frame_name = str(task["frame_name"])
    frame_index = int(task["frame_index"])
    repo_root = Path(task["repo_root"])
    stl_path = Path(task["stl_path"])
    stl_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mesh_cmd = [
            str(task["python_cmd"]),
            str(task["open3d_script"]),
            "--input-npz",
            str(task["npz_path"]),
            "--output-stl",
            str(stl_path),
            "--frame-index",
            str(frame_index),
        ]
        run_cmd_checked(
            mesh_cmd,
            cwd=repo_root,
            timeout_sec=int(task["meshing_timeout_sec"]),
            stage=f"npz_to_stl[{frame_name}]",
        )
        return {
            "status": "ok",
            "episode_id": episode_id,
            "scale_mm": scale_mm,
            "frame_name": frame_name,
            "frame_index": frame_index,
            "stl_path": str(stl_path),
            "sample": task["sample"],
        }
    except Exception as exc:
        return {
            "status": "error",
            "episode_id": episode_id,
            "scale_mm": scale_mm,
            "frame_name": frame_name,
            "frame_index": frame_index,
            "stl_path": str(stl_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_render_task(task: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = int(task["episode_id"])
    scale_mm = int(task["scale_mm"])
    frame_name = str(task["frame_name"])
    frame_dir = Path(task["frame_dir"])
    frame_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(task["repo_root"])
    stl_path = Path(task["stl_path"])
    keep_intermediates = bool(task["keep_intermediates"])
    resume_mode = bool(task.get("resume", False))
    expected_marker_files = [str(v) for v in task.get("expected_marker_files", [])]
    if not expected_marker_files:
        raise ValueError("run_render_task requires non-empty expected_marker_files")

    try:
        frame_complete = frame_outputs_complete(frame_dir, expected_marker_files)
        if resume_mode and frame_complete:
            rendered = sorted(expected_marker_files)
            return {
                "status": "ok",
                "episode_id": episode_id,
                "scale_mm": scale_mm,
                "frame_name": frame_name,
                "rendered_markers": rendered,
                "reused": True,
            }

        if not stl_path.exists():
            raise FileNotFoundError(f"Missing STL for render: {stl_path}")

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
            str(frame_dir),
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
            "--render-samples",
            str(task["render_samples"]),
        ]
        run_cmd_checked(
            render_cmd,
            cwd=repo_root,
            timeout_sec=int(task["render_timeout_sec"]),
            stage=f"blender_render[{frame_name}]",
        )

        rendered = sorted(p.name for p in frame_dir.glob("marker_*.jpg"))
        if not rendered:
            raise RuntimeError(f"No marker renders produced in {frame_dir}")

        side_border_px = int(task.get("force_black_side_border_px", 0))
        if side_border_px > 0:
            for name in rendered:
                img_path = frame_dir / name
                with Image.open(img_path) as im:
                    rgb = np.array(im.convert("RGB"), dtype=np.uint8)
                px = min(side_border_px, max(1, rgb.shape[1] // 2))
                rgb[:, :px, :] = 0
                rgb[:, -px:, :] = 0
                Image.fromarray(rgb, mode="RGB").save(img_path, format="JPEG", quality=100)

        rendered = sorted(p.name for p in frame_dir.glob("marker_*.jpg"))
        missing_after_render = [name for name in expected_marker_files if not (frame_dir / name).exists()]
        if missing_after_render:
            raise RuntimeError(
                f"Missing marker renders after render for {frame_name}: {missing_after_render[:3]}"
            )

        return {
            "status": "ok",
            "episode_id": episode_id,
            "scale_mm": scale_mm,
            "frame_name": frame_name,
            "rendered_markers": rendered,
            "reused": False,
        }
    except Exception as exc:
        return {
            "status": "error",
            "episode_id": episode_id,
            "scale_mm": scale_mm,
            "frame_name": frame_name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if (not keep_intermediates) and stl_path.exists():
            stl_path.unlink(missing_ok=True)


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
    p.add_argument("--dataset-root", type=Path, default=Path("datasets/usa_static_v1"))

    p.add_argument("--particle", type=str, default="100000")
    p.add_argument("--episodes-per-indenter", type=int, default=1)
    p.add_argument("--scales-mm", type=int, nargs="+", default=DEFAULT_SCALES_MM)
    p.add_argument(
        "--patch-grid",
        type=str,
        default=DEFAULT_PATCH_GRID,
        help="Patch grid in HxW format for adapter_coord_map export (for example 16x16).",
    )
    p.add_argument("--objects", nargs="*", default=None, help="Optional subset of indenter names")

    p.add_argument("--x-min", type=float, default=DEFAULT_X_RANGE[0])
    p.add_argument("--x-max", type=float, default=DEFAULT_X_RANGE[1])
    p.add_argument("--y-min", type=float, default=DEFAULT_Y_RANGE[0])
    p.add_argument("--y-max", type=float, default=DEFAULT_Y_RANGE[1])
    p.add_argument(
        "--position-clean-ratio",
        type=float,
        default=DEFAULT_POSITION_SAMPLE_RATIOS["clean"],
        help="Fraction of episodes sampled as clean fully in-bound samples.",
    )
    p.add_argument(
        "--position-near-boundary-ratio",
        type=float,
        default=DEFAULT_POSITION_SAMPLE_RATIOS["near_boundary"],
        help="Fraction of episodes sampled near the smallest-scale boundary but still mostly inside.",
    )
    p.add_argument(
        "--position-partial-crop-ratio",
        type=float,
        default=DEFAULT_POSITION_SAMPLE_RATIOS["partial_crop"],
        help="Fraction of episodes allowed to be partially cropped on the smallest scale.",
    )
    p.add_argument(
        "--near-boundary-max-margin-mm",
        type=float,
        default=DEFAULT_NEAR_BOUNDARY_MAX_MARGIN_MM,
        help="Maximum smallest-scale margin (mm) for near-boundary samples.",
    )
    p.add_argument(
        "--partial-crop-min-overhang-mm",
        type=float,
        default=DEFAULT_PARTIAL_CROP_MIN_OVERHANG_MM,
        help="Minimum smallest-scale out-of-bounds amount (mm) for partial-crop samples.",
    )
    p.add_argument(
        "--partial-crop-max-overhang-mm",
        type=float,
        default=DEFAULT_PARTIAL_CROP_MAX_OVERHANG_MM,
        help="Maximum smallest-scale out-of-bounds amount (mm) for partial-crop samples.",
    )
    p.add_argument("--depth-min", type=float, default=DEFAULT_DEPTH_RANGE[0])
    p.add_argument("--depth-max", type=float, default=DEFAULT_DEPTH_RANGE[1])
    p.add_argument(
        "--frame-sampling-mode",
        choices=["depth_random", "fraction"],
        default=DEFAULT_FRAME_SAMPLING_MODE,
        help="Frame sampling policy. depth_random: stratified-random max-down targets per episode.",
    )
    p.add_argument(
        "--frame-count",
        type=int,
        default=DEFAULT_FRAME_COUNT,
        help="Number of sampled frames per scale when using depth_random mode.",
    )
    p.add_argument(
        "--frame-depth-start-mm",
        type=float,
        default=DEFAULT_FRAME_DEPTH_START_MM,
        help="Shallow-depth anchor (mm) for random depth sampling per episode.",
    )
    p.add_argument(
        "--frame-depth-end-mm",
        type=float,
        default=DEFAULT_FRAME_DEPTH_END_MM,
        help="Deep-depth anchor (mm) for random depth sampling per episode (typically 2.2).",
    )
    p.add_argument(
        "--frame-fractions",
        type=float,
        nargs="+",
        default=DEFAULT_FRAME_FRACTIONS,
        help="Sampled deformation fractions in (0,1], mapped to trajectory indices (fraction mode only).",
    )
    p.add_argument(
        "--frame-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit deformation frame indices (fraction mode only). If set, overrides --frame-fractions.",
    )
    p.add_argument(
        "--deduplicate-frame-indices",
        dest="deduplicate_frame_indices",
        action="store_true",
        default=True,
        help="Deduplicate repeated sampled frame indices after fraction->index mapping (default: enabled).",
    )
    p.add_argument(
        "--no-deduplicate-frame-indices",
        dest="deduplicate_frame_indices",
        action="store_false",
        help="Keep repeated sampled frame indices.",
    )

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
    p.add_argument(
        "--force-black-side-border-px",
        type=int,
        default=2,
        help="Force left/right border columns to black after render to remove white matte seams.",
    )

    p.add_argument("--max-physics-workers", type=int, default=7)
    p.add_argument("--max-meshing-workers", type=int, default=3)
    p.add_argument("--max-render-workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    p.add_argument("--physics-timeout-sec", type=int, default=240)
    p.add_argument("--meshing-timeout-sec", type=int, default=240)
    p.add_argument("--render-timeout-sec", type=int, default=300)
    p.add_argument("--render-samples", type=int, default=32)

    p.add_argument("--python-cmd", type=str, default=sys.executable)
    p.add_argument("--blender-cmd", type=str, default="blender")

    p.add_argument("--clean-output", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted generation by reusing existing per-scale intermediates and finished frame renders.",
    )
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
    if args.max_meshing_workers <= 0:
        raise ValueError("--max-meshing-workers must be > 0")
    if args.max_render_workers <= 0:
        raise ValueError("--max-render-workers must be > 0")
    if args.depth_min <= 0 or args.depth_max <= 0 or args.depth_min > args.depth_max:
        raise ValueError("Invalid depth range")
    if args.position_clean_ratio < 0 or args.position_near_boundary_ratio < 0 or args.position_partial_crop_ratio < 0:
        raise ValueError("Position sample ratios must be >= 0")
    if (
        args.position_clean_ratio
        + args.position_near_boundary_ratio
        + args.position_partial_crop_ratio
    ) <= 0:
        raise ValueError("At least one position sample ratio must be > 0")
    if args.near_boundary_max_margin_mm < 0:
        raise ValueError("--near-boundary-max-margin-mm must be >= 0")
    if args.partial_crop_min_overhang_mm < 0:
        raise ValueError("--partial-crop-min-overhang-mm must be >= 0")
    if args.partial_crop_max_overhang_mm < args.partial_crop_min_overhang_mm:
        raise ValueError("--partial-crop-max-overhang-mm must be >= --partial-crop-min-overhang-mm")
    if args.reference_scale_mm <= 0:
        raise ValueError("--reference-scale-mm must be > 0")
    if args.distance_safety <= 0:
        raise ValueError("--distance-safety must be > 0")
    if args.fov_deg <= 0 or args.fov_deg >= 179.0:
        raise ValueError("--fov-deg must be in (0, 179)")
    if args.uv_inset_ratio < 0.0 or args.uv_inset_ratio >= 0.49:
        raise ValueError("--uv-inset-ratio must be in [0.0, 0.49)")
    if args.force_black_side_border_px < 0:
        raise ValueError("--force-black-side-border-px must be >= 0")
    if args.meshing_timeout_sec <= 0:
        raise ValueError("--meshing-timeout-sec must be > 0")
    if args.render_samples <= 0:
        raise ValueError("--render-samples must be > 0")
    if args.resume and args.clean_output:
        raise ValueError("--resume and --clean-output cannot be used together")
    if args.frame_sampling_mode == "depth_random":
        if args.frame_count <= 0:
            raise ValueError("--frame-count must be > 0 in depth_random mode")
        if args.frame_depth_start_mm <= 0:
            raise ValueError("--frame-depth-start-mm must be > 0")
        if args.frame_depth_end_mm <= args.frame_depth_start_mm:
            raise ValueError("--frame-depth-end-mm must be > --frame-depth-start-mm")
        if args.depth_max < args.frame_depth_end_mm:
            raise ValueError(
                "--depth-max must be >= --frame-depth-end-mm in depth_random mode "
                f"(got depth-max={args.depth_max}, frame-depth-end-mm={args.frame_depth_end_mm})"
            )
    else:
        if args.frame_indices is not None and len(args.frame_indices) == 0:
            raise ValueError("--frame-indices cannot be empty")
        if args.frame_indices is None:
            if args.frame_fractions is None or len(args.frame_fractions) == 0:
                raise ValueError("--frame-fractions cannot be empty when --frame-indices is not set")
            for f in args.frame_fractions:
                if not (0.0 < float(f) <= 1.0):
                    raise ValueError(f"--frame-fractions values must be in (0, 1], got {f}")
    patch_h, patch_w = parse_patch_grid(args.patch_grid)

    if args.camera_distance_m is not None:
        if args.camera_distance_m <= 0:
            raise ValueError("--camera-distance-m must be > 0")
        fixed_camera_distance_m = float(args.camera_distance_m)
    else:
        ref_scale_m = float(args.reference_scale_mm) / 1000.0
        ref_fit_scale_m = compute_full_sensor_fit_scale_m(
            ref_scale_m,
            DEFAULT_IMAGE_RES[0],
            DEFAULT_IMAGE_RES[1],
        )
        ref_fov_rad = math.radians(float(args.fov_deg))
        fixed_camera_distance_m = (ref_fit_scale_m / 2.0) / math.tan(ref_fov_rad / 2.0)
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
    expected_marker_files = sorted({f"marker_{p.stem}.jpg" for p in textures})
    contact_reference_scale_mm = float(max(args.scales_mm))
    contact_constraint_scale_mm = float(min(args.scales_mm))

    with open(parameters_path, "r", encoding="utf-8") as f:
        parameters_cfg = yaml.safe_load(f)
    indenter_pose = (
        parameters_cfg.get("indenter", {}).get("pose", {})
        if isinstance(parameters_cfg, dict)
        else {}
    )
    indenter_rotation = rotation_matrix_from_pose(indenter_pose, degrees=False)
    indenter_position_sampling_specs: Dict[str, Dict[str, Any]] = {}
    impossible_indenters: List[str] = []
    degraded_indenters: List[str] = []
    for indenter_name in indenter_names:
        bbox_mm = compute_indenter_contact_bbox_mm(indenter_dir / f"{indenter_name}.npy", indenter_rotation)
        inside_bounds_min_scale = compute_margin_constrained_normalized_sampling_bounds(
            bbox_mm=bbox_mm,
            scale_mm=contact_constraint_scale_mm,
            reference_scale_mm=contact_reference_scale_mm,
            requested_x_min_mm=float(args.x_min),
            requested_x_max_mm=float(args.x_max),
            requested_y_min_mm=float(args.y_min),
            requested_y_max_mm=float(args.y_max),
            min_margin_mm=0.0,
        )
        clean_bounds = compute_margin_constrained_normalized_sampling_bounds(
            bbox_mm=bbox_mm,
            scale_mm=contact_constraint_scale_mm,
            reference_scale_mm=contact_reference_scale_mm,
            requested_x_min_mm=float(args.x_min),
            requested_x_max_mm=float(args.x_max),
            requested_y_min_mm=float(args.y_min),
            requested_y_max_mm=float(args.y_max),
            min_margin_mm=float(args.near_boundary_max_margin_mm),
        )
        inside_bounds_reference_scale = compute_margin_constrained_normalized_sampling_bounds(
            bbox_mm=bbox_mm,
            scale_mm=contact_reference_scale_mm,
            reference_scale_mm=contact_reference_scale_mm,
            requested_x_min_mm=float(args.x_min),
            requested_x_max_mm=float(args.x_max),
            requested_y_min_mm=float(args.y_min),
            requested_y_max_mm=float(args.y_max),
            min_margin_mm=0.0,
        )
        indenter_position_sampling_specs[indenter_name] = {
            "bbox_mm": bbox_mm,
            "inside_bounds_min_scale": inside_bounds_min_scale,
            "clean_bounds": clean_bounds,
            "inside_bounds_reference_scale": inside_bounds_reference_scale,
        }
        if (
            inside_bounds_reference_scale["x_min_norm"] > inside_bounds_reference_scale["x_max_norm"]
            or inside_bounds_reference_scale["y_min_norm"] > inside_bounds_reference_scale["y_max_norm"]
        ):
            impossible_indenters.append(indenter_name)
        if (
            inside_bounds_min_scale["x_min_norm"] > inside_bounds_min_scale["x_max_norm"]
            or inside_bounds_min_scale["y_min_norm"] > inside_bounds_min_scale["y_max_norm"]
        ):
            degraded_indenters.append(indenter_name)
    if impossible_indenters:
        problems = []
        for indenter_name in impossible_indenters:
            spec = indenter_position_sampling_specs[indenter_name]
            bbox_mm = spec["bbox_mm"]
            bounds = spec["inside_bounds_reference_scale"]
            problems.append(
                f"{indenter_name}: bbox=({bbox_mm['width_mm']:.3f}mm x {bbox_mm['height_mm']:.3f}mm) "
                f"requested_norm_x=[{bounds['x_min_norm']:.4f},{bounds['x_max_norm']:.4f}] "
                f"requested_norm_y=[{bounds['y_min_norm']:.4f},{bounds['y_max_norm']:.4f}]"
            )
        raise ValueError(
            "Reference-scale contact sampling is impossible for the following indenters:\n"
            + "\n".join(problems)
        )
    if degraded_indenters:
        logging.warning(
            "Smallest-scale fully in-bound region is empty for %d indenters; stratified sampling will fall back "
            "to best-effort categories when needed: %s",
            len(degraded_indenters),
            degraded_indenters,
        )

    if args.clean_output and dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    backup_path = create_backup(parameters_path)
    logging.info("Backup created: %s", backup_path)

    script_tmp_dir = Path(tempfile.mkdtemp(prefix="multiscale_scripts_"))
    work_tmp_root = dataset_root / "_intermediates"
    work_tmp_root.mkdir(parents=True, exist_ok=True)

    open3d_script = script_tmp_dir / "tmp_npz2stl.py"
    blender_script = script_tmp_dir / "tmp_blender_multirender.py"
    open3d_script.write_text(OPEN3D_TEMP_SCRIPT, encoding="utf-8")
    blender_script.write_text(BLENDER_TEMP_SCRIPT, encoding="utf-8")

    rng = random.Random(args.seed)

    episode_states: Dict[int, EpisodeState] = {}
    physics_tasks: List[Dict[str, Any]] = []
    scale_progress: Dict[str, Dict[str, Any]] = {}
    resume_scales_from_metadata = 0
    episode_id = 0
    for indenter in indenter_names:
        requested_sample_types = assign_position_sample_types(
            episodes_per_indenter=int(args.episodes_per_indenter),
            clean_ratio=float(args.position_clean_ratio),
            near_boundary_ratio=float(args.position_near_boundary_ratio),
            partial_crop_ratio=float(args.position_partial_crop_ratio),
            rng=rng,
        )
        sampling_spec = indenter_position_sampling_specs[indenter]
        for requested_sample_type in requested_sample_types:
            sampled_contact = sample_stratified_contact_position(
                rng=rng,
                requested_sample_type=str(requested_sample_type),
                bbox_mm=sampling_spec["bbox_mm"],
                clean_bounds=sampling_spec["clean_bounds"],
                inside_bounds_min_scale=sampling_spec["inside_bounds_min_scale"],
                inside_bounds_reference_scale=sampling_spec["inside_bounds_reference_scale"],
                min_scale_mm=contact_constraint_scale_mm,
                reference_scale_mm=contact_reference_scale_mm,
                near_boundary_max_margin_mm=float(args.near_boundary_max_margin_mm),
                partial_crop_min_overhang_mm=float(args.partial_crop_min_overhang_mm),
                partial_crop_max_overhang_mm=float(args.partial_crop_max_overhang_mm),
            )
            x_norm = float(sampled_contact["x_norm"])
            y_norm = float(sampled_contact["y_norm"])
            x_mm, y_mm = contact_mm_from_normalized(
                x_norm=x_norm,
                y_norm=y_norm,
                scale_mm=contact_reference_scale_mm,
            )
            x_mm = round(x_mm, 4)
            y_mm = round(y_mm, 4)
            episode_frame_depth_targets_mm: List[float] | None = None
            if args.frame_sampling_mode == "depth_random":
                episode_frame_depth_targets_mm = make_stratified_random_frame_depth_targets_mm(
                    frame_count=int(args.frame_count),
                    depth_start_mm=float(args.frame_depth_start_mm),
                    depth_end_mm=float(args.frame_depth_end_mm),
                    rng=rng,
                )
                deep_target = float(episode_frame_depth_targets_mm[-1])
                # Bias final press depth to be close to deep target (around 2.2mm by default).
                depth_low = max(float(args.depth_min), deep_target - 0.2)
                depth_high = max(depth_low, float(args.depth_max))
                depth_mm = round(rng.uniform(depth_low, depth_high), 1)
            else:
                depth_mm = round(rng.uniform(args.depth_min, args.depth_max), 1)

            episode_dir = dataset_root / f"episode_{episode_id:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            episode_states[episode_id] = EpisodeState(
                episode_id=episode_id,
                indenter=indenter,
                x_mm=x_mm,
                y_mm=y_mm,
                x_norm=x_norm,
                y_norm=y_norm,
                position_sample_type_requested=str(sampled_contact["requested_sample_type"]),
                position_sample_type_actual=str(sampled_contact["actual_sample_type"]),
                contact_margin_min_scale_mm=float(sampled_contact["min_scale_margin_mm"]),
                contact_margin_reference_scale_mm=float(sampled_contact["reference_scale_margin_mm"]),
                reference_scale_mm=contact_reference_scale_mm,
                depth_mm=depth_mm,
                episode_dir=episode_dir,
            )

            existing_scales: Dict[str, Any] = {}
            meta_path = episode_dir / "metadata.json"
            if args.resume and meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        existing_meta = json.load(f)
                    maybe_scales = existing_meta.get("scales", {})
                    if isinstance(maybe_scales, dict):
                        existing_scales = maybe_scales
                except Exception as exc:
                    logging.warning("Failed to parse existing metadata for resume (%s): %s", meta_path, exc)

            prepared_scales = 0
            skipped_scales = 0
            for scale_mm in args.scales_mm:
                scale_contact_x_mm, scale_contact_y_mm = contact_mm_from_normalized(
                    x_norm=x_norm,
                    y_norm=y_norm,
                    scale_mm=float(scale_mm),
                )
                scale_contact_x_mm = round(scale_contact_x_mm, 4)
                scale_contact_y_mm = round(scale_contact_y_mm, 4)
                scale_key = f"scale_{int(scale_mm)}mm"
                progress_key = make_scale_progress_key(episode_id, int(scale_mm))
                scale_progress[progress_key] = make_scale_progress_entry(
                    episode_id=episode_id,
                    scale_mm=int(scale_mm),
                    indenter=indenter,
                    x_mm=scale_contact_x_mm,
                    y_mm=scale_contact_y_mm,
                    x_norm=x_norm,
                    y_norm=y_norm,
                    position_sample_type_requested=episode_states[episode_id].position_sample_type_requested,
                    position_sample_type_actual=episode_states[episode_id].position_sample_type_actual,
                    depth_mm=depth_mm,
                    frame_sampling_mode=str(args.frame_sampling_mode),
                    frame_depth_targets_mm=episode_frame_depth_targets_mm,
                )
                if args.resume and scale_key in existing_scales:
                    scale_meta = existing_scales[scale_key]
                    if validate_existing_scale_metadata(
                        episode_dir,
                        scale_key,
                        scale_meta,
                        expected_marker_files=expected_marker_files,
                        expected_frame_sampling_mode=str(args.frame_sampling_mode),
                        expected_frame_depth_targets_mm=episode_frame_depth_targets_mm,
                        expected_contact_x_mm=scale_contact_x_mm,
                        expected_contact_y_mm=scale_contact_y_mm,
                    ):
                        episode_states[episode_id].scales[scale_key] = scale_meta
                        resume_scales_from_metadata += 1
                        skipped_scales += 1
                        frame_count_existing = 0
                        frames_existing = scale_meta.get("frames", {})
                        if isinstance(frames_existing, dict):
                            frame_count_existing = len(frames_existing)
                        scale_progress[progress_key]["restored_from_metadata"] = True
                        scale_progress[progress_key]["physics"] = {
                            "status": "reused_from_metadata",
                            "trajectory_length": scale_meta.get("trajectory_length"),
                            "npz_path": "",
                            "reused": True,
                        }
                        scale_progress[progress_key]["meshing"] = {
                            "status": "reused_from_metadata",
                            "planned_frames": int(frame_count_existing),
                            "completed_frames": int(frame_count_existing),
                            "failed_frames": 0,
                        }
                        scale_progress[progress_key]["render"] = {
                            "status": "reused_from_metadata",
                            "planned_frames": int(frame_count_existing),
                            "completed_frames": int(frame_count_existing),
                            "reused_frames": int(frame_count_existing),
                            "failed_frames": 0,
                        }
                        scale_progress[progress_key]["scale_complete"] = True
                        if not args.keep_intermediates:
                            completed_scale_tmp_dir = (
                                work_tmp_root / f"episode_{episode_id:06d}" / f"scale_{int(scale_mm)}mm"
                            )
                            if completed_scale_tmp_dir.exists():
                                shutil.rmtree(completed_scale_tmp_dir, ignore_errors=True)
                        continue

                physics_tasks.append(
                    {
                        "episode_id": episode_id,
                        "scale_mm": int(scale_mm),
                        "indenter": indenter,
                        "x_mm": scale_contact_x_mm,
                        "y_mm": scale_contact_y_mm,
                        "depth_mm": depth_mm,
                        "repo_root": str(repo_root),
                        "particle": str(args.particle),
                        "python_cmd": str(args.python_cmd),
                        "base_parameters": str(parameters_path),
                        "temp_scale_dir": str(
                            work_tmp_root / f"episode_{episode_id:06d}" / f"scale_{int(scale_mm)}mm"
                        ),
                        "physics_timeout_sec": int(args.physics_timeout_sec),
                        "resume": bool(args.resume),
                        "frame_sampling_mode": str(args.frame_sampling_mode),
                        "frame_fractions": (
                            None
                            if (args.frame_sampling_mode != "fraction" or args.frame_indices is not None)
                            else [float(v) for v in args.frame_fractions]
                        ),
                        "frame_indices": (
                            None
                            if (args.frame_sampling_mode != "fraction" or args.frame_indices is None)
                            else [int(v) for v in args.frame_indices]
                        ),
                        "deduplicate_frame_indices": bool(args.deduplicate_frame_indices),
                        "frame_depth_targets_mm": (
                            [float(v) for v in episode_frame_depth_targets_mm]
                            if episode_frame_depth_targets_mm is not None
                            else None
                        ),
                    }
                )
                prepared_scales += 1

            logging.info(
                "Prepared episode_%06d | indenter=%s | sample=%s->%s | margin_min_scale=%.4fmm | "
                "contact_ref=(x=%.4f, y=%.4f @ %.1fmm) | contact_norm=(x=%.4f, y=%.4f) | queued_scales=%d skipped_scales=%d%s",
                episode_id,
                indenter,
                episode_states[episode_id].position_sample_type_requested,
                episode_states[episode_id].position_sample_type_actual,
                episode_states[episode_id].contact_margin_min_scale_mm,
                x_mm,
                y_mm,
                contact_reference_scale_mm,
                x_norm,
                y_norm,
                prepared_scales,
                skipped_scales,
                (
                    f" | depth_targets_mm={episode_frame_depth_targets_mm}"
                    if episode_frame_depth_targets_mm is not None
                    else ""
                ),
            )
            episode_id += 1

    total_episodes = len(episode_states)
    total_scale_tasks = len(physics_tasks)
    total_scale_slots = int(total_episodes * len(args.scales_mm))

    logging.info("=" * 72)
    logging.info("Dataset root: %s", dataset_root)
    logging.info("Indenters: %d", len(indenter_names))
    logging.info("Episodes: %d", total_episodes)
    logging.info("Scales: %s", args.scales_mm)
    logging.info("Patch grid: %dx%d", patch_h, patch_w)
    logging.info(
        "Position sampling: clean=%.2f near_boundary=%.2f partial_crop=%.2f | near_margin<=%.3fmm | "
        "partial_overhang=[%.3f, %.3f]mm",
        float(args.position_clean_ratio),
        float(args.position_near_boundary_ratio),
        float(args.position_partial_crop_ratio),
        float(args.near_boundary_max_margin_mm),
        float(args.partial_crop_min_overhang_mm),
        float(args.partial_crop_max_overhang_mm),
    )
    logging.info(
        "Indenter bbox: mode=bottom_slice bottom_height=%.3fmm",
        float(DEFAULT_INDENTER_BBOX_BOTTOM_HEIGHT_MM),
    )
    logging.info(
        "Contact constraint: layered_sampling=True min_scale=%.1fmm reference_scale=%.1fmm requested_ref_bounds=x[%.3f,%.3f] y[%.3f,%.3f]",
        contact_constraint_scale_mm,
        contact_reference_scale_mm,
        float(args.x_min),
        float(args.x_max),
        float(args.y_min),
        float(args.y_max),
    )
    logging.info("Resume mode: %s", args.resume)
    logging.info("Intermediates root: %s", work_tmp_root)
    if args.frame_sampling_mode == "depth_random":
        logging.info(
            "Frame sampling: mode=depth_random strategy=stratified_random frame_count=%d start=%.3fmm end=%.3fmm deduplicate=%s",
            int(args.frame_count),
            float(args.frame_depth_start_mm),
            float(args.frame_depth_end_mm),
            args.deduplicate_frame_indices,
        )
    elif args.frame_indices is not None:
        logging.info("Frame sampling: mode=fraction explicit indices=%s deduplicate=%s", args.frame_indices, args.deduplicate_frame_indices)
    else:
        logging.info("Frame sampling: mode=fraction fractions=%s deduplicate=%s", args.frame_fractions, args.deduplicate_frame_indices)
    logging.info("Total (episode,scale) physics tasks: %d", total_scale_tasks)
    if args.resume:
        logging.info("Scales restored directly from existing metadata: %d", resume_scales_from_metadata)
    logging.info("Marker textures: %d (%s)", len(textures), texture_dir)
    logging.info(
        "Workers: physics=%d (GPU), meshing=%d (CPU), render=%d (CPU)",
        args.max_physics_workers,
        args.max_meshing_workers,
        args.max_render_workers,
    )
    logging.info("Render samples (Cycles CPU): %d", args.render_samples)
    logging.info(
        "Camera: mode=%s fixed_distance=%.6fm base_fov=%.3fdeg ref_scale=%.2fmm uv_mode=%s uv_inset=%.3f border_black_px=%d",
        args.camera_mode,
        fixed_camera_distance_m,
        float(args.fov_deg),
        float(args.reference_scale_mm),
        str(args.uv_mode),
        float(args.uv_inset_ratio),
        int(args.force_black_side_border_px),
    )
    logging.info("=" * 72)

    physics_ok = 0
    meshing_ok = 0
    render_ok = 0
    physics_done = 0
    meshing_done = 0
    render_done = 0
    physics_failed = 0
    meshing_failed = 0
    render_failed = 0
    physics_reused = 0
    total_meshing_tasks_planned = 0
    total_render_tasks_planned = 0
    total_frames_rendered = 0
    total_frames_reused = 0
    allow_intermediate_cleanup = False
    run_start_ts = time.time()
    run_start_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    physics_stage_start_ts = run_start_ts
    meshing_stage_start_ts: float | None = None
    render_stage_start_ts: float | None = None
    run_status_path = dataset_root / "run_status.json"
    last_run_status_write_ts = 0.0
    last_event: Dict[str, Any] = {}
    stage_last_event: Dict[str, Dict[str, Any]] = {}

    scale_states: Dict[Tuple[int, int], ScaleState] = {}

    def update_last_event(stage: str, status: str, **payload: Any) -> None:
        nonlocal last_event
        event = {
            "time": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stage": str(stage),
            "status": str(status),
            **_json_safe(payload),
        }
        last_event = event
        stage_last_event[str(stage)] = event

    def persist_run_status(run_status: str, force: bool = False, error: str | None = None) -> None:
        nonlocal last_run_status_write_ts
        now_ts = time.time()
        if (not force) and (now_ts - last_run_status_write_ts < 2.0):
            return

        completed_scales = sum(1 for item in scale_progress.values() if item.get("scale_complete"))
        failed_scales = sum(1 for item in scale_progress.values() if item.get("last_error"))
        payload = {
            "status": str(run_status),
            "started_at": run_start_iso,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "completed_at": (
                dt.datetime.now(dt.timezone.utc).isoformat()
                if run_status in {"completed", "failed"}
                else None
            ),
            "dataset_root": str(dataset_root),
            "run_status_json": str(run_status_path),
            "command": shlex.join([str(sys.executable), *sys.argv]),
            "argv": list(sys.argv),
            "args": _json_safe(vars(args)),
            "paths": {
                "manifest_json": str(dataset_root / "manifest.json"),
                "image_index_csv": str(dataset_root / "image_index.csv"),
                "run_status_json": str(run_status_path),
                "intermediates_root": str(work_tmp_root),
            },
            "totals": {
                "episodes": int(total_episodes),
                "scale_slots": int(total_scale_slots),
                "physics_tasks_planned": int(total_scale_tasks),
                "meshing_tasks_planned": int(total_meshing_tasks_planned),
                "render_tasks_planned": int(total_render_tasks_planned),
                "textures": int(len(textures)),
                "scales_per_episode": int(len(args.scales_mm)),
            },
            "progress": {
                "resume_mode": bool(args.resume),
                "scales_restored_from_metadata": int(resume_scales_from_metadata),
                "completed_scales": int(completed_scales),
                "failed_scales": int(failed_scales),
                "physics": {
                    "completed": int(physics_done),
                    "succeeded": int(physics_ok),
                    "failed": int(physics_failed),
                    "reused": int(physics_reused),
                },
                "meshing": {
                    "completed": int(meshing_done),
                    "succeeded": int(meshing_ok),
                    "failed": int(meshing_failed),
                },
                "render": {
                    "completed": int(render_done),
                    "succeeded": int(render_ok),
                    "failed": int(render_failed),
                    "frames_rendered": int(total_frames_rendered),
                    "frames_reused": int(total_frames_reused),
                },
            },
            "last_event": last_event,
            "last_stage_event": stage_last_event,
            "error": str(error) if error else "",
            "scales": _json_safe(scale_progress),
        }
        write_json_atomic(run_status_path, payload)
        last_run_status_write_ts = now_ts

    def persist_episode_metadata(state: EpisodeState) -> None:
        metadata = {
            "episode_id": state.episode_id,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "indenter": state.indenter,
            "particle": str(args.particle),
            "contact": {
                "x_mm": state.x_mm,
                "y_mm": state.y_mm,
                "depth_mm": state.depth_mm,
                "reference_scale_mm": state.reference_scale_mm,
                "x_norm": state.x_norm,
                "y_norm": state.y_norm,
                "position_sample_type_requested": state.position_sample_type_requested,
                "position_sample_type_actual": state.position_sample_type_actual,
                "min_scale_margin_mm": state.contact_margin_min_scale_mm,
                "reference_scale_margin_mm": state.contact_margin_reference_scale_mm,
                "constraint_scale_mm": float(contact_constraint_scale_mm),
            },
            "image_resolution": {"width": DEFAULT_IMAGE_RES[0], "height": DEFAULT_IMAGE_RES[1]},
            "scales": state.scales,
        }
        with open(state.episode_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def maybe_finalize_scale(scale_state: ScaleState) -> None:
        if scale_state.sealed or scale_state.failed:
            return
        if scale_state.pending_mesh_count != 0 or scale_state.pending_render_count != 0:
            return
        if len(scale_state.frames) != len(scale_state.sampled_frames):
            return

        ep_state = episode_states[scale_state.episode_id]
        progress_key = make_scale_progress_key(scale_state.episode_id, scale_state.scale_mm)
        ordered_frames: Dict[str, Dict[str, Any]] = {}
        for sample in scale_state.sampled_frames:
            frame_name = str(sample["frame_name"])
            if frame_name not in scale_state.frames:
                return
            ordered_frames[frame_name] = scale_state.frames[frame_name]

        ep_state.scales[scale_state.scale_key] = {
            "scale_mm": int(scale_state.scale_mm),
            "contact_x_mm": float(scale_state.contact_x_mm),
            "contact_y_mm": float(scale_state.contact_y_mm),
            "contact_x_norm": float(ep_state.x_norm),
            "contact_y_norm": float(ep_state.y_norm),
            "position_sample_type_requested": str(ep_state.position_sample_type_requested),
            "position_sample_type_actual": str(ep_state.position_sample_type_actual),
            "contact_depth_mm": float(scale_state.contact_depth_mm),
            "deformation_stats_final": scale_state.deformation_stats_final,
            "trajectory_length": int(scale_state.trajectory_length),
            "frame_sampling_mode": str(scale_state.frame_sampling_mode),
            "frame_depth_targets_mm": scale_state.frame_depth_targets_mm,
            "camera_mode": str(scale_state.camera_mode),
            "camera_distance_m": float(scale_state.camera_distance_m),
            "camera_fov_deg": float(scale_state.camera_fov_deg),
            "adapter_coord_map": str(scale_state.adapter_coord_map_path.relative_to(ep_state.episode_dir)),
            "adapter_coord_map_shape": [int(v) for v in scale_state.adapter_coord_map_shape],
            "frames": ordered_frames,
        }
        try:
            # Persist per-scale progress immediately so resume can skip physics next time.
            persist_episode_metadata(ep_state)
        except Exception as exc:
            msg = (
                f"failed to persist metadata after scale complete: "
                f"ep={scale_state.episode_id} scale={scale_state.scale_mm} err={exc}"
            )
            ep_state.errors.append(msg)
            scale_state.failed = True
            logging.error(msg)
            scale_progress[progress_key]["last_error"] = msg
            update_last_event(
                "metadata",
                "error",
                episode_id=scale_state.episode_id,
                scale_mm=scale_state.scale_mm,
                message=msg,
            )
            persist_run_status("running", force=True)
            return
        scale_state.sealed = True
        scale_progress[progress_key]["scale_complete"] = True
        scale_progress[progress_key]["meshing"]["status"] = "complete"
        scale_progress[progress_key]["render"]["status"] = "complete"
        update_last_event(
            "scale",
            "complete",
            episode_id=scale_state.episode_id,
            scale_mm=scale_state.scale_mm,
            frames=len(ordered_frames),
            rendered=scale_state.frames_rendered,
            reused=scale_state.frames_reused,
        )

        logging.info(
            "Scale COMPLETE | ep=%06d scale=%d | frames=%d (rendered=%d reused=%d)",
            scale_state.episode_id,
            scale_state.scale_mm,
            len(ordered_frames),
            scale_state.frames_rendered,
            scale_state.frames_reused,
        )
        if (not args.keep_intermediates) and scale_state.temp_scale_dir.exists():
            shutil.rmtree(scale_state.temp_scale_dir, ignore_errors=True)
            logging.info(
                "Intermediates cleaned | ep=%06d scale=%d | dir=%s",
                scale_state.episode_id,
                scale_state.scale_mm,
                scale_state.temp_scale_dir,
            )
        persist_run_status("running", force=True)

    persist_run_status("running", force=True)

    try:
        with (
            cf.ProcessPoolExecutor(max_workers=args.max_physics_workers) as physics_pool,
            cf.ProcessPoolExecutor(max_workers=args.max_meshing_workers) as meshing_pool,
            cf.ProcessPoolExecutor(max_workers=args.max_render_workers) as render_pool,
        ):
            physics_futures: Dict[cf.Future, Dict[str, Any]] = {}
            meshing_futures: Dict[cf.Future, Dict[str, Any]] = {}
            render_futures: Dict[cf.Future, Dict[str, Any]] = {}

            for task in physics_tasks:
                f = physics_pool.submit(run_physics_task, task)
                physics_futures[f] = task

            while physics_futures or meshing_futures or render_futures:
                pending: List[cf.Future] = []
                pending.extend(list(physics_futures.keys()))
                pending.extend(list(meshing_futures.keys()))
                pending.extend(list(render_futures.keys()))
                done_set, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)

                for done_future in done_set:
                    if done_future in physics_futures:
                        task = physics_futures.pop(done_future)
                        physics_done += 1
                        episode_id = int(task["episode_id"])
                        scale_mm = int(task["scale_mm"])
                        progress_key = make_scale_progress_key(episode_id, scale_mm)
                        ep_state = episode_states[episode_id]
                        physics_eta = _progress_eta_text(physics_stage_start_ts, physics_done, max(total_scale_tasks, 1))
                        physics_thr = _throughput_text(physics_stage_start_ts, physics_done)

                        try:
                            result = done_future.result()
                        except Exception as exc:
                            msg = f"physics future crashed: ep={episode_id} scale={scale_mm} err={exc}"
                            ep_state.errors.append(msg)
                            physics_failed += 1
                            scale_progress[progress_key]["physics"]["status"] = "error"
                            scale_progress[progress_key]["last_error"] = msg
                            update_last_event("physics", "error", episode_id=episode_id, scale_mm=scale_mm, message=msg)
                            logging.error("%s | %s | %s", msg, physics_eta, physics_thr)
                            persist_run_status("running", force=True)
                            continue

                        if result.get("status") != "ok":
                            msg = (
                                f"physics failed: ep={episode_id} scale={scale_mm} err={result.get('error')}\n"
                                f"{result.get('traceback', '')}"
                            )
                            ep_state.errors.append(msg)
                            err_short = (
                                str(result.get("error", "")).strip().splitlines()[0]
                                if result.get("error")
                                else "(no error text)"
                            )
                            physics_failed += 1
                            scale_progress[progress_key]["physics"]["status"] = "error"
                            scale_progress[progress_key]["last_error"] = err_short
                            update_last_event(
                                "physics",
                                "error",
                                episode_id=episode_id,
                                scale_mm=scale_mm,
                                message=err_short,
                            )
                            logging.error(
                                "Physics failed | ep=%06d scale=%d | %s | %s | %s",
                                episode_id,
                                scale_mm,
                                err_short,
                                physics_eta,
                                physics_thr,
                            )
                            persist_run_status("running", force=True)
                            continue

                        physics_ok += 1
                        scale_progress[progress_key]["physics"]["status"] = "ok"
                        scale_progress[progress_key]["physics"]["trajectory_length"] = int(result["trajectory_length"])
                        scale_progress[progress_key]["physics"]["npz_path"] = str(result["npz_path"])
                        scale_progress[progress_key]["physics"]["reused"] = bool(result.get("physics_reused"))
                        frame_count = len(result["sampled_frames"])
                        scale_progress[progress_key]["meshing"]["planned_frames"] = int(frame_count)
                        scale_progress[progress_key]["render"]["planned_frames"] = int(frame_count)
                        if result.get("physics_reused"):
                            physics_reused += 1
                        update_last_event(
                            "physics",
                            "ok",
                            episode_id=episode_id,
                            scale_mm=scale_mm,
                            reused=bool(result.get("physics_reused")),
                            trajectory_length=int(result["trajectory_length"]),
                            sampled_frames=int(frame_count),
                        )
                        if result.get("physics_reused"):
                            logging.info(
                                "Physics REUSE | ep=%06d scale=%d | max_down=%.4fmm mean_down=%.4fmm | %s | %s",
                                episode_id,
                                scale_mm,
                                result["deformation_stats_final"]["surface_max_down_mm"],
                                result["deformation_stats_final"]["surface_mean_down_mm"],
                                physics_eta,
                                physics_thr,
                            )
                        else:
                            logging.info(
                                "Physics OK | ep=%06d scale=%d | max_down=%.4fmm mean_down=%.4fmm | %s | %s",
                                episode_id,
                                scale_mm,
                                result["deformation_stats_final"]["surface_max_down_mm"],
                                result["deformation_stats_final"]["surface_mean_down_mm"],
                                physics_eta,
                                physics_thr,
                            )

                        scale_dir = ep_state.episode_dir / f"scale_{scale_mm}mm"
                        scale_dir.mkdir(parents=True, exist_ok=True)

                        coord_map = make_adapter_coord_map(float(scale_mm), int(patch_h), int(patch_w))
                        adapter_coord_map_path = scale_dir / "adapter_coord_map.npy"
                        np.save(adapter_coord_map_path, coord_map)
                        adapter_coord_map_shape = [int(v) for v in coord_map.shape]
                        fit_scale_m = compute_full_sensor_fit_scale_m(
                            float(scale_mm) / 1000.0,
                            DEFAULT_IMAGE_RES[0],
                            DEFAULT_IMAGE_RES[1],
                        )

                        if args.camera_mode == "fixed_distance_variable_fov":
                            camera_fov_deg = math.degrees(
                                2.0 * math.atan((fit_scale_m / 2.0) / float(fixed_camera_distance_m))
                            )
                            camera_distance_m = float(fixed_camera_distance_m)
                        else:
                            camera_fov_deg = float(args.fov_deg)
                            camera_distance_m = (
                                (fit_scale_m / 2.0) / math.tan(math.radians(float(args.fov_deg)) / 2.0)
                            )
                            camera_distance_m *= float(args.distance_safety)

                        scale_id = (episode_id, scale_mm)
                        scale_state = ScaleState(
                            episode_id=episode_id,
                            scale_mm=scale_mm,
                            scale_key=f"scale_{scale_mm}mm",
                            scale_dir=scale_dir,
                            temp_scale_dir=Path(result["temp_scale_dir"]),
                            contact_x_mm=float(result["x_mm"]),
                            contact_y_mm=float(result["y_mm"]),
                            contact_depth_mm=float(result["depth_mm"]),
                            trajectory_length=int(result["trajectory_length"]),
                            deformation_stats_final=result["deformation_stats_final"],
                            frame_sampling_mode=str(result.get("frame_sampling_mode", args.frame_sampling_mode)),
                            frame_depth_targets_mm=result.get("frame_depth_targets_mm"),
                            sampled_frames=[dict(v) for v in result["sampled_frames"]],
                            camera_mode=str(args.camera_mode),
                            camera_distance_m=float(camera_distance_m),
                            camera_fov_deg=float(camera_fov_deg),
                            adapter_coord_map_path=adapter_coord_map_path,
                            adapter_coord_map_shape=adapter_coord_map_shape,
                        )
                        scale_states[scale_id] = scale_state

                        for sample in scale_state.sampled_frames:
                            frame_name = str(sample["frame_name"])
                            frame_dir = scale_dir / frame_name

                            if args.resume and frame_outputs_complete(frame_dir, expected_marker_files):
                                scale_state.frames[frame_name] = make_frame_metadata(sample, expected_marker_files)
                                scale_state.frames_reused += 1
                                total_frames_reused += 1
                                scale_progress[progress_key]["render"]["completed_frames"] += 1
                                scale_progress[progress_key]["render"]["reused_frames"] += 1
                                scale_progress[progress_key]["render"]["status"] = "reused"
                                update_last_event(
                                    "render",
                                    "reused",
                                    episode_id=episode_id,
                                    scale_mm=scale_mm,
                                    frame_name=frame_name,
                                    markers=len(expected_marker_files),
                                )
                                logging.info(
                                    "Render REUSE | ep=%06d scale=%d frame=%s | markers=%d",
                                    episode_id,
                                    scale_mm,
                                    frame_name,
                                    len(expected_marker_files),
                                )
                                continue

                            stl_path = scale_state.temp_scale_dir / "stl_frames" / f"{frame_name}.stl"
                            mesh_task = {
                                "episode_id": episode_id,
                                "scale_mm": scale_mm,
                                "frame_name": frame_name,
                                "frame_index": int(sample["frame_index"]),
                                "sample": sample,
                                "repo_root": str(repo_root),
                                "python_cmd": str(args.python_cmd),
                                "open3d_script": str(open3d_script),
                                "npz_path": str(result["npz_path"]),
                                "stl_path": str(stl_path),
                                "meshing_timeout_sec": int(args.meshing_timeout_sec),
                            }
                            mf = meshing_pool.submit(run_meshing_task, mesh_task)
                            meshing_futures[mf] = {
                                "scale_id": scale_id,
                            }
                            scale_state.pending_mesh_count += 1
                            total_meshing_tasks_planned += 1
                            scale_progress[progress_key]["meshing"]["status"] = "in_progress"
                            scale_progress[progress_key]["render"]["status"] = "pending"
                            if meshing_stage_start_ts is None:
                                meshing_stage_start_ts = time.time()

                        maybe_finalize_scale(scale_state)
                        persist_run_status("running")

                    elif done_future in meshing_futures:
                        payload = meshing_futures.pop(done_future)
                        meshing_done += 1
                        meshing_stage_start = meshing_stage_start_ts if meshing_stage_start_ts is not None else run_start_ts
                        meshing_eta = _progress_eta_text(
                            meshing_stage_start,
                            meshing_done,
                            max(total_meshing_tasks_planned, meshing_done, 1),
                        )
                        meshing_thr = _throughput_text(meshing_stage_start, meshing_done)

                        scale_id = payload["scale_id"]
                        scale_state = scale_states.get(scale_id)
                        if scale_state is None:
                            continue
                        progress_key = make_scale_progress_key(scale_state.episode_id, scale_state.scale_mm)
                        ep_state = episode_states[scale_state.episode_id]
                        scale_state.pending_mesh_count = max(0, scale_state.pending_mesh_count - 1)

                        try:
                            rr = done_future.result()
                        except Exception as exc:
                            msg = (
                                f"meshing future crashed: ep={scale_state.episode_id} "
                                f"scale={scale_state.scale_mm} err={exc}"
                            )
                            ep_state.errors.append(msg)
                            scale_state.failed = True
                            meshing_failed += 1
                            scale_progress[progress_key]["meshing"]["failed_frames"] += 1
                            scale_progress[progress_key]["meshing"]["status"] = "error"
                            scale_progress[progress_key]["last_error"] = msg
                            update_last_event(
                                "meshing",
                                "error",
                                episode_id=scale_state.episode_id,
                                scale_mm=scale_state.scale_mm,
                                message=msg,
                            )
                            logging.error("%s | %s | %s", msg, meshing_eta, meshing_thr)
                            maybe_finalize_scale(scale_state)
                            persist_run_status("running", force=True)
                            continue

                        if rr.get("status") != "ok":
                            msg = (
                                f"meshing failed: ep={scale_state.episode_id} scale={scale_state.scale_mm} "
                                f"frame={rr.get('frame_name')} err={rr.get('error')}\n{rr.get('traceback', '')}"
                            )
                            ep_state.errors.append(msg)
                            scale_state.failed = True
                            meshing_failed += 1
                            scale_progress[progress_key]["meshing"]["failed_frames"] += 1
                            scale_progress[progress_key]["meshing"]["status"] = "error"
                            scale_progress[progress_key]["last_error"] = str(rr.get("error", ""))
                            update_last_event(
                                "meshing",
                                "error",
                                episode_id=scale_state.episode_id,
                                scale_mm=scale_state.scale_mm,
                                frame_name=rr.get("frame_name"),
                                message=rr.get("error", ""),
                            )
                            logging.error(
                                "Meshing failed | ep=%06d scale=%d frame=%s | %s | %s",
                                scale_state.episode_id,
                                scale_state.scale_mm,
                                rr.get("frame_name"),
                                meshing_eta,
                                meshing_thr,
                            )
                            maybe_finalize_scale(scale_state)
                            persist_run_status("running", force=True)
                            continue

                        meshing_ok += 1
                        frame_name = str(rr["frame_name"])
                        frame_dir = scale_state.scale_dir / frame_name
                        scale_progress[progress_key]["meshing"]["completed_frames"] += 1
                        scale_progress[progress_key]["meshing"]["status"] = "in_progress"
                        update_last_event(
                            "meshing",
                            "ok",
                            episode_id=scale_state.episode_id,
                            scale_mm=scale_state.scale_mm,
                            frame_name=frame_name,
                            frame_index=int(rr["frame_index"]),
                        )

                        render_task = {
                            "episode_id": int(scale_state.episode_id),
                            "scale_mm": int(scale_state.scale_mm),
                            "frame_name": frame_name,
                            "frame_dir": str(frame_dir),
                            "repo_root": str(repo_root),
                            "stl_path": str(rr["stl_path"]),
                            "keep_intermediates": bool(args.keep_intermediates),
                            "resume": bool(args.resume),
                            "expected_marker_files": list(expected_marker_files),
                            "blender_cmd": str(args.blender_cmd),
                            "blender_script": str(blender_script),
                            "textures_dir": str(texture_dir),
                            "camera_mode": str(args.camera_mode),
                            "base_fov_deg": float(args.fov_deg),
                            "fixed_distance_m": float(fixed_camera_distance_m),
                            "distance_safety": float(args.distance_safety),
                            "uv_mode": str(args.uv_mode),
                            "uv_inset_ratio": float(args.uv_inset_ratio),
                            "force_black_side_border_px": int(args.force_black_side_border_px),
                            "render_samples": int(args.render_samples),
                            "render_timeout_sec": int(args.render_timeout_sec),
                        }

                        rf = render_pool.submit(run_render_task, render_task)
                        render_futures[rf] = {
                            "scale_id": scale_id,
                            "sample": rr["sample"],
                        }
                        total_render_tasks_planned += 1
                        scale_state.pending_render_count += 1
                        scale_progress[progress_key]["render"]["status"] = "in_progress"
                        if render_stage_start_ts is None:
                            render_stage_start_ts = time.time()

                        logging.info(
                            "Meshing OK | ep=%06d scale=%d frame=%s idx=%d | %s | %s",
                            scale_state.episode_id,
                            scale_state.scale_mm,
                            frame_name,
                            int(rr["frame_index"]),
                            meshing_eta,
                            meshing_thr,
                        )

                        maybe_finalize_scale(scale_state)
                        persist_run_status("running")

                    elif done_future in render_futures:
                        payload = render_futures.pop(done_future)
                        render_done += 1
                        render_stage_start = render_stage_start_ts if render_stage_start_ts is not None else run_start_ts
                        render_eta = _progress_eta_text(
                            render_stage_start,
                            render_done,
                            max(total_render_tasks_planned, render_done, 1),
                        )
                        render_thr = _throughput_text(render_stage_start, render_done)

                        scale_id = payload["scale_id"]
                        scale_state = scale_states.get(scale_id)
                        if scale_state is None:
                            continue
                        progress_key = make_scale_progress_key(scale_state.episode_id, scale_state.scale_mm)
                        ep_state = episode_states[scale_state.episode_id]
                        scale_state.pending_render_count = max(0, scale_state.pending_render_count - 1)

                        try:
                            rr = done_future.result()
                        except Exception as exc:
                            msg = (
                                f"render future crashed: ep={scale_state.episode_id} "
                                f"scale={scale_state.scale_mm} err={exc}"
                            )
                            ep_state.errors.append(msg)
                            scale_state.failed = True
                            render_failed += 1
                            scale_progress[progress_key]["render"]["failed_frames"] += 1
                            scale_progress[progress_key]["render"]["status"] = "error"
                            scale_progress[progress_key]["last_error"] = msg
                            update_last_event(
                                "render",
                                "error",
                                episode_id=scale_state.episode_id,
                                scale_mm=scale_state.scale_mm,
                                message=msg,
                            )
                            logging.error("%s | %s | %s", msg, render_eta, render_thr)
                            maybe_finalize_scale(scale_state)
                            persist_run_status("running", force=True)
                            continue

                        if rr.get("status") != "ok":
                            msg = (
                                f"render failed: ep={scale_state.episode_id} scale={scale_state.scale_mm} "
                                f"frame={rr.get('frame_name')} err={rr.get('error')}\n{rr.get('traceback', '')}"
                            )
                            ep_state.errors.append(msg)
                            scale_state.failed = True
                            render_failed += 1
                            scale_progress[progress_key]["render"]["failed_frames"] += 1
                            scale_progress[progress_key]["render"]["status"] = "error"
                            scale_progress[progress_key]["last_error"] = str(rr.get("error", ""))
                            update_last_event(
                                "render",
                                "error",
                                episode_id=scale_state.episode_id,
                                scale_mm=scale_state.scale_mm,
                                frame_name=rr.get("frame_name"),
                                message=rr.get("error", ""),
                            )
                            logging.error(
                                "Render failed | ep=%06d scale=%d frame=%s | %s | %s",
                                scale_state.episode_id,
                                scale_state.scale_mm,
                                rr.get("frame_name"),
                                render_eta,
                                render_thr,
                            )
                            maybe_finalize_scale(scale_state)
                            persist_run_status("running", force=True)
                            continue

                        render_ok += 1
                        frame_name = str(rr["frame_name"])
                        sample = payload["sample"]
                        rendered_markers = [str(v) for v in rr.get("rendered_markers", [])]
                        scale_state.frames[frame_name] = make_frame_metadata(sample, rendered_markers)
                        if rr.get("reused"):
                            scale_state.frames_reused += 1
                            total_frames_reused += 1
                        else:
                            scale_state.frames_rendered += 1
                            total_frames_rendered += 1
                        scale_progress[progress_key]["render"]["completed_frames"] += 1
                        if rr.get("reused"):
                            scale_progress[progress_key]["render"]["reused_frames"] += 1
                        scale_progress[progress_key]["render"]["status"] = "in_progress"
                        update_last_event(
                            "render",
                            "ok" if not rr.get("reused") else "reused",
                            episode_id=scale_state.episode_id,
                            scale_mm=scale_state.scale_mm,
                            frame_name=frame_name,
                            markers=len(rendered_markers),
                            reused=bool(rr.get("reused", False)),
                        )

                        logging.info(
                            "Render OK | ep=%06d scale=%d frame=%s | markers=%d reused=%s | %s | %s",
                            scale_state.episode_id,
                            scale_state.scale_mm,
                            frame_name,
                            len(rendered_markers),
                            bool(rr.get("reused", False)),
                            render_eta,
                            render_thr,
                        )

                        maybe_finalize_scale(scale_state)
                        persist_run_status("running")

            total_frames_rendered = int(sum(s.frames_rendered for s in scale_states.values()))
            total_frames_reused = int(sum(s.frames_reused for s in scale_states.values()))
            logging.info(
                "Stage summary | physics=%d/%d meshing=%d/%d render=%d/%d",
                physics_ok,
                total_scale_tasks,
                meshing_ok,
                total_meshing_tasks_planned,
                render_ok,
                total_render_tasks_planned,
            )
            logging.info(
                "Frame summary | rendered=%d reused=%d total=%d",
                total_frames_rendered,
                total_frames_reused,
                total_frames_rendered + total_frames_reused,
            )

        manifest_episodes: List[Dict[str, Any]] = []
        failed_episodes: List[Dict[str, Any]] = []
        successful_frame_counts: List[int] = []
        position_sample_counts_requested = {
            "clean": 0,
            "near_boundary": 0,
            "partial_crop": 0,
        }
        position_sample_counts_actual: Dict[str, int] = {}

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
                if state.episode_dir.exists() and not args.resume:
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
                    "reference_scale_mm": state.reference_scale_mm,
                    "x_norm": state.x_norm,
                    "y_norm": state.y_norm,
                    "position_sample_type_requested": state.position_sample_type_requested,
                    "position_sample_type_actual": state.position_sample_type_actual,
                    "min_scale_margin_mm": state.contact_margin_min_scale_mm,
                    "reference_scale_margin_mm": state.contact_margin_reference_scale_mm,
                    "constraint_scale_mm": float(contact_constraint_scale_mm),
                },
                "image_resolution": {"width": DEFAULT_IMAGE_RES[0], "height": DEFAULT_IMAGE_RES[1]},
                "scales": state.scales,
            }
            with open(state.episode_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            for scale_meta in state.scales.values():
                successful_frame_counts.append(len(scale_meta.get("frames", {})))
            position_sample_counts_requested[state.position_sample_type_requested] = (
                position_sample_counts_requested.get(state.position_sample_type_requested, 0) + 1
            )
            position_sample_counts_actual[state.position_sample_type_actual] = (
                position_sample_counts_actual.get(state.position_sample_type_actual, 0) + 1
            )

            manifest_episodes.append(
                {
                    "episode_id": state.episode_id,
                    "path": state.episode_dir.name,
                    "indenter": state.indenter,
                    "contact": {
                        "x_mm": state.x_mm,
                        "y_mm": state.y_mm,
                        "depth_mm": state.depth_mm,
                        "reference_scale_mm": state.reference_scale_mm,
                        "x_norm": state.x_norm,
                        "y_norm": state.y_norm,
                        "position_sample_type_requested": state.position_sample_type_requested,
                        "position_sample_type_actual": state.position_sample_type_actual,
                        "min_scale_margin_mm": state.contact_margin_min_scale_mm,
                        "reference_scale_margin_mm": state.contact_margin_reference_scale_mm,
                        "constraint_scale_mm": float(contact_constraint_scale_mm),
                    },
                }
            )

        unique_frame_counts = sorted(set(successful_frame_counts))
        if not unique_frame_counts:
            frames_per_scale: int | Dict[str, Any] = 0
        elif len(unique_frame_counts) == 1:
            frames_per_scale = int(unique_frame_counts[0])
        else:
            frames_per_scale = {
                "min": int(min(unique_frame_counts)),
                "max": int(max(unique_frame_counts)),
                "unique_values": [int(v) for v in unique_frame_counts],
            }

        manifest = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_root": str(dataset_root),
            "dataset_variant": "static_multiframe_v1",
            "run_status_json": str(run_status_path.relative_to(dataset_root)),
            "command": shlex.join([str(sys.executable), *sys.argv]),
            "particle": str(args.particle),
            "scales_mm": [int(s) for s in args.scales_mm],
            "patch_grid": [int(patch_h), int(patch_w)],
            "coordinate_convention": "X right positive, Y down positive, row-major",
            "frame_sampling_mode": str(args.frame_sampling_mode),
            "frame_count": int(args.frame_count),
            "frame_depth_start_mm": float(args.frame_depth_start_mm),
            "frame_depth_end_mm": float(args.frame_depth_end_mm),
            "frame_fractions": (
                None
                if (args.frame_sampling_mode != "fraction" or args.frame_indices is not None)
                else [float(v) for v in args.frame_fractions]
            ),
            "frame_indices": (
                None
                if (args.frame_sampling_mode != "fraction" or args.frame_indices is None)
                else [int(v) for v in args.frame_indices]
            ),
            "contact_sampling_mode": "normalized_layered_min_scale",
            "indenter_bbox_mode": "bottom_slice",
            "indenter_bbox_bottom_height_mm": float(DEFAULT_INDENTER_BBOX_BOTTOM_HEIGHT_MM),
            "position_sample_ratios": {
                "clean": float(args.position_clean_ratio),
                "near_boundary": float(args.position_near_boundary_ratio),
                "partial_crop": float(args.position_partial_crop_ratio),
            },
            "near_boundary_max_margin_mm": float(args.near_boundary_max_margin_mm),
            "partial_crop_min_overhang_mm": float(args.partial_crop_min_overhang_mm),
            "partial_crop_max_overhang_mm": float(args.partial_crop_max_overhang_mm),
            "contact_reference_scale_mm": float(contact_reference_scale_mm),
            "contact_constraint_scale_mm": float(contact_constraint_scale_mm),
            "position_sample_counts_requested": {
                key: int(value) for key, value in position_sample_counts_requested.items()
            },
            "position_sample_counts_actual": {
                str(key): int(value) for key, value in position_sample_counts_actual.items()
            },
            "frames_per_scale": frames_per_scale,
            "episodes_per_indenter": int(args.episodes_per_indenter),
            "indenter_count": len(indenter_names),
            "texture_count": len(textures),
            "total_episodes_planned": total_episodes,
            "total_scale_tasks_planned": total_scale_tasks,
            "total_meshing_tasks_planned": int(total_meshing_tasks_planned),
            "total_render_tasks_planned": int(total_render_tasks_planned),
            "scales_restored_from_metadata": int(resume_scales_from_metadata),
            "physics_tasks_succeeded": physics_ok,
            "meshing_tasks_succeeded": meshing_ok,
            "render_tasks_succeeded": render_ok,
            "frames_rendered": int(total_frames_rendered),
            "frames_reused": int(total_frames_reused),
            "successful_episodes": len(manifest_episodes),
            "failed_episodes": failed_episodes,
            "episodes": manifest_episodes,
        }

        with open(dataset_root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        image_index_path, image_index_rows = build_image_index_csv(dataset_root)
        manifest["image_index_csv"] = str(image_index_path.relative_to(dataset_root))
        manifest["image_index_rows"] = int(image_index_rows)
        with open(dataset_root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        update_last_event(
            "dataset",
            "completed",
            successful_episodes=len(manifest_episodes),
            failed_episodes=len(failed_episodes),
            image_index_rows=image_index_rows,
        )
        persist_run_status("completed", force=True)

        allow_intermediate_cleanup = (len(failed_episodes) == 0)

        logging.info("=" * 72)
        logging.info("DONE | successful episodes: %d | failed episodes: %d", len(manifest_episodes), len(failed_episodes))
        logging.info("Total wall time: %s", _format_duration(time.time() - run_start_ts))
        logging.info("Manifest: %s", dataset_root / "manifest.json")
        logging.info("Image index: %s (%d rows)", image_index_path, image_index_rows)
        logging.info("=" * 72)

    except BaseException as exc:
        err_text = str(exc) if str(exc) else exc.__class__.__name__
        update_last_event("dataset", "failed", message=err_text)
        persist_run_status("failed", force=True, error=err_text)
        raise
    finally:
        restore_backup(parameters_path, backup_path)
        logging.info("Restored %s from backup", parameters_path)

        if script_tmp_dir.exists():
            shutil.rmtree(script_tmp_dir, ignore_errors=True)
        if (
            work_tmp_root.exists()
            and (not args.keep_intermediates)
            and allow_intermediate_cleanup
        ):
            shutil.rmtree(work_tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
