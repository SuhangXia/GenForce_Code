#!/usr/bin/env python3
"""Generate a multiscale tactile dataset with synchronized contact physics.

Pipeline per episode/scale:
1) Update elastomer size in sim/parameters.yml.
2) Run gel_press.py to generate deformation (.npz).
3) Run inline Open3D script (temp file) to convert .npz -> .stl.
4) Run inline Blender script (temp file) to render marker image.
5) Save metadata.json with patch coordinates for each scale.

The script keeps x/y/depth and marker fixed across scales for each episode,
which creates "parallel universe" samples for cross-scale transfer.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import fcntl
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml


DEFAULT_SCALES_MM = [15, 18, 20, 22, 25]
DEFAULT_FOV_DEG = 40.0

OPEN3D_TEMP_SCRIPT = r'''
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description="Convert deformation npz to STL.")
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-stl", required=True)
    parser.add_argument("--poisson-depth", type=int, default=9)
    return parser.parse_args()


def load_points(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    required = ["p_xpos_list", "p_ypos_list", "p_zpos_list"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {npz_path}")

    x = np.asarray(data["p_xpos_list"])
    y = np.asarray(data["p_ypos_list"])
    z = np.asarray(data["p_zpos_list"])

    x_last = x[-1] if x.ndim > 1 else x
    y_last = y[-1] if y.ndim > 1 else y
    z_last = z[-1] if z.ndim > 1 else z

    if not (x_last.shape == y_last.shape == z_last.shape):
        raise ValueError("Shape mismatch among x/y/z arrays in npz data")

    # Convert mm -> m so Blender camera trig uses physical metric units.
    points = np.stack([x_last, y_last, z_last], axis=1).astype(np.float64) / 1000.0
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    if points.shape[0] < 100:
        raise ValueError(f"Too few valid points ({points.shape[0]}) in {npz_path}")
    return points


def main() -> None:
    args = parse_args()
    npz_path = Path(args.input_npz)
    stl_path = Path(args.output_stl)
    stl_path.parent.mkdir(parents=True, exist_ok=True)

    points = load_points(npz_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    span = points.max(axis=0) - points.min(axis=0)
    diag = float(np.linalg.norm(span))
    radius = max(diag * 0.03, 1e-4)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except RuntimeError:
        # Some point clouds are sparse near borders; continue with estimated normals.
        pass

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.poisson_depth
    )
    if len(mesh.triangles) == 0:
        raise RuntimeError("Poisson reconstruction produced an empty mesh")

    densities = np.asarray(densities)
    if densities.size:
        threshold = float(np.quantile(densities, 0.02))
        mesh.remove_vertices_by_mask(densities < threshold)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    ok = o3d.io.write_triangle_mesh(str(stl_path), mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write STL: {stl_path}")

    print(f"Converted {npz_path} -> {stl_path} with {len(mesh.triangles)} triangles")


if __name__ == "__main__":
    main()
'''

BLENDER_TEMP_SCRIPT = r'''
import argparse
import bmesh
import math
from pathlib import Path
import bpy
import numpy as np


def parse_args():
    argv = bpy.app.driver_namespace.get("argv")
    if argv is None:
        import sys
        argv = sys.argv
    user_argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser(description="Render STL with marker texture")
    parser.add_argument("--stl", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale-mm", required=True, type=float)
    parser.add_argument("--fov-deg", default=40.0, type=float)
    return parser.parse_args(user_argv)


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)


def import_stl(stl_path: str):
    try:
        bpy.ops.import_mesh.stl(filepath=stl_path)
    except Exception:
        bpy.ops.wm.stl_import(filepath=stl_path)

    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        raise RuntimeError("STL import failed or did not produce a mesh object")
    return obj


def robust_bbox_world(obj):
    verts_world = np.array(
        [tuple(obj.matrix_world @ v.co) for v in obj.data.vertices], dtype=np.float64
    )
    if verts_world.shape[0] < 8:
        raise RuntimeError("Mesh has too few vertices")

    q_low = np.quantile(verts_world, 0.02, axis=0)
    q_high = np.quantile(verts_world, 0.98, axis=0)

    min_x, min_y, min_z = [float(v) for v in q_low]
    max_x, max_y, max_z = [float(v) for v in q_high]

    width = max(max_x - min_x, 1e-8)
    height = max(max_y - min_y, 1e-8)

    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    bottom_z = min_z

    return min_x, min_y, width, height, center_x, center_y, bottom_z


def create_material(marker_path: str):
    material = bpy.data.materials.new(name="AdapterPrincipled")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex_coord = nodes.new("ShaderNodeTexCoord")

    tex.image = bpy.data.images.load(marker_path)
    # Keep out-of-range UVs black to preserve black sensor coating background.
    tex.extension = "CLIP"
    tex.interpolation = "Cubic"
    tex.image.colorspace_settings.name = "sRGB"

    # Marker appearance is driven by texture itself (black paint + white markers),
    # not by scene lighting/shadows.
    bsdf.inputs["Specular"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Emission Strength"].default_value = 1.0

    links.new(tex_coord.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex.outputs["Color"], bsdf.inputs["Emission"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return material


def apply_uv_relative(obj, center_x: float, center_y: float, scale_mm: float):
    scale_m = max(scale_mm / 1000.0, 1e-8)
    min_x = center_x - scale_m * 0.5
    min_y = center_y - scale_m * 0.5
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)

    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        uv_layer = bm.loops.layers.uv.new("UVMap")

    for face in bm.faces:
        for loop in face.loops:
            world_co = obj.matrix_world @ loop.vert.co
            u = (world_co.x - min_x) / scale_m
            v = (world_co.y - min_y) / scale_m
            u = min(max(u, 0.0), 1.0)
            v = min(max(v, 0.0), 1.0)
            loop[uv_layer].uv = (u, v)

    bm.to_mesh(me)
    bm.free()
    me.update()


def setup_world_light():
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True

    nodes = scene.world.node_tree.nodes
    bg = nodes.get("Background")
    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
    # Match GenForce-style black background.
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0

def setup_camera(center_x: float, center_y: float, bottom_z: float, scale_mm: float, fov_deg: float):
    scene = bpy.context.scene

    camera_data = bpy.data.cameras.new("AdapterCamera")
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"

    fov_h = math.radians(fov_deg)
    camera_data.angle = fov_h

    scale_m = scale_mm / 1000.0
    distance = (scale_m / 2.0) / math.tan(fov_h / 2.0)
    # Important for mm-scale scenes: Blender default clip_start (0.1m) would clip
    # the entire tactile surface because camera distance is around 0.02-0.03m.
    camera_data.clip_start = 1e-4
    camera_data.clip_end = 10.0

    cam_obj = bpy.data.objects.new("AdapterCamera", camera_data)
    cam_obj.location = (center_x, center_y, bottom_z - distance)
    cam_obj.rotation_euler = (math.pi, 0.0, 0.0)

    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj


def configure_render(output_path: str):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 64
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.sample_clamp_direct = 2.0
        scene.cycles.sample_clamp_indirect = 2.0
        scene.cycles.max_bounces = 4
    scene.render.image_settings.file_format = "JPEG"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    # Keep texture contrast stable for marker visibility.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.filepath = output_path


def main():
    args = parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clean_scene()
    obj = import_stl(args.stl)

    min_x, min_y, width, height, center_x, center_y, bottom_z = robust_bbox_world(obj)

    material = create_material(args.marker)
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

    apply_uv_relative(obj, center_x, center_y, args.scale_mm)
    setup_world_light()
    setup_camera(center_x, center_y, bottom_z, args.scale_mm, args.fov_deg)
    configure_render(str(out_path))

    bpy.ops.render.render(write_still=True)
    print(f"Rendered image to {out_path}")


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class EpisodeJob:
    global_idx: int
    indenter_name: str
    repeat_idx: int


@contextmanager
def locked_file(lock_path: Path):
    """Cross-process exclusive file lock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate parallel-universe multiscale dataset from GenForce simulation."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to GenForce repository root.",
    )
    parser.add_argument("--particle", type=str, default="100000")
    parser.add_argument("--episodes-per-indenter", type=int, default=1)
    parser.add_argument("--scales-mm", type=int, nargs="+", default=DEFAULT_SCALES_MM)

    parser.add_argument("--x-min", type=int, default=-4)
    parser.add_argument("--x-max", type=int, default=4)
    parser.add_argument("--y-min", type=int, default=-4)
    parser.add_argument("--y-max", type=int, default=4)
    parser.add_argument("--xy-step", type=int, default=1)

    parser.add_argument("--depth-min", type=float, default=0.3)
    parser.add_argument("--depth-max", type=float, default=1.5)
    parser.add_argument("--depth-step", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG)
    parser.add_argument(
        "--scale-mode",
        type=str,
        default="virtual",
        choices=["virtual", "physical"],
        help="`virtual`: one physics run per episode, per-scale camera/marker remap only; "
        "`physical`: rerun physics for every scale with elastomer size patching.",
    )
    parser.add_argument(
        "--physics-scale-mm",
        type=int,
        default=None,
        help="Only used in `virtual` mode. If set, force elastomer.size.l/w to this value before physics.",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("adapter_dataset"),
        help="Relative to repo root unless absolute path is provided.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Shard total jobs for multi-process execution.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Current worker id in [0, num_workers-1].",
    )

    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Additive offset for episode index in folder naming.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episode folder if it already exists.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep temporary npz/stl and temp scripts for debugging.",
    )
    parser.add_argument(
        "--blender-cmd",
        type=str,
        default="blender",
        help="Blender executable command.",
    )
    parser.add_argument(
        "--python-cmd",
        type=str,
        default=sys.executable,
        help="Python executable used for subprocess scripts.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    if args.episodes_per_indenter <= 0:
        parser.error("--episodes-per-indenter must be > 0")
    if args.xy_step <= 0:
        parser.error("--xy-step must be > 0")
    if args.depth_step <= 0:
        parser.error("--depth-step must be > 0")
    if args.num_workers <= 0:
        parser.error("--num-workers must be > 0")
    if not (0 <= args.worker_id < args.num_workers):
        parser.error("--worker-id must satisfy 0 <= worker_id < num_workers")
    if args.physics_scale_mm is not None and args.physics_scale_mm <= 0:
        parser.error("--physics-scale-mm must be > 0 when provided")

    return args


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(base_root: Path, path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else (base_root / path_value)


def list_indenter_npys(indenter_npy_dir: Path) -> List[Path]:
    npys = sorted(p for p in indenter_npy_dir.glob("*.npy") if p.is_file())
    if not npys:
        raise FileNotFoundError(
            f"No .npy indenters found in {indenter_npy_dir}. "
            "Run sim/deformation/1_stl2npy.py first."
        )
    return npys


def list_marker_images(marker_dir: Path) -> List[Path]:
    markers = sorted(
        p
        for p in marker_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith(".")
    )
    if not markers:
        raise FileNotFoundError(f"No marker images found in {marker_dir}")
    return markers


def build_jobs(indenter_files: Sequence[Path], episodes_per_indenter: int) -> List[EpisodeJob]:
    jobs: List[EpisodeJob] = []
    global_idx = 0
    for indenter in indenter_files:
        indenter_name = indenter.stem
        for repeat_idx in range(episodes_per_indenter):
            jobs.append(EpisodeJob(global_idx=global_idx, indenter_name=indenter_name, repeat_idx=repeat_idx))
            global_idx += 1
    return jobs


def split_jobs_for_worker(jobs: Sequence[EpisodeJob], worker_id: int, num_workers: int) -> List[EpisodeJob]:
    return [job for i, job in enumerate(jobs) if i % num_workers == worker_id]


def depth_candidates(depth_min: float, depth_max: float, depth_step: float) -> np.ndarray:
    values = np.arange(depth_min, depth_max + 1e-8, depth_step, dtype=np.float64)
    if values.size == 0:
        raise ValueError("No depth candidates generated. Check depth range/step.")
    return np.round(values, 1)


def sample_contact(rng: random.Random, xs: Sequence[int], ys: Sequence[int], depths: Sequence[float]) -> Tuple[int, int, float]:
    x = rng.choice(xs)
    y = rng.choice(ys)
    depth = float(rng.choice(depths))
    return x, y, round(depth, 1)


def write_temp_script(script_text: str, prefix: str) -> Path:
    fd, script_path = tempfile.mkstemp(prefix=prefix, suffix=".py")
    os.close(fd)
    path = Path(script_path)
    path.write_text(script_text, encoding="utf-8")
    return path


def validate_embedded_script(script_text: str, script_name: str) -> None:
    try:
        ast.parse(script_text)
    except SyntaxError as exc:
        raise SyntaxError(f"Embedded script '{script_name}' has invalid syntax: {exc}") from exc


def run_subprocess(cmd: Sequence[str], cwd: Path, stage_name: str) -> None:
    logging.debug("Running [%s]: %s", stage_name, " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_elastomer_scale(parameters_yml: Path, scale_mm: int) -> None:
    data = read_yaml(parameters_yml)
    try:
        data["elastomer"]["size"]["l"] = int(scale_mm)
        data["elastomer"]["size"]["w"] = int(scale_mm)
    except Exception as exc:
        raise KeyError("Expected keys elastomer.size.l and elastomer.size.w in parameters.yml") from exc
    write_yaml(parameters_yml, data)


def patch_coords_16x16(scale_mm: int) -> List[List[List[float]]]:
    axis = np.linspace(-scale_mm / 2.0, scale_mm / 2.0, 16, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coords = np.stack([xx, yy], axis=-1)
    coords = np.round(coords, 6)
    return coords.tolist()


def expected_npz_path(npz_output_root: Path, indenter_name: str, x: int, y: int, depth: float) -> Path:
    suffix = f"{x}_{y}_{round(depth, 1)}.npz"
    return npz_output_root / indenter_name / suffix


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def to_metadata_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def run_episode(
    repo_root: Path,
    job: EpisodeJob,
    episode_id: int,
    scales_mm: Sequence[int],
    marker_path: Path,
    x: int,
    y: int,
    depth: float,
    args: argparse.Namespace,
    parameters_yml: Path,
    lock_path: Path,
    npz2stl_script: Path,
    blender_script: Path,
    output_root: Path,
) -> None:
    episode_dir = output_root / f"episode_{episode_id:06d}"
    if episode_dir.exists() and args.skip_existing:
        logging.info("Skipping existing episode folder: %s", episode_dir)
        return
    episode_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix=f"multiscale_ep_{episode_id:06d}_"))
    npz_output_root = work_dir / "npz"

    metadata: Dict[str, object] = {
        "episode_id": episode_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "worker": {"worker_id": args.worker_id, "num_workers": args.num_workers},
        "indenter": job.indenter_name,
        "particle": str(args.particle),
        "scale_mode": args.scale_mode,
        "physics_scale_mm": args.physics_scale_mm,
        "contact": {"x_mm": x, "y_mm": y, "depth_mm": depth},
        "marker_image": to_metadata_path(marker_path, repo_root),
        "scales": {},
    }

    try:
        if args.scale_mode == "virtual":
            # In virtual mode, run one physics simulation and render it with multiple
            # virtual physical windows (`scale_mm`) via UV/camera remapping.
            logging.info(
                "Episode %06d | indenter=%s | virtual physics | contact=(x=%d, y=%d, d=%.1f)",
                episode_id,
                job.indenter_name,
                x,
                y,
                depth,
            )
            with locked_file(lock_path):
                if args.physics_scale_mm is not None:
                    set_elastomer_scale(parameters_yml, int(args.physics_scale_mm))
                run_subprocess(
                    [
                        args.python_cmd,
                        "sim/deformation/gel_press.py",
                        "--config",
                        "sim/parameters.yml",
                        "--particle",
                        str(args.particle),
                        "--dir_output",
                        str(npz_output_root),
                        "--dataset",
                        "sim/assets/indenters/input",
                        "--object",
                        job.indenter_name,
                        "--x",
                        str(x),
                        "--y",
                        str(y),
                        "--depth",
                        f"{depth:.1f}",
                    ],
                    cwd=repo_root,
                    stage_name="gel_press",
                )

            npz_path = expected_npz_path(npz_output_root, job.indenter_name, x, y, depth)
            if not npz_path.exists():
                fallback = sorted((npz_output_root / job.indenter_name).glob(f"{x}_{y}_*.npz"))
                if not fallback:
                    raise FileNotFoundError(f"Expected deformation npz not found at {npz_path}")
                npz_path = fallback[-1]

            stl_path = work_dir / f"{job.indenter_name}_ep{episode_id:06d}.stl"
            run_subprocess(
                [
                    args.python_cmd,
                    str(npz2stl_script),
                    "--input-npz",
                    str(npz_path),
                    "--output-stl",
                    str(stl_path),
                    "--poisson-depth",
                    "9",
                ],
                cwd=repo_root,
                stage_name="npz_to_stl",
            )

            for scale_mm in scales_mm:
                scale_dir = episode_dir / f"scale_{scale_mm}mm"
                scale_dir.mkdir(parents=True, exist_ok=True)
                render_path = scale_dir / "render.jpg"

                logging.info(
                    "Episode %06d | indenter=%s | render scale=%dmm",
                    episode_id,
                    job.indenter_name,
                    scale_mm,
                )
                run_subprocess(
                    [
                        args.blender_cmd,
                        "-b",
                        "--python",
                        str(blender_script),
                        "--",
                        "--stl",
                        str(stl_path),
                        "--marker",
                        str(marker_path),
                        "--output",
                        str(render_path),
                        "--scale-mm",
                        str(scale_mm),
                        "--fov-deg",
                        str(args.fov_deg),
                    ],
                    cwd=repo_root,
                    stage_name="blender_render",
                )

                metadata["scales"][f"{scale_mm}mm"] = {
                    "scale_mm": int(scale_mm),
                    "render_image": str(render_path.relative_to(episode_dir)),
                    "patch_coords_16x16": patch_coords_16x16(int(scale_mm)),
                }

            if not args.keep_intermediates:
                safe_unlink(npz_path)
                safe_unlink(stl_path)

        else:
            for scale_mm in scales_mm:
                scale_dir = episode_dir / f"scale_{scale_mm}mm"
                scale_dir.mkdir(parents=True, exist_ok=True)
                render_path = scale_dir / "render.jpg"

                logging.info(
                    "Episode %06d | indenter=%s | physical scale=%dmm | contact=(x=%d, y=%d, d=%.1f)",
                    episode_id,
                    job.indenter_name,
                    scale_mm,
                    x,
                    y,
                    depth,
                )

                # Step A + B are inside one lock so multiple workers do not race on sim/parameters.yml.
                with locked_file(lock_path):
                    set_elastomer_scale(parameters_yml, scale_mm)
                    run_subprocess(
                        [
                            args.python_cmd,
                            "sim/deformation/gel_press.py",
                            "--config",
                            "sim/parameters.yml",
                            "--particle",
                            str(args.particle),
                            "--dir_output",
                            str(npz_output_root),
                            "--dataset",
                            "sim/assets/indenters/input",
                            "--object",
                            job.indenter_name,
                            "--x",
                            str(x),
                            "--y",
                            str(y),
                            "--depth",
                            f"{depth:.1f}",
                        ],
                        cwd=repo_root,
                        stage_name="gel_press",
                    )

                npz_path = expected_npz_path(npz_output_root, job.indenter_name, x, y, depth)
                if not npz_path.exists():
                    fallback = sorted((npz_output_root / job.indenter_name).glob(f"{x}_{y}_*.npz"))
                    if not fallback:
                        raise FileNotFoundError(f"Expected deformation npz not found at {npz_path}")
                    npz_path = fallback[-1]

                stl_path = work_dir / f"{job.indenter_name}_ep{episode_id:06d}_s{scale_mm}.stl"
                run_subprocess(
                    [
                        args.python_cmd,
                        str(npz2stl_script),
                        "--input-npz",
                        str(npz_path),
                        "--output-stl",
                        str(stl_path),
                        "--poisson-depth",
                        "9",
                    ],
                    cwd=repo_root,
                    stage_name="npz_to_stl",
                )
                run_subprocess(
                    [
                        args.blender_cmd,
                        "-b",
                        "--python",
                        str(blender_script),
                        "--",
                        "--stl",
                        str(stl_path),
                        "--marker",
                        str(marker_path),
                        "--output",
                        str(render_path),
                        "--scale-mm",
                        str(scale_mm),
                        "--fov-deg",
                        str(args.fov_deg),
                    ],
                    cwd=repo_root,
                    stage_name="blender_render",
                )

                metadata["scales"][f"{scale_mm}mm"] = {
                    "scale_mm": int(scale_mm),
                    "render_image": str(render_path.relative_to(episode_dir)),
                    "patch_coords_16x16": patch_coords_16x16(int(scale_mm)),
                }

                if not args.keep_intermediates:
                    safe_unlink(npz_path)
                    safe_unlink(stl_path)

        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    finally:
        if not args.keep_intermediates:
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    repo_root = args.repo_root.resolve()
    output_root = resolve_path(repo_root, args.dataset_root)
    output_root.mkdir(parents=True, exist_ok=True)

    parameters_yml = repo_root / "sim" / "parameters.yml"
    backup_path = parameters_yml.with_suffix(parameters_yml.suffix + ".bak")
    lock_path = parameters_yml.with_suffix(parameters_yml.suffix + ".lock")

    marker_dir = repo_root / "sim" / "marker" / "marker_pattern"
    indenter_npy_dir = repo_root / "sim" / "assets" / "indenters" / "input" / f"npy_{args.particle}"

    if not parameters_yml.exists():
        raise FileNotFoundError(f"Missing parameters.yml: {parameters_yml}")
    if not indenter_npy_dir.exists():
        raise FileNotFoundError(
            f"Missing indenter npy directory: {indenter_npy_dir}. "
            "Run sim/deformation/1_stl2npy.py first."
        )

    indenter_npys = list_indenter_npys(indenter_npy_dir)
    markers = list_marker_images(marker_dir)

    x_candidates = list(range(args.x_min, args.x_max + 1, args.xy_step))
    y_candidates = list(range(args.y_min, args.y_max + 1, args.xy_step))
    if not x_candidates or not y_candidates:
        raise ValueError("Empty x/y candidate list. Check x/y range and xy-step.")
    z_candidates = depth_candidates(args.depth_min, args.depth_max, args.depth_step).tolist()

    all_jobs = build_jobs(indenter_npys, args.episodes_per_indenter)
    jobs = split_jobs_for_worker(all_jobs, args.worker_id, args.num_workers)

    if not jobs:
        logging.warning("No jobs assigned to worker_id=%d (num_workers=%d)", args.worker_id, args.num_workers)
        return

    logging.info("Repo root: %s", repo_root)
    logging.info("Total jobs: %d | This worker jobs: %d", len(all_jobs), len(jobs))
    logging.info("Scales(mm): %s", list(args.scales_mm))
    logging.info("Scale mode: %s | physics_scale_mm=%s", args.scale_mode, args.physics_scale_mm)
    logging.info("Found %d markers in %s", len(markers), marker_dir)

    validate_embedded_script(OPEN3D_TEMP_SCRIPT, "OPEN3D_TEMP_SCRIPT")
    validate_embedded_script(BLENDER_TEMP_SCRIPT, "BLENDER_TEMP_SCRIPT")
    npz2stl_script = write_temp_script(OPEN3D_TEMP_SCRIPT, prefix="tmp_npz2stl_")
    blender_script = write_temp_script(BLENDER_TEMP_SCRIPT, prefix="tmp_blender_render_")

    if backup_path.exists():
        raise FileExistsError(
            f"Backup already exists at {backup_path}. Resolve/remove it before running to avoid overwriting." 
        )

    with locked_file(lock_path):
        shutil.copy2(parameters_yml, backup_path)
        logging.info("Created backup: %s", backup_path)

    try:
        for idx, job in enumerate(jobs, start=1):
            # Deterministic episode sampling by global index, independent of worker sharding.
            rng = random.Random(args.seed + job.global_idx)
            marker_path = rng.choice(markers)
            x, y, depth = sample_contact(rng, x_candidates, y_candidates, z_candidates)

            episode_id = args.episode_offset + job.global_idx
            logging.info(
                "[%d/%d] episode_%06d | indenter=%s | repeat=%d | marker=%s",
                idx,
                len(jobs),
                episode_id,
                job.indenter_name,
                job.repeat_idx,
                marker_path.name,
            )

            run_episode(
                repo_root=repo_root,
                job=job,
                episode_id=episode_id,
                scales_mm=args.scales_mm,
                marker_path=marker_path,
                x=x,
                y=y,
                depth=depth,
                args=args,
                parameters_yml=parameters_yml,
                lock_path=lock_path,
                npz2stl_script=npz2stl_script,
                blender_script=blender_script,
                output_root=output_root,
            )

    finally:
        # Restore parameters.yml from .bak backup.
        with locked_file(lock_path):
            if backup_path.exists():
                shutil.copy2(backup_path, parameters_yml)
                if not args.keep_intermediates:
                    safe_unlink(backup_path)
                logging.info("Restored %s from backup", parameters_yml)

        if not args.keep_intermediates:
            safe_unlink(npz2stl_script)
            safe_unlink(blender_script)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error("Interrupted by user")
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        logging.error("Subprocess failed (exit=%s): %s", exc.returncode, exc.cmd)
        sys.exit(exc.returncode)
    except Exception as exc:
        logging.exception("Generation failed: %s", exc)
        sys.exit(1)
