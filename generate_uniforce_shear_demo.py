#!/usr/bin/env python3
"""Minimal z-press + x/y shear demo exported in uniforce format.

This file is intentionally additive. It does not modify the existing GenForce
or uniforce_corl code paths. It creates a small self-contained demo:

z press -> x shear -> y shear -> release from sheared endpoint

and exports the result as a marker-only uniforce-style dataset.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR / "sim"
PARAMETERS_PATH = SIM_DIR / "parameters.yml"
INDENTER_DATASET_DIR = SIM_DIR / "assets" / "indenters" / "input"
MARKER_PATTERN_DIR = SIM_DIR / "marker" / "marker_pattern"

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
RAW_IMAGE_EXT = ".jpg"
JPEG_SAVE_KWARGS = {"quality": 100, "subsampling": 0}
PIL_RESAMPLING = getattr(Image, "Resampling", Image)

DEFAULT_SCALE_MM = 15.0
DEFAULT_SENSOR_L = "digit"
DEFAULT_SENSOR_R = "gelsight"
DEFAULT_OBJECT = "cylinder"
DEFAULT_MARKER = "Array1"
DEFAULT_DEPTH_MM = 1.5
DEFAULT_SHEAR_X_MM = 0.5
DEFAULT_SHEAR_Y_MM = 0.5
DEFAULT_CONTACT_X_MM = 0.0
DEFAULT_CONTACT_Y_MM = 0.0
DEFAULT_SAFETY_MARGIN_MM = 0.1
DEFAULT_DATE_TAG = dt.date.today().strftime("%Y%m%d")
DEFAULT_RENDER_DEVICE = "cpu"
DEFAULT_RENDER_SAMPLES = 1
DEFAULT_CAMERA_MODE = "fixed_distance_variable_fov"
DEFAULT_PRECONTACT_FRAMES = 1
DEFAULT_PRESS_FRAMES = 10
DEFAULT_SHEAR_X_FRAMES = 5
DEFAULT_SHEAR_Y_FRAMES = 5
DEFAULT_RELEASE_FRAMES = 7
DEFAULT_WARMUP_STEPS = 20
DEFAULT_STEPS_PER_FRAME = 0
DEFAULT_MESHING_TIMEOUT_SEC = 300
DEFAULT_RENDER_TIMEOUT_SEC = 600
DEFAULT_MIN_BAD_DEPTH_MM = {
    "dots": 1.510032,
    "moon": 1.711395,
    "hemisphere": 1.790733,
    "cylinder_si": 1.810065,
    "cylinder_sh": 1.910038,
}


PHYSICS_TEMP_SCRIPT = textwrap.dedent(
    r"""
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import taichi as ti
    import yaml
    from pytransform3d import rotations


    def parse_args():
        p = argparse.ArgumentParser(description="Run a pinned-rigidbody shear demo in Taichi MPM.")
        p.add_argument("--config", required=True)
        p.add_argument("--dataset", required=True)
        p.add_argument("--object", required=True)
        p.add_argument("--particle", default="100000")
        p.add_argument("--contact-x-mm", type=float, required=True)
        p.add_argument("--contact-y-mm", type=float, required=True)
        p.add_argument("--schedule-json", required=True)
        p.add_argument("--output-npz", required=True)
        p.add_argument("--warmup-steps", type=int, default=20)
        p.add_argument("--steps-per-frame", type=int, default=30)
        p.add_argument("--show-gui", action="store_true")
        return p.parse_args()


    def read_yaml(path: Path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)


    def init_taichi():
        try:
            ti.init(arch=ti.gpu)
            return "gpu"
        except Exception:
            ti.init(arch=ti.cpu)
            return "cpu"


    args = parse_args()
    config = read_yaml(Path(args.config))
    schedule_payload = json.loads(Path(args.schedule_json).read_text(encoding="utf-8"))
    frame_commands = schedule_payload["frame_commands"]
    warmup_steps = int(args.warmup_steps)
    fixed_steps_per_frame = int(args.steps_per_frame)

    selected_arch = init_taichi()

    dt = float(config["world"]["dt"])
    world_speed = abs(float(config["world"]["speed"]))
    x_offset = float(config["world"]["coordin_offset"]["x_offset"])
    y_offset = float(config["world"]["coordin_offset"]["y_offset"])
    z_offset = float(config["world"]["coordin_offset"]["z_offset"])
    dx_display = float(config["world"]["display_shift"]["dx"])
    dy_display = float(config["world"]["display_shift"]["dy"])
    scale_display = float(config["world"]["display_shift"]["scale"])

    num_l = int(config["elastomer"]["particle"]["num_l"])
    num_w = int(config["elastomer"]["particle"]["num_w"])
    num_h = int(config["elastomer"]["particle"]["num_h"])
    l = float(config["elastomer"]["size"]["l"])
    w = float(config["elastomer"]["size"]["w"])
    h = float(config["elastomer"]["size"]["h"])

    pose_x = float(config["indenter"]["pose"]["x"])
    pose_y = float(config["indenter"]["pose"]["y"])
    pose_z = float(config["indenter"]["pose"]["z"])
    pose_R = float(config["indenter"]["pose"]["R"])
    pose_P = float(config["indenter"]["pose"]["P"])
    pose_Y = float(config["indenter"]["pose"]["Y"])

    n_grid = int(config["grid"]["n_grid"])
    l_grid = float(config["grid"]["length_grid"])
    dx = l_grid / n_grid
    inv_dx = 1.0 / dx

    particle_dis_l = l / (num_l - 1)
    particle_dis_w = w / (num_w - 1)
    particle_dis_h = h / (num_h - 1)

    object_path = Path(args.dataset) / f"npy_{args.particle}" / f"{args.object}.npy"
    raw_points = np.load(object_path).astype(np.float32)
    rotation_m = rotations.matrix_from_euler((pose_R, pose_P, pose_Y), 0, 1, 2, True)
    base_points = (rotation_m @ raw_points.T).T.astype(np.float32)
    base_points += np.array(
        [[x_offset + pose_x + float(args.contact_x_mm), y_offset + pose_y + float(args.contact_y_mm), pose_z]],
        dtype=np.float32,
    )
    indenter_count = int(base_points.shape[0])
    initial_min_z = float(np.min(base_points[:, 2]))
    gel_surface_h = z_offset + h / 2.0

    frame_targets = []
    for item in frame_commands:
        command_x_mm = float(item["command_x_mm"])
        command_y_mm = float(item["command_y_mm"])
        command_depth_mm = float(item["command_depth_mm"])
        tx = command_x_mm - float(args.contact_x_mm)
        ty = command_y_mm - float(args.contact_y_mm)
        target_min_z = gel_surface_h - command_depth_mm
        tz = target_min_z - initial_min_z
        frame_targets.append([tx, ty, tz])
    frame_targets = np.asarray(frame_targets, dtype=np.float32)
    if world_speed <= 0.0:
        raise ValueError(f"world.speed must have non-zero magnitude, got {config['world']['speed']}")
    max_step_distance_mm = world_speed * dt

    n_particles = num_l * num_w * num_h + indenter_count

    grid_v = ti.Vector.field(n=3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

    x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    x_2d = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
    F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
    material = ti.field(dtype=ti.i32, shape=n_particles)
    rigid_rest = ti.Vector.field(3, dtype=ti.f32, shape=indenter_count)

    rigid_tx = ti.field(dtype=ti.f32, shape=())
    rigid_ty = ti.field(dtype=ti.f32, shape=())
    rigid_tz = ti.field(dtype=ti.f32, shape=())
    rigid_vx = ti.field(dtype=ti.f32, shape=())
    rigid_vy = ti.field(dtype=ti.f32, shape=())
    rigid_vz = ti.field(dtype=ti.f32, shape=())

    p_vol = (dx * 0.5) ** 2
    p_rho = 1.0
    p_mass = p_vol * p_rho
    E = float(config["elastomer"]["modulus"])
    nu = float(config["elastomer"]["poisson_ratio"])
    mu_0 = E / (2.0 * (1.0 + nu))
    lambda_0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    gui = None
    particle_colors = None
    if args.show_gui:
        gui = ti.GUI("Shear Demo", res=config["world"]["resolution"], background_color=0x112F41)
        particle_colors = np.array([0x808080, 0x00FF00], dtype=np.uint32)


    @ti.func
    def is_valid_vec(p):
        return p[0] >= 0 and p[0] < n_grid and p[1] >= 0 and p[1] < n_grid and p[2] >= 0 and p[2] < n_grid


    @ti.func
    def rigid_translation():
        return ti.Vector([rigid_tx[None], rigid_ty[None], rigid_tz[None]])


    @ti.func
    def rigid_velocity():
        return ti.Vector([rigid_vx[None], rigid_vy[None], rigid_vz[None]])


    @ti.func
    def project_to_gui(p3):
        return ti.Vector([p3[1] * scale_display + dx_display, p3[2] * scale_display + dy_display])


    @ti.kernel
    def initialize(rest_points: ti.types.ndarray()):
        for i, j, k in ti.ndrange(num_l, num_w, num_h):
            m = i + j * num_l + k * num_l * num_w
            offset = ti.Vector([x_offset - l / 2.0, y_offset - w / 2.0, z_offset - h / 2.0])
            x[m] = ti.Vector([i * particle_dis_l, j * particle_dis_w, k * particle_dis_h]) + offset
            x_2d[m] = project_to_gui(x[m])
            v[m] = [0.0, 0.0, 0.0]
            material[m] = 0
            F[m] = ti.Matrix.identity(ti.f32, 3)
            C[m] = ti.Matrix.zero(ti.f32, 3, 3)

        base_index = num_l * num_w * num_h
        for i in range(indenter_count):
            rigid_rest[i] = ti.Vector([rest_points[i, 0], rest_points[i, 1], rest_points[i, 2]])
            x[base_index + i] = rigid_rest[i]
            x_2d[base_index + i] = project_to_gui(x[base_index + i])
            v[base_index + i] = [0.0, 0.0, 0.0]
            material[base_index + i] = 1
            F[base_index + i] = ti.Matrix.identity(ti.f32, 3)
            C[base_index + i] = ti.Matrix.zero(ti.f32, 3, 3)


    @ti.kernel
    def substep():
        for i, j, k in grid_m:
            grid_v[i, j, k] = [0.0, 0.0, 0.0]
            grid_m[i, j, k] = 0.0

        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            weights = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

            mu, la = mu_0, lambda_0
            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(3)):
                J *= sig[d, d]

            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose()
            stress += ti.Matrix.identity(ti.f32, 3) * la * J * (J - 1.0)
            stress = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * stress
            affine = stress + p_mass * C[p]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * dx
                weight = weights[i][0] * weights[j][1] * weights[k][2]
                pose = base + offset
                if is_valid_vec(pose):
                    grid_m[pose] += weight * p_mass
                    grid_v[pose] += weight * (p_mass * v[p] + affine @ dpos)

        for i, j, k in grid_m:
            if grid_m[i, j, k] > 0.0:
                grid_v[i, j, k] = (1.0 / grid_m[i, j, k]) * grid_v[i, j, k]
                if i < 3 and grid_v[i, j, k][0] < 0:
                    grid_v[i, j, k][0] = 0
                if i > n_grid - 3 and grid_v[i, j, k][0] > 0:
                    grid_v[i, j, k][0] = 0
                if j < 3 and grid_v[i, j, k][1] < 0:
                    grid_v[i, j, k][1] = 0
                if j > n_grid - 3 and grid_v[i, j, k][1] > 0:
                    grid_v[i, j, k][1] = 0
                if k < 3 and grid_v[i, j, k][2] < 0:
                    grid_v[i, j, k][2] = 0
                if k > n_grid - 3 and grid_v[i, j, k][2] > 0:
                    grid_v[i, j, k][2] = 0

        base_index = num_l * num_w * num_h
        rigid_t = rigid_translation()
        rigid_v = rigid_velocity()

        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            weights = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

            cached_v = ti.Vector.zero(ti.f32, 3)
            cached_C = ti.Matrix.zero(ti.f32, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                pose = base + ti.Vector([i, j, k])
                if is_valid_vec(pose):
                    dpos = ti.Vector([i, j, k]).cast(float) - fx
                    g_v = grid_v[pose]
                    weight = weights[i][0] * weights[j][1] * weights[k][2]
                    cached_v += weight * g_v
                    cached_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos)

            if material[p] == 0:
                v[p], C[p] = cached_v, cached_C
                if p < num_l * num_w * 3:
                    v[p] = ti.Vector([0.0, 0.0, 0.0])
                x[p] += dt * v[p]
                F[p] = (ti.Matrix.identity(ti.f32, 3) + dt * C[p]) @ F[p]
            else:
                rp = p - base_index
                C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                v[p] = rigid_v
                x[p] = rigid_rest[rp] + rigid_t
                F[p] = ti.Matrix.identity(ti.f32, 3)

            x_2d[p] = project_to_gui(x[p])


    def capture_surface_frame():
        x_np = x.to_numpy()
        start = num_l * num_w * (num_h - 1)
        end = num_l * num_w * num_h
        surface = x_np[start:end]
        return surface[:, 0].astype(np.float32), surface[:, 1].astype(np.float32), surface[:, 2].astype(np.float32)


    def render_gui():
        if gui is None:
            return
        display_scale = float(config["world"]["display_scale"])
        gui.circles(
            x_2d.to_numpy() / display_scale,
            radius=1,
            color=particle_colors[material.to_numpy()],
        )
        gui.show()


    initialize(base_points)

    x_frames = []
    y_frames = []
    z_frames = []

    prev_t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(max(0, warmup_steps)):
        rigid_tx[None] = float(prev_t[0])
        rigid_ty[None] = float(prev_t[1])
        rigid_tz[None] = float(prev_t[2])
        rigid_vx[None] = 0.0
        rigid_vy[None] = 0.0
        rigid_vz[None] = 0.0
        substep()
        render_gui()

    sx, sy, sz = capture_surface_frame()
    x_frames.append(sx)
    y_frames.append(sy)
    z_frames.append(sz)

    for frame_idx in range(1, int(frame_targets.shape[0])):
        next_t = frame_targets[frame_idx]
        start_t = prev_t.copy()
        segment_distance_mm = float(np.linalg.norm(next_t - start_t))
        adaptive_steps = 1 if segment_distance_mm <= 0.0 else int(np.ceil(segment_distance_mm / max_step_distance_mm))
        segment_steps = max(1, adaptive_steps)
        if fixed_steps_per_frame > 0:
            segment_steps = max(segment_steps, fixed_steps_per_frame)
        for step in range(1, segment_steps + 1):
            alpha = float(step) / float(segment_steps)
            curr_t = start_t + alpha * (next_t - start_t)
            prev_step_t = start_t + float(step - 1) / float(segment_steps) * (next_t - start_t)
            vel = (curr_t - prev_step_t) / dt
            rigid_tx[None] = float(curr_t[0])
            rigid_ty[None] = float(curr_t[1])
            rigid_tz[None] = float(curr_t[2])
            rigid_vx[None] = float(vel[0])
            rigid_vy[None] = float(vel[1])
            rigid_vz[None] = float(vel[2])
            substep()
            render_gui()
        prev_t = next_t.copy()
        sx, sy, sz = capture_surface_frame()
        x_frames.append(sx)
        y_frames.append(sy)
        z_frames.append(sz)

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        p_xpos_list=np.stack(x_frames, axis=0),
        p_ypos_list=np.stack(y_frames, axis=0),
        p_zpos_list=np.stack(z_frames, axis=0),
        frame_commands=np.asarray(frame_targets, dtype=np.float32),
        taichi_arch=np.asarray([selected_arch]),
    )
    print(f"Saved shear demo NPZ: {output_path}")
    """
)


@dataclass(frozen=True)
class DemoFrame:
    frame_name: str
    phase_name: str
    phase_index: int
    phase_progress: float
    command_x_mm: float
    command_y_mm: float
    command_depth_mm: float


@dataclass(frozen=True)
class DepthCapInfo:
    requested_depth_mm: float
    effective_depth_mm: float
    safe_depth_cap_mm: float | None
    min_bad_depth_mm: float | None
    safety_margin_mm: float
    capped: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal z-press + x/y shear uniforce demo.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sensor-l-name", type=str, default=DEFAULT_SENSOR_L)
    parser.add_argument("--sensor-r-name", type=str, default=DEFAULT_SENSOR_R)
    parser.add_argument("--scale-mm", type=float, default=DEFAULT_SCALE_MM)
    parser.add_argument("--object", type=str, default=DEFAULT_OBJECT)
    parser.add_argument("--marker-texture-name", type=str, default=DEFAULT_MARKER)
    parser.add_argument("--depth-mm", type=float, default=DEFAULT_DEPTH_MM)
    parser.add_argument("--shear-x-mm", type=float, default=DEFAULT_SHEAR_X_MM)
    parser.add_argument("--shear-y-mm", type=float, default=DEFAULT_SHEAR_Y_MM)
    parser.add_argument("--contact-x-mm", type=float, default=DEFAULT_CONTACT_X_MM)
    parser.add_argument("--contact-y-mm", type=float, default=DEFAULT_CONTACT_Y_MM)
    parser.add_argument("--safety-margin-mm", type=float, default=DEFAULT_SAFETY_MARGIN_MM)
    parser.add_argument("--render-device", choices=["cpu", "gpu"], default=DEFAULT_RENDER_DEVICE)
    parser.add_argument("--render-samples", type=int, default=DEFAULT_RENDER_SAMPLES)
    parser.add_argument("--show-gui", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-intermediates", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.expanduser().resolve()
    folder = f"uniforce_shear_demo_{args.object}_{args.marker_texture_name}_{str(args.scale_mm).replace('.', 'p')}mm"
    return (Path("/home/suhang/datasets") / folder).resolve()


def ensure_empty_dir(path: Path, label: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"{label} exists but is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"{label} must not already contain data: {path}")
    path.mkdir(parents=True, exist_ok=True)


def cover_crop_to_target(image: Image.Image) -> Image.Image:
    converted = image.convert("L")
    return ImageOps.fit(
        converted,
        (TARGET_WIDTH, TARGET_HEIGHT),
        method=PIL_RESAMPLING.LANCZOS,
        centering=(0.5, 0.5),
    )


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, **JPEG_SAVE_KWARGS)


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def append_log_line(log_path: Path, indenter: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{timestamp_now()} - [indenter: {indenter}] - {message}\n")


def resolve_effective_depth(args: argparse.Namespace) -> DepthCapInfo:
    requested = float(args.depth_mm)
    margin = float(args.safety_margin_mm)
    if margin < 0:
        raise ValueError("--safety-margin-mm must be >= 0")

    min_bad_depth = DEFAULT_MIN_BAD_DEPTH_MM.get(str(args.object))
    if min_bad_depth is None:
        return DepthCapInfo(
            requested_depth_mm=requested,
            effective_depth_mm=requested,
            safe_depth_cap_mm=None,
            min_bad_depth_mm=None,
            safety_margin_mm=margin,
            capped=False,
        )

    safe_cap = float(min_bad_depth) - margin
    if safe_cap <= 0:
        raise ValueError(f"Computed non-positive safe depth cap for {args.object}: {safe_cap}")

    effective = min(requested, safe_cap)
    return DepthCapInfo(
        requested_depth_mm=requested,
        effective_depth_mm=effective,
        safe_depth_cap_mm=safe_cap,
        min_bad_depth_mm=float(min_bad_depth),
        safety_margin_mm=margin,
        capped=effective < requested - 1e-9,
    )


def build_demo_frames(args: argparse.Namespace, depth_info: DepthCapInfo) -> list[DemoFrame]:
    frames: list[DemoFrame] = []
    frame_counter = 0
    effective_depth_mm = float(depth_info.effective_depth_mm)

    def add_frame(phase_name: str, phase_index: int, phase_progress: float, x_mm: float, y_mm: float, depth_mm: float) -> None:
        nonlocal frame_counter
        frames.append(
            DemoFrame(
                frame_name=f"frame_{frame_counter:06d}",
                phase_name=phase_name,
                phase_index=phase_index,
                phase_progress=float(phase_progress),
                command_x_mm=float(x_mm),
                command_y_mm=float(y_mm),
                command_depth_mm=float(depth_mm),
            )
        )
        frame_counter += 1

    add_frame(
        phase_name="precontact",
        phase_index=0,
        phase_progress=0.0,
        x_mm=args.contact_x_mm,
        y_mm=args.contact_y_mm,
        depth_mm=0.0,
    )

    for idx in range(DEFAULT_PRESS_FRAMES):
        progress = float(idx + 1) / float(DEFAULT_PRESS_FRAMES)
        add_frame(
            phase_name="press",
            phase_index=idx,
            phase_progress=progress,
            x_mm=args.contact_x_mm,
            y_mm=args.contact_y_mm,
            depth_mm=effective_depth_mm * progress,
        )

    for idx in range(DEFAULT_SHEAR_X_FRAMES):
        progress = float(idx + 1) / float(DEFAULT_SHEAR_X_FRAMES)
        add_frame(
            phase_name="shear_x",
            phase_index=idx,
            phase_progress=progress,
            x_mm=args.contact_x_mm + args.shear_x_mm * progress,
            y_mm=args.contact_y_mm,
            depth_mm=effective_depth_mm,
        )

    for idx in range(DEFAULT_SHEAR_Y_FRAMES):
        progress = float(idx + 1) / float(DEFAULT_SHEAR_Y_FRAMES)
        add_frame(
            phase_name="shear_y",
            phase_index=idx,
            phase_progress=progress,
            x_mm=args.contact_x_mm + args.shear_x_mm,
            y_mm=args.contact_y_mm + args.shear_y_mm * progress,
            depth_mm=effective_depth_mm,
        )

    for idx in range(DEFAULT_RELEASE_FRAMES):
        progress = float(idx + 1) / float(DEFAULT_RELEASE_FRAMES)
        add_frame(
            phase_name="release",
            phase_index=idx,
            phase_progress=progress,
            x_mm=args.contact_x_mm + args.shear_x_mm,
            y_mm=args.contact_y_mm + args.shear_y_mm,
            depth_mm=effective_depth_mm * (1.0 - progress),
        )

    return frames


def write_schedule_json(path: Path, frames: Sequence[DemoFrame]) -> None:
    payload = {
        "frame_commands": [
            {
                "frame_name": frame.frame_name,
                "phase_name": frame.phase_name,
                "phase_index": frame.phase_index,
                "phase_progress": frame.phase_progress,
                "command_x_mm": frame.command_x_mm,
                "command_y_mm": frame.command_y_mm,
                "command_depth_mm": frame.command_depth_mm,
            }
            for frame in frames
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_temp_scripts(script_root: Path, *, render_device: str) -> tuple[Path, Path]:
    import generate_multiscale_dataset as legacy
    import generate_multiscale_dataset_sequence as seq

    script_root.mkdir(parents=True, exist_ok=True)
    physics_script = script_root / "tmp_shear_physics.py"
    open3d_script = script_root / "tmp_npz2stl.py"
    blender_script = script_root / "tmp_blender_multirender.py"

    physics_script.write_text(PHYSICS_TEMP_SCRIPT, encoding="utf-8")
    open3d_script.write_text(legacy.OPEN3D_TEMP_SCRIPT, encoding="utf-8")
    blender_script.write_text(
        seq.build_sequence_blender_script(render_device=render_device, render_gpu_backend="auto"),
        encoding="utf-8",
    )
    return open3d_script, blender_script


def copy_selected_texture(texture_name: str, output_dir: Path) -> Path:
    src = MARKER_PATTERN_DIR / f"{texture_name}.jpg"
    if not src.exists():
        raise FileNotFoundError(f"Missing marker texture: {src}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / src.name
    shutil.copy2(src, dst)
    return dst


def run_demo_physics(
    *,
    python_cmd: str,
    physics_script: Path,
    output_npz: Path,
    schedule_json: Path,
    args: argparse.Namespace,
) -> None:
    import generate_multiscale_dataset as legacy

    cmd = [
        python_cmd,
        str(physics_script),
        "--config",
        str(PARAMETERS_PATH),
        "--dataset",
        str(INDENTER_DATASET_DIR),
        "--object",
        str(args.object),
        "--particle",
        "100000",
        "--contact-x-mm",
        str(args.contact_x_mm),
        "--contact-y-mm",
        str(args.contact_y_mm),
        "--schedule-json",
        str(schedule_json),
        "--output-npz",
        str(output_npz),
        "--warmup-steps",
        str(DEFAULT_WARMUP_STEPS),
        "--steps-per-frame",
        str(DEFAULT_STEPS_PER_FRAME),
    ]
    if args.show_gui:
        cmd.append("--show-gui")
    legacy.run_cmd_checked(cmd, cwd=SCRIPT_DIR, timeout_sec=DEFAULT_RENDER_TIMEOUT_SEC, stage="shear_demo_physics")


def build_demo_manifest_payload(
    *,
    args: argparse.Namespace,
    depth_info: DepthCapInfo,
    frames: Sequence[DemoFrame],
    output_root: Path,
    dataset_root: Path,
    rendered_paths: Sequence[Path],
) -> dict[str, Any]:
    return {
        "demo_type": "z_press_xy_shear",
        "pair_mode": "duplicate_single_scale",
        "date_tag": DEFAULT_DATE_TAG,
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "object": args.object,
        "marker_texture_name": args.marker_texture_name,
        "scale_mm": float(args.scale_mm),
        "sensor_l_name": args.sensor_l_name,
        "sensor_r_name": args.sensor_r_name,
        "contact_x_mm": float(args.contact_x_mm),
        "contact_y_mm": float(args.contact_y_mm),
        "requested_depth_mm": float(depth_info.requested_depth_mm),
        "effective_depth_mm": float(depth_info.effective_depth_mm),
        "safe_depth_cap_mm": None if depth_info.safe_depth_cap_mm is None else float(depth_info.safe_depth_cap_mm),
        "min_bad_depth_mm": None if depth_info.min_bad_depth_mm is None else float(depth_info.min_bad_depth_mm),
        "safety_margin_mm": float(depth_info.safety_margin_mm),
        "depth_was_capped": bool(depth_info.capped),
        "shear_x_mm": float(args.shear_x_mm),
        "shear_y_mm": float(args.shear_y_mm),
        "warmup_steps": int(DEFAULT_WARMUP_STEPS),
        "steps_per_frame": int(DEFAULT_STEPS_PER_FRAME),
        "render_device": str(args.render_device),
        "render_samples": int(args.render_samples),
        "show_gui": bool(args.show_gui),
        "frame_count": int(len(frames)),
        "frames": [
            {
                "frame_name": frame.frame_name,
                "phase_name": frame.phase_name,
                "phase_index": int(frame.phase_index),
                "phase_progress": float(frame.phase_progress),
                "command_x_mm": float(frame.command_x_mm),
                "command_y_mm": float(frame.command_y_mm),
                "command_depth_mm": float(frame.command_depth_mm),
                "rendered_stage_file": str(rendered_paths[idx]),
                "export_left_relpath": str(Path("marker") / args.sensor_l_name / args.object / f"{idx:06d}.jpg"),
                "export_right_relpath": str(Path("marker") / args.sensor_r_name / args.object / f"{idx:06d}.jpg"),
            }
            for idx, frame in enumerate(frames)
        ],
    }


def export_uniforce_dataset(
    *,
    args: argparse.Namespace,
    depth_info: DepthCapInfo,
    frames: Sequence[DemoFrame],
    render_root: Path,
    output_root: Path,
) -> tuple[Path, list[Path]]:
    dataset_root = output_root / args.marker_texture_name / DEFAULT_DATE_TAG / f"{args.sensor_l_name}_{args.sensor_r_name}"
    dataset_root.mkdir(parents=True, exist_ok=True)
    log_path = dataset_root / "collection_log.txt"
    if log_path.exists():
        log_path.unlink()

    append_log_line(
        log_path,
        args.object,
        (
            f"Synthetic shear demo export started. Marker={args.marker_texture_name} "
            f"scale={args.scale_mm:g}mm requested_depth={depth_info.requested_depth_mm:g}mm "
            f"effective_depth={depth_info.effective_depth_mm:g}mm "
            f"shear_x={args.shear_x_mm:g}mm shear_y={args.shear_y_mm:g}mm"
        ),
    )
    if depth_info.safe_depth_cap_mm is not None:
        append_log_line(
            log_path,
            args.object,
            (
                f"Depth safety cap active. min_bad_depth={depth_info.min_bad_depth_mm:.6f}mm "
                f"safety_margin={depth_info.safety_margin_mm:.3f}mm "
                f"safe_cap={depth_info.safe_depth_cap_mm:.6f}mm capped={depth_info.capped}"
            ),
        )

    left_dir = dataset_root / "marker" / args.sensor_l_name / args.object
    right_dir = dataset_root / "marker" / args.sensor_r_name / args.object
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    rendered_paths: list[Path] = []
    for idx, frame in enumerate(frames):
        src = render_root / frame.frame_name / f"marker_{args.marker_texture_name}.jpg"
        if not src.exists():
            raise FileNotFoundError(f"Missing rendered marker for {frame.frame_name}: {src}")
        rendered_paths.append(src)
        with Image.open(src) as img:
            prepared = cover_crop_to_target(img)
        filename = f"{idx:06d}{RAW_IMAGE_EXT}"
        save_image(prepared, left_dir / filename)
        save_image(prepared, right_dir / filename)

    append_log_line(log_path, args.object, f"Synthetic shear demo export completed. Final frames: {len(frames)}")
    return dataset_root, rendered_paths


def verify_export(dataset_root: Path, frame_count: int, *, sensor_l_name: str, sensor_r_name: str, object_name: str) -> None:
    left_dir = dataset_root / "marker" / sensor_l_name / object_name
    right_dir = dataset_root / "marker" / sensor_r_name / object_name
    left_files = sorted(left_dir.glob("*.jpg"))
    right_files = sorted(right_dir.glob("*.jpg"))
    if len(left_files) != frame_count or len(right_files) != frame_count:
        raise RuntimeError(
            f"Unexpected exported frame count: left={len(left_files)} right={len(right_files)} expected={frame_count}"
        )
    with Image.open(left_files[0]) as probe:
        if probe.size != (TARGET_WIDTH, TARGET_HEIGHT):
            raise RuntimeError(f"Unexpected export size: {probe.size}")


def main() -> None:
    args = parse_args()
    if args.scale_mm <= 0:
        raise ValueError("--scale-mm must be > 0")
    if args.depth_mm <= 0:
        raise ValueError("--depth-mm must be > 0")
    if args.safety_margin_mm < 0:
        raise ValueError("--safety-margin-mm must be >= 0")
    if args.render_samples <= 0:
        raise ValueError("--render-samples must be > 0")

    output_root = resolve_output_root(args)
    ensure_empty_dir(output_root, "Output root")

    intermediates_root = output_root / "_intermediates"
    intermediates_root.mkdir(parents=True, exist_ok=True)
    script_root = intermediates_root / "scripts"
    render_root = intermediates_root / "render"
    stl_root = intermediates_root / "stl"
    texture_root = intermediates_root / "textures"
    physics_root = intermediates_root / "physics"
    physics_root.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)
    stl_root.mkdir(parents=True, exist_ok=True)

    depth_info = resolve_effective_depth(args)
    if depth_info.safe_depth_cap_mm is not None:
        print(
            f"Depth safety for {args.object}: requested={depth_info.requested_depth_mm:.6f}mm "
            f"safe_cap={depth_info.safe_depth_cap_mm:.6f}mm effective={depth_info.effective_depth_mm:.6f}mm "
            f"capped={depth_info.capped}"
        )

    frames = build_demo_frames(args, depth_info)
    schedule_json = physics_root / "shear_schedule.json"
    write_schedule_json(schedule_json, frames)

    open3d_script, blender_script = write_temp_scripts(script_root, render_device=args.render_device)
    copy_selected_texture(args.marker_texture_name, texture_root)

    npz_path = physics_root / f"{args.object}_shear_demo.npz"
    run_demo_physics(
        python_cmd=sys.executable,
        physics_script=script_root / "tmp_shear_physics.py",
        output_npz=npz_path,
        schedule_json=schedule_json,
        args=args,
    )

    import generate_multiscale_dataset as legacy
    import generate_multiscale_dataset_sequence as seq

    fixed_distance_m = seq.compute_fixed_camera_distance_m(
        reference_scale_mm=float(args.scale_mm),
        base_fov_deg=float(legacy.DEFAULT_FOV_DEG),
        distance_safety=float(legacy.DEFAULT_DISTANCE_SAFETY),
    )

    for idx, frame in enumerate(frames):
        stl_path = stl_root / f"{frame.frame_name}.stl"
        mesh_cmd = [
            sys.executable,
            str(open3d_script),
            "--input-npz",
            str(npz_path),
            "--output-stl",
            str(stl_path),
            "--frame-index",
            str(idx),
        ]
        legacy.run_cmd_checked(
            mesh_cmd,
            cwd=SCRIPT_DIR,
            timeout_sec=DEFAULT_MESHING_TIMEOUT_SEC,
            stage=f"shear_demo_npz_to_stl[{frame.frame_name}]",
        )

        frame_render_dir = render_root / frame.frame_name
        frame_render_dir.mkdir(parents=True, exist_ok=True)
        render_cmd = [
            "blender",
            "-b",
            "--python",
            str(blender_script),
            "--",
            "--stl",
            str(stl_path),
            "--textures-dir",
            str(texture_root),
            "--output-dir",
            str(frame_render_dir),
            "--scale-mm",
            str(args.scale_mm),
            "--camera-mode",
            DEFAULT_CAMERA_MODE,
            "--base-fov-deg",
            str(legacy.DEFAULT_FOV_DEG),
            "--fixed-distance-m",
            str(fixed_distance_m),
            "--distance-safety",
            str(legacy.DEFAULT_DISTANCE_SAFETY),
            "--uv-mode",
            "unwrap_genforce",
            "--uv-inset-ratio",
            "0.01",
            "--render-samples",
            str(args.render_samples),
            "--render-device",
            str(args.render_device),
            "--render-gpu-backend",
            "auto",
        ]
        legacy.run_cmd_checked(
            render_cmd,
            cwd=SCRIPT_DIR,
            timeout_sec=DEFAULT_RENDER_TIMEOUT_SEC,
            stage=f"shear_demo_render[{frame.frame_name}]",
        )

    dataset_root, rendered_paths = export_uniforce_dataset(
        args=args,
        depth_info=depth_info,
        frames=frames,
        render_root=render_root,
        output_root=output_root,
    )

    demo_manifest = build_demo_manifest_payload(
        args=args,
        depth_info=depth_info,
        frames=frames,
        output_root=output_root,
        dataset_root=dataset_root,
        rendered_paths=rendered_paths,
    )
    (dataset_root / "demo_manifest.json").write_text(json.dumps(demo_manifest, indent=2), encoding="utf-8")

    verify_export(
        dataset_root,
        len(frames),
        sensor_l_name=args.sensor_l_name,
        sensor_r_name=args.sensor_r_name,
        object_name=args.object,
    )

    if not args.keep_intermediates and intermediates_root.exists():
        shutil.rmtree(intermediates_root)

    print(f"Shear demo export complete: {dataset_root}")


if __name__ == "__main__":
    main()
