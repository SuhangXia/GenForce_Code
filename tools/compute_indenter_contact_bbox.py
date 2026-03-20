#!/usr/bin/env python3
"""
遍历所有压头，为每个压头计算一个 2D bbox（矩形或正方形），用于完全包住压头。

默认：从下往上只取 5mm 高度的点做 xy 投影再算 bbox（--bottom-height-mm 5），排除上方法兰。
可选 --full-footprint 用全点云（会含法兰）；--contact-face 用 z 最低一层点（与深度相关）。

坐标系说明:
  bbox 和图中坐标均为「压头局部坐标系（旋转后）」的 xy 平面，单位 mm。
  - 与 gel_press.py 中使用的旋转一致：parameters.yml 里 indenter.pose 的 R,P,Y
    （默认 P=-π 绕 Y 轴 180°），旋转后 z 轴朝下为接触方向。
  - 原点：压头 .npy 的几何中心未平移，仅做了旋转。
  - 用于仿真时，世界坐标 = (20+x_mm, 20+y_mm) + 本 bbox 的 (x,y)。

用法:
  python tools/compute_indenter_contact_bbox.py --indenter-dir sim/assets/indenters/input/npy_100000
  python tools/compute_indenter_contact_bbox.py ... --square --output indenters_bbox.json
  python tools/compute_indenter_contact_bbox.py ... --visualize --vis-dir indenter_bbox_vis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

# 与 gel_press.py 一致的旋转：用 pytransform3d，角度与 parameters.yml 一致
try:
    from pytransform3d import rotations
except ImportError:
    rotations = None


# 18 种压头（与 sim/marker/4_render.py 一致）
DEFAULT_INDENTER_NAMES = [
    "cone", "curface", "cylinder_sh", "cylinder_si", "cylinder",
    "dot_in", "dots", "hexagon", "line", "moon", "pacman", "prism",
    "random", "sphere_s", "sphere", "torus", "triangle", "wave",
]


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rotation_matrix_from_pose(pose: dict[str, float], degrees: bool = False) -> np.ndarray:
    """从 pose 的 R, P, Y 构建 3x3 旋转矩阵，与 gel_press.py 一致。"""
    if rotations is None:
        # 无 pytransform3d 时用 (0, -π, 0) 的简单实现：绕 Y 轴 180°，(x,y,z)->(-x,y,-z)
        R, P, Y = pose.get("R", 0), pose.get("P", -np.pi), pose.get("Y", 0)
        if degrees:
            P = np.radians(P)
        c, s = np.cos(P), np.sin(P)
        return np.array([
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ], dtype=np.float64)
    R = pose.get("R", 0)
    P = pose.get("P", -np.pi)
    Y = pose.get("Y", 0)
    return rotations.matrix_from_euler((R, P, Y), 0, 1, 2, degrees)


def get_full_footprint_xy(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """压头全点云旋转后投影到 xy 平面，返回 (N, 2)。会包含上方法兰等，通常偏大。"""
    points_rot = (rot @ points.T).T
    return points_rot[:, :2]


def get_bottom_slice_footprint_xy(
    points: np.ndarray,
    rot: np.ndarray,
    bottom_height_mm: float,
) -> np.ndarray:
    """
    从下往上只取压头底部一段（z_min 到 z_min + bottom_height_mm）的点，投影到 xy，返回 (N, 2)。
    用于排除上方法兰，只算接触端头的投影。旋转后 z 轴朝下，故 z 最小处为尖端。
    """
    if bottom_height_mm <= 0:
        raise ValueError("bottom_height_mm must be > 0")
    points_rot = (rot @ points.T).T
    z = points_rot[:, 2]
    z_min = float(np.nanmin(z))
    z_max_slice = z_min + bottom_height_mm
    mask = (z >= z_min) & (z <= z_max_slice)
    return points_rot[mask, :2]


def get_contact_points_xy(
    points: np.ndarray,
    rot: np.ndarray,
    z_fraction: float = 0.08,
) -> np.ndarray:
    """
    取压头点云中与硅胶接触的那一层的点，投影到 xy 平面，返回 (N, 2)。
    接触面定义为旋转后 z 最小的那一部分点（占 z 范围的 z_fraction）。与下压深度相关（球等会随深度变化）。
    """
    points_rot = (rot @ points.T).T
    z = points_rot[:, 2]
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    z_span = z_max - z_min
    if z_span <= 0:
        threshold = z_min
    else:
        threshold = z_min + z_fraction * z_span
    contact = points_rot[z <= threshold]
    return contact[:, :2]


def bbox_from_points_xy(xy: np.ndarray) -> tuple[float, float, float, float]:
    """(N,2) xy 点 → (x_min, y_min, x_max, y_max)。"""
    if xy.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    x_min = float(np.nanmin(xy[:, 0]))
    x_max = float(np.nanmax(xy[:, 0]))
    y_min = float(np.nanmin(xy[:, 1]))
    y_max = float(np.nanmax(xy[:, 1]))
    return x_min, y_min, x_max, y_max


def bbox_to_square(x_min: float, y_min: float, x_max: float, y_max: float) -> tuple[float, float, float, float]:
    """将矩形 bbox 扩展为能完全覆盖的正方形 (x_min, y_min, x_max, y_max)。"""
    w = x_max - x_min
    h = y_max - y_min
    side = max(w, h)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half = side / 2.0
    return cx - half, cy - half, cx + half, cy + half


def apply_bbox_margin(
    x_min: float, y_min: float, x_max: float, y_max: float,
    margin_mm: float,
) -> tuple[float, float, float, float]:
    """各边外扩 margin_mm，使 bbox 完全包住接触面。"""
    return x_min - margin_mm, y_min - margin_mm, x_max + margin_mm, y_max + margin_mm


def compute_indenter_bbox(
    npy_path: Path,
    rot: np.ndarray,
    bottom_height_mm: float | None = 5.0,
    use_full_footprint: bool = False,
    z_fraction: float = 0.08,
    square: bool = False,
    margin_mm: float = 0.0,
) -> dict[str, Any]:
    """
    对单个压头 .npy 计算 2D bbox。
    bottom_height_mm>0（默认 5）：从下往上只取该高度(mm)的点做 xy 投影，排除上方法兰。
    bottom_height_mm=None 且 use_full_footprint=True：用全点云（会含法兰）。
    use_full_footprint=False 且 bottom_height_mm=None：用 z_fraction 接触面。
    """
    points = np.load(npy_path).astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array in {npy_path}, got shape {points.shape}")
    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid):
        return {
            "bbox_mm": [0.0, 0.0, 0.0, 0.0],
            "width_mm": 0.0,
            "height_mm": 0.0,
            "square_side_mm": 0.0,
            "points_count": 0,
            "contact_points_count": 0,
            "error": "no finite points",
        }
    points = points[valid]

    if bottom_height_mm is not None and bottom_height_mm > 0:
        xy = get_bottom_slice_footprint_xy(points, rot, bottom_height_mm)
    elif use_full_footprint:
        xy = get_full_footprint_xy(points, rot)
    else:
        xy = get_contact_points_xy(points, rot, z_fraction=z_fraction)
    if xy.shape[0] < 3:
        n_pts = int(xy.shape[0])
        return {
            "bbox_mm": [0.0, 0.0, 0.0, 0.0],
            "width_mm": 0.0,
            "height_mm": 0.0,
            "square_side_mm": 0.0,
            "points_count": n_pts,
            "contact_points_count": n_pts,
            "error": "too few points",
        }

    x_min, y_min, x_max, y_max = bbox_from_points_xy(xy)
    if margin_mm > 0:
        x_min, y_min, x_max, y_max = apply_bbox_margin(x_min, y_min, x_max, y_max, margin_mm)
    width = x_max - x_min
    height = y_max - y_min

    if square:
        x_min, y_min, x_max, y_max = bbox_to_square(x_min, y_min, x_max, y_max)
        side = max(width, height)
        width = height = side

    n_pts = int(xy.shape[0])
    result = {
        "bbox_mm": [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)],
        "width_mm": round(width, 4),
        "height_mm": round(height, 4),
        "points_count": n_pts,
        "contact_points_count": n_pts,  # 兼容旧 JSON
    }
    if square:
        result["square_side_mm"] = round(max(result["width_mm"], result["height_mm"]), 4)
    return result


def draw_bbox_vis(
    indenter_dir: Path,
    names: list[str],
    results: dict[str, Any],
    rot: np.ndarray,
    vis_dir: Path,
    bottom_height_mm: float | None = 5.0,
    use_full_footprint: bool = False,
    z_fraction: float = 0.08,
    show_full_footprint: bool = True,
) -> None:
    """为每个压头画一张图：参与 bbox 的点 + 红色 bbox；可选灰色全点云。"""
    if not _HAS_MATPLOTLIB:
        print("Warning: matplotlib not found, skip --visualize")
        return
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        r = results.get(name)
        if not r or "error" in r:
            continue
        npy_path = indenter_dir / f"{name}.npy"
        if not npy_path.exists():
            continue
        points = np.load(npy_path).astype(np.float64)
        valid = np.isfinite(points).all(axis=1)
        if not np.any(valid):
            continue
        points = points[valid]
        points_rot = (rot @ points.T).T
        full_xy = points_rot[:, :2]
        x_min, y_min, x_max, y_max = r["bbox_mm"]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        if show_full_footprint:
            ax.scatter(
                full_xy[:, 0], full_xy[:, 1],
                s=0.3, c="lightgray", alpha=0.5, label="full indenter (xy)",
            )
        # 画参与 bbox 的点
        if bottom_height_mm is not None and bottom_height_mm > 0:
            slice_xy = get_bottom_slice_footprint_xy(points, rot, bottom_height_mm)
            ax.scatter(
                slice_xy[:, 0], slice_xy[:, 1],
                s=1.5, c="C0", alpha=0.9, label=f"bottom {bottom_height_mm}mm (used for bbox)",
            )
            title = f"{name}  bottom {bottom_height_mm}mm bbox (mm)"
        elif use_full_footprint:
            ax.scatter(
                full_xy[:, 0], full_xy[:, 1],
                s=0.5, c="C0", alpha=0.6, label="full (used for bbox)",
            )
            title = f"{name}  full footprint bbox (mm)"
        else:
            contact_xy = get_contact_points_xy(points, rot, z_fraction=z_fraction)
            ax.scatter(
                contact_xy[:, 0], contact_xy[:, 1],
                s=2, c="C0", alpha=0.8, label="contact face (z_fraction)",
            )
            title = f"{name}  contact-face bbox (mm)"
        rect_x = [x_min, x_max, x_max, x_min, x_min]
        rect_y = [y_min, y_min, y_max, y_max, y_min]
        ax.plot(rect_x, rect_y, "r-", linewidth=2, label="bbox")
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        out_file = vis_dir / f"{name}.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_file}")

    print(f"Visualization saved under {vis_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算每个压头与硅胶接触面的 2D 物理 bbox（矩形或正方形）",
    )
    parser.add_argument(
        "--indenter-dir",
        type=Path,
        default=Path("sim/assets/indenters/input/npy_100000"),
        help="压头 .npy 所在目录（例如 npy_100000）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("sim/parameters.yml"),
        help="仿真参数 yaml，用于读取 indenter pose",
    )
    parser.add_argument(
        "--bottom-height-mm",
        type=float,
        default=5.0,
        metavar="MM",
        help="从下往上只取该高度(mm)的点做 xy 投影算 bbox，排除上方法兰（默认 5）",
    )
    parser.add_argument(
        "--full-footprint",
        action="store_true",
        help="用压头全点云 xy 投影（会含法兰）；与 --bottom-height-mm 二选一",
    )
    parser.add_argument(
        "--contact-face",
        action="store_true",
        help="用旋转后 z 最低的一层点（z_fraction）算 bbox；与 --bottom-height-mm / --full-footprint 二选一",
    )
    parser.add_argument(
        "--z-fraction",
        type=float,
        default=0.08,
        help="仅 --contact-face 时有效：接触面取 z 最低的该比例点（默认 0.08）",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="输出能完全覆盖接触面的正方形 bbox（边长取宽高较大值）",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=0.0,
        help="bbox 四边外扩的边距 (mm)，便于完全包住接触面",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="结果写入 JSON 文件；不指定则只打印",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="只处理这些压头名；不指定则处理目录下所有 .npy",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="为每个压头生成一张图（接触点+bbox），便于肉眼确认 bbox 是否包住压头",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=Path("indenter_bbox_vis"),
        help="可视化图片保存目录（默认 indenter_bbox_vis）",
    )
    parser.add_argument(
        "--no-full-footprint",
        action="store_true",
        help="可视化时不画压头全点云投影，只画接触点与 bbox",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    indenter_dir = args.indenter_dir if args.indenter_dir.is_absolute() else repo_root / args.indenter_dir
    config_path = args.config if args.config.is_absolute() else repo_root / args.config

    if not indenter_dir.exists():
        raise FileNotFoundError(f"压头目录不存在: {indenter_dir}")

    config = load_config(config_path)
    pose = config.get("indenter", {}).get("pose", {"R": 0, "P": -np.pi, "Y": 0})
    # parameters.yml 里 P 为 -3.14，实为弧度；与 gel_press 一致用弧度
    rot = rotation_matrix_from_pose(pose, degrees=False)

    if args.objects:
        names = sorted(set(args.objects))
    else:
        names = sorted(p.stem for p in indenter_dir.glob("*.npy") if p.is_file())
    if not names:
        raise FileNotFoundError(f"在 {indenter_dir} 下未找到任何 .npy 压头文件")

    # 三种模式：底部一段 / 全点云 / 接触面
    if args.contact_face:
        bottom_height_mm = None
        use_full_footprint = False
    elif args.full_footprint:
        bottom_height_mm = None
        use_full_footprint = True
    else:
        bottom_height_mm = float(args.bottom_height_mm)
        use_full_footprint = False
        if bottom_height_mm <= 0:
            raise ValueError("--bottom-height-mm must be > 0 (e.g. 5)")

    results = {}
    for name in names:
        npy_path = indenter_dir / f"{name}.npy"
        if not npy_path.exists():
            results[name] = {"error": f"file not found: {npy_path}"}
            continue
        try:
            results[name] = compute_indenter_bbox(
                npy_path,
                rot,
                bottom_height_mm=bottom_height_mm,
                use_full_footprint=use_full_footprint,
                z_fraction=args.z_fraction,
                square=args.square,
                margin_mm=args.margin_mm,
            )
        except Exception as e:
            results[name] = {"error": str(e)}

    # 打印摘要
    if args.contact_face:
        mode = "contact-face"
    elif args.full_footprint:
        mode = "full footprint"
    else:
        mode = f"bottom slice {args.bottom_height_mm}mm"
    print(f"Indenter 2D bbox (mm) [mode={mode}]:")
    print("-" * 60)
    for name in names:
        r = results[name]
        if "error" in r and len(r) == 1:
            print(f"  {name}: ERROR {r['error']}")
        else:
            b = r["bbox_mm"]
            print(f"  {name}: bbox=[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}] "
                  f"W={r['width_mm']:.3f} H={r['height_mm']:.3f} mm  n_points={r['points_count']}")
    print("-" * 60)

    if args.output:
        out_path = args.output if args.output.is_absolute() else repo_root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "indenter_dir": str(indenter_dir),
            "config": str(config_path),
            "bottom_height_mm": args.bottom_height_mm if not (args.contact_face or args.full_footprint) else None,
            "use_full_footprint": args.full_footprint,
            "z_fraction": args.z_fraction,
            "square_bbox": args.square,
            "margin_mm": args.margin_mm,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "indenters": results}, f, indent=2, ensure_ascii=False)
        print(f"Results written to {out_path}")

    if args.visualize:
        vis_dir = args.vis_dir if args.vis_dir.is_absolute() else repo_root / args.vis_dir
        draw_bbox_vis(
            indenter_dir,
            names,
            results,
            rot,
            vis_dir,
            bottom_height_mm=bottom_height_mm,
            use_full_footprint=use_full_footprint,
            z_fraction=args.z_fraction,
            show_full_footprint=not args.no_full_footprint,
        )


if __name__ == "__main__":
    main()
