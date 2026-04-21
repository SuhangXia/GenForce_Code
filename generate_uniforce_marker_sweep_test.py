#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE_UNIFORCE_SCRIPT = SCRIPT_DIR / "generate_uniforce.py"
ANALYZE_MARKER_COUNT_SCRIPT = SCRIPT_DIR / "scripts" / "analyze_marker_count_limits.py"
DEFAULT_DATASETS_ROOT = Path("/home/suhang/datasets")
DEFAULT_DATE_TAG = dt.date.today().strftime("%Y%m%d")
DEFAULT_OUTPUT_BASENAME = "uniforce_all12_array1_depth2p2_test"
DEFAULT_SENSOR_L_NAME = "digit"
DEFAULT_SENSOR_R_NAME = "gelsight"
DEFAULT_SCALE_L_MM = 15.0
DEFAULT_SCALE_R_MM = 17.0
DEFAULT_EPISODES_PER_INDENTER = 1
DEFAULT_SEED = 42
DEFAULT_WORKERS = 8
DEFAULT_MIN_AREA = 20
DEFAULT_MAX_PHYSICS_WORKERS = 1
DEFAULT_MAX_MESHING_WORKERS = 1
DEFAULT_MAX_RENDER_WORKERS = 2
TARGET_MARKER_NAME = "Array1"
TARGET_MARKER_FILENAME = "marker_Array1.jpg"
TARGET_DEPTH_MAX_MM = 2.2
TARGET_INDENTERS = [
    "cone",
    "cylinder",
    "cylinder_sh",
    "cylinder_si",
    "dotin",
    "dots",
    "hemisphere",
    "line",
    "moon",
    "prism",
    "random",
    "sphere",
]
EXPECTED_COUNTS_JSON = json.dumps({TARGET_MARKER_FILENAME: 225}, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single-marker uniforce test dataset across the 12 train indenters, "
            "keep the intermediate stage dataset, and run marker-count QA on that stage tree."
        )
    )
    parser.add_argument("--output-root", type=Path, default=None, help="Final export root under /home/suhang/datasets.")
    parser.add_argument("--stage-root", type=Path, default=None, help="Optional override for the kept _stage directory.")
    parser.add_argument("--date-tag", type=str, default=DEFAULT_DATE_TAG)
    parser.add_argument("--episodes-per-indenter", type=int, default=DEFAULT_EPISODES_PER_INDENTER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale-l-mm", type=float, default=DEFAULT_SCALE_L_MM)
    parser.add_argument("--scale-r-mm", type=float, default=DEFAULT_SCALE_R_MM)
    parser.add_argument("--sensor-l-name", type=str, default=DEFAULT_SENSOR_L_NAME)
    parser.add_argument("--sensor-r-name", type=str, default=DEFAULT_SENSOR_R_NAME)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="QA worker count.")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA, help="QA minimum component area.")
    parser.add_argument(
        "--max-physics-workers",
        type=int,
        default=DEFAULT_MAX_PHYSICS_WORKERS,
        help="Physics worker count forwarded to generate_multiscale_dataset_sequence.py.",
    )
    parser.add_argument(
        "--max-meshing-workers",
        type=int,
        default=DEFAULT_MAX_MESHING_WORKERS,
        help="Meshing worker count forwarded to generate_multiscale_dataset_sequence.py.",
    )
    parser.add_argument(
        "--max-render-workers",
        type=int,
        default=DEFAULT_MAX_RENDER_WORKERS,
        help="Render worker count forwarded to generate_multiscale_dataset_sequence.py.",
    )
    return parser.parse_args()


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.expanduser().resolve()
    return (DEFAULT_DATASETS_ROOT / f"{DEFAULT_OUTPUT_BASENAME}_{args.date_tag}").resolve()


def resolve_stage_root(args: argparse.Namespace, output_root: Path) -> Path:
    if args.stage_root is not None:
        return args.stage_root.expanduser().resolve()
    return (output_root / "_stage").resolve()


def resolve_qa_output_dir(output_root: Path) -> Path:
    return (output_root / "_marker_count_qc_stage").resolve()


def ensure_scripts_exist() -> None:
    for path in (GENERATE_UNIFORCE_SCRIPT, ANALYZE_MARKER_COUNT_SCRIPT):
        if not path.exists():
            raise FileNotFoundError(f"Missing required script: {path}")


def ensure_runtime_dependencies() -> None:
    if importlib.util.find_spec("taichi") is None:
        raise RuntimeError(
            "Missing Python dependency 'taichi'. "
            "Activate the environment that can run sim/deformation/gel_press.py before generating this dataset."
        )


def run_command(cmd: list[str], *, cwd: Path) -> None:
    print("Running:")
    print("  " + shlex.join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def build_generate_command(args: argparse.Namespace, output_root: Path, stage_root: Path) -> list[str]:
    return [
        sys.executable,
        str(GENERATE_UNIFORCE_SCRIPT),
        "--output-root",
        str(output_root),
        "--stage-root",
        str(stage_root),
        "--sensor-l-name",
        str(args.sensor_l_name),
        "--sensor-r-name",
        str(args.sensor_r_name),
        "--scale-l-mm",
        str(float(args.scale_l_mm)),
        "--scale-r-mm",
        str(float(args.scale_r_mm)),
        "--date-tag",
        str(args.date_tag),
        "--objects",
        *TARGET_INDENTERS,
        "--episodes-per-indenter",
        str(int(args.episodes_per_indenter)),
        "--seed",
        str(int(args.seed)),
        "--keep-stage",
        "--genforce-args",
        "--max-physics-workers",
        str(int(args.max_physics_workers)),
        "--max-meshing-workers",
        str(int(args.max_meshing_workers)),
        "--max-render-workers",
        str(int(args.max_render_workers)),
        "--depth-max",
        str(float(TARGET_DEPTH_MAX_MM)),
        "--marker-texture-names",
        TARGET_MARKER_NAME,
        "--train-indenters",
        *TARGET_INDENTERS,
        "--val-indenters",
        "--test-indenters",
    ]


def build_qa_command(args: argparse.Namespace, stage_root: Path, qa_output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ANALYZE_MARKER_COUNT_SCRIPT),
        "--image-root",
        str(stage_root),
        "--metadata-root",
        str(stage_root),
        "--output-dir",
        str(qa_output_dir),
        "--workers",
        str(int(args.workers)),
        "--min-area",
        str(int(args.min_area)),
        "--expected-counts-json",
        EXPECTED_COUNTS_JSON,
    ]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def verify_export_layout(output_root: Path, args: argparse.Namespace) -> Path:
    export_root = output_root / TARGET_MARKER_NAME / str(args.date_tag) / f"{args.sensor_l_name}_{args.sensor_r_name}"
    log_path = export_root / "collection_log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing export log: {log_path}")
    return export_root


def verify_stage_dataset(stage_root: Path) -> None:
    manifest_path = stage_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing stage manifest: {manifest_path}")

    manifest = load_json(manifest_path)
    selected = [str(name) for name in manifest.get("marker_textures_selected", [])]
    if selected != [TARGET_MARKER_NAME]:
        raise ValueError(f"Unexpected marker_textures_selected in {manifest_path}: {selected}")

    metadata_files = sorted(stage_root.glob("episode_*/metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No episode metadata found under stage root: {stage_root}")

    seen_indenters: set[str] = set()
    for metadata_path in metadata_files:
        episode_meta = load_json(metadata_path)
        seen_indenters.add(str(episode_meta.get("indenter", "")))
    if seen_indenters != set(TARGET_INDENTERS):
        raise ValueError(
            "Stage dataset indenter set mismatch: "
            f"expected={sorted(TARGET_INDENTERS)} actual={sorted(seen_indenters)}"
        )

    sample_path = stage_root / "episode_000000"
    if not sample_path.exists():
        sample_path = metadata_files[0].parent

    scale_dirs = sorted(sample_path.glob("scale_*"))
    if not scale_dirs:
        raise FileNotFoundError(f"No scale directories found under {sample_path}")

    sample_image = scale_dirs[0] / "frame_000000" / TARGET_MARKER_FILENAME
    if not sample_image.exists():
        raise FileNotFoundError(f"Missing sample marker image: {sample_image}")


def verify_qa_outputs(qa_output_dir: Path) -> tuple[Path, Path, Path]:
    image_csv = qa_output_dir / "image_level_counts.csv"
    marker_csv = qa_output_dir / "limit_by_indenter_marker.csv"
    indenter_csv = qa_output_dir / "limit_by_indenter.csv"
    for path in (image_csv, marker_csv, indenter_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing QA artifact: {path}")
    return image_csv, marker_csv, indenter_csv


def print_limit_summary(indenter_csv: Path) -> None:
    rows: list[dict[str, str]] = []
    with indenter_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({str(k): str(v) for k, v in row.items()})

    if len(rows) != len(TARGET_INDENTERS):
        raise ValueError(
            f"Expected {len(TARGET_INDENTERS)} indenter rows in {indenter_csv}, found {len(rows)}"
        )

    def sort_key(row: dict[str, str]) -> tuple[int, float, str]:
        raw = str(row.get("min_bad_depth_mm", "")).strip()
        if not raw:
            return (1, float("inf"), str(row.get("indenter_name", "")))
        return (0, float(raw), str(row.get("indenter_name", "")))

    print("limit_by_indenter.csv summary:")
    for row in sorted(rows, key=sort_key):
        indenter_name = row.get("indenter_name", "")
        min_bad = row.get("min_bad_depth_mm", "") or "NA"
        max_safe = row.get("max_safe_depth_mm", "") or "NA"
        worst_marker = row.get("worst_marker_name", "") or "-"
        bad = row.get("bad_count", "0")
        total = row.get("total_images", "0")
        non_monotonic = row.get("non_monotonic_flag", "")
        print(
            "  "
            f"{indenter_name}: min_bad_depth_mm={min_bad} max_safe_depth_mm={max_safe} "
            f"worst_marker={worst_marker} bad={bad}/{total} non_monotonic={non_monotonic}"
        )


def main() -> None:
    args = parse_args()
    if args.episodes_per_indenter <= 0:
        raise ValueError("--episodes-per-indenter must be > 0")
    if args.scale_l_mm <= 0 or args.scale_r_mm <= 0:
        raise ValueError("--scale-l-mm and --scale-r-mm must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.min_area <= 0:
        raise ValueError("--min-area must be > 0")
    if args.max_physics_workers <= 0:
        raise ValueError("--max-physics-workers must be > 0")
    if args.max_meshing_workers <= 0:
        raise ValueError("--max-meshing-workers must be > 0")
    if args.max_render_workers <= 0:
        raise ValueError("--max-render-workers must be > 0")

    ensure_scripts_exist()
    ensure_runtime_dependencies()

    output_root = resolve_output_root(args)
    stage_root = resolve_stage_root(args, output_root)
    qa_output_dir = resolve_qa_output_dir(output_root)

    try:
        run_command(build_generate_command(args, output_root, stage_root), cwd=SCRIPT_DIR)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Dataset generation failed. Partial outputs were preserved at output_root={output_root} stage_root={stage_root}"
        ) from exc

    export_root = verify_export_layout(output_root, args)
    verify_stage_dataset(stage_root)

    try:
        run_command(build_qa_command(args, stage_root, qa_output_dir), cwd=SCRIPT_DIR)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Marker QA failed. Generated data were preserved at output_root={output_root} stage_root={stage_root} "
            f"qa_output_dir={qa_output_dir}"
        ) from exc

    _image_csv, _marker_csv, indenter_csv = verify_qa_outputs(qa_output_dir)

    print(f"Export root: {export_root}")
    print(f"Stage root: {stage_root}")
    print(f"QA output dir: {qa_output_dir}")
    print_limit_summary(indenter_csv)


if __name__ == "__main__":
    main()
