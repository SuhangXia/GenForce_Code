#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Iterable

import cv2

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class _TqdmFallback:  # pragma: no cover
        def __init__(self, iterable, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def update(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return _TqdmFallback(iterable, *args, **kwargs)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_ROOT = Path(
    "/home/suhang/datasets/usa_static_v1_large_run_blackborder_3px/scheme_s_lite_adapter_train_seq"
)
DEFAULT_METADATA_ROOT = Path("/home/suhang/datasets/usa_static_v1_large_run/scheme_s_lite_adapter_train_seq")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "marker_count_qc"
DEFAULT_EXPECTED_COUNTS = {
    "marker_Array1.jpg": 225,
    "marker_Array2.jpg": 100,
    "marker_Array4.jpg": 98,
    "marker_Circle1.jpg": 319,
    "marker_Diamond3.jpg": 80,
    "marker_Diamond4.jpg": 80,
}
VALID_STATUSES = ("ok", "count_mismatch", "missing", "read_error")


@dataclass(frozen=True)
class FrameSource:
    episode_id: int
    episode_dir: str
    indenter_name: str
    scale_key: str
    scale_mm: float | None
    scale_requested_mm: float | None
    phase_name: str
    phase_index: int
    frame_name: str
    frame_actual_max_down_mm: float | None
    rendered_markers: tuple[str, ...]


@dataclass(frozen=True)
class WorkItem:
    episode_id: int
    episode_dir: str
    indenter_name: str
    scale_key: str
    scale_mm: float | None
    scale_requested_mm: float | None
    phase_name: str
    phase_index: int
    frame_name: str
    frame_actual_max_down_mm: float | None
    marker_name: str
    image_relpath: str


@dataclass(frozen=True)
class ImageResult:
    episode_id: int
    episode_dir: str
    indenter_name: str
    scale_key: str
    scale_mm: float | None
    scale_requested_mm: float | None
    phase_name: str
    phase_index: int
    frame_name: str
    frame_actual_max_down_mm: float | None
    marker_name: str
    image_relpath: str
    expected_count: int
    observed_count: int | None
    status: str
    error: str


@dataclass
class SummaryAccumulator:
    expected_count: int | None = None
    total_images: int = 0
    ok_count: int = 0
    count_mismatch_count: int = 0
    missing_count: int = 0
    read_error_count: int = 0
    max_safe_depth_mm: float | None = None
    min_bad_depth_mm: float | None = None
    first_bad_depth_mm: float | None = None
    first_bad_status: str = ""
    first_bad_episode_dir: str = ""
    first_bad_scale_key: str = ""
    first_bad_frame_name: str = ""
    first_bad_phase_name: str = ""
    first_bad_image_relpath: str = ""
    first_bad_observed_count: int | None = None
    first_bad_expected_count: int | None = None

    @property
    def bad_count(self) -> int:
        return self.count_mismatch_count + self.missing_count + self.read_error_count

    @property
    def non_monotonic_flag(self) -> bool:
        return (
            self.min_bad_depth_mm is not None
            and self.max_safe_depth_mm is not None
            and self.min_bad_depth_mm <= self.max_safe_depth_mm
        )

    def update(self, result: ImageResult) -> None:
        self.total_images += 1
        if self.expected_count is None:
            self.expected_count = result.expected_count

        if result.status == "ok":
            self.ok_count += 1
            if result.frame_actual_max_down_mm is not None:
                if self.max_safe_depth_mm is None or result.frame_actual_max_down_mm > self.max_safe_depth_mm:
                    self.max_safe_depth_mm = result.frame_actual_max_down_mm
            return

        if result.status == "count_mismatch":
            self.count_mismatch_count += 1
        elif result.status == "missing":
            self.missing_count += 1
        elif result.status == "read_error":
            self.read_error_count += 1
        else:
            raise ValueError(f"Unsupported status: {result.status}")

        if result.frame_actual_max_down_mm is not None:
            if self.min_bad_depth_mm is None or result.frame_actual_max_down_mm < self.min_bad_depth_mm:
                self.min_bad_depth_mm = result.frame_actual_max_down_mm

        if self._should_replace_first_bad(result):
            self.first_bad_depth_mm = result.frame_actual_max_down_mm
            self.first_bad_status = result.status
            self.first_bad_episode_dir = result.episode_dir
            self.first_bad_scale_key = result.scale_key
            self.first_bad_frame_name = result.frame_name
            self.first_bad_phase_name = result.phase_name
            self.first_bad_image_relpath = result.image_relpath
            self.first_bad_observed_count = result.observed_count
            self.first_bad_expected_count = result.expected_count

    def _should_replace_first_bad(self, result: ImageResult) -> bool:
        if not self.first_bad_status:
            return True
        current_key = _first_bad_sort_key(self.first_bad_depth_mm, self.first_bad_image_relpath)
        candidate_key = _first_bad_sort_key(result.frame_actual_max_down_mm, result.image_relpath)
        return candidate_key < current_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan marker images under the black-border scheme_s_lite_adapter_train_seq dataset, "
            "count rendered markers in each image, and estimate per-indenter depth limits."
        )
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Black-border dataset root.")
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
        help="Original dataset root that contains metadata.json and sequence_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV reports will be written.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads for image counting.")
    parser.add_argument("--min-area", type=int, default=20, help="Minimum connected-component area to keep.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--expected-counts-json",
        default="",
        help="Optional JSON string or JSON file path that overrides the default expected marker counts.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def format_optional_int(value: int | None) -> str:
    return "" if value is None else str(int(value))


def _first_bad_sort_key(depth_mm: float | None, relpath: str) -> tuple[int, float, str]:
    if depth_mm is None:
        return (1, float("inf"), relpath)
    return (0, float(depth_mm), relpath)


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_expected_counts(raw: str, discovered_markers: set[str]) -> dict[str, int]:
    if not raw:
        source = DEFAULT_EXPECTED_COUNTS
    else:
        maybe_path = Path(raw)
        if maybe_path.exists():
            source = load_json_file(maybe_path)
        else:
            source = json.loads(raw)

    if not isinstance(source, dict):
        raise ValueError("Expected marker counts must be a JSON object.")

    counts: dict[str, int] = {}
    for key, value in source.items():
        marker_name = str(key).strip()
        if not marker_name:
            raise ValueError("Encountered an empty marker name in expected counts.")
        marker_count = int(value)
        if marker_count <= 0:
            raise ValueError(f"Expected marker count must be > 0 for {marker_name}: {marker_count}")
        counts[marker_name] = marker_count

    missing_markers = sorted(discovered_markers.difference(counts))
    if missing_markers:
        raise ValueError(f"Expected counts are missing markers: {missing_markers}")
    return counts


def build_frame_sources(metadata_root: Path, max_episodes: int | None) -> tuple[list[FrameSource], set[str]]:
    metadata_files = sorted(metadata_root.glob("episode_*/metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No episode metadata found under {metadata_root}")
    if max_episodes is not None:
        metadata_files = metadata_files[:max_episodes]

    frame_sources: list[FrameSource] = []
    discovered_markers: set[str] = set()

    for metadata_path in tqdm(metadata_files, desc="Load metadata", unit="episode"):
        episode_meta = load_json_file(metadata_path)
        if not isinstance(episode_meta, dict):
            raise ValueError(f"Episode metadata is not a JSON object: {metadata_path}")

        episode_dir = metadata_path.parent.name
        episode_id = int(episode_meta.get("episode_id", episode_dir.split("_")[-1]))
        indenter_name = str(episode_meta.get("indenter", "")).strip()
        scales = episode_meta.get("scales", {})
        if not isinstance(scales, dict) or not scales:
            raise ValueError(f"Missing or invalid scales entry in {metadata_path}")

        for scale_key in sorted(scales):
            scale_meta = scales[scale_key]
            if not isinstance(scale_meta, dict):
                raise ValueError(f"Scale metadata for {scale_key} is not a JSON object in {metadata_path}")

            seq_rel = scale_meta.get("sequence_metadata", f"{scale_key}/sequence_metadata.json")
            seq_path = metadata_path.parent / str(seq_rel)
            if not seq_path.exists():
                raise FileNotFoundError(f"Missing sequence metadata: {seq_path}")
            sequence_meta = load_json_file(seq_path)
            if not isinstance(sequence_meta, dict):
                raise ValueError(f"Sequence metadata is not a JSON object: {seq_path}")

            rendered_defaults = tuple(str(v) for v in sequence_meta.get("marker_files_selected", []))
            if not rendered_defaults:
                raise ValueError(f"No marker_files_selected recorded in {seq_path}")
            discovered_markers.update(rendered_defaults)

            frames = sequence_meta.get("frames", [])
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"Sequence metadata has no frames: {seq_path}")

            for frame in frames:
                if not isinstance(frame, dict):
                    raise ValueError(f"Frame entry is not a JSON object in {seq_path}")

                frame_name = str(frame.get("frame_name", "")).strip()
                if not frame_name:
                    raise ValueError(f"Frame is missing frame_name in {seq_path}")

                rendered_markers = tuple(str(v) for v in frame.get("rendered_markers", rendered_defaults))
                if not rendered_markers:
                    rendered_markers = rendered_defaults
                discovered_markers.update(rendered_markers)

                frame_sources.append(
                    FrameSource(
                        episode_id=episode_id,
                        episode_dir=episode_dir,
                        indenter_name=indenter_name or str(sequence_meta.get("indenter", "")).strip(),
                        scale_key=str(sequence_meta.get("scale_key", scale_key)),
                        scale_mm=safe_float(sequence_meta.get("scale_mm")),
                        scale_requested_mm=safe_float(sequence_meta.get("scale_requested_mm")),
                        phase_name=str(frame.get("phase_name", "")),
                        phase_index=int(frame.get("phase_index", -1)),
                        frame_name=frame_name,
                        frame_actual_max_down_mm=safe_float(frame.get("frame_actual_max_down_mm")),
                        rendered_markers=rendered_markers,
                    )
                )

    if not frame_sources:
        raise RuntimeError(f"No frame sources were discovered under {metadata_root}")
    return frame_sources, discovered_markers


def iter_work_items(frame_sources: Iterable[FrameSource]) -> Iterable[WorkItem]:
    for frame in frame_sources:
        frame_dir = Path(frame.episode_dir) / frame.scale_key / frame.frame_name
        for marker_name in frame.rendered_markers:
            yield WorkItem(
                episode_id=frame.episode_id,
                episode_dir=frame.episode_dir,
                indenter_name=frame.indenter_name,
                scale_key=frame.scale_key,
                scale_mm=frame.scale_mm,
                scale_requested_mm=frame.scale_requested_mm,
                phase_name=frame.phase_name,
                phase_index=frame.phase_index,
                frame_name=frame.frame_name,
                frame_actual_max_down_mm=frame.frame_actual_max_down_mm,
                marker_name=marker_name,
                image_relpath=(frame_dir / marker_name).as_posix(),
            )


def count_markers(image: Any, min_area: int) -> int:
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats((binary > 0).astype("uint8"), 8)

    height, width = image.shape[:2]
    count = 0
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if int(area) < min_area:
            continue
        if int(x) == 0 or int(y) == 0 or int(x + w) >= width or int(y + h) >= height:
            continue
        count += 1
    return count


def process_image(
    item: WorkItem,
    image_root: Path,
    expected_counts: dict[str, int],
    min_area: int,
) -> ImageResult:
    expected_count = expected_counts[item.marker_name]
    image_path = image_root / item.image_relpath

    if not image_path.exists():
        return ImageResult(
            episode_id=item.episode_id,
            episode_dir=item.episode_dir,
            indenter_name=item.indenter_name,
            scale_key=item.scale_key,
            scale_mm=item.scale_mm,
            scale_requested_mm=item.scale_requested_mm,
            phase_name=item.phase_name,
            phase_index=item.phase_index,
            frame_name=item.frame_name,
            frame_actual_max_down_mm=item.frame_actual_max_down_mm,
            marker_name=item.marker_name,
            image_relpath=item.image_relpath,
            expected_count=expected_count,
            observed_count=None,
            status="missing",
            error="Image file not found",
        )

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    except Exception as exc:
        return ImageResult(
            episode_id=item.episode_id,
            episode_dir=item.episode_dir,
            indenter_name=item.indenter_name,
            scale_key=item.scale_key,
            scale_mm=item.scale_mm,
            scale_requested_mm=item.scale_requested_mm,
            phase_name=item.phase_name,
            phase_index=item.phase_index,
            frame_name=item.frame_name,
            frame_actual_max_down_mm=item.frame_actual_max_down_mm,
            marker_name=item.marker_name,
            image_relpath=item.image_relpath,
            expected_count=expected_count,
            observed_count=None,
            status="read_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    if image is None:
        return ImageResult(
            episode_id=item.episode_id,
            episode_dir=item.episode_dir,
            indenter_name=item.indenter_name,
            scale_key=item.scale_key,
            scale_mm=item.scale_mm,
            scale_requested_mm=item.scale_requested_mm,
            phase_name=item.phase_name,
            phase_index=item.phase_index,
            frame_name=item.frame_name,
            frame_actual_max_down_mm=item.frame_actual_max_down_mm,
            marker_name=item.marker_name,
            image_relpath=item.image_relpath,
            expected_count=expected_count,
            observed_count=None,
            status="read_error",
            error="cv2.imread returned None",
        )

    try:
        observed_count = count_markers(image, min_area=min_area)
    except Exception as exc:
        return ImageResult(
            episode_id=item.episode_id,
            episode_dir=item.episode_dir,
            indenter_name=item.indenter_name,
            scale_key=item.scale_key,
            scale_mm=item.scale_mm,
            scale_requested_mm=item.scale_requested_mm,
            phase_name=item.phase_name,
            phase_index=item.phase_index,
            frame_name=item.frame_name,
            frame_actual_max_down_mm=item.frame_actual_max_down_mm,
            marker_name=item.marker_name,
            image_relpath=item.image_relpath,
            expected_count=expected_count,
            observed_count=None,
            status="read_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    status = "ok" if observed_count == expected_count else "count_mismatch"
    return ImageResult(
        episode_id=item.episode_id,
        episode_dir=item.episode_dir,
        indenter_name=item.indenter_name,
        scale_key=item.scale_key,
        scale_mm=item.scale_mm,
        scale_requested_mm=item.scale_requested_mm,
        phase_name=item.phase_name,
        phase_index=item.phase_index,
        frame_name=item.frame_name,
        frame_actual_max_down_mm=item.frame_actual_max_down_mm,
        marker_name=item.marker_name,
        image_relpath=item.image_relpath,
        expected_count=expected_count,
        observed_count=observed_count,
        status=status,
        error="",
    )


def write_image_level_row(writer: csv.DictWriter, result: ImageResult) -> None:
    writer.writerow(
        {
            "episode_id": result.episode_id,
            "episode_dir": result.episode_dir,
            "indenter_name": result.indenter_name,
            "scale_key": result.scale_key,
            "scale_mm": format_optional_float(result.scale_mm),
            "scale_requested_mm": format_optional_float(result.scale_requested_mm),
            "phase_name": result.phase_name,
            "phase_index": result.phase_index,
            "frame_name": result.frame_name,
            "frame_actual_max_down_mm": format_optional_float(result.frame_actual_max_down_mm),
            "marker_name": result.marker_name,
            "expected_count": result.expected_count,
            "observed_count": format_optional_int(result.observed_count),
            "status": result.status,
            "error": result.error,
            "image_relpath": result.image_relpath,
        }
    )


def write_summary_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_marker_summary_rows(
    marker_summaries: dict[tuple[str, str], SummaryAccumulator]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (indenter_name, marker_name), summary in sorted(marker_summaries.items()):
        rows.append(
            {
                "indenter_name": indenter_name,
                "marker_name": marker_name,
                "expected_count": summary.expected_count if summary.expected_count is not None else "",
                "total_images": summary.total_images,
                "ok_count": summary.ok_count,
                "bad_count": summary.bad_count,
                "count_mismatch_count": summary.count_mismatch_count,
                "missing_count": summary.missing_count,
                "read_error_count": summary.read_error_count,
                "max_safe_depth_mm": format_optional_float(summary.max_safe_depth_mm),
                "min_bad_depth_mm": format_optional_float(summary.min_bad_depth_mm),
                "non_monotonic_flag": str(summary.non_monotonic_flag),
                "first_bad_status": summary.first_bad_status,
                "first_bad_episode_dir": summary.first_bad_episode_dir,
                "first_bad_scale_key": summary.first_bad_scale_key,
                "first_bad_frame_name": summary.first_bad_frame_name,
                "first_bad_phase_name": summary.first_bad_phase_name,
                "first_bad_image_relpath": summary.first_bad_image_relpath,
                "first_bad_observed_count": format_optional_int(summary.first_bad_observed_count),
                "first_bad_expected_count": format_optional_int(summary.first_bad_expected_count),
            }
        )
    return rows


def build_indenter_summary_rows(
    indenter_summaries: dict[str, SummaryAccumulator],
    marker_summaries: dict[tuple[str, str], SummaryAccumulator],
) -> list[dict[str, Any]]:
    marker_groups: dict[str, list[tuple[str, SummaryAccumulator]]] = defaultdict(list)
    for (indenter_name, marker_name), summary in marker_summaries.items():
        marker_groups[indenter_name].append((marker_name, summary))

    rows: list[dict[str, Any]] = []
    for indenter_name in sorted(indenter_summaries):
        summary = indenter_summaries[indenter_name]
        grouped_markers = marker_groups.get(indenter_name, [])
        worst_marker_name = ""
        worst_marker_min_bad_depth_mm: float | None = None

        candidate_markers = [(marker_name, marker_summary) for marker_name, marker_summary in grouped_markers if marker_summary.bad_count > 0]
        if candidate_markers:
            worst_marker_name, worst_marker_summary = min(
                candidate_markers,
                key=lambda item: _first_bad_sort_key(item[1].min_bad_depth_mm, item[0]),
            )
            worst_marker_min_bad_depth_mm = worst_marker_summary.min_bad_depth_mm

        rows.append(
            {
                "indenter_name": indenter_name,
                "total_images": summary.total_images,
                "ok_count": summary.ok_count,
                "bad_count": summary.bad_count,
                "count_mismatch_count": summary.count_mismatch_count,
                "missing_count": summary.missing_count,
                "read_error_count": summary.read_error_count,
                "marker_groups": len(grouped_markers),
                "max_safe_depth_mm": format_optional_float(summary.max_safe_depth_mm),
                "min_bad_depth_mm": format_optional_float(summary.min_bad_depth_mm),
                "non_monotonic_flag": str(summary.non_monotonic_flag),
                "worst_marker_name": worst_marker_name,
                "worst_marker_min_bad_depth_mm": format_optional_float(worst_marker_min_bad_depth_mm),
                "first_bad_status": summary.first_bad_status,
                "first_bad_episode_dir": summary.first_bad_episode_dir,
                "first_bad_scale_key": summary.first_bad_scale_key,
                "first_bad_frame_name": summary.first_bad_frame_name,
                "first_bad_phase_name": summary.first_bad_phase_name,
                "first_bad_image_relpath": summary.first_bad_image_relpath,
            }
        )
    return rows


def print_terminal_summary(indenter_rows: list[dict[str, Any]], image_count: int, output_dir: Path) -> None:
    ordered_rows = sorted(
        indenter_rows,
        key=lambda row: _first_bad_sort_key(safe_float(row["min_bad_depth_mm"]), str(row["indenter_name"])),
    )
    print(f"Processed {image_count} images")
    print(f"Wrote reports to: {output_dir}")
    print("Per-indenter depth summary:")
    for row in ordered_rows:
        min_bad = row["min_bad_depth_mm"] or "NA"
        max_safe = row["max_safe_depth_mm"] or "NA"
        worst_marker = row["worst_marker_name"] or "-"
        print(
            "  "
            f"{row['indenter_name']}: "
            f"min_bad_depth_mm={min_bad} "
            f"max_safe_depth_mm={max_safe} "
            f"worst_marker={worst_marker} "
            f"bad={row['bad_count']}/{row['total_images']} "
            f"non_monotonic={row['non_monotonic_flag']}"
        )


def main() -> None:
    args = parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.min_area <= 0:
        raise ValueError("--min-area must be > 0")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be > 0 when provided")
    if not args.image_root.exists():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")
    if not args.metadata_root.exists():
        raise FileNotFoundError(f"Metadata root not found: {args.metadata_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_sources, discovered_markers = build_frame_sources(args.metadata_root, args.max_episodes)
    expected_counts = load_expected_counts(args.expected_counts_json, discovered_markers)
    total_images = sum(len(frame.rendered_markers) for frame in frame_sources)

    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(1)

    image_csv_path = args.output_dir / "image_level_counts.csv"
    marker_csv_path = args.output_dir / "limit_by_indenter_marker.csv"
    indenter_csv_path = args.output_dir / "limit_by_indenter.csv"

    marker_summaries: dict[tuple[str, str], SummaryAccumulator] = defaultdict(SummaryAccumulator)
    indenter_summaries: dict[str, SummaryAccumulator] = defaultdict(SummaryAccumulator)

    image_fields = [
        "episode_id",
        "episode_dir",
        "indenter_name",
        "scale_key",
        "scale_mm",
        "scale_requested_mm",
        "phase_name",
        "phase_index",
        "frame_name",
        "frame_actual_max_down_mm",
        "marker_name",
        "expected_count",
        "observed_count",
        "status",
        "error",
        "image_relpath",
    ]

    with image_csv_path.open("w", encoding="utf-8", newline="") as image_file:
        image_writer = csv.DictWriter(image_file, fieldnames=image_fields)
        image_writer.writeheader()

        work_items = iter_work_items(frame_sources)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            iterator = executor.map(
                process_image,
                work_items,
                repeat(args.image_root),
                repeat(expected_counts),
                repeat(args.min_area),
            )
            for result in tqdm(iterator, total=total_images, desc="Count markers", unit="image"):
                if result.status not in VALID_STATUSES:
                    raise ValueError(f"Unexpected status produced: {result.status}")
                write_image_level_row(image_writer, result)
                marker_summaries[(result.indenter_name, result.marker_name)].update(result)
                indenter_summaries[result.indenter_name].update(result)

    marker_rows = build_marker_summary_rows(marker_summaries)
    indenter_rows = build_indenter_summary_rows(indenter_summaries, marker_summaries)

    write_summary_csv(
        marker_csv_path,
        fieldnames=[
            "indenter_name",
            "marker_name",
            "expected_count",
            "total_images",
            "ok_count",
            "bad_count",
            "count_mismatch_count",
            "missing_count",
            "read_error_count",
            "max_safe_depth_mm",
            "min_bad_depth_mm",
            "non_monotonic_flag",
            "first_bad_status",
            "first_bad_episode_dir",
            "first_bad_scale_key",
            "first_bad_frame_name",
            "first_bad_phase_name",
            "first_bad_image_relpath",
            "first_bad_observed_count",
            "first_bad_expected_count",
        ],
        rows=marker_rows,
    )
    write_summary_csv(
        indenter_csv_path,
        fieldnames=[
            "indenter_name",
            "total_images",
            "ok_count",
            "bad_count",
            "count_mismatch_count",
            "missing_count",
            "read_error_count",
            "marker_groups",
            "max_safe_depth_mm",
            "min_bad_depth_mm",
            "non_monotonic_flag",
            "worst_marker_name",
            "worst_marker_min_bad_depth_mm",
            "first_bad_status",
            "first_bad_episode_dir",
            "first_bad_scale_key",
            "first_bad_frame_name",
            "first_bad_phase_name",
            "first_bad_image_relpath",
        ],
        rows=indenter_rows,
    )

    print_terminal_summary(indenter_rows, total_images, args.output_dir)


if __name__ == "__main__":
    main()
