#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFile

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class _TqdmFallback:  # pragma: no cover
        def __init__(self, iterable, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args, **kwargs) -> None:
            return None

    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return _TqdmFallback(iterable, *args, **kwargs)


DEFAULT_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
DEFAULT_MIN_EDGE_COVERAGE = 0.60
DEFAULT_STRATEGY = "detect"

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class BorderWidths:
    top: int
    bottom: int
    left: int
    right: int


@dataclass(frozen=True)
class ProcessResult:
    file_name: str
    relative_path: Path
    widths: BorderWidths
    modified: bool
    masked_pixels: int
    output_path: Path


@dataclass(frozen=True)
class FailureResult:
    relative_path: Path
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect bright marker-image border artifacts and replace them with black pixels in a new output directory."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing source marker images.")
    parser.add_argument("--output-dir", required=True, help="Directory to write processed images into.")
    parser.add_argument("--debug-dir", default="", help="Optional directory for debug visualizations.")
    parser.add_argument("--pattern", default="*", help="Filename pattern filter, for example '*.jpg' or 'Circle*'.")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated file extensions to include. Default: jpg,jpeg,png,webp",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=245,
        help="Brightness threshold for white-border detection. Default: 245",
    )
    parser.add_argument(
        "--max-border-width",
        type=int,
        default=6,
        help="Maximum width, in pixels, to inspect near each image edge. Default: 6",
    )
    parser.add_argument(
        "--strategy",
        choices=("detect", "fixed"),
        default=DEFAULT_STRATEGY,
        help="Border handling strategy. 'detect' masks detected bright edge artifacts; 'fixed' paints a fixed-width black border.",
    )
    parser.add_argument(
        "--fixed-border-width",
        type=int,
        default=3,
        help="Fixed black border width in pixels when --strategy fixed is used. Default: 3",
    )
    parser.add_argument(
        "--mode",
        choices=("debug", "full"),
        default="debug",
        help="Debug writes extra visualizations; full only writes processed images unless debug-dir is also given.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in output/debug directories.",
    )
    parser.add_argument(
        "--save-contact-sheet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save original|mask|processed contact sheets when debug outputs are enabled.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively traverse subdirectories under --input-dir. Default: true",
    )
    parser.add_argument(
        "--skip-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue processing if an image fails to decode or save, and record it in a failure log. Default: true",
    )
    return parser.parse_args()


def parse_extensions(raw: str) -> tuple[str, ...]:
    values = [item.strip().lower().lstrip(".") for item in raw.split(",")]
    extensions = tuple(item for item in values if item)
    if not extensions:
        raise ValueError("No valid extensions were provided.")
    return extensions


def list_input_files(input_dir: Path, pattern: str, extensions: Iterable[str], *, recursive: bool) -> list[Path]:
    allowed = {f".{ext.lower()}" for ext in extensions}
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = [
        path
        for path in sorted(iterator)
        if path.is_file() and path.suffix.lower() in allowed and fnmatch.fnmatch(path.name, pattern)
    ]
    return files


def ensure_writable_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return
    existing_files = [child for child in path.iterdir() if child.is_file()]
    if existing_files:
        raise FileExistsError(
            f"Output directory already contains files and --overwrite was not set: {path}"
        )


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    return 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]


def build_edge_zone(height: int, width: int, max_border_width: int) -> np.ndarray:
    zone = np.zeros((height, width), dtype=bool)
    if max_border_width <= 0:
        return zone
    band = min(max_border_width, height, width)
    zone[:band, :] = True
    zone[-band:, :] = True
    zone[:, :band] = True
    zone[:, -band:] = True
    return zone


def connected_edge_mask(bright_mask: np.ndarray, edge_zone: np.ndarray) -> np.ndarray:
    allowed = bright_mask & edge_zone
    height, width = allowed.shape
    connected = np.zeros_like(allowed, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def maybe_push(y: int, x: int) -> None:
        if 0 <= y < height and 0 <= x < width and allowed[y, x] and not connected[y, x]:
            connected[y, x] = True
            queue.append((y, x))

    for x in range(width):
        maybe_push(0, x)
        maybe_push(height - 1, x)
    for y in range(height):
        maybe_push(y, 0)
        maybe_push(y, width - 1)

    while queue:
        y, x = queue.popleft()
        maybe_push(y - 1, x)
        maybe_push(y + 1, x)
        maybe_push(y, x - 1)
        maybe_push(y, x + 1)
    return connected


def contiguous_band_widths(mask: np.ndarray, max_border_width: int, min_edge_coverage: float) -> BorderWidths:
    height, width = mask.shape
    band = min(max_border_width, height, width)

    def scan_rows(indices: Iterable[int]) -> int:
        count = 0
        for idx in indices:
            coverage = float(mask[idx, :].mean())
            if coverage >= min_edge_coverage:
                count += 1
            else:
                break
        return count

    def scan_cols(indices: Iterable[int]) -> int:
        count = 0
        for idx in indices:
            coverage = float(mask[:, idx].mean())
            if coverage >= min_edge_coverage:
                count += 1
            else:
                break
        return count

    top = scan_rows(range(0, band))
    bottom = scan_rows(range(height - 1, height - band - 1, -1))
    left = scan_cols(range(0, band))
    right = scan_cols(range(width - 1, width - band - 1, -1))
    return BorderWidths(top=top, bottom=bottom, left=left, right=right)


def mask_from_widths(edge_connected_mask: np.ndarray, widths: BorderWidths) -> np.ndarray:
    height, width = edge_connected_mask.shape
    final_mask = np.zeros_like(edge_connected_mask, dtype=bool)
    if widths.top > 0:
        final_mask[: widths.top, :] |= edge_connected_mask[: widths.top, :]
    if widths.bottom > 0:
        final_mask[height - widths.bottom :, :] |= edge_connected_mask[height - widths.bottom :, :]
    if widths.left > 0:
        final_mask[:, : widths.left] |= edge_connected_mask[:, : widths.left]
    if widths.right > 0:
        final_mask[:, width - widths.right :] |= edge_connected_mask[:, width - widths.right :]
    return final_mask


def fixed_border_mask(height: int, width: int, border_width: int) -> tuple[np.ndarray, BorderWidths]:
    width_px = max(0, min(int(border_width), height, width))
    widths = BorderWidths(
        top=width_px,
        bottom=width_px,
        left=width_px,
        right=width_px,
    )
    mask = np.zeros((height, width), dtype=bool)
    if width_px <= 0:
        return mask, widths
    mask[:width_px, :] = True
    mask[-width_px:, :] = True
    mask[:, :width_px] = True
    mask[:, -width_px:] = True
    return mask, widths


def detect_border_mask(
    image: Image.Image,
    *,
    white_threshold: int,
    max_border_width: int,
    min_edge_coverage: float = DEFAULT_MIN_EDGE_COVERAGE,
) -> tuple[np.ndarray, BorderWidths]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    luma = rgb_to_luma(rgb)
    edge_zone = build_edge_zone(rgb.shape[0], rgb.shape[1], max_border_width)
    bright = luma >= float(white_threshold)
    edge_connected = connected_edge_mask(bright, edge_zone)
    widths = contiguous_band_widths(edge_connected, max_border_width=max_border_width, min_edge_coverage=min_edge_coverage)
    final_mask = mask_from_widths(edge_connected, widths)
    return final_mask, widths


def apply_black_border(image: Image.Image, mask: np.ndarray) -> Image.Image:
    if image.mode == "L":
        arr = np.asarray(image).copy()
        arr[mask] = 0
        return Image.fromarray(arr, mode="L")
    if image.mode == "RGBA":
        arr = np.asarray(image).copy()
        arr[mask, 0:3] = 0
        return Image.fromarray(arr, mode="RGBA")
    if image.mode == "RGB":
        arr = np.asarray(image).copy()
        arr[mask] = 0
        return Image.fromarray(arr, mode="RGB")
    rgb = np.asarray(image.convert("RGB")).copy()
    rgb[mask] = 0
    return Image.fromarray(rgb, mode="RGB")


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def save_debug_outputs(
    original: Image.Image,
    processed: Image.Image,
    mask: np.ndarray,
    relative_path: Path,
    debug_dir: Path,
    *,
    overwrite: bool,
    save_contact_sheet: bool,
) -> None:
    original_dir = debug_dir / "original"
    mask_dir = debug_dir / "mask"
    processed_dir = debug_dir / "processed"
    sheet_dir = debug_dir / "contact_sheet"
    for path in (original_dir, mask_dir, processed_dir):
        path.mkdir(parents=True, exist_ok=True)
    if save_contact_sheet:
        sheet_dir.mkdir(parents=True, exist_ok=True)

    original_path = original_dir / relative_path
    processed_path = processed_dir / relative_path
    mask_path = mask_dir / relative_path.parent / f"{relative_path.stem}_mask.png"
    for path in (original_path.parent, processed_path.parent, mask_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        for path in (original_path, processed_path, mask_path):
            if path.exists():
                raise FileExistsError(f"Debug output already exists and --overwrite was not set: {path}")
    original.save(original_path)
    processed.save(processed_path)
    mask_img = mask_to_image(mask)
    mask_img.save(mask_path)

    if save_contact_sheet:
        sheet_path = sheet_dir / relative_path.parent / f"{relative_path.stem}_sheet.png"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        if sheet_path.exists() and not overwrite:
            raise FileExistsError(f"Contact sheet already exists and --overwrite was not set: {sheet_path}")
        original_rgb = original.convert("RGB")
        processed_rgb = processed.convert("RGB")
        mask_rgb = mask_img.convert("RGB")
        width, height = original_rgb.size
        sheet = Image.new("RGB", (width * 3, height), color=(0, 0, 0))
        sheet.paste(original_rgb, (0, 0))
        sheet.paste(mask_rgb, (width, 0))
        sheet.paste(processed_rgb, (width * 2, 0))
        sheet.save(sheet_path)


def process_file(
    path: Path,
    input_root: Path,
    output_dir: Path,
    *,
    strategy: str,
    white_threshold: int,
    max_border_width: int,
    fixed_border_width: int,
    mode: str,
    debug_dir: Path | None,
    overwrite: bool,
    save_contact_sheet: bool,
) -> ProcessResult:
    image = Image.open(path)
    try:
        original = image.copy()
    finally:
        image.close()

    if strategy == "fixed":
        mask, widths = fixed_border_mask(original.height, original.width, fixed_border_width)
    else:
        mask, widths = detect_border_mask(
            original,
            white_threshold=white_threshold,
            max_border_width=max_border_width,
        )
    masked_pixels = int(mask.sum())
    modified = masked_pixels > 0
    processed = apply_black_border(original, mask) if modified else original.copy()

    relative_path = path.relative_to(input_root)
    output_path = output_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists and --overwrite was not set: {output_path}")
    processed.save(output_path)

    if mode == "debug" or debug_dir is not None:
        if debug_dir is None:
            raise ValueError("Internal error: debug_dir must exist when debug outputs are requested.")
        save_debug_outputs(
            original,
            processed,
            mask,
            relative_path,
            debug_dir,
            overwrite=overwrite,
            save_contact_sheet=save_contact_sheet,
        )

    return ProcessResult(
        file_name=path.name,
        relative_path=relative_path,
        widths=widths,
        modified=modified,
        masked_pixels=masked_pixels,
        output_path=output_path,
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}", file=sys.stderr)
        return 2

    try:
        extensions = parse_extensions(args.extensions)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    files = list_input_files(input_dir, args.pattern, extensions, recursive=bool(args.recursive))
    if not files:
        print(
            f"No files matched in {input_dir} with pattern={args.pattern!r} and extensions={','.join(extensions)}",
            file=sys.stderr,
        )
        return 1

    try:
        ensure_writable_directory(output_dir, overwrite=args.overwrite)
        if args.mode == "debug" and debug_dir is None:
            debug_dir = output_dir / "_debug"
        if debug_dir is not None:
            ensure_writable_directory(debug_dir, overwrite=args.overwrite)
    except (ValueError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    modified_count = 0
    results: list[ProcessResult] = []
    failures: list[FailureResult] = []
    iterator = tqdm(files, desc="Preprocess marker borders", unit="image", disable=False)
    for path in iterator:
        try:
            result = process_file(
                path,
                input_dir,
                output_dir,
                strategy=args.strategy,
                white_threshold=args.white_threshold,
                max_border_width=args.max_border_width,
                fixed_border_width=args.fixed_border_width,
                mode=args.mode,
                debug_dir=debug_dir,
                overwrite=args.overwrite,
                save_contact_sheet=bool(args.save_contact_sheet),
            )
        except Exception as exc:
            if not bool(args.skip_errors):
                print(f"[ERROR] {path.name}: {exc}", file=sys.stderr)
                return 1
            relative_path = path.relative_to(input_dir)
            failures.append(FailureResult(relative_path=relative_path, error=str(exc)))
            print(f"[WARN] {relative_path}: {exc}", file=sys.stderr)
            iterator.set_postfix({"modified": modified_count, "failed": len(failures)})
            continue
        modified_count += int(result.modified)
        results.append(result)
        iterator.set_postfix({"modified": modified_count, "failed": len(failures)})
        if args.mode == "debug":
            print(
                f"{result.relative_path}: top={result.widths.top} bottom={result.widths.bottom} "
                f"left={result.widths.left} right={result.widths.right} "
                f"modified={result.modified} masked_pixels={result.masked_pixels} "
                f"output={result.output_path}"
            )

    print(
        f"Processed {len(results)} image(s); modified {modified_count}; "
        f"output_dir={output_dir}"
    )
    if failures:
        failure_log_path = output_dir / "_preprocess_failures.txt"
        lines = [f"{item.relative_path}\t{item.error}" for item in failures]
        failure_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Encountered {len(failures)} failure(s); log written to: {failure_log_path}", file=sys.stderr)
    if debug_dir is not None:
        print(f"Debug outputs written to: {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
