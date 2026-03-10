#!/usr/bin/env python3
"""Parse USA training logs and plot train/val curves.

Expected epoch-summary line example:
[2026-03-10 03:40:40] INFO  Epoch   1/120  train_loss=5.940131 train_mse=5.700995 train_cos_loss=0.478271 train_cos=0.7607  val_loss=3.821976 val_mse=3.685715 val_cos_loss=0.272522 val_cos=0.8783  lr=4.01e-05  epoch_t=04m27s elapsed=04m27s eta=8h49m34s

Expected test-only line example:
[2026-03-10 11:26:38] INFO  TEST-ONLY (zero-shot indenters): loss=2.269199 mse=2.190980 cos_loss=0.156438 cos_sim=0.9145
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_KEYS = ["train_loss", "train_mse", "train_cos_loss", "train_cos"]
VAL_KEYS = ["val_loss", "val_mse", "val_cos_loss", "val_cos"]
CSV_KEYS = ["epoch", "total_epochs", *TRAIN_KEYS, *VAL_KEYS, "lr"]


# Match plain numeric values including scientific notation.
KV_RE = re.compile(r"\b([a-zA-Z_]+)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
EPOCH_RE = re.compile(r"\bEpoch\s+(\d+)\s*/\s*(\d+)\b")
TEST_RE = re.compile(r"\bTEST(?:-ONLY)?\s*\(zero-shot indenters\):")


def parse_epoch_record(line: str) -> dict | None:
    """Parse one epoch summary line. Returns None for non-summary lines."""
    if "train_loss=" not in line:
        return None

    epoch_match = EPOCH_RE.search(line)
    if not epoch_match:
        return None

    epoch = int(epoch_match.group(1))
    total_epochs = int(epoch_match.group(2))

    metrics: dict[str, float] = {}
    for key, val in KV_RE.findall(line):
        try:
            metrics[key] = float(val)
        except ValueError:
            continue

    if "train_loss" not in metrics:
        return None

    record: dict[str, float | int | bool | None] = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "lr": metrics.get("lr"),
    }
    for key in TRAIN_KEYS + VAL_KEYS:
        record[key] = metrics.get(key)

    train_complete = all(record[k] is not None for k in TRAIN_KEYS)
    # lr is expected in summary lines from train_adapter.py, include it in completeness.
    train_complete = train_complete and record["lr"] is not None

    has_any_val = any(record[k] is not None for k in VAL_KEYS)
    has_full_val = all(record[k] is not None for k in VAL_KEYS)
    val_complete = has_full_val if has_any_val else True

    record["_complete"] = bool(train_complete and val_complete)
    record["_has_val"] = bool(has_full_val)
    return record


def parse_test_summary(line: str) -> dict | None:
    """Parse one TEST-ONLY/TEST summary line."""
    if not TEST_RE.search(line):
        return None

    metrics: dict[str, float] = {}
    for key, val in KV_RE.findall(line):
        try:
            metrics[key] = float(val)
        except ValueError:
            continue

    needed = ["loss", "mse", "cos_loss", "cos_sim"]
    if not all(k in metrics for k in needed):
        return None

    mode = "test_only" if "TEST-ONLY" in line else "test"
    return {
        "mode": mode,
        "loss": metrics["loss"],
        "mse": metrics["mse"],
        "cos_loss": metrics["cos_loss"],
        "cos_sim": metrics["cos_sim"],
    }


def parse_log_file(log_file: Path) -> tuple[list[dict], list[dict]]:
    """Parse epoch summaries and test summaries from log file."""
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    # epoch -> record; duplicate handling keeps the last complete summary.
    by_epoch: dict[int, dict] = {}
    test_summaries: list[dict] = []

    with log_file.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            test_rec = parse_test_summary(line)
            if test_rec is not None:
                test_summaries.append(test_rec)

            rec = parse_epoch_record(line)
            if rec is None:
                continue

            epoch = int(rec["epoch"])
            existing = by_epoch.get(epoch)
            if existing is None:
                by_epoch[epoch] = rec
                continue

            new_complete = bool(rec.get("_complete", False))
            old_complete = bool(existing.get("_complete", False))

            if new_complete:
                # Always replace with newer complete summary.
                by_epoch[epoch] = rec
            elif not old_complete:
                # No complete summary yet for this epoch: keep latest seen.
                by_epoch[epoch] = rec
            # else: keep existing complete summary, ignore newer incomplete one.

    records = [by_epoch[e] for e in sorted(by_epoch.keys())]
    if not records:
        raise ValueError(
            "No epoch summary found in log. Expected lines containing "
            "'Epoch ... train_loss=...'."
        )

    return records, test_summaries


def write_metrics_csv(records: list[dict], output_path: Path) -> None:
    """Write parsed epoch metrics to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_KEYS)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in CSV_KEYS}
            writer.writerow(row)


def extract_series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Extract (epochs, values) for entries with non-None metric."""
    epochs: list[int] = []
    vals: list[float] = []
    for rec in records:
        v = rec.get(key)
        if v is None:
            continue
        epochs.append(int(rec["epoch"]))
        vals.append(float(v))
    return epochs, vals


def plot_metric(
    records: list[dict],
    train_key: str,
    val_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    best_epoch: int | None = None,
) -> None:
    """Plot one metric figure (train + optional val)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    tr_epochs, tr_vals = extract_series(records, train_key)
    if tr_epochs:
        ax.plot(tr_epochs, tr_vals, label=train_key)

    val_epochs, val_vals = extract_series(records, val_key)
    if val_epochs:
        ax.plot(val_epochs, val_vals, label=val_key)

    if best_epoch is not None:
        ax.axvline(best_epoch, linestyle="--", label=f"best val epoch ({best_epoch})")

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overview(records: list[dict], out_path: Path) -> None:
    """Create 2x2 overview plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    configs = [
        ("train_loss", "val_loss", "Loss", "loss"),
        ("train_mse", "val_mse", "MSE", "mse"),
        ("train_cos_loss", "val_cos_loss", "Cosine Loss", "cos_loss"),
        ("train_cos", "val_cos", "Cosine Similarity", "cos"),
    ]

    for ax, (train_key, val_key, title, ylabel) in zip(axes.flat, configs):
        tr_epochs, tr_vals = extract_series(records, train_key)
        if tr_epochs:
            ax.plot(tr_epochs, tr_vals, label=train_key)

        val_epochs, val_vals = extract_series(records, val_key)
        if val_epochs:
            ax.plot(val_epochs, val_vals, label=val_key)

        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def best_val_epoch(records: list[dict], key: str, mode: str) -> int | None:
    """Return best val epoch by min/max over key, or None if val not present."""
    vals = [(int(r["epoch"]), r.get(key)) for r in records if r.get(key) is not None]
    if not vals:
        return None

    if mode == "min":
        return min(vals, key=lambda x: float(x[1]))[0]
    if mode == "max":
        return max(vals, key=lambda x: float(x[1]))[0]
    raise ValueError(f"Unknown mode: {mode}")


def print_test_summaries(test_summaries: list[dict]) -> None:
    """Print parsed TEST-ONLY/TEST summaries to terminal."""
    if not test_summaries:
        return

    print("\nParsed test summaries:")
    for i, rec in enumerate(test_summaries, start=1):
        print(
            f"  [{i}] mode={rec['mode']} "
            f"loss={rec['loss']:.6f} mse={rec['mse']:.6f} "
            f"cos_loss={rec['cos_loss']:.6f} cos_sim={rec['cos_sim']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot USA train/val curves from training log.")
    parser.add_argument("--log-file", type=Path, required=True, help="Path to training log file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save output images/csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        records, test_summaries = parse_log_file(args.log_file)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # CSV export
    csv_path = args.output_dir / "metrics.csv"
    write_metrics_csv(records, csv_path)

    # Best-val epochs for vertical markers (if val is available)
    best_loss_ep = best_val_epoch(records, key="val_loss", mode="min")
    best_cos_ep = best_val_epoch(records, key="val_cos", mode="max")

    # Individual curves
    plot_metric(
        records,
        train_key="train_loss",
        val_key="val_loss",
        title="USA Training Curve: Loss",
        ylabel="loss",
        out_path=args.output_dir / "loss_curve.png",
        best_epoch=best_loss_ep,
    )
    plot_metric(
        records,
        train_key="train_mse",
        val_key="val_mse",
        title="USA Training Curve: MSE",
        ylabel="mse",
        out_path=args.output_dir / "mse_curve.png",
    )
    plot_metric(
        records,
        train_key="train_cos_loss",
        val_key="val_cos_loss",
        title="USA Training Curve: Cosine Loss",
        ylabel="cos_loss",
        out_path=args.output_dir / "cos_loss_curve.png",
    )
    plot_metric(
        records,
        train_key="train_cos",
        val_key="val_cos",
        title="USA Training Curve: Cosine Similarity",
        ylabel="cos",
        out_path=args.output_dir / "cos_curve.png",
        best_epoch=best_cos_ep,
    )

    # Overview
    plot_overview(records, args.output_dir / "overview.png")

    # Console summary
    print(f"Parsed epochs: {len(records)}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plots under: {args.output_dir}")
    if best_loss_ep is not None:
        print(f"Best val_loss epoch: {best_loss_ep}")
    if best_cos_ep is not None:
        print(f"Best val_cos epoch: {best_cos_ep}")

    # Optional TEST-ONLY / TEST summary print
    print_test_summaries(test_summaries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
