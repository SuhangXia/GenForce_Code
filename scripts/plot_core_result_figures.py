#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCCWM_COLOR = "#3567B7"
DWL_COLOR = "#C97A2B"
GRID_COLOR = "#D9D9D9"
BEST_FILL = "#EAF4E3"
TEXT_COLOR = "#222222"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "analysis" / "core_result_figures"


FALLBACK_RESULTS: dict[str, dict[str, Any]] = {
    "sccwm_hard": {
        "protocol_name": "unseen_indenters_heldout_scales",
        "metrics": {
            "mae_x": 0.2789422869682312,
            "mae_y": 0.22461916506290436,
            "mae_depth": 0.06076561287045479,
            "mae_mean": 0.18810901045799255,
            "mse_x": 0.133926659822464,
            "mse_y": 0.08334209024906158,
            "mse_depth": 0.011982838623225689,
            "mse_mean": 0.07641719281673431,
            "sass_x": 0.0008508202925796367,
            "sass_y": 0.0006500983510779724,
            "sass_depth": 0.002084093157491675,
            "sass_mean": 0.001195003933716428,
            "ccauc": 0.9003069400787354,
            "sample_count": 40000,
            "ccauc_sample_count": 2048,
        },
    },
    "dwl_hard": {
        "protocol_name": "unseen_indenters_heldout_scales",
        "metrics": {
            "mae_x": 0.22365958988666534,
            "mae_y": 0.17549987137317657,
            "mae_depth": 0.12903577089309692,
            "mae_mean": 0.1760650873184204,
            "mse_x": 0.12091822177171707,
            "mse_y": 0.07146874815225601,
            "mse_depth": 0.04388098791241646,
            "mse_mean": 0.07875598967075348,
            "sass_x": 0.0005065840201047873,
            "sass_y": 0.0008042283005347615,
            "sass_depth": 0.00144155295699954,
            "sass_mean": 0.000917455092546363,
            "ccauc": 0.8867588043212891,
            "sample_count": 40000,
            "ccauc_sample_count": 2048,
        },
    },
    "sccwm_val": {
        "protocol_name": "heldout_scale_bands",
        "metrics": {
            "mae_x": 0.15937118232250214,
            "mae_y": 0.15774506330490112,
            "mae_depth": 0.12013013660907745,
            "mae_mean": 0.1457487940788269,
            "mse_x": 0.05825047194957733,
            "mse_y": 0.06836320459842682,
            "mse_depth": 0.03444799780845642,
            "mse_mean": 0.05368722602725029,
            "sass_x": 0.00035683747871386454,
            "sass_y": 0.00019699207423962942,
            "sass_depth": 0.005103928692260694,
            "sass_mean": 0.001885919415071396,
            "ccauc": 0.835047721862793,
            "sample_count": 33292,
            "ccauc_sample_count": 2048,
        },
    },
    "dwl_val": {
        "protocol_name": "heldout_scale_bands",
        "metrics": {
            "mae_x": 0.14643195271492004,
            "mae_y": 0.14521218836307526,
            "mae_depth": 0.17967374622821808,
            "mae_mean": 0.1571059674024582,
            "mse_x": 0.05945616215467453,
            "mse_y": 0.06707826256752014,
            "mse_depth": 0.06266168504953384,
            "mse_mean": 0.06306537240743637,
            "sass_x": 0.00030948786684817115,
            "sass_y": 0.0002134901661785881,
            "sass_depth": 0.004911275480058465,
            "sass_mean": 0.0018114178376950748,
            "ccauc": 0.8242948055267334,
            "sample_count": 33292,
            "ccauc_sample_count": 2048,
        },
    },
}


DEFAULT_JSON_PATHS = {
    "sccwm_hard": Path("/home/suhang/datasets/checkpoints/sccwm_120h_full/eval_stage2_test_unseen_indenters_heldout_scales_limit40k.json"),
    "dwl_hard": Path("/home/suhang/datasets/checkpoints/dwl_tr_120h/eval_test_unseen_indenters_heldout_scales_limit40k.json"),
    "sccwm_val": Path("/home/suhang/datasets/checkpoints/sccwm_120h_full/eval_stage2_val_heldout_scale_bands_limit40k.json"),
    "dwl_val": Path("/home/suhang/datasets/checkpoints/dwl_tr_120h/eval_val_heldout_scale_bands_limit40k.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SCCWM vs DWL-TR core benchmark figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_result(name: str) -> tuple[dict[str, Any], str]:
    path = DEFAULT_JSON_PATHS[name]
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    return FALLBACK_RESULTS[name], f"fallback:{name}"


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def style_axis(ax: plt.Axes, ylabel: str | None = None) -> None:
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(ax: plt.Axes, bars: list[Any], fmt: str = "{:.3f}", rotation: int = 0) -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + ymax * 0.015,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=rotation,
        )


def grouped_metric_bars(
    title: str,
    metrics: list[str],
    labels: list[str],
    sccwm: dict[str, Any],
    dwl: dict[str, Any],
    footnote: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.36
    s_vals = [sccwm["metrics"][m] for m in metrics]
    d_vals = [dwl["metrics"][m] for m in metrics]
    bars1 = ax.bar(x - width / 2, s_vals, width, label="SCCWM", color=SCCWM_COLOR)
    bars2 = ax.bar(x + width / 2, d_vals, width, label="DWL-TR", color=DWL_COLOR)
    ax.set_xticks(x, labels)
    ax.set_title(title, pad=16)
    style_axis(ax, ylabel="Error")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.text(0.0, 1.02, footnote, transform=ax.transAxes, fontsize=11, ha="left", va="bottom")
    annotate_bars(ax, list(bars1))
    annotate_bars(ax, list(bars2))
    return fig


def confound_figure(title: str, sccwm: dict[str, Any], dwl: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    entries = [
        ("ccauc", "CCAUC", "Higher is better"),
        ("sass_mean", "SASS Mean", "Lower is better"),
    ]
    for ax, (metric, label, note) in zip(axes, entries):
        vals = [sccwm["metrics"][metric], dwl["metrics"][metric]]
        bars = ax.bar(["SCCWM", "DWL-TR"], vals, color=[SCCWM_COLOR, DWL_COLOR], width=0.55)
        ax.set_title(label)
        style_axis(ax, ylabel=label)
        ax.text(0.0, 1.02, note, transform=ax.transAxes, fontsize=10, ha="left", va="bottom")
        annotate_bars(ax, list(bars))
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


def percent_reduction(baseline: float, improved: float) -> float:
    return (baseline - improved) / baseline * 100.0


def key_takeaways_figure(sccwm_hard: dict[str, Any], dwl_hard: dict[str, Any], sccwm_val: dict[str, Any], dwl_val: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("A. Hardest Test Depth", "mae_depth", sccwm_hard, dwl_hard, "Lower is better"),
        ("B. Held-out Val Depth", "mae_depth", sccwm_val, dwl_val, "Lower is better"),
        ("C. Hardest Test CCAUC", "ccauc", sccwm_hard, dwl_hard, "Higher is better"),
        ("D. Held-out Val CCAUC", "ccauc", sccwm_val, dwl_val, "Higher is better"),
    ]
    for ax, (title, metric, sccwm, dwl, note) in zip(axes.flat, panels):
        vals = [sccwm["metrics"][metric], dwl["metrics"][metric]]
        bars = ax.bar(["SCCWM", "DWL-TR"], vals, color=[SCCWM_COLOR, DWL_COLOR], width=0.55)
        style_axis(ax)
        ax.set_title(title)
        ax.text(0.0, 1.02, note, transform=ax.transAxes, fontsize=10, ha="left", va="bottom")
        annotate_bars(ax, list(bars))

    hard_reduction = percent_reduction(dwl_hard["metrics"]["mae_depth"], sccwm_hard["metrics"]["mae_depth"])
    val_reduction = percent_reduction(dwl_val["metrics"]["mae_depth"], sccwm_val["metrics"]["mae_depth"])
    axes[0, 0].text(
        0.5,
        0.82,
        f"Depth error reduction: {hard_reduction:.1f}%",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F3F7FD", "edgecolor": SCCWM_COLOR},
    )
    axes[0, 1].text(
        0.5,
        0.82,
        f"Depth error reduction: {val_reduction:.1f}%",
        transform=axes[0, 1].transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F3F7FD", "edgecolor": SCCWM_COLOR},
    )
    fig.suptitle(
        "Key Takeaway: SCCWM improves depth and counterfactual discrimination\nwhile DWL-TR remains a strong x/y baseline",
        fontsize=17,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def summary_table_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    columns = ["mae_x", "mae_y", "mae_depth", "mae_mean", "sass_mean", "ccauc"]
    headers = ["Run", "mae_x", "mae_y", "mae_depth", "mae_mean", "sass_mean", "ccauc"]
    cell_text = []
    for row in rows:
        cell_text.append(
            [
                f"{row['model']} / {row['protocol']}",
                *(f"{row['metrics'][c]:.3f}" for c in columns[:-1]),
                f"{row['metrics']['ccauc']:.3f}",
            ]
        )

    table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.7)

    for col_idx in range(len(headers)):
        table[0, col_idx].set_facecolor("#F0F2F5")
        table[0, col_idx].set_text_props(weight="bold")

    best_low = {col: min(row["metrics"][col] for row in rows) for col in columns[:-1]}
    best_high = max(row["metrics"]["ccauc"] for row in rows)

    for row_idx, row in enumerate(rows, start=1):
        for col_idx, col in enumerate(columns, start=1):
            value = row["metrics"][col]
            is_best = abs(value - (best_high if col == "ccauc" else best_low[col])) < 1e-12
            if is_best:
                table[row_idx, col_idx].set_facecolor(BEST_FILL)
                table[row_idx, col_idx].set_text_props(weight="bold")

    ax.set_title("SCCWM vs DWL-TR: Summary Table", fontsize=16, pad=18)
    ax.text(
        0.0,
        -0.12,
        "Best values highlighted. Lower is better for error/SASS; higher is better for CCAUC.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    return fig


def write_readme(output_dir: Path, results: dict[str, tuple[dict[str, Any], str]]) -> None:
    hard_reduction = percent_reduction(results["dwl_hard"][0]["metrics"]["mae_depth"], results["sccwm_hard"][0]["metrics"]["mae_depth"])
    val_reduction = percent_reduction(results["dwl_val"][0]["metrics"]["mae_depth"], results["sccwm_val"][0]["metrics"]["mae_depth"])
    lines = [
        "# Core Result Figures",
        "",
        "## Files",
        "",
        "- `fig1_hardest_test_main_metrics.(png|pdf)`: Hardest test main errors. Shows DWL-TR leads on x/y and overall MAE, while SCCWM leads on depth.",
        "- `fig2_hardest_test_confound_metrics.(png|pdf)`: Hardest test confound-sensitive metrics. Shows SCCWM has higher CCAUC, DWL-TR slightly lower SASS.",
        "- `fig3_heldout_val_main_metrics.(png|pdf)`: Held-out validation main errors. Shows DWL-TR is slightly better on x/y, SCCWM is better on depth and overall MAE.",
        "- `fig4_heldout_val_confound_metrics.(png|pdf)`: Held-out validation confound-sensitive metrics. Shows SCCWM has higher CCAUC and similar SASS.",
        f"- `fig5_key_takeaways.(png|pdf)`: PPT-ready conclusion panel. Highlights SCCWM depth error reduction of {hard_reduction:.1f}% on hardest test and {val_reduction:.1f}% on held-out val, plus stronger CCAUC.",
        "- `fig6_summary_table.(png|pdf)`: Compact summary table with best values highlighted.",
        "",
        "## Recommended Main Result Slide",
        "",
        "- Use `fig5_key_takeaways.png` as the primary supervisor presentation slide.",
        "- Use `fig6_summary_table.png` as the backup summary slide if a compact comparison table is needed.",
        "",
        "## Data Sources",
        "",
    ]
    for key, (_, source) in results.items():
        lines.append(f"- `{key}`: `{source}`")
    (output_dir / "README_core_figures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_style()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {name: load_result(name) for name in ["sccwm_hard", "dwl_hard", "sccwm_val", "dwl_val"]}
    sccwm_hard, dwl_hard, sccwm_val, dwl_val = (results[k][0] for k in ["sccwm_hard", "dwl_hard", "sccwm_val", "dwl_val"])

    fig1 = grouped_metric_bars(
        "Hardest Test: Unseen Indenters + Held-out Scales",
        ["mae_x", "mae_y", "mae_depth", "mae_mean"],
        ["mae_x", "mae_y", "mae_depth", "mae_mean"],
        sccwm_hard,
        dwl_hard,
        "Lower is better",
    )
    save_figure(fig1, output_dir, "fig1_hardest_test_main_metrics")

    fig2 = confound_figure("Hardest Test: Confound-sensitive Metrics", sccwm_hard, dwl_hard)
    save_figure(fig2, output_dir, "fig2_hardest_test_confound_metrics")

    fig3 = grouped_metric_bars(
        "Held-out Scale Bands (Val)",
        ["mae_x", "mae_y", "mae_depth", "mae_mean"],
        ["mae_x", "mae_y", "mae_depth", "mae_mean"],
        sccwm_val,
        dwl_val,
        "Lower is better",
    )
    save_figure(fig3, output_dir, "fig3_heldout_val_main_metrics")

    fig4 = confound_figure("Held-out Val: Confound-sensitive Metrics", sccwm_val, dwl_val)
    save_figure(fig4, output_dir, "fig4_heldout_val_confound_metrics")

    fig5 = key_takeaways_figure(sccwm_hard, dwl_hard, sccwm_val, dwl_val)
    save_figure(fig5, output_dir, "fig5_key_takeaways")

    table_rows = [
        {"model": "SCCWM", "protocol": "hardest test", "metrics": sccwm_hard["metrics"]},
        {"model": "DWL-TR", "protocol": "hardest test", "metrics": dwl_hard["metrics"]},
        {"model": "SCCWM", "protocol": "heldout val", "metrics": sccwm_val["metrics"]},
        {"model": "DWL-TR", "protocol": "heldout val", "metrics": dwl_val["metrics"]},
    ]
    fig6 = summary_table_figure(table_rows)
    save_figure(fig6, output_dir, "fig6_summary_table")

    write_readme(output_dir, results)

    generated = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    print(f"output_dir={output_dir}")
    for item in generated:
        print(item)


if __name__ == "__main__":
    main()
