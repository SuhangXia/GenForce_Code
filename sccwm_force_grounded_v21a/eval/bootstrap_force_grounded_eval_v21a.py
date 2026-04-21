#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import _summarize_direct_records, save_eval_result


METRIC_KEYS = ["mae_x", "mae_y", "mae_depth", "mae_mean", "ccauc"]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in record file: {file_path}")
    return rows


def _choose_cluster_key(rows: list[dict[str, Any]], requested: str) -> str:
    if requested != "auto":
        if all(row.get(requested) not in {None, ""} for row in rows):
            return requested
        raise RuntimeError(f"Requested cluster key {requested!r} is unavailable in all rows.")
    for key in ("episode_id", "pair_id", "event_key"):
        if all(row.get(key) not in {None, ""} for row in rows):
            return key
    raise RuntimeError("Could not determine a usable cluster key. None of episode_id, pair_id, event_key were fully available.")


def _summarize_records(rows: list[dict[str, Any]], *, protocol: str, ccauc_max_samples: int) -> dict[str, float]:
    summary = _summarize_direct_records(rows, protocol=protocol, cfg={"eval": {"ccauc_max_samples": int(ccauc_max_samples)}})
    metrics = summary["metrics"]
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def _bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    protocol: str,
    ccauc_max_samples: int,
    n_bootstrap: int,
    seed: int,
    cluster_key: str,
) -> dict[str, Any]:
    point = _summarize_records(rows, protocol=protocol, ccauc_max_samples=ccauc_max_samples)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(row)
    cluster_ids = sorted(grouped.keys())
    rng = np.random.default_rng(int(seed))
    draws: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    for _ in range(int(n_bootstrap)):
        sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_rows: list[dict[str, Any]] = []
        for cluster_id in sampled_clusters.tolist():
            sampled_rows.extend(grouped[str(cluster_id)])
        boot = _summarize_records(sampled_rows, protocol=protocol, ccauc_max_samples=ccauc_max_samples)
        for key in METRIC_KEYS:
            draws[key].append(float(boot[key]))
    metrics = {}
    for key in METRIC_KEYS:
        arr = np.asarray(draws[key], dtype=np.float64)
        metrics[key] = {
            "point_estimate": float(point[key]),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }
    return {
        "metrics": metrics,
        "n_bootstrap": int(n_bootstrap),
        "cluster_count": len(cluster_ids),
        "cluster_key_used": cluster_key,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustered bootstrap confidence intervals for force-grounded v21a direct-eval record files.")
    parser.add_argument("record_files", nargs="+", help="One or more --save-records-jsonl outputs from direct eval.")
    parser.add_argument("--protocol", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-key", type=str, default="auto", choices=["auto", "episode_id", "pair_id", "event_key"])
    parser.add_argument("--ccauc-max-samples", type=int, default=2048)
    args = parser.parse_args()

    per_run: list[dict[str, Any]] = []
    for record_file in args.record_files:
        rows = _read_jsonl(record_file)
        cluster_key_used = _choose_cluster_key(rows, args.cluster_key)
        result = _bootstrap_ci(
            rows,
            protocol=args.protocol,
            ccauc_max_samples=int(args.ccauc_max_samples),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed),
            cluster_key=cluster_key_used,
        )
        result.update(
            {
                "protocol": args.protocol,
                "record_file": str(Path(record_file).expanduser().resolve()),
            }
        )
        per_run.append(result)

    aggregate: dict[str, Any] = {}
    if len(per_run) > 1:
        for key in METRIC_KEYS:
            points = np.asarray([run["metrics"][key]["point_estimate"] for run in per_run], dtype=np.float64)
            aggregate[key] = {
                "point_estimate_mean": float(points.mean()),
                "point_estimate_std": float(points.std(ddof=0)),
            }

    payload = {
        "protocol": args.protocol,
        "record_files": [str(Path(path).expanduser().resolve()) for path in args.record_files],
        "n_bootstrap": int(args.n_bootstrap),
        "requested_cluster_key": args.cluster_key,
        "per_run": per_run,
        "across_run_point_estimate_stats": aggregate,
    }
    save_eval_result(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
