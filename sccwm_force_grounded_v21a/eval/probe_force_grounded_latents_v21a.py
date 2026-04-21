#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sccwm.eval.common import _batch_item, _move_eval_batch, _protocol_filter, save_eval_result
from sccwm.eval.eval_cross_band_any_to_any import _build_cross_band_pair_specs, _build_loader_from_pair_specs
from sccwm.train.common import default_device
from sccwm.utils.config import load_config_with_overrides
from sccwm_force_grounded_v21a.eval.eval_sccwm_force_grounded_direct_v21a import (
    EMBEDDING_VIEW_CHOICES,
    STRICT_CROSS_BAND_PROTOCOLS,
    STANDARD_PROTOCOLS,
    build_force_grounded_v21a_eval_loader,
    load_force_grounded_v21a_for_eval,
)
from sccwm_force_grounded_v21a.models import select_embedding_view_v21a

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


try:  # pragma: no cover - availability depends on env
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score

    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    LogisticRegression = None  # type: ignore
    confusion_matrix = None  # type: ignore
    f1_score = None  # type: ignore
    HAVE_SKLEARN = False


PROBE_TASK_CHOICES = ["scale", "branch", "marker"]


def _scale_bucket_labels(scale_mm: torch.Tensor, edges_mm: list[float]) -> torch.Tensor:
    edges = scale_mm.new_tensor(edges_mm, dtype=torch.float32)
    return torch.bucketize(scale_mm.to(torch.float32), edges)


def _metadata_field(batch: dict[str, Any], key: str, index: int) -> Any | None:
    metadata = batch.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return _batch_item(metadata[key], index)
    if key in batch:
        return _batch_item(batch[key], index)
    return None


def _pair_row_common(batch: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    source_episode = int(batch["source_episode_id"][sample_idx].item()) if "source_episode_id" in batch else None
    target_episode = int(batch["target_episode_id"][sample_idx].item()) if "target_episode_id" in batch else None
    episode_id = int(batch["episode_id"][sample_idx].item()) if "episode_id" in batch else None
    pair_id = str(_batch_item(batch["pair_id"], sample_idx)) if "pair_id" in batch else None
    if episode_id is None and source_episode is not None and target_episode is not None and source_episode == target_episode:
        episode_id = source_episode
    return {
        "event_key": pair_id
        or (
            f"{episode_id}::{int(batch['global_seq_index'][sample_idx].item())}"
            if "global_seq_index" in batch
            else f"{source_episode}::{target_episode}::{sample_idx}"
        ),
        "pair_id": pair_id,
        "episode_id": episode_id,
        "source_episode_id": source_episode if source_episode is not None else episode_id,
        "target_episode_id": target_episode if target_episode is not None else episode_id,
        "source_marker_name": _metadata_field(batch, "source_marker_name", sample_idx),
        "target_marker_name": _metadata_field(batch, "target_marker_name", sample_idx),
        "source_scale_mm": float(batch["source_scale_mm"][sample_idx].item()),
        "target_scale_mm": float(batch["target_scale_mm"][sample_idx].item()),
        "source_scale_split": str(_batch_item(batch["source_scale_split"], sample_idx)),
        "target_scale_split": str(_batch_item(batch["target_scale_split"], sample_idx)),
        "boundary_subset": str(_batch_item(batch["boundary_subset"], sample_idx)),
        "is_unseen_indenter": bool(batch["is_unseen_indenter"][sample_idx].item()),
        "is_unseen_scale_target": bool(batch["is_unseen_scale_target"][sample_idx].item()),
    }


def _collect_pair_rows(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    split: str,
    protocol: str | None,
    sequence_length: int | None,
    embedding_view: str,
) -> list[dict[str, Any]]:
    device = default_device(cfg)
    model = load_force_grounded_v21a_for_eval(cfg, checkpoint_path, device)
    model.eval()
    loss_cfg = cfg.get("loss", {})
    scale_bucket_edges_mm = [float(v) for v in loss_cfg.get("scale_bucket_edges_mm", [14.0, 18.0, 22.0])]
    strict_protocol = protocol in STRICT_CROSS_BAND_PROTOCOLS if protocol else False
    if strict_protocol:
        pair_specs, _ = _build_cross_band_pair_specs(cfg, protocol=str(protocol), limit=0)
        loader = _build_loader_from_pair_specs(cfg, pair_specs, sequence_length=sequence_length)
    else:
        loader = build_force_grounded_v21a_eval_loader(cfg, split=split, sequence_length=sequence_length, device=device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"Collect probe rows {split} {protocol or 'all'} {embedding_view}", total=len(loader), leave=False)
        for batch in iterator:
            batch = _move_eval_batch(batch, device)
            absolute_contact = batch["absolute_contact_xy_mm"] if bool(batch["has_absolute_contact_xy_mm"].any()) else None
            source = model.forward_single(
                batch["source_obs"],
                batch["source_coord_map"],
                batch["source_scale_mm"],
                absolute_contact_xy_mm=absolute_contact,
                world_origin_xy_mm=batch["world_origin_xy_mm"],
                valid_mask=batch["seq_valid_mask"],
            )
            target = model.forward_single(
                batch["target_obs"],
                batch["target_coord_map"],
                batch["target_scale_mm"],
                absolute_contact_xy_mm=absolute_contact,
                world_origin_xy_mm=batch["world_origin_xy_mm"],
                valid_mask=batch["seq_valid_mask"],
            )
            source_embedding = select_embedding_view_v21a(source, embedding_view)
            target_embedding = select_embedding_view_v21a(target, embedding_view)
            source_scale_bucket = _scale_bucket_labels(batch["source_scale_mm"], scale_bucket_edges_mm)
            target_scale_bucket = _scale_bucket_labels(batch["target_scale_mm"], scale_bucket_edges_mm)
            for sample_idx in range(source_embedding.shape[0]):
                row = {
                    **_pair_row_common(batch, sample_idx),
                    "source_embedding": source_embedding[sample_idx].detach().cpu().tolist(),
                    "target_embedding": target_embedding[sample_idx].detach().cpu().tolist(),
                    "source_scale_bucket": int(source_scale_bucket[sample_idx].item()),
                    "target_scale_bucket": int(target_scale_bucket[sample_idx].item()),
                }
                rows.append(row)
    if protocol and not strict_protocol:
        rows = [row for row in rows if _protocol_filter(row, protocol, cfg)]
    return rows


def _strict_partition(rows: list[dict[str, Any]], *, partition: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("pair_id") or row.get("event_key"))
        bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10
        if partition == "train" and bucket < 7:
            selected.append(row)
        elif partition == "eval" and bucket >= 7:
            selected.append(row)
    return selected


def _expand_branch_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pair_rows:
        common = {
            "pair_id": row.get("pair_id"),
            "event_key": row.get("event_key"),
            "episode_id": row.get("episode_id"),
            "source_episode_id": row.get("source_episode_id"),
            "target_episode_id": row.get("target_episode_id"),
        }
        rows.append(
            {
                **common,
                "branch": "source",
                "branch_label": 0,
                "embedding": row["source_embedding"],
                "scale_bucket": row["source_scale_bucket"],
                "marker_label": row.get("source_marker_name"),
            }
        )
        rows.append(
            {
                **common,
                "branch": "target",
                "branch_label": 1,
                "embedding": row["target_embedding"],
                "scale_bucket": row["target_scale_bucket"],
                "marker_label": row.get("target_marker_name"),
            }
        )
    return rows


def _balanced_sample(rows: list[dict[str, Any]], *, label_key: str, seed: int, max_samples: int = 0) -> list[dict[str, Any]]:
    by_label: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row[label_key]].append(row)
    if len(by_label) <= 1:
        return rows
    rng = np.random.default_rng(int(seed))
    per_class = min(len(values) for values in by_label.values())
    if max_samples > 0:
        per_class = min(per_class, max(max_samples // max(len(by_label), 1), 1))
    sampled: list[dict[str, Any]] = []
    for values in by_label.values():
        if len(values) > per_class:
            indices = rng.choice(len(values), size=per_class, replace=False)
            sampled.extend(values[int(idx)] for idx in indices.tolist())
        else:
            sampled.extend(values)
    rng.shuffle(sampled)
    return sampled


def _rows_to_arrays(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    label_to_index: dict[Any, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[Any, int], list[Any]]:
    if label_to_index is None:
        labels = sorted({row[label_key] for row in rows})
        label_to_index = {label: idx for idx, label in enumerate(labels)}
    else:
        labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    x = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    y = np.asarray([label_to_index[row[label_key]] for row in rows], dtype=np.int64)
    return x, y, label_to_index, labels


def _manual_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: list[float] = []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        if tp == 0.0 and fp == 0.0 and fn == 0.0:
            f1s.append(0.0)
            continue
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1s.append(0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else float("nan")


def _manual_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(truth), int(pred)] += 1
    return matrix.tolist()


def _train_torch_probe(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, *, num_classes: int, seed: int) -> np.ndarray:
    torch.manual_seed(int(seed))
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_eval_t = torch.from_numpy(x_eval)
    mean = x_train_t.mean(dim=0, keepdim=True)
    std = x_train_t.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x_train_t = (x_train_t - mean) / std
    x_eval_t = (x_eval_t - mean) / std
    probe = torch.nn.Linear(int(x_train_t.shape[1]), int(num_classes))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    best_loss = float("inf")
    patience = 20
    bad_epochs = 0
    for _ in range(200):
        logits = probe(x_train_t)
        loss = torch.nn.functional.cross_entropy(logits, y_train_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        current = float(loss.item())
        if current + 1e-6 < best_loss:
            best_loss = current
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    with torch.no_grad():
        return probe(x_eval_t).argmax(dim=1).cpu().numpy()


def _fit_probe(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    label_key: str,
    seed: int,
) -> dict[str, Any]:
    x_train, y_train, label_to_index, labels = _rows_to_arrays(train_rows, label_key=label_key)
    x_eval, y_eval, _, _ = _rows_to_arrays(eval_rows, label_key=label_key, label_to_index=label_to_index)
    if HAVE_SKLEARN:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_eval_s = scaler.transform(x_eval)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(seed), multi_class="auto")
        clf.fit(x_train_s, y_train)
        y_pred = clf.predict(x_eval_s)
        macro_f1 = float(f1_score(y_eval, y_pred, average="macro"))
        matrix = confusion_matrix(y_eval, y_pred, labels=np.arange(len(labels))).tolist()
        backend = "sklearn_logistic_regression"
    else:
        y_pred = _train_torch_probe(x_train, y_train, x_eval, num_classes=len(labels), seed=seed)
        macro_f1 = _manual_macro_f1(y_eval, y_pred, len(labels))
        matrix = _manual_confusion_matrix(y_eval, y_pred, len(labels))
        backend = "torch_linear_probe"
    accuracy = float((y_pred == y_eval).mean()) if y_eval.size > 0 else float("nan")
    return {
        "backend": backend,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_count": len(labels),
        "support": int(y_eval.size),
        "labels": [str(label) for label in labels],
        "label_to_index": {str(label): int(index) for label, index in label_to_index.items()},
        "confusion_matrix": matrix,
        "train_label_distribution": {str(k): int(v) for k, v in Counter(row[label_key] for row in train_rows).items()},
        "eval_label_distribution": {str(k): int(v) for k, v in Counter(row[label_key] for row in eval_rows).items()},
    }


def _run_probe_task(
    branch_rows_train: list[dict[str, Any]],
    branch_rows_eval: list[dict[str, Any]],
    *,
    task: str,
    seed: int,
    max_train_samples: int,
    max_eval_samples: int,
) -> dict[str, Any]:
    if task == "scale":
        label_key = "scale_bucket"
    elif task == "branch":
        label_key = "branch_label"
    elif task == "marker":
        label_key = "marker_label"
    else:  # pragma: no cover
        raise ValueError(f"Unsupported probe task: {task}")
    train_rows = [row for row in branch_rows_train if row.get(label_key) is not None]
    eval_rows = [row for row in branch_rows_eval if row.get(label_key) is not None]
    if task == "marker" and (not train_rows or not eval_rows):
        return {"status": "skipped", "reason": "marker_labels_unavailable"}
    if len({row[label_key] for row in train_rows}) <= 1 or len({row[label_key] for row in eval_rows}) <= 1:
        return {"status": "skipped", "reason": f"{task}_labels_insufficient_variation"}
    train_rows = _balanced_sample(train_rows, label_key=label_key, seed=seed, max_samples=max_train_samples)
    eval_rows = _balanced_sample(eval_rows, label_key=label_key, seed=seed + 1, max_samples=max_eval_samples)
    result = _fit_probe(train_rows, eval_rows, label_key=label_key, seed=seed)
    result["status"] = "ok"
    result["label_key"] = label_key
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen post-hoc leakage probes on force-grounded SCCWM v2.1a embeddings.")
    parser.add_argument("--config", type=str, default="sccwm_force_grounded_v21a/configs/sccwm_stage2_force_grounded_v21a.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--protocol", type=str, default="", choices=[""] + STANDARD_PROTOCOLS + STRICT_CROSS_BAND_PROTOCOLS)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--embedding-view", type=str, default="latent_only", choices=EMBEDDING_VIEW_CHOICES)
    parser.add_argument("--probe-tasks", type=str, default="scale,branch,marker")
    parser.add_argument("--max-train-samples", type=int, default=4000)
    parser.add_argument("--max-eval-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config_with_overrides(args.config, args.override)
    protocol = args.protocol or None
    if protocol in STRICT_CROSS_BAND_PROTOCOLS:
        strict_rows = _collect_pair_rows(
            cfg,
            checkpoint_path=args.checkpoint,
            split=args.eval_split,
            protocol=protocol,
            sequence_length=args.sequence_length,
            embedding_view=args.embedding_view,
        )
        train_pair_rows = _strict_partition(strict_rows, partition="train")
        eval_pair_rows = _strict_partition(strict_rows, partition="eval")
        split_note = "strict protocol rows were partitioned 70/30 by deterministic hash of pair_id/event_key"
    else:
        train_pair_rows = _collect_pair_rows(
            cfg,
            checkpoint_path=args.checkpoint,
            split=args.train_split,
            protocol=protocol,
            sequence_length=args.sequence_length,
            embedding_view=args.embedding_view,
        )
        eval_pair_rows = _collect_pair_rows(
            cfg,
            checkpoint_path=args.checkpoint,
            split=args.eval_split,
            protocol=protocol,
            sequence_length=args.sequence_length,
            embedding_view=args.embedding_view,
        )
        split_note = "standard split loaders used directly"

    branch_rows_train = _expand_branch_rows(train_pair_rows)
    branch_rows_eval = _expand_branch_rows(eval_pair_rows)
    tasks = [task.strip() for task in args.probe_tasks.split(",") if task.strip()]
    invalid = sorted(set(tasks) - set(PROBE_TASK_CHOICES))
    if invalid:
        raise ValueError(f"Unsupported probe tasks: {invalid}")

    results: dict[str, Any] = {}
    for task in tasks:
        results[task] = _run_probe_task(
            branch_rows_train,
            branch_rows_eval,
            task=task,
            seed=int(args.seed),
            max_train_samples=int(args.max_train_samples),
            max_eval_samples=int(args.max_eval_samples),
        )

    payload = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "embedding_view": args.embedding_view,
        "protocol": protocol,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "split_note": split_note,
        "train_pair_count": len(train_pair_rows),
        "eval_pair_count": len(eval_pair_rows),
        "train_branch_count": len(branch_rows_train),
        "eval_branch_count": len(branch_rows_eval),
        "probe_tasks": tasks,
        "results": results,
    }
    save_eval_result(payload, args.output)
    print(json.dumps(payload["results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
