from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def compute_sass(predictions: Iterable[dict[str, object]]) -> dict[str, float]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for item in predictions:
        key = str(item["event_key"])
        value = np.asarray(item["pred"], dtype=np.float32)
        if value.shape != (3,):
            raise ValueError(f"SASS pred must be shape (3,), got {value.shape} for key={key}")
        grouped[key].append(value)
    var_x: list[float] = []
    var_y: list[float] = []
    var_depth: list[float] = []
    for values in grouped.values():
        if len(values) < 2:
            continue
        arr = np.stack(values, axis=0)
        axis_var = arr.var(axis=0)
        var_x.append(float(axis_var[0]))
        var_y.append(float(axis_var[1]))
        var_depth.append(float(axis_var[2]))
    if not var_x:
        return {"sass_x": 0.0, "sass_y": 0.0, "sass_depth": 0.0, "sass_mean": 0.0}
    sass_x = float(np.mean(var_x))
    sass_y = float(np.mean(var_y))
    sass_depth = float(np.mean(var_depth))
    return {
        "sass_x": sass_x,
        "sass_y": sass_y,
        "sass_depth": sass_depth,
        "sass_mean": float((sass_x + sass_y + sass_depth) / 3.0),
    }
