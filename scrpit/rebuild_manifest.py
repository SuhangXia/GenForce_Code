#!/usr/bin/env python3
"""Rebuild manifest.json from existing episode directories."""
import json
import sys
from pathlib import Path

SCALES_MM = [15, 18, 20, 22, 25]


def _infer_scales_from_dir(ep_dir: Path, ep_name: str) -> dict:
    """Build scales dict from scale_*mm directories (for episodes without metadata)."""
    scales = {}
    for s in SCALES_MM:
        scale_dir = ep_dir / f"scale_{s}mm"
        render = scale_dir / "render.jpg"
        if render.exists():
            scales[f"scale_{s}mm"] = {
                "physical_width_mm": s,
                "physical_height_mm": s,
                "image": f"{ep_name}/scale_{s}mm/render.jpg",
            }
    return scales


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("adapter_dataset_v2")
    root = root.resolve()
    manifest_path = root / "manifest.json"

    episodes = []
    for ep_dir in sorted(root.glob("episode_*")):
        if not ep_dir.is_dir():
            continue
        ep_name = ep_dir.name
        ep_num = int(ep_dir.name.split("_")[1])

        meta_path = ep_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            eid = ep_num
            indenter = meta.get("indenter", "unknown")
            marker = meta.get("marker_pattern", "?")
            ep_entry = {
                "episode_id": eid,
                "dir": ep_name,
                "indenter": indenter,
                "marker": marker,
            }
        else:
            scales = _infer_scales_from_dir(ep_dir, ep_name)
            if len(scales) < 2:
                continue
            eid = ep_num
            ep_entry = {
                "episode_id": eid,
                "dir": ep_name,
                "indenter": "unknown",
                "marker": "?",
                "scales": scales,
            }
        episodes.append(ep_entry)

    data = {"total_episodes": len(episodes), "total_failures": 0, "scales_mm": SCALES_MM, "episodes": episodes}
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Rebuilt manifest: {len(episodes)} episodes -> {manifest_path}")

if __name__ == "__main__":
    main()
