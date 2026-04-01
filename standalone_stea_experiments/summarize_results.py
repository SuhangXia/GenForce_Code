#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from standalone_stea_experiments.utils import default_results_dir

LINE_RE = re.compile(
    r'^Eval scale=(?P<scale>[\d.]+) split=(?P<split>\w+) use_stea=(?P<use_stea>True|False) '
    r'adapter_kind=(?P<adapter_kind>\w+) \| '
    r'mse_x=(?P<mse_x>[0-9.eE+-]+) mse_y=(?P<mse_y>[0-9.eE+-]+) '
    r'mse_depth=(?P<mse_depth>[0-9.eE+-]+) mse_total=(?P<mse_total>[0-9.eE+-]+)$'
)

DEFAULT_LOGS = {
    'A.no_adapter_baseline_trained_on_16_20_23': 'no_adapter_baseline_matrix.txt',
    'B.canonical20_no_adapter': 'canonical20_no_adapter_matrix.txt',
    'C.canonical20_with_stea': 'canonical20_with_stea_matrix.txt',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize supplemental STEA experiment matrix logs into markdown tables.')
    parser.add_argument('--results-dir', type=str, default=str(default_results_dir()))
    return parser.parse_args()


def load_log(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    results: dict[tuple[str, str], dict[str, float]] = {}
    if not path.exists():
        return results
    for line in path.read_text().splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        key = (match.group('scale'), match.group('split'))
        results[key] = {
            'mse_x': float(match.group('mse_x')),
            'mse_y': float(match.group('mse_y')),
            'mse_depth': float(match.group('mse_depth')),
            'mse_total': float(match.group('mse_total')),
        }
    return results


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser()

    loaded = {
        config_name: load_log(results_dir / filename)
        for config_name, filename in DEFAULT_LOGS.items()
    }
    baseline = loaded['A.no_adapter_baseline_trained_on_16_20_23']

    print('| configuration | scale | split | mse_x | mse_y | mse_depth | mse_total | rel_vs_fair_no_adapter_% |')
    print('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |')
    for config_name, config_results in loaded.items():
        for scale in ('16', '20', '23'):
            for split in ('seen', 'unseen'):
                key = (scale, split)
                metrics = config_results.get(key)
                if metrics is None:
                    print(f'| {config_name} | {scale} | {split} | NA | NA | NA | NA | NA |')
                    continue
                baseline_metrics = baseline.get(key)
                rel = 0.0
                if baseline_metrics is not None and baseline_metrics['mse_total'] > 0:
                    rel = (baseline_metrics['mse_total'] - metrics['mse_total']) / baseline_metrics['mse_total'] * 100.0
                print(
                    f'| {config_name} | {scale} | {split} | '
                    f'{metrics["mse_x"]:.6f} | {metrics["mse_y"]:.6f} | '
                    f'{metrics["mse_depth"]:.6f} | {metrics["mse_total"]:.6f} | {rel:.2f} |'
                )


if __name__ == '__main__':
    main()
