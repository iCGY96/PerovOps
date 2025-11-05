"""Analyze scope attempt files and plot layer-count distribution."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def _iter_scope_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.json")):
        if path.is_file():
            yield path


def _load_layer_count(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    scope = payload.get("scope") or {}
    layers = scope.get("layers")
    if isinstance(layers, list):
        return len(layers)
    return None


def _collect_counts(directory: Path) -> List[int]:
    counts: List[int] = []
    for file_path in _iter_scope_files(directory):
        count = _load_layer_count(file_path)
        if count is not None:
            counts.append(count)
    return counts


def _save_plot(counts: List[int], output_path: Path) -> None:
    if not counts:
        print("No valid layer counts found; skipping plot generation.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=range(min(counts), max(counts) + 2), edgecolor="black", align="left")
    plt.xlabel("Layer count")
    plt.ylabel("Frequency")
    plt.title("Scope layer count distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scope layer counts and plot distribution.")
    parser.add_argument("--directory", type=Path, default='./history/run_20251007/02_parse/scope_attempts', help="Directory containing scope_attempt_*.json files")
    parser.add_argument(
        "--output",
        type=Path,
        default='./scripts/results.png',
        help="Path for the output plot image (default: <directory>/scope_layers_distribution.png)",
    )
    args = parser.parse_args()

    directory = args.directory.expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    counts = _collect_counts(directory)
    if counts:
        stats = Counter(counts)
        print("Layer count distribution (count -> frequency):")
        for count, freq in sorted(stats.items()):
            print(f"  {count}: {freq}")
        print(f"Total files with layers: {len(counts)}")
        print(f"Min layers: {min(counts)}")
        print(f"Max layers: {max(counts)}")
        print(f"Average layers: {sum(counts) / len(counts):.2f}")
    else:
        print("No valid layer counts found in the provided directory.")

    output_path = args.output
    if output_path is None:
        output_path = directory / "scope_layers_distribution.png"
    else:
        output_path = output_path.expanduser().resolve()
    _save_plot(counts, output_path)
    if counts:
        print(f"Saved distribution plot to: {output_path}")


if __name__ == "__main__":
    main()

