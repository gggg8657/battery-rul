"""Unified evaluation protocol for V7 experiments.

This module defines dataset/cell discovery, deterministic split manifests, and
small metric helpers used by later baseline and protocol-conditioned model
goals. It intentionally keeps CALCE as an external sanity check because the
current repository has only one CALCE cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EvalSplit:
    """One train/eval split under a named evaluation mode."""

    name: str
    mode: str
    train_datasets: list[str]
    eval_datasets: list[str]
    train_cells: list[str]
    eval_cells: list[str]
    metric_scope: str
    notes: str = ""

    @property
    def disjoint(self) -> bool:
        return not (set(self.train_cells) & set(self.eval_cells))


def _stable_fraction(key: str, seed: int = 42) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _read_cell_key(path: Path, fallback_pattern: re.Pattern[str]) -> str | None:
    if path.name.endswith("_index.json"):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        key = data.get("cell_key")
        if key:
            return str(key)
    except Exception:
        pass
    match = fallback_pattern.search(path.name)
    return match.group(1) if match else None


def discover_cells(root: Path = PROJECT_ROOT) -> dict[str, list[str]]:
    """Discover unique cells for TRI, HUST, and CALCE from label files."""

    specs = {
        "tri_severson": (
            root / "data" / "fixed_d_labels",
            re.compile(r"((?:batch\d+)_cell\d+)_cycle\d+\.json$"),
        ),
        "hust_ma": (
            root / "data" / "hust_labels",
            re.compile(r"(hust_cell[\d-]+)_cycle\d+\.json$"),
        ),
        "calce_a123": (
            root / "data" / "calce_labels_v2"
            if (root / "data" / "calce_labels_v2").exists()
            else root / "data" / "calce_labels",
            re.compile(r"(calce_[A-Za-z0-9_]+)_cycle\d+\.json$"),
        ),
    }

    cells: dict[str, list[str]] = {}
    for dataset, (directory, pattern) in specs.items():
        found: set[str] = set()
        if directory.exists():
            for path in directory.glob("*.json"):
                key = _read_cell_key(path, pattern)
                if key:
                    found.add(key)
        cells[dataset] = sorted(found)
    return cells


def deterministic_holdout(
    cells: Sequence[str],
    *,
    eval_fraction: float = 0.2,
    seed: int = 42,
    min_eval: int = 1,
) -> tuple[list[str], list[str]]:
    """Create deterministic train/eval cells from a cell list."""

    ordered = sorted(cells, key=lambda c: _stable_fraction(c, seed))
    if not ordered:
        return [], []
    n_eval = max(min_eval, int(round(len(ordered) * eval_fraction)))
    n_eval = min(n_eval, max(1, len(ordered) - 1)) if len(ordered) > 1 else 1
    eval_cells = sorted(ordered[:n_eval])
    train_cells = sorted(ordered[n_eval:])
    return train_cells, eval_cells


def _prefixed(dataset: str, cells: Iterable[str]) -> list[str]:
    return [f"{dataset}:{cell}" for cell in sorted(cells)]


def build_split_manifest(seed: int = 42, root: Path = PROJECT_ROOT) -> dict:
    """Build the V7 split manifest."""

    cells = discover_cells(root)
    splits: list[EvalSplit] = []

    for dataset in ("tri_severson", "hust_ma"):
        train, eval_ = deterministic_holdout(cells[dataset], seed=seed)
        splits.append(
            EvalSplit(
                name=f"within_{dataset}",
                mode="within_dataset",
                train_datasets=[dataset],
                eval_datasets=[dataset],
                train_cells=_prefixed(dataset, train),
                eval_cells=_prefixed(dataset, eval_),
                metric_scope="statistical",
                notes="Deterministic 80/20 cell-disjoint split.",
            )
        )

    train, eval_ = deterministic_holdout(cells["tri_severson"] + cells["hust_ma"], seed=seed)
    # Recover dataset prefixes for pooled cells by membership.
    tri_set = set(cells["tri_severson"])
    train_prefixed = [
        f"{'tri_severson' if cell in tri_set else 'hust_ma'}:{cell}" for cell in train
    ]
    eval_prefixed = [
        f"{'tri_severson' if cell in tri_set else 'hust_ma'}:{cell}" for cell in eval_
    ]
    splits.append(
        EvalSplit(
            name="pooled_tri_hust",
            mode="pooled",
            train_datasets=["tri_severson", "hust_ma"],
            eval_datasets=["tri_severson", "hust_ma"],
            train_cells=sorted(train_prefixed),
            eval_cells=sorted(eval_prefixed),
            metric_scope="statistical",
            notes="Pooled TRI+HUST deterministic cell-disjoint split.",
        )
    )

    splits.extend(
        [
            EvalSplit(
                name="tri_to_hust",
                mode="cross_dataset",
                train_datasets=["tri_severson"],
                eval_datasets=["hust_ma"],
                train_cells=_prefixed("tri_severson", cells["tri_severson"]),
                eval_cells=_prefixed("hust_ma", cells["hust_ma"]),
                metric_scope="statistical",
                notes="Zero-shot TRI to HUST transfer.",
            ),
            EvalSplit(
                name="hust_to_tri",
                mode="cross_dataset",
                train_datasets=["hust_ma"],
                eval_datasets=["tri_severson"],
                train_cells=_prefixed("hust_ma", cells["hust_ma"]),
                eval_cells=_prefixed("tri_severson", cells["tri_severson"]),
                metric_scope="statistical",
                notes="Zero-shot HUST to TRI transfer.",
            ),
            EvalSplit(
                name="leave_tri_out",
                mode="leave_one_dataset_out",
                train_datasets=["hust_ma"],
                eval_datasets=["tri_severson"],
                train_cells=_prefixed("hust_ma", cells["hust_ma"]),
                eval_cells=_prefixed("tri_severson", cells["tri_severson"]),
                metric_scope="statistical",
                notes="Equivalent to HUST to TRI with current two statistical domains.",
            ),
            EvalSplit(
                name="leave_hust_out",
                mode="leave_one_dataset_out",
                train_datasets=["tri_severson"],
                eval_datasets=["hust_ma"],
                train_cells=_prefixed("tri_severson", cells["tri_severson"]),
                eval_cells=_prefixed("hust_ma", cells["hust_ma"]),
                metric_scope="statistical",
                notes="Equivalent to TRI to HUST with current two statistical domains.",
            ),
            EvalSplit(
                name="calce_sanity",
                mode="external_sanity_check",
                train_datasets=["tri_severson", "hust_ma"],
                eval_datasets=["calce_a123"],
                train_cells=_prefixed("tri_severson", cells["tri_severson"])
                + _prefixed("hust_ma", cells["hust_ma"]),
                eval_cells=_prefixed("calce_a123", cells["calce_a123"]),
                metric_scope="sanity_only",
                notes="CALCE is excluded from statistical claims in this loop.",
            ),
        ]
    )

    manifest = {
        "version": "v7_eval_protocol_2026-05-21",
        "seed": seed,
        "datasets": {
            dataset: {"n_cells": len(values), "cells": values}
            for dataset, values in cells.items()
        },
        "metric_policy": {
            "main_metrics": ["eol_mape", "eol_rmse", "rul_error", "q_mape"],
            "support_metrics": ["kp_mape", "kp_detection_status"],
            "calce_scope": "external_sanity_check_only",
        },
        "splits": [asdict(split) | {"disjoint": split.disjoint} for split in splits],
    }
    violations = [split.name for split in splits if not split.disjoint]
    manifest["batt_h09_all_pass"] = not violations
    manifest["batt_h09_violations"] = violations
    return manifest


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    pairs = [(float(t), float(p)) for t, p in zip(y_true, y_pred) if float(t) != 0.0]
    if not pairs:
        return math.nan
    return sum(abs((t - p) / t) for t, p in pairs) / len(pairs) * 100.0


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    pairs = [(float(t), float(p)) for t, p in zip(y_true, y_pred)]
    if not pairs:
        return math.nan
    return math.sqrt(sum((t - p) ** 2 for t, p in pairs) / len(pairs))


def rul_error(
    true_eol: Sequence[float],
    pred_eol: Sequence[float],
    observation_cycle: float,
) -> list[float]:
    """Return predicted-minus-true RUL errors at a fixed observation cycle."""

    return [
        (float(pred) - observation_cycle) - (float(true) - observation_cycle)
        for true, pred in zip(true_eol, pred_eol)
    ]


def write_manifest(path: Path, *, seed: int = 42, root: Path = PROJECT_ROOT) -> dict:
    manifest = build_split_manifest(seed=seed, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2)
        fout.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write V7 unified split manifest.")
    parser.add_argument(
        "--output",
        default="results/v7_eval_protocol/split_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = write_manifest(PROJECT_ROOT / args.output, seed=args.seed)
    if not manifest["batt_h09_all_pass"]:
        raise SystemExit(f"BATT-H-09 split overlap: {manifest['batt_h09_violations']}")
    print(
        f"Wrote {args.output}: "
        f"{len(manifest['splits'])} splits, "
        f"datasets={ {k: v['n_cells'] for k, v in manifest['datasets'].items()} }"
    )


if __name__ == "__main__":
    main()
