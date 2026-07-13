"""
V7 phase label extraction for real TRI, HUST, and CALCE trajectories.

This module is analysis-only. It reads measured capacity trajectories, applies
the existing KP detector with sensitivity checks, and writes labels for
downstream evaluation/generation decisions. Failed and ambiguous KP records are
kept explicitly in the output.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kp_detection import detect_knee_point

OUT_DIR = PROJECT_ROOT / "results" / "phase_labels"
OUT_JSON = OUT_DIR / "phase_labels.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "phase_label_extraction_report.md"
PREREQ_JSON = PROJECT_ROOT / "results" / "v7_prereq" / "prerequisite_validation.json"

TRI_BATCHES = [
    PROJECT_ROOT / "data" / "raw" / "batch1.pkl",
    PROJECT_ROOT / "data" / "raw" / "batch2.pkl",
]
HUST_ZIP = PROJECT_ROOT / "data" / "raw" / "hust" / "hust_data.zip"
CALCE_ZIP = PROJECT_ROOT / "data" / "raw" / "calce" / "A123_094.zip"

EOL_THRESHOLD = 0.80
KP_RANGES = {
    "default_035_070": (0.35, 0.70),
    "early_030_065": (0.30, 0.65),
    "late_040_075": (0.40, 0.75),
    "wide_025_080": (0.25, 0.80),
}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_tri_qn() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    qn: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pkl_path in TRI_BATCHES:
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            batch = pickle.load(f)
        for cell_key, cell_data in batch.items():
            summary = cell_data.get("summary", {})
            qd = np.asarray(summary.get("QD", []), dtype=float)
            cycles = np.asarray(summary.get("cycle", []), dtype=float)
            if len(cycles) != len(qd) or len(qd) == 0:
                cycles = np.arange(1, len(qd) + 1, dtype=float)
            cleaned = clean_trajectory(cycles, qd)
            if cleaned is not None:
                qn[cell_key] = cleaned
    return qn


def load_hust_qn() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    qn: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not HUST_ZIP.exists():
        return qn
    with zipfile.ZipFile(HUST_ZIP) as zf:
        names = [n for n in zf.namelist() if n.startswith("our_data/") and n.endswith(".pkl")]
        for name in sorted(names):
            raw_id = Path(name).stem
            try:
                with zf.open(name) as f:
                    raw = pickle.load(f)
                cell = raw[raw_id]
                dq = cell["dq"]
                cycle_keys = sorted(dq.keys())
                cycles = np.asarray(cycle_keys, dtype=float)
                qd = np.asarray([float(dq[c]) / 1000.0 for c in cycle_keys], dtype=float)
            except Exception:
                continue
            cleaned = clean_trajectory(cycles, qd)
            if cleaned is not None:
                qn[f"hust_{raw_id}"] = cleaned
    return qn


def load_calce_qn() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not CALCE_ZIP.exists():
        return {}
    try:
        from scripts.etl_calce import load_calce_raw

        raw = load_calce_raw()
        cycles = np.asarray(raw["cycle_nums"], dtype=float)
        qd = np.asarray(raw["QD_Ah"], dtype=float)
        cleaned = clean_trajectory(cycles, qd)
        return {"calce_A123_094": cleaned} if cleaned is not None else {}
    except Exception as exc:
        return {"_load_error": (np.asarray([], dtype=float), np.asarray([np.nan, str(exc)], dtype=object))}


def clean_trajectory(cycles: np.ndarray, qd: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.isfinite(cycles) & np.isfinite(qd) & (qd > 0)
    cycles, qd = cycles[mask], qd[mask]
    if len(qd) < 10:
        return None
    order = np.argsort(cycles)
    return cycles[order], qd[order]


def estimate_eol(cycles: np.ndarray, qd: np.ndarray, threshold: float = EOL_THRESHOLD) -> int | None:
    if len(qd) < 2 or qd[0] <= 0:
        return None
    below = np.where(qd < threshold * qd[0])[0]
    if len(below) == 0:
        return None
    return int(round(float(cycles[int(below[0])])))


def linear_slope(cycles: np.ndarray, qd: np.ndarray) -> float | None:
    if len(qd) < 3:
        return None
    try:
        return float(np.polyfit(cycles.astype(float), qd.astype(float), 1)[0])
    except Exception:
        return None


def robust_noise_estimate(cycles: np.ndarray, qd: np.ndarray) -> float | None:
    if len(qd) < 6:
        return None
    slope = linear_slope(cycles, qd)
    if slope is None:
        return None
    intercept = float(np.mean(qd) - slope * np.mean(cycles))
    residuals = qd - (slope * cycles + intercept)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    return 1.4826 * mad


def summarize_array(values: list[float | None]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def classify_detection(default: dict[str, Any], eol: int | None, sensitivity: dict[str, Any], n_points: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if n_points < 20:
        return "failure", ["fewer than 20 capacity points"]
    if eol is None:
        reasons.append("EOL not reached in observed trajectory")
    if default.get("method") == "fallback_midpoint":
        reasons.append("midpoint fallback")
    if default.get("method") == "max_curvature":
        reasons.append("Bacon-Watts failed; max-curvature fallback used")
    if not default.get("success", False):
        reasons.append(default.get("reason") or "Bacon-Watts unsuccessful")
    if eol is not None and default.get("knee_cycle") is not None:
        ratio = float(default["knee_cycle"]) / float(eol)
        if ratio < 0.25 or ratio > 0.80:
            reasons.append(f"KP/EOL ratio outside broad physical range: {ratio:.3f}")
    shift = sensitivity.get("max_abs_shift_over_eol")
    if shift is not None and shift > 0.10:
        reasons.append("KP shifts by >10% EOL under search-range sensitivity")

    if any("midpoint" in reason for reason in reasons):
        return "failure", reasons
    if reasons:
        return "ambiguous", reasons
    return "success", []


def sensitivity_check(qd: np.ndarray, cycles: np.ndarray, eol: int | None, kp_default: int | None) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for name, kp_range in KP_RANGES.items():
        try:
            result = detect_knee_point(qd, eol_cycle=eol, cycles=cycles, knee_search_range=kp_range)
            variants[name] = {
                "knee_cycle": int(result["knee_cycle"]) if result.get("knee_cycle") is not None else None,
                "method": result.get("method"),
                "success": bool(result.get("success", False)),
                "reason": result.get("reason", ""),
            }
        except Exception as exc:
            variants[name] = {"knee_cycle": None, "method": "error", "success": False, "reason": str(exc)}

    kps = [v["knee_cycle"] for v in variants.values() if v["knee_cycle"] is not None]
    if kps and eol is not None and kp_default is not None:
        max_abs_shift = float(max(abs(kp - kp_default) for kp in kps))
        max_abs_shift_over_eol = float(max_abs_shift / eol)
    else:
        max_abs_shift = None
        max_abs_shift_over_eol = None
    return {
        "variant_kps": variants,
        "max_abs_shift_cycles": max_abs_shift,
        "max_abs_shift_over_eol": max_abs_shift_over_eol,
    }


def extract_cell_label(dataset: str, cell_key: str, cycles: np.ndarray, qd: np.ndarray) -> dict[str, Any]:
    if cell_key == "_load_error":
        return {
            "dataset": dataset,
            "cell_key": cell_key,
            "n_points": 0,
            "detection_status": "failure",
            "failure_reasons": [str(qd[-1])],
            "ambiguous_reasons": [],
        }

    eol = estimate_eol(cycles, qd)
    default = detect_knee_point(qd, eol_cycle=eol, cycles=cycles, knee_search_range=KP_RANGES["default_035_070"])
    kp = int(default["knee_cycle"]) if default.get("knee_cycle") is not None else None
    sensitivity = sensitivity_check(qd, cycles, eol, kp)
    status, reasons = classify_detection(default, eol, sensitivity, len(qd))

    first_cycle = int(round(float(cycles[0])))
    last_cycle = int(round(float(cycles[-1])))
    early_limit = max(first_cycle, int(round(0.10 * eol))) if eol else int(round(float(cycles[min(len(cycles) - 1, max(2, len(cycles) // 10))])))
    early_mask = cycles <= early_limit
    post_mask = cycles >= kp if kp is not None else np.zeros_like(cycles, dtype=bool)
    pre_mask = cycles <= kp if kp is not None else np.zeros_like(cycles, dtype=bool)
    plateau_duration = int(max(kp - first_cycle, 0)) if kp is not None else None

    return {
        "dataset": dataset,
        "cell_key": cell_key,
        "n_points": int(len(qd)),
        "first_cycle": first_cycle,
        "last_cycle": last_cycle,
        "eol_definition": f"first cycle below {EOL_THRESHOLD:.0%} of first observed discharge capacity",
        "eol_threshold": EOL_THRESHOLD,
        "eol_cycle": eol,
        "kp_cycle": kp,
        "kp_over_eol": float(kp / eol) if kp is not None and eol else None,
        "early_slope": linear_slope(cycles[early_mask], qd[early_mask]),
        "plateau_duration": plateau_duration,
        "pre_kp_slope": linear_slope(cycles[pre_mask], qd[pre_mask]) if kp is not None else None,
        "post_kp_slope": linear_slope(cycles[post_mask], qd[post_mask]) if kp is not None else None,
        "noise_estimate": robust_noise_estimate(cycles, qd),
        "detection_status": status,
        "kp_method": default.get("method"),
        "bw_success": bool(default.get("success", False)),
        "fit_rmse": finite_float(default.get("fit_rmse")),
        "failure_reasons": reasons if status == "failure" else [],
        "ambiguous_reasons": reasons if status == "ambiguous" else [],
        "sensitivity": sensitivity,
    }


def summarize_dataset(dataset: str, labels: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(label["detection_status"] for label in labels)
    n = len(labels)
    return {
        "dataset": dataset,
        "n_cells": n,
        "counts": dict(counts),
        "rates": {key: float(value / n) for key, value in counts.items()} if n else {},
        "kp_over_eol": summarize_array([label.get("kp_over_eol") for label in labels]),
        "early_slope": summarize_array([label.get("early_slope") for label in labels]),
        "plateau_duration": summarize_array([label.get("plateau_duration") for label in labels]),
        "post_kp_slope": summarize_array([label.get("post_kp_slope") for label in labels]),
        "noise_estimate": summarize_array([label.get("noise_estimate") for label in labels]),
    }


def kp_generation_decision(summaries: dict[str, Any], prereq: dict[str, Any] | None) -> dict[str, Any]:
    if prereq is not None:
        rec = prereq.get("recommendation", {})
        return {
            "unblocks_generation": rec.get("decision") == "allow",
            "decision": rec.get("decision"),
            "summary": rec.get("summary"),
            "source": str(PREREQ_JSON.relative_to(PROJECT_ROOT)),
            "reasons": rec.get("reasons", []),
        }

    reasons = []
    for dataset in ("TRI", "HUST"):
        success_rate = summaries.get(dataset, {}).get("rates", {}).get("success", 0.0)
        if success_rate < 0.80:
            reasons.append(f"{dataset} strict KP success below 80% ({success_rate:.1%}).")
    return {
        "unblocks_generation": not reasons,
        "decision": "allow" if not reasons else "block",
        "summary": "Fallback decision from extracted labels only.",
        "source": "phase label extraction",
        "reasons": reasons,
    }


def extract_phase_labels() -> dict[str, Any]:
    qn = {
        "TRI": load_tri_qn(),
        "HUST": load_hust_qn(),
        "CALCE": load_calce_qn(),
    }
    labels = {
        dataset: [extract_cell_label(dataset, cell_key, cycles, qd) for cell_key, (cycles, qd) in sorted(cells.items())]
        for dataset, cells in qn.items()
    }
    summaries = {dataset: summarize_dataset(dataset, rows) for dataset, rows in labels.items()}
    prereq = json.loads(PREREQ_JSON.read_text()) if PREREQ_JSON.exists() else None
    return {
        "meta": {
            "goal": "docs/v7_goals/04_phase_label_extraction.md",
            "analysis_only": True,
            "generation_run": False,
            "inputs": {
                "tri_batches": [str(p.relative_to(PROJECT_ROOT)) for p in TRI_BATCHES],
                "hust_zip": str(HUST_ZIP.relative_to(PROJECT_ROOT)),
                "calce_zip": str(CALCE_ZIP.relative_to(PROJECT_ROOT)),
                "prerequisite_validation": str(PREREQ_JSON.relative_to(PROJECT_ROOT)),
            },
            "kp_search_ranges": KP_RANGES,
        },
        "summaries": summaries,
        "kp_labels_unblock_generation": kp_generation_decision(summaries, prereq),
        "labels": labels,
    }


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_report(payload)


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# V7 Phase Label Extraction Report",
        "",
        "This report extracts real-data phase labels only. No synthetic trajectory generation was run.",
        "",
        "## Generation Gate",
        "",
    ]
    gate = payload["kp_labels_unblock_generation"]
    lines += [
        f"- KP labels unblock generation: **{gate['unblocks_generation']}**",
        f"- Decision source: `{gate['source']}`",
        f"- Decision: `{gate['decision']}`",
        f"- Summary: {gate['summary']}",
    ]
    if gate.get("reasons"):
        lines += ["- Reasons:"] + [f"  - {reason}" for reason in gate["reasons"]]

    lines += [
        "",
        "## Per-Dataset Counts",
        "",
        "| Dataset | n cells | success | ambiguous | failure | median KP/EOL | median plateau duration | median post-KP slope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("TRI", "HUST", "CALCE"):
        summary = payload["summaries"][dataset]
        rates = summary.get("rates", {})
        counts = summary.get("counts", {})
        lines.append(
            f"| {dataset} | {summary['n_cells']} | {counts.get('success', 0)} ({fmt_pct(rates.get('success', 0.0))}) | "
            f"{counts.get('ambiguous', 0)} ({fmt_pct(rates.get('ambiguous', 0.0))}) | "
            f"{counts.get('failure', 0)} ({fmt_pct(rates.get('failure', 0.0))}) | "
            f"{summary['kp_over_eol'].get('median')} | {summary['plateau_duration'].get('median')} | "
            f"{summary['post_kp_slope'].get('median')} |"
        )

    lines += [
        "",
        "## Failure And Ambiguity Policy",
        "",
        "KP failures and ambiguous labels are retained in `results/phase_labels/phase_labels.json` with `failure_reasons` or `ambiguous_reasons`. They are not filtered out of the dataset summaries.",
        "",
        "## Schema",
        "",
        "Each cell label includes `eol_cycle`, `kp_cycle`, `kp_over_eol`, `early_slope`, `plateau_duration`, `post_kp_slope`, `noise_estimate`, and `detection_status`.",
        "",
        "## Output",
        "",
        "- Machine-readable labels: `results/phase_labels/phase_labels.json`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    payload = extract_phase_labels()
    write_outputs(payload)
    compact = {
        "output": str(OUT_JSON),
        "report": str(REPORT_PATH),
        "counts": {dataset: summary["counts"] for dataset, summary in payload["summaries"].items()},
        "kp_labels_unblock_generation": payload["kp_labels_unblock_generation"]["unblocks_generation"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
