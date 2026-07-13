"""Validity and uncertainty utilities for V7 evidence synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SignalSummary:
    name: str
    status: str
    n: int
    pearson_q_error: float | None = None
    spearman_q_error: float | None = None
    pearson_eol_error: float | None = None
    spearman_eol_error: float | None = None
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "n": self.n,
            "pearson_q_error": self.pearson_q_error,
            "spearman_q_error": self.spearman_q_error,
            "pearson_eol_error": self.pearson_eol_error,
            "spearman_eol_error": self.spearman_eol_error,
            "interpretation": self.interpretation,
        }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def safe_corr(x: list[float], y: list[float], *, method: str = "pearson") -> float | None:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return None
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if method == "spearman":
        x_arr = rankdata(x_arr)
        y_arr = rankdata(y_arr)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def summarize_signal(records: list[dict[str, Any]], signal_name: str) -> SignalSummary:
    signal_values: list[float] = []
    q_errors: list[float] = []
    eol_signal_values: list[float] = []
    eol_errors: list[float] = []
    for record in records:
        signal = record.get(signal_name)
        q_error = record.get("q_abs_error_pct")
        eol_error = record.get("eol_abs_error_pct")
        if signal is not None and q_error is not None:
            signal_values.append(float(signal))
            q_errors.append(float(q_error))
        if signal is not None and eol_error is not None:
            eol_signal_values.append(float(signal))
            eol_errors.append(float(eol_error))

    pearson_q = safe_corr(signal_values, q_errors, method="pearson")
    spearman_q = safe_corr(signal_values, q_errors, method="spearman")
    pearson_eol = safe_corr(eol_signal_values, eol_errors, method="pearson")
    spearman_eol = safe_corr(eol_signal_values, eol_errors, method="spearman")
    useful = any(abs(v) >= 0.3 for v in [pearson_q, spearman_q, pearson_eol, spearman_eol] if v is not None)
    if not signal_values:
        status = "not_available"
        interpretation = "Signal was not available in the current artifacts."
    elif useful:
        status = "candidate"
        interpretation = "Signal shows at least moderate correlation with observed error and may support caution/rejection."
    else:
        status = "not_predictive"
        interpretation = "Signal is available but does not show a useful error correlation in this evidence set."
    return SignalSummary(
        name=signal_name,
        status=status,
        n=len(signal_values),
        pearson_q_error=pearson_q,
        spearman_q_error=spearman_q,
        pearson_eol_error=pearson_eol,
        spearman_eol_error=spearman_eol,
        interpretation=interpretation,
    )


def build_uncertainty_bundle(records: list[dict[str, Any]], unavailable: dict[str, str] | None = None) -> dict[str, Any]:
    unavailable = unavailable or {}
    signal_names = [
        "dataset_shift_score",
        "protocol_ood_score",
        "domain_gap_score",
        "fewshot_label_count_score",
        "dataset_identity_confounding_score",
    ]
    summaries = {name: summarize_signal(records, name).to_dict() for name in signal_names}
    for name, reason in unavailable.items():
        summaries[name] = {
            "name": name,
            "status": "not_available",
            "n": 0,
            "pearson_q_error": None,
            "spearman_q_error": None,
            "pearson_eol_error": None,
            "spearman_eol_error": None,
            "interpretation": reason,
        }
    useful = [name for name, summary in summaries.items() if summary["status"] == "candidate"]
    return {
        "n_records": len(records),
        "signals": summaries,
        "useful_signals": useful,
        "decision": "use caution/rejection only for candidate signals; do not infer accuracy improvement from uncertainty alone",
    }
