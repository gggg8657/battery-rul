"""Train/evaluate a PyTorch protocol-conditioned DeepONet for V7."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_baseline_table import build_dataset
from scripts.v7_prereq_validation import estimate_eol, load_calce_qn, load_hust_qn, load_tri_qn
from src.deeponet_protocol import ProtocolDeepONet
from src.eval_protocol import build_split_manifest, mape, rmse

OUT_DIR = PROJECT_ROOT / "results" / "v7_protocol_deeponet"
OUT_JSON = OUT_DIR / "protocol_deeponet_eval.json"
OUT_DOC = PROJECT_ROOT / "docs" / "protocol_conditioned_deeponet_report.md"


def _load_qn_maps() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    return {"tri_severson": load_tri_qn(), "hust_ma": load_hust_qn(), "calce_a123": load_calce_qn()}


def _feature(row: dict[str, Any], variant: str) -> list[float]:
    if variant == "theta_only":
        return row["features"]["theta"]
    if variant == "theta_protocol":
        return row["features"]["theta_protocol"]
    if variant == "protocol_only":
        return row["features"]["protocol"]
    if variant == "epsilon_neg_only":
        return row["features"]["epsilon_neg"]
    raise KeyError(variant)


def _cell_payload(cell_id: str) -> tuple[str, str]:
    return tuple(cell_id.split(":", 1))  # type: ignore[return-value]


def _training_samples(rows, qn_maps, train_cells, variant, *, max_points_per_cell=120, max_cycle=2500.0):
    xs, cycles, qs = [], [], []
    for cell_id in train_cells:
        if cell_id not in rows:
            continue
        dataset, cell = _cell_payload(cell_id)
        if cell not in qn_maps.get(dataset, {}):
            continue
        cyc, q = qn_maps[dataset][cell]
        if len(cyc) > max_points_per_cell:
            idx = np.linspace(0, len(cyc) - 1, max_points_per_cell).astype(int)
            cyc, q = cyc[idx], q[idx]
        feat = _feature(rows[cell_id], variant)
        xs.extend([feat] * len(cyc))
        cycles.extend((cyc / max_cycle).tolist())
        qs.extend(q.tolist())
    return np.asarray(xs, dtype=np.float32), np.asarray(cycles, dtype=np.float32), np.asarray(qs, dtype=np.float32)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _fit_model(x_train, n_train, q_train, *, epochs, batch_size, lr, seed, device):
    start = time.perf_counter()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)
    np.random.seed(seed)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(n_train.reshape(-1, 1)), torch.from_numpy(q_train.reshape(-1, 1)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ProtocolDeepONet(branch_in=x_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    for _epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for xb, nb, qb in loader:
            xb = xb.to(device, non_blocking=True)
            nb = nb.to(device, non_blocking=True)
            qb = qb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(xb, nb), qb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        losses.append(total / max(count, 1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    train_info = {
        "final_train_loss": float(losses[-1]),
        "min_train_loss": float(min(losses)),
        "train_seconds": float(train_seconds),
        "device": str(device),
        "model_on_cuda": bool(next(model.parameters()).is_cuda),
        "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
    }
    return model.eval(), scaler, train_info


def _predict(model, scaler, x, cycles_norm):
    x_scaled = scaler.transform(x).astype(np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(
            torch.from_numpy(x_scaled).to(device),
            torch.from_numpy(cycles_norm.astype(np.float32).reshape(-1, 1)).to(device),
        ).cpu().numpy().ravel()
    return pred


def _eval_cell(model, scaler, row, cycles, q, variant, *, max_cycle):
    x = np.asarray([_feature(row, variant)] * len(cycles), dtype=np.float32)
    pred = _predict(model, scaler, x, cycles / max_cycle)
    true_eol = estimate_eol(cycles, q)
    pred_eol = estimate_eol(cycles, pred)
    out = {"q_mape_pct": float(mape(q, pred)), "q_rmse_ah": float(rmse(q, pred))}
    if true_eol is not None and pred_eol is not None:
        out.update({
            "true_eol": float(true_eol),
            "pred_eol": float(pred_eol),
            "eol_abs_error": abs(float(true_eol) - float(pred_eol)),
            "eol_ape_pct": abs(float(true_eol) - float(pred_eol)) / float(true_eol) * 100.0,
        })
    return out


def evaluate_variant(variant, *, split, rows, qn_maps, epochs, batch_size, lr, seed, max_cycle, device):
    print(f"[deeponet] variant={variant} split={split['name']} device={device}", flush=True)
    x_train, n_train, q_train = _training_samples(rows, qn_maps, split["train_cells"], variant, max_cycle=max_cycle)
    if len(q_train) < 100:
        return {"status": "skipped", "reason": "insufficient training samples", "n_samples": int(len(q_train))}
    model, scaler, train_info = _fit_model(x_train, n_train, q_train, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, device=device)
    cell_metrics = []
    for cell_id in split["eval_cells"]:
        if cell_id not in rows:
            continue
        dataset, cell = _cell_payload(cell_id)
        if cell not in qn_maps.get(dataset, {}):
            continue
        cyc, q = qn_maps[dataset][cell]
        metric = _eval_cell(model, scaler, rows[cell_id], cyc, q, variant, max_cycle=max_cycle)
        metric["cell_id"] = cell_id
        cell_metrics.append(metric)
    q_mapes = [m["q_mape_pct"] for m in cell_metrics if np.isfinite(m.get("q_mape_pct", np.nan))]
    eol_apes = [m["eol_ape_pct"] for m in cell_metrics if "eol_ape_pct" in m]
    return {
        "status": "ok",
        "variant": variant,
        "split": split["name"],
        "n_train_samples": int(len(q_train)),
        "n_eval_cells": len(cell_metrics),
        "q_mape_mean_pct": float(np.mean(q_mapes)) if q_mapes else None,
        "q_mape_median_pct": float(np.median(q_mapes)) if q_mapes else None,
        "eol_mape_mean_pct": float(np.mean(eol_apes)) if eol_apes else None,
        "train_info": train_info,
        "cell_metrics": cell_metrics,
    }


def run(args):
    device = _resolve_device(args.device)
    cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    rows = build_dataset()
    qn_maps = _load_qn_maps()
    manifest = build_split_manifest(seed=42)
    split = next(s for s in manifest["splits"] if s["name"] == args.split)
    return {
        "meta": {
            "model": "ProtocolDeepONet",
            "framework": "pytorch",
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": cuda_name,
            "cuda_available": bool(torch.cuda.is_available()),
            "split": args.split,
            "epochs": args.epochs,
            "kp_generation_status": "blocked",
            "note": "First PyTorch protocol-conditioned operator scaffold.",
        },
        "results": {variant: evaluate_variant(variant, split=split, rows=rows, qn_maps=qn_maps, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, max_cycle=args.max_cycle, device=device) for variant in args.variants},
    }


def _fmt(x):
    return "NA" if x is None else f"{x:.2f}"


def write_report(results):
    lines = [
        "# Protocol-Conditioned DeepONet PyTorch Report",
        "",
        f"PyTorch: `{results['meta']['torch_version']}`",
        f"Device: `{results['meta']['device']}`" + (f" / `{results['meta']['cuda_device_name']}`" if results["meta"].get("cuda_device_name") else ""),
        "",
        "| Variant | Split | Train samples | Eval cells | Q MAPE % | EOL MAPE % | Final loss | Peak CUDA MB | Train sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, res in results["results"].items():
        if res.get("status") != "ok":
            lines.append(f"| {variant} | - | - | - | skipped | skipped | - | - | - |")
            continue
        lines.append(
            f"| {variant} | {res['split']} | {res['n_train_samples']} | {res['n_eval_cells']} | "
            f"{_fmt(res['q_mape_mean_pct'])} | {_fmt(res['eol_mape_mean_pct'])} | {res['train_info']['final_train_loss']:.6f} | "
            f"{res['train_info'].get('cuda_peak_allocated_mb', 0.0):.1f} | {res['train_info'].get('train_seconds', 0.0):.2f} |"
        )
    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="pooled_tri_hust")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cycle", type=float, default=2500.0)
    parser.add_argument("--device", default="auto", help="PyTorch device, e.g. auto, cuda, cuda:0, cpu")
    parser.add_argument("--variants", nargs="+", default=["theta_only", "protocol_only", "theta_protocol", "epsilon_neg_only"])
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run(args)
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_report(results)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_DOC}")


if __name__ == "__main__":
    main()
