# Reproducibility

## 1. Environment

- **Python:** 3.11
- **Env name:** `pybamm-inv`

```bash
conda env create -f environment.yml
conda activate pybamm-inv

# verify
python -V
python -c "import pybamm, torch, sklearn; print('PyBaMM', pybamm.__version__)"
```

`pip install -r requirements.txt` also works, but PyBaMM/PyBOP/BEEP install
most reliably from `conda-forge` (used by `environment.yml`).

## 2. Datasets (public — download separately)

None of these are bundled. Download and place under `data/raw/`.

| Dataset | Source | Notes |
|---|---|---|
| TRI / Severson (A123 LFP) | https://data.matr.io/1 | Severson et al., *Nature Energy* 2019. `batch1`/`batch2` `.mat` files. |
| HUST LFP | https://data.mendeley.com/datasets/nsc7hnsg4s | Ma et al., 2022. |
| CALCE A123 | https://calce.umd.edu/battery-data | University of Maryland CALCE. |

The TRI loader (`src/download_tri_data.py`) expects the Severson batch files in
`data/raw/`. Adjust the paths in that module if you store them elsewhere.

## 3. Run commands

```bash
# Unit tests (no data needed)
pytest tests/ -m "not slow"

# Physics-based multi-cycle inverse (recovers physical theta)
#   smoke (single cell, reduced iterations):
python scripts/run_multi_cycle_inverse.py \
    --cells batch1_cell5 --cycles 10,100 --maxiter 10 --workers 1
#   fuller run:
python scripts/run_multi_cycle_inverse.py \
    --cells batch1_cell5,batch1_cell9 --cycles 10,100,200 --maxiter 50 --workers 1

# Density boundary over theta-space (GMM validity prior)
python scripts/fit_theta_gmm.py --K 5 \
    --input-dir data/fixed_d_labels --output-dir results/density_estimator

# Protocol-conditioned DeepONet operator training
python scripts/train_deeponet_protocol.py
```

## 4. Seeds

- Global seed **42** is used throughout (NumPy `default_rng(42)`, sklearn
  `random_state=42`, torch manual seed in the training scripts). Test fixtures
  use fixed seeds so guard and unit tests are deterministic.
- The multi-cycle inverse uses `scipy.differential_evolution`; pass a fixed
  `--seed`/`--workers 1` for deterministic optimization runs.
- Forward simulations are single-threaded (`--workers 1`) for reproducibility;
  raising `--workers` speeds runs up but reorders floating-point reductions.

## 5. Notes

- Quantitative results (accuracy, cross-batch transfer, ablations) are reported
  in a manuscript in preparation and are not included in this repository.
- Runtime: 200-cycle SPM forward simulations are the dominant cost; the inverse
  uses a two-stage strategy (see `scripts/run_multi_cycle_inverse.py` header) to
  keep per-cell time practical on CPU.
