# Battery Life Prediction — Physics-Based Inverse + Operator Learning

Physics-grounded remaining-useful-life (RUL) and end-of-life (EOL) prediction
for LFP (lithium iron phosphate) cells, using **open cycling datasets**.

## Problem

Data-driven battery-life models tend to learn per-cell shortcuts: they fit the
idiosyncrasies of individual cells (and of a single manufacturing batch or
charging protocol) rather than the underlying degradation physics. Such models
look accurate in-distribution but collapse when transferred to a new batch,
chemistry supplier, or fast-charge protocol. This project attacks that
**circularity** by anchoring predictions to identifiable physical parameters
recovered from voltage curves, then learning the cycle-to-degradation *operator*
on top of those parameters — with explicit guards and a density boundary that
flag out-of-distribution inputs instead of silently extrapolating.

## Method / Architecture

```
   Voltage curves V(t)              Charge protocol
   (open datasets)                  (C-rates, cutoffs)
          │                                │
          ▼                                │
 ┌──────────────────────┐                  │
 │ PyBaMM SPM inverse    │  differential    │
 │ (physics-based)       │  evolution +     │
 │  θ = (SoC, ε_neg,     │  multi-cycle     │
 │       ε_pos, k_SEI)   │  V(t) fitting    │
 └──────────┬───────────┘                  │
            │ identifiable physical θ       │
            ▼                                ▼
 ┌───────────────────────────────────────────────┐
 │ Protocol-conditioned DeepONet (operator learn) │
 │  branch: θ + protocol features                 │
 │  trunk : cycle index (Fourier-encoded)         │
 │  → capacity trajectory Q(n)                     │
 └──────────┬────────────────────────────────────┘
            │
            ▼
 ┌───────────────────────────┐   in-distribution?
 │ Density boundary (GMM)     │──────────────┐
 │  θ-space validity prior    │              │
 └──────────┬────────────────┘               ▼
            │                          OOD → abstain / flag
            ▼
   EOL / capacity-trajectory prediction  (+ uncertainty)
```

Components (each maps to code in `src/` and `scripts/`):

1. **PyBaMM SPM inverse** — recover identifiable physical parameters
   θ = (initial SoC, negative/positive active-material volume fractions, and a
   log SEI rate constant) by fitting a Single Particle Model (with an
   EC-reaction-limited SEI submodel) to observed voltage curves across multiple
   cycles. A two-stage strategy keeps the forward-simulation cost tractable.
   See `scripts/run_multi_cycle_inverse.py`.
2. **Protocol-conditioned DeepONet** — an operator-learning network whose branch
   consumes θ plus charge-protocol features and whose trunk consumes a
   Fourier-encoded cycle index, producing the capacity-fade trajectory Q(n).
   See `src/deeponet_protocol.py` and `scripts/train_deeponet_protocol.py`.
3. **Density boundary (GMM)** — a Gaussian-mixture density over the physical
   θ-space provides a validity prior, so inputs that fall outside the region
   supported by the training cells are flagged rather than extrapolated.
   See `scripts/fit_theta_gmm.py`.
4. **Domain guards** — `src/battery_guards.py` encodes hard assertions against
   common failure modes (e.g. cell-disjoint train/eval splits, C-rate→current
   conversion per chemistry, grid coverage of the real θ range), so leakage and
   silent domain shift raise errors instead of inflating scores.

## Datasets (all public)

| Dataset | Chemistry | Reference | Link |
|---|---|---|---|
| Toyota Research Institute (TRI / Severson) | LFP/graphite (A123) | Severson et al., *Nature Energy* 2019 | https://data.matr.io/1 |
| HUST | LFP | Ma et al., 2022 | https://data.mendeley.com/datasets/nsc7hnsg4s |
| CALCE A123 | LFP (A123) | CALCE, University of Maryland | https://calce.umd.edu/battery-data |

No raw data or derived binaries are bundled. See `REPRODUCIBILITY.md` for
acquisition steps.

## How to run

```bash
# 1) environment (Python 3.11)
conda env create -f environment.yml
conda activate pybamm-inv

# 2) fast unit tests (no data required)
pytest tests/ -m "not slow"

# 3) physics-based multi-cycle inverse (needs a downloaded dataset)
python scripts/run_multi_cycle_inverse.py --cells batch1_cell5 --cycles 10,100 --maxiter 10 --workers 1

# 4) fit the θ-space density boundary (GMM)
python scripts/fit_theta_gmm.py --K 5

# 5) train the protocol-conditioned DeepONet operator
python scripts/train_deeponet_protocol.py
```

Full commands, dataset paths, and seed guidance are in **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

## Results

Quantitative accuracy, cross-batch transfer, and ablations are reported in a
**manuscript in preparation** and are intentionally omitted here. This
repository documents the method and provides a runnable reference
implementation; it does not publish headline metrics ahead of the paper.

## Layout

```
src/       core method modules (inverse data prep, DeepONet, GMM inputs, guards)
scripts/   representative entry points (inverse / GMM / operator training)
tests/     self-contained unit tests (no private data)
configs/   battery-guard lookup tables (chemistry specs, θ bounds)
```

## License

MIT — see [LICENSE](LICENSE).
