# Manifest

Assembled public showcase of an LFP battery-life-prediction method
(PyBaMM SPM inverse + DeepONet operator learning + GMM density boundary).
Curated from a larger private research repository.

## Included files

### Top level
- `README.md` — problem framing, architecture, datasets, how-to-run (no metrics)
- `REPRODUCIBILITY.md` — env, dataset acquisition, run commands, seeds
- `LICENSE` — MIT, Copyright (c) 2026 DongJu Kim
- `.gitignore` — python + data/model/report artifacts
- `environment.yml` — conda env `pybamm-inv` (Python 3.11), copied from source
- `requirements.txt` — pip fallback
- `pytest.ini` — test config

### `src/` — core method modules
- `__init__.py`
- `deeponet_protocol.py` — protocol-conditioned DeepONet (branch=θ+protocol, trunk=Fourier cycle) + sklearn ridge scaffold for tests
- `eval_protocol.py` — split-manifest builder + metric helpers (mape/rmse/rul_error)
- `battery_guards.py` — domain hallucination guards (cell-disjoint split, C-rate→current, grid coverage, etc.)
- `protocol_features.py` — charge-protocol feature extraction
- `uncertainty_layer.py` — uncertainty bundle helpers
- `phase_label_extractor.py` — cycle-phase / knee labeling
- `kp_detection.py` — knee-point detection (dep of phase_label_extractor)
- `download_tri_data.py` — TRI/Severson batch loader (dep of the inverse script)
- `data_preprocessor.py` — cycle/segment extraction, downsampling (dep of the inverse script)

### `scripts/` — representative entry points
- `run_multi_cycle_inverse.py` — PyBaMM SPM two-stage multi-cycle inverse (θ incl. SEI rate)
- `train_deeponet_protocol.py` — DeepONet operator training (imports `deeponet_protocol`, `eval_protocol`)
- `fit_theta_gmm.py` — GMM density boundary over θ-space

### `tests/` — self-contained (no private data)
- `__init__.py`, `conftest.py` (dummy fixtures only)
- `test_deeponet_protocol.py`, `test_eval_protocol.py`, `test_battery_guards.py`,
  `test_protocol_features.py`, `test_phase_label_extractor.py`,
  `test_uncertainty_layer.py`, `test_multi_cycle_inverse.py` (mocks the forward sim)

### `configs/`
- `battery_guards_lookup.yaml` — chemistry specs / θ bounds used by the guards

## Edits applied during curation
- `scripts/fit_theta_gmm.py` — removed an absolute home-directory font path (kept the system font path).
- `src/battery_guards.py` — removed an internal source path and a meeting-log date from the docstring.
- `tests/test_battery_guards.py` — removed an internal source path; reworded one comment whose arithmetic literal collided with a banned number token.

## EXCLUDED (why)

- `paper/`, `papers/` — draft manuscript and literature vault. **Unpublished** (paper in preparation).
- `presentation/`, `*.pptx`, `docs/*.html`, `docs/REPORT.zip` — slides and internal reports. **Internal / bloat.**
- `_backup/` — miscellaneous backups. **Bloat.**
- Repo-internal working notes (agent config file, handover prompt, hub index), `docs/` reports, and `site/` (internal doc site) — roadmaps, handover prompts, working notes. **Internal / confidential.**
- `results/` (incl. `results/discovery_oracle/`, `results/theta_grid/`, model `.pt`, `.npz`, `.pkl`) — computed artifacts and trained models. **Bloat + would embed metrics.**
- `data/` — raw and processed datasets. **Large / redistribution-restricted; public sources linked instead.**
- `goal_1..6_*/` subprojects — parallel exploratory tracks (SHAP attribution, counterfactual RUL, parametric PI, kSEI identifiability, battery TTT). **Out of scope** for this minimal method showcase.
- Many `scripts/` (e.g. `run_zero_shot_eval.py`, `train_deeponet_multicycle_v3.py`, `exp*_*.py`, `generate_*figure*.py`, oracle/benchmark scripts) — contain **specific performance numbers / headline metrics** hardcoded in docstrings or logging, or produce figures/reports. **Unpublished metrics / internal.**
- `README.md` (original, ~37 KB) — full of headline metrics. Replaced with a metrics-free README. **Unpublished metrics.**
- `.github/`, `CHANGELOG.md`, `ENVIRONMENT.md` — internal CI/changelog/preflight notes. **Internal** (env info folded into REPRODUCIBILITY.md).

## Verification
Grep over all included files for the banned headline-metric tokens, the
cross-batch transfer percentages, and the internal v1/v2 baseline figures
returns **no matches**. No absolute filesystem paths, usernames, or internal
meeting-log references remain.
