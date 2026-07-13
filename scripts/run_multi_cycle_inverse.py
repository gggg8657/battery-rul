"""
P5-D: Multi-Cycle Inverse Parameter Estimation
===============================================
Cycle 10/100/200 V(t)를 동시에 fit하여 SEI rate 상수(k_SEI) 등
동적 파라미터(theta_4)를 추정하는 prototype.

theta_4 = (SoC_init, eps_neg, eps_pos, log10_k_SEI)  — 4D

2-Stage 역산 전략 (계산 비용 현실화):
    Stage-1: cycle 10 단일 V(t) → (SoC_init, eps_neg, eps_pos) DE 추정
             (SEI 성장 무시 — 초기 cycle은 k_SEI 영향 미미)
    Stage-2: stage-1 파라미터 고정, cycle 100/200 V(t) 비교
             → log10_k_SEI만 scipy.optimize.minimize_scalar (1D) 최적화

이유: 200-cycle forward sim ~29초/회 → 4D 전체 DE는 비현실적.
     k_SEI는 cycle 100/200 용량 fade에 직접 반영 → 1D로 식별 가능.

Multi-cycle loss:
    L = sum_i w_i * ||V_sim(theta, cycle_i) - V_obs(cycle_i)||^2

계산 비용 (실측):
    - cycle 10 forward sim: ~2.7s/eval
    - cycle 100/200 forward sim: ~12s / ~29s per eval
    - Stage-1: ~2.7s × DE(maxiter=50, popsize=5, 3D) ≈ ~20min/셀
    - Stage-2: ~29s × grid(15) + Brent(~10) ≈ ~12min/셀
    - 총 prototype 3셀 (직렬): 약 1~1.5시간 예상

Usage:
    conda activate pybamm-inv

    # 전체 실행 (3 셀, cycle 10/100/200, 2-stage)
    python scripts/run_multi_cycle_inverse.py \\
        --cells batch1_cell5,batch1_cell9,hust_1-1 \\
        --cycles 10,100,200 \\
        --maxiter 50 --workers 1

    # 빠른 smoke test (단일 셀, maxiter 축소)
    python scripts/run_multi_cycle_inverse.py \\
        --cells batch1_cell5 --cycles 10,100 \\
        --maxiter 10 --workers 1

산출물:
    data/multicycle_labels/cell_*_theta4_mc.json
    results/phase5/multicycle_eol_comparison.json
    docs/figures/p5d_multicycle_v_fit.png
"""

import argparse
import json
import logging
import sys
import time as time_module
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_LABELS_DIR = PROJECT_ROOT / "data" / "multicycle_labels"
OUTPUT_RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

# A123 APR18650M1A 고정 파라미터 (calibrate_a123.py 결과)
GEOMETRY_OVERRIDES = {
    "Electrode height [m]": 0.287,
    "Electrode width [m]": 0.3,
}
A123_FIXED_DIFFUSIVITY = {
    "Negative particle diffusivity [m2.s-1]": 1.491e-14,
    "Positive particle diffusivity [m2.s-1]": 5.9e-18,
}
PRADA2013_MAX_NEG_CONC = 30555.0  # mol/m3

# 기본 k_SEI (degradation_config.yaml 재보정값)
K_SEI_BASELINE = 5.3e-16  # m/s

# 4D 파라미터 경계: (SoC_init, eps_neg, eps_pos, log10_k_SEI)
THETA4_BOUNDS = [
    (0.80, 1.00),   # SoC_init
    (0.20, 0.55),   # eps_neg (음극 active material vol frac)
    (0.20, 0.55),   # eps_pos (양극 active material vol frac)
    (-17.0, -14.5), # log10(k_SEI): 5.3e-16 ≈ -15.28 → 범위 여유 확보
]

# Cycle weight (초기 사이클은 기준, 후기 사이클은 k_SEI 정보)
CYCLE_WEIGHTS: dict[int, float] = {
    10: 1.0,
    100: 1.5,
    200: 2.0,
}

# SEI 관련 degradation_params (degradation_config.yaml 동일)
SEI_STATIC_PARAMS = {
    "Ratio of lithium moles to SEI moles": 2.0,
    "SEI partial molar volume [m3.mol-1]": 9.585e-05,
    "SEI reaction exchange current density [A.m-2]": 1.5e-07,
    "SEI resistivity [Ohm.m]": 200000.0,
    "SEI solvent diffusivity [m2.s-1]": 2.5e-22,
    "SEI open-circuit potential [V]": 0.4,
    "SEI electron conductivity [S.m-1]": 8.95e-14,
    "SEI lithium interstitial diffusivity [m2.s-1]": 1.0e-20,
    "Initial SEI thickness [m]": 5.0e-09,
    "SEI growth activation energy [J.mol-1]": 0.0,
    "EC diffusivity [m2.s-1]": 2.0e-18,
    "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
}


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_tri_multi_cycle(
    batch_name: str,
    cell_idx: int,
    target_cycles: list[int],
    max_points: int = 60,
    cc_threshold: float = 0.85,
    v_min: float = 2.05,
) -> tuple[dict[int, dict], dict]:
    """TRI 셀의 여러 사이클 방전 V(t) 로드.

    Returns:
        V_obs_dict: {cycle_num: {"time_s": arr, "voltage_V": arr, "I_mean_A": float}}
        cell_info: cell_key, cycle_life, charge_policy 등 메타데이터
    """
    from src.download_tri_data import load_batch
    from src.data_preprocessor import (
        extract_cycle,
        extract_discharge_segment,
        trim_initial_ramp,
        downsample,
    )

    cells = load_batch(batch_name)
    cell_key = f"{batch_name}_cell{cell_idx}"
    cell = cells[cell_key]

    logger.info(
        f"TRI 셀 로드: {cell_key}, cycle_life={cell['cycle_life']}, "
        f"policy={cell['charge_policy']}"
    )

    available = {int(k) for k in cell["cycles"].keys()}
    V_obs_dict: dict[int, dict] = {}

    for cyc in target_cycles:
        if cyc not in available:
            logger.warning(f"  사이클 {cyc} 없음 — 스킵")
            continue
        try:
            cycle_data = extract_cycle(cell, cyc)
            discharge = extract_discharge_segment(cycle_data, negate_current=True)
            discharge = trim_initial_ramp(discharge, threshold_fraction=0.80)

            I_max = discharge["current_A"].max()
            cc_mask = discharge["current_A"] > I_max * cc_threshold
            v_mask = discharge["voltage_V"] > v_min
            cc_data = {k: v[cc_mask & v_mask] for k, v in discharge.items()}
            cc_data["time_s"] = cc_data["time_s"] - cc_data["time_s"][0]
            ds = downsample(cc_data, max_points)

            V_obs_dict[cyc] = {
                "time_s": ds["time_s"],
                "voltage_V": ds["voltage_V"],
                "I_mean_A": float(ds["current_A"].mean()),
                "n_pts": len(ds["time_s"]),
            }
            logger.info(
                f"  사이클 {cyc}: {len(ds['time_s'])}pts, "
                f"I={ds['current_A'].mean():.2f}A, "
                f"V=[{ds['voltage_V'].min():.3f}, {ds['voltage_V'].max():.3f}]"
            )
        except Exception as e:
            logger.warning(f"  사이클 {cyc} 전처리 실패: {e}")

    cell_info = {
        "cell_key": cell_key,
        "batch": batch_name,
        "cell_idx": cell_idx,
        "cycle_life": float(cell["cycle_life"]),
        "charge_policy": cell["charge_policy"],
        "source": "TRI",
    }
    return V_obs_dict, cell_info


def load_hust_multi_cycle(
    cell_id: str,
    target_cycles: list[int],
    max_points: int = 60,
    cc_threshold: float = 0.85,
    v_min: float = 2.05,
) -> tuple[dict[int, dict], dict]:
    """HUST 셀의 여러 사이클 방전 V(t) 로드.

    Returns:
        V_obs_dict: {cycle_num: {"time_s": arr, "voltage_V": arr, "I_mean_A": float}}
        cell_info: 메타데이터
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.etl_hust import load_raw_cell, extract_discharge_cycle_hust

    raw = load_raw_cell(cell_id)
    available = sorted(raw["data"].keys())

    # cycle_life 추정: max(rul) + cycle_num, 또는 마지막 사이클
    # HUST schema: rul[cycle] = 잔여 사이클
    if raw.get("rul"):
        max_cycle = max(raw["rul"].keys())
        rul_at_max = raw["rul"][max_cycle]
        cycle_life = max_cycle + rul_at_max
    else:
        cycle_life = max(available)

    # dq (방전 용량) 첫 번째 사이클 기준
    dq_dict = raw.get("dq", {})
    init_cap = 0.0
    if dq_dict:
        first_cyc = min(dq_dict.keys())
        init_cap = dq_dict[first_cyc] / 1000.0  # mAh → Ah

    logger.info(
        f"HUST 셀 로드: hust_{cell_id}, cycle_life≈{cycle_life}, "
        f"init_cap={init_cap:.3f}Ah, avail_cycles={len(available)}"
    )

    from src.data_preprocessor import downsample

    V_obs_dict: dict[int, dict] = {}
    for cyc in target_cycles:
        if cyc not in available:
            # 가장 가까운 사이클로 대체
            closest = min(available, key=lambda x: abs(x - cyc))
            logger.warning(
                f"  HUST 사이클 {cyc} 없음 → {closest} 사용"
            )
            actual_cyc = closest
        else:
            actual_cyc = cyc

        try:
            discharge = extract_discharge_cycle_hust(raw, actual_cyc, negate_current=True)
            # CC 필터
            I_max = discharge["current_A"].max()
            if I_max <= 0:
                logger.warning(f"  사이클 {cyc}: I_max≤0, 스킵")
                continue
            cc_mask = discharge["current_A"] > I_max * cc_threshold
            v_mask = discharge["voltage_V"] > v_min
            combined = cc_mask & v_mask
            cc_data = {k: v[combined] for k, v in discharge.items()}
            if len(cc_data["time_s"]) < 10:
                logger.warning(f"  사이클 {cyc}: CC 필터 후 포인트 부족, 스킵")
                continue
            cc_data["time_s"] = cc_data["time_s"] - cc_data["time_s"][0]
            ds = downsample(cc_data, max_points)

            V_obs_dict[cyc] = {
                "time_s": ds["time_s"],
                "voltage_V": ds["voltage_V"],
                "I_mean_A": float(ds["current_A"].mean()),
                "n_pts": len(ds["time_s"]),
                "actual_cycle": actual_cyc,
            }
            logger.info(
                f"  사이클 {cyc} (실사용 {actual_cyc}): {len(ds['time_s'])}pts, "
                f"I={ds['current_A'].mean():.2f}A"
            )
        except Exception as e:
            logger.warning(f"  HUST 사이클 {cyc} 전처리 실패: {e}")

    cell_info = {
        "cell_key": f"hust_{cell_id}",
        "cell_id": cell_id,
        "cycle_life": float(cycle_life),
        "source": "HUST",
        "init_cap_Ah": init_cap,
    }
    return V_obs_dict, cell_info


# ---------------------------------------------------------------------------
# Forward simulator
# ---------------------------------------------------------------------------

def multi_cycle_forward(
    theta: np.ndarray,
    target_cycles: list[int],
    I_discharge: float = 4.4,
    v_cutoff: float = 2.0,
    n_points: int = 60,
) -> Optional[dict[int, dict]]:
    """SEI degradation 포함 multi-cycle 순방향 시뮬레이션.

    theta: (SoC_init, eps_neg, eps_pos, log10_k_SEI)

    Returns:
        dict of {cycle_num: {"time_s": arr, "voltage_V": arr}} or None on failure
    """
    import pybamm

    soc_init, eps_neg, eps_pos, log10_k_sei = theta
    k_sei = 10.0 ** log10_k_sei

    # 물리적 타당성 확인
    if not (0.0 < eps_neg < 1.0 and 0.0 < eps_pos < 1.0):
        return None
    if not (0.6 <= soc_init <= 1.05):
        return None
    if not (1e-20 <= k_sei <= 1e-10):
        return None

    max_cycle = max(target_cycles)

    try:
        # SEI 모델 빌드
        model = pybamm.lithium_ion.SPM(options={"SEI": "ec reaction limited"})

        pv = pybamm.ParameterValues("Prada2013")
        for k, v in GEOMETRY_OVERRIDES.items():
            pv[k] = v

        # A123 재보정 diffusivity
        pv["Negative particle diffusivity [m2.s-1]"] = A123_FIXED_DIFFUSIVITY[
            "Negative particle diffusivity [m2.s-1]"
        ]
        pv["Positive particle diffusivity [m2.s-1]"] = A123_FIXED_DIFFUSIVITY[
            "Positive particle diffusivity [m2.s-1]"
        ]

        # theta 파라미터 적용
        pv["Initial concentration in negative electrode [mol.m-3]"] = (
            soc_init * PRADA2013_MAX_NEG_CONC
        )
        pv["Negative electrode active material volume fraction"] = eps_neg
        pv["Positive electrode active material volume fraction"] = eps_pos
        pv["Current function [A]"] = I_discharge

        # SEI static params 적용
        for k, v in SEI_STATIC_PARAMS.items():
            try:
                pv[k] = v
            except Exception:
                pass
        pv["SEI kinetic rate constant [m.s-1]"] = k_sei

        # Temperature
        pv["Ambient temperature [K]"] = 303.15
        pv["Initial temperature [K]"] = 303.15

        # CC-CV cycling experiment (TRI 프로토콜)
        cycle_steps = (
            f"Discharge at {I_discharge} A until {v_cutoff}V",
            "Rest for 5 minutes",
            "Charge at 1.1 A until 3.6V",
            "Hold at 3.6V until 0.11 A",
            "Rest for 5 minutes",
        )
        experiment = pybamm.Experiment(
            [cycle_steps] * max_cycle,
            termination=[f"{v_cutoff * 0.95}V"],
        )

        solver = pybamm.CasadiSolver(dt_max=60)
        sim = pybamm.Simulation(
            model,
            parameter_values=pv,
            experiment=experiment,
            solver=solver,
        )
        solution = sim.solve()

        if not hasattr(solution, "cycles") or not solution.cycles:
            logger.debug("solution.cycles 없음")
            return None

        n_simulated = len(solution.cycles)
        results: dict[int, dict] = {}

        for cyc in target_cycles:
            # cycle 인덱스 (0-based)
            idx = cyc - 1
            if idx >= n_simulated:
                logger.debug(f"사이클 {cyc} 시뮬레이션 미달성 (총 {n_simulated})")
                continue

            cycle_sol = solution.cycles[idx]
            try:
                t_vals = cycle_sol["Time [s]"].entries
                V_vals = cycle_sol["Voltage [V]"](t_vals)

                # 유효 구간 (방전: I>0 상태에서 V가 cutoff 이상)
                valid = np.isfinite(V_vals) & (V_vals > v_cutoff - 0.1)
                if valid.sum() < 5:
                    continue

                t_v = t_vals[valid]
                V_v = V_vals[valid]
                t_v = t_v - t_v[0]  # 사이클 내 상대 시간

                # 다운샘플
                if len(t_v) > n_points:
                    idx_ds = np.linspace(0, len(t_v) - 1, n_points, dtype=int)
                    t_v = t_v[idx_ds]
                    V_v = V_v[idx_ds]

                results[cyc] = {"time_s": t_v, "voltage_V": V_v}
            except Exception as e:
                logger.debug(f"사이클 {cyc} 전압 추출 오류: {e}")

        return results if results else None

    except Exception as e:
        logger.debug(f"multi_cycle_forward 실패: {e}")
        return None


# ---------------------------------------------------------------------------
# Multi-cycle loss
# ---------------------------------------------------------------------------

def multi_cycle_loss(
    theta: np.ndarray,
    V_obs_dict: dict[int, dict],
    weights: Optional[dict[int, float]] = None,
) -> float:
    """Multi-cycle squared-error loss.

    L = sum_i w_i * mean((V_sim(theta, cycle_i) - V_obs(cycle_i))^2)
    """
    if weights is None:
        weights = {c: 1.0 for c in V_obs_dict}

    target_cycles = list(V_obs_dict.keys())
    I_discharge = float(np.mean([V_obs_dict[c]["I_mean_A"] for c in target_cycles]))

    V_sim_dict = multi_cycle_forward(theta, target_cycles, I_discharge=I_discharge)
    if V_sim_dict is None:
        return 1e6

    total_loss = 0.0
    n_terms = 0

    for cyc, obs in V_obs_dict.items():
        if cyc not in V_sim_dict:
            total_loss += 1.0  # 시뮬레이션 미달성 패널티
            n_terms += 1
            continue

        t_obs = np.array(obs["time_s"])
        V_obs_arr = np.array(obs["voltage_V"])
        t_sim = V_sim_dict[cyc]["time_s"]
        V_sim_arr = V_sim_dict[cyc]["voltage_V"]

        if len(t_sim) < 3 or len(t_obs) < 3:
            total_loss += 1.0
            n_terms += 1
            continue

        # 공통 시간 구간에서 보간
        t_max = min(t_sim[-1], t_obs[-1])
        if t_max < 10.0:
            total_loss += 1.0
            n_terms += 1
            continue

        t_eval = np.linspace(0, t_max, 40)
        V_obs_interp = np.interp(t_eval, t_obs, V_obs_arr)
        V_sim_interp = np.interp(t_eval, t_sim, V_sim_arr)

        mse = float(np.mean((V_obs_interp - V_sim_interp) ** 2))
        w = weights.get(cyc, 1.0)
        total_loss += w * mse
        n_terms += 1

    return total_loss / max(n_terms, 1)


# ---------------------------------------------------------------------------
# Stage-1: 단일 cycle V(t) 역산 (SoC, eps_neg, eps_pos)
# ---------------------------------------------------------------------------

def _stage1_single_cycle_inverse(
    V_obs_stage1: dict,
    I_discharge: float,
    k_sei_fixed: float,
    maxiter: int,
    popsize: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Stage-1: cycle_10 기준 3D DE 역산 (k_SEI 고정).

    k_SEI는 cycle 10에서는 영향이 미미하므로 baseline으로 고정.
    x = (SoC_init, eps_neg, eps_pos)

    Returns:
        x_opt: 최적 3D 파라미터
        rmse_mV: 최종 RMSE
    """
    t_obs = np.array(V_obs_stage1["time_s"])
    V_obs_arr = np.array(V_obs_stage1["voltage_V"])
    log10_k_fixed = float(np.log10(k_sei_fixed))

    bounds_3d = [
        THETA4_BOUNDS[0],  # SoC_init
        THETA4_BOUNDS[1],  # eps_neg
        THETA4_BOUNDS[2],  # eps_pos
    ]

    def _cost(x3: np.ndarray) -> float:
        theta = np.array([x3[0], x3[1], x3[2], log10_k_fixed])
        # cycle 10만 시뮬레이션 (빠름)
        res = multi_cycle_forward(theta, [10], I_discharge=I_discharge, n_points=40)
        if res is None or 10 not in res:
            return 1e6
        t_sim = res[10]["time_s"]
        V_sim = res[10]["voltage_V"]
        if len(t_sim) < 3:
            return 1e6
        t_max = min(t_sim[-1], t_obs[-1])
        if t_max < 10.0:
            return 1e6
        t_eval = np.linspace(0, t_max, 40)
        V_o = np.interp(t_eval, t_obs, V_obs_arr)
        V_s = np.interp(t_eval, t_sim, V_sim)
        return float(np.mean((V_o - V_s) ** 2))

    result = differential_evolution(
        _cost,
        bounds=bounds_3d,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=1e-5,
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=True,
        workers=1,
    )

    # RMSE 계산
    rmse_mV = float(np.sqrt(result.fun) * 1000)
    return result.x, rmse_mV


# ---------------------------------------------------------------------------
# Stage-2: k_SEI 1D 최적화 (SoC/eps 고정, cycle 100/200 사용)
# ---------------------------------------------------------------------------

def _stage2_kseiinverse_1d(
    x_stage1: np.ndarray,
    V_obs_late: dict[int, dict],
    I_discharge: float,
    bounds_log10: tuple[float, float] = (-17.0, -14.5),
    n_grid: int = 20,
) -> tuple[float, float, dict]:
    """Stage-2: k_SEI 1D golden-section 탐색.

    SoC/eps_neg/eps_pos 고정, cycle 100/200 V(t)로 log10_k_SEI 최적화.

    Strategy:
        1. log10_k_SEI 범위를 n_grid 포인트 격자 탐색 (coarse)
        2. 최소값 주변 구간에서 scipy.optimize.minimize_scalar (Brent)

    Returns:
        log10_k_opt: 최적 log10(k_SEI)
        loss_opt: 최소 loss
        per_cycle_data: {cycle: {V_sim, V_obs, rmse_mV}}
    """
    from scipy.optimize import minimize_scalar

    log10_lo, log10_hi = bounds_log10

    def _loss_1d(log10_k: float) -> float:
        theta = np.array([
            x_stage1[0], x_stage1[1], x_stage1[2], log10_k
        ])
        target = list(V_obs_late.keys())
        if not target:
            return 1e6
        max_cyc = max(target)
        V_sim_dict = multi_cycle_forward(theta, target, I_discharge=I_discharge, n_points=40)
        if V_sim_dict is None:
            return 1e6

        total = 0.0
        n = 0
        for cyc, obs in V_obs_late.items():
            if cyc not in V_sim_dict:
                total += 1.0
                n += 1
                continue
            t_obs = np.array(obs["time_s"])
            V_obs_arr = np.array(obs["voltage_V"])
            t_sim = V_sim_dict[cyc]["time_s"]
            V_sim_arr = V_sim_dict[cyc]["voltage_V"]
            t_max = min(t_sim[-1], t_obs[-1])
            if t_max < 10.0:
                total += 1.0
                n += 1
                continue
            t_eval = np.linspace(0, t_max, 40)
            V_o = np.interp(t_eval, t_obs, V_obs_arr)
            V_s = np.interp(t_eval, t_sim, V_sim_arr)
            w = CYCLE_WEIGHTS.get(cyc, 1.0)
            total += w * float(np.mean((V_o - V_s) ** 2))
            n += 1
        return total / max(n, 1)

    # Coarse grid scan
    log10_grid = np.linspace(log10_lo, log10_hi, n_grid)
    losses = np.array([_loss_1d(lk) for lk in log10_grid])
    best_idx = int(np.argmin(losses))
    bracket_lo = log10_grid[max(0, best_idx - 2)]
    bracket_hi = log10_grid[min(n_grid - 1, best_idx + 2)]

    logger.info(
        f"  Stage-2 coarse: best log10_k={log10_grid[best_idx]:.3f} "
        f"loss={losses[best_idx]:.6f} bracket=[{bracket_lo:.3f},{bracket_hi:.3f}]"
    )

    # Fine 1D minimize_scalar (Brent)
    try:
        res = minimize_scalar(
            _loss_1d,
            bounds=(bracket_lo, bracket_hi),
            method="bounded",
            options={"xatol": 0.02, "maxiter": 20},
        )
        log10_k_opt = float(res.x)
        loss_opt = float(res.fun)
    except Exception as e:
        logger.warning(f"  Stage-2 minimize_scalar 실패: {e} — coarse best 사용")
        log10_k_opt = float(log10_grid[best_idx])
        loss_opt = float(losses[best_idx])

    logger.info(f"  Stage-2 fine: log10_k_opt={log10_k_opt:.4f} loss={loss_opt:.6f}")

    # 최종 V_sim 계산 (플롯/RMSE용)
    theta_opt = np.array([x_stage1[0], x_stage1[1], x_stage1[2], log10_k_opt])
    V_sim_final = multi_cycle_forward(
        theta_opt, list(V_obs_late.keys()), I_discharge=I_discharge, n_points=40
    )

    per_cycle_data: dict[int, dict] = {}
    for cyc, obs in V_obs_late.items():
        t_obs = np.array(obs["time_s"])
        V_obs_arr = np.array(obs["voltage_V"])
        cyc_sim = V_sim_final.get(cyc) if V_sim_final else None
        if cyc_sim is None:
            per_cycle_data[cyc] = {
                "V_obs": {"time_s": t_obs.tolist(), "voltage_V": V_obs_arr.tolist()},
                "V_sim": None,
                "rmse_mV": float("nan"),
            }
        else:
            t_sim = cyc_sim["time_s"]
            V_sim_arr = cyc_sim["voltage_V"]
            t_max = min(t_sim[-1], t_obs[-1])
            t_eval = np.linspace(0, t_max, 40)
            V_o = np.interp(t_eval, t_obs, V_obs_arr)
            V_s = np.interp(t_eval, t_sim, V_sim_arr)
            rmse_mv = float(np.sqrt(np.mean((V_o - V_s) ** 2)) * 1000)
            per_cycle_data[cyc] = {
                "V_obs": {"time_s": t_obs.tolist(), "voltage_V": V_obs_arr.tolist()},
                "V_sim": {
                    "time_s": cyc_sim["time_s"].tolist(),
                    "voltage_V": cyc_sim["voltage_V"].tolist(),
                },
                "rmse_mV": rmse_mv,
            }

    return log10_k_opt, loss_opt, per_cycle_data


# ---------------------------------------------------------------------------
# 단일 셀 역산 (Worker 함수 — ProcessPoolExecutor 호환)
# ---------------------------------------------------------------------------

def _run_single_cell_mc_inverse(args_tuple: tuple) -> dict:
    """ProcessPoolExecutor worker: 단일 셀 multi-cycle 2-stage 역산.

    args_tuple: (cell_spec, target_cycles, maxiter, popsize, seed, stage2_n_grid)
    cell_spec: {"type": "tri"|"hust", "batch"?: str, "cell_idx"?: int, "cell_id"?: str}

    2-Stage 전략:
        Stage-1: cycle 10 단일 V(t), k_SEI=baseline 고정 → (SoC, eps_neg, eps_pos) DE
        Stage-2: stage-1 파라미터 고정, cycle 100/200 V(t) → log10_k_SEI 1D 탐색
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    if len(args_tuple) == 6:
        cell_spec, target_cycles, maxiter, popsize, seed, stage2_n_grid = args_tuple
    else:
        cell_spec, target_cycles, maxiter, popsize, seed = args_tuple
        stage2_n_grid = 15
    t0 = time_module.time()

    # 데이터 로드
    try:
        if cell_spec["type"] == "tri":
            V_obs_dict, cell_info = load_tri_multi_cycle(
                cell_spec["batch"], cell_spec["cell_idx"], target_cycles
            )
        else:
            V_obs_dict, cell_info = load_hust_multi_cycle(
                cell_spec["cell_id"], target_cycles
            )
    except Exception as e:
        return {
            "cell_key": cell_spec.get("cell_key", "unknown"),
            "success": False,
            "error": f"data_load_failed: {e}",
            "elapsed_s": time_module.time() - t0,
        }

    if len(V_obs_dict) < 2:
        return {
            "cell_key": cell_info.get("cell_key", "unknown"),
            "success": False,
            "error": f"insufficient_cycles: {len(V_obs_dict)}/{len(target_cycles)} 로드됨",
            "elapsed_s": time_module.time() - t0,
            **cell_info,
        }

    cell_key = cell_info["cell_key"]
    sorted_cycles = sorted(V_obs_dict.keys())
    earliest_cyc = sorted_cycles[0]
    late_cycles = sorted_cycles[1:]  # stage-2에 사용
    I_mean = float(np.mean([V_obs_dict[c]["I_mean_A"] for c in V_obs_dict]))

    logger.info(
        f"2-Stage 역산 시작: {cell_key} "
        f"Stage-1 cyc={earliest_cyc}, Stage-2 cyc={late_cycles} "
        f"maxiter={maxiter}"
    )

    # ── Stage-1: SoC/eps_neg/eps_pos 추정 ───────────────────────────────────
    t1 = time_module.time()
    try:
        x_s1, rmse_s1 = _stage1_single_cycle_inverse(
            V_obs_dict[earliest_cyc],
            I_discharge=I_mean,
            k_sei_fixed=K_SEI_BASELINE,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
        )
    except Exception as e:
        return {
            "cell_key": cell_key,
            "success": False,
            "error": f"stage1_failed: {e}",
            "elapsed_s": time_module.time() - t0,
            **cell_info,
        }

    elapsed_s1 = time_module.time() - t1
    logger.info(
        f"  Stage-1 완료: {elapsed_s1:.0f}s "
        f"SoC={x_s1[0]:.4f} eps_neg={x_s1[1]:.4f} eps_pos={x_s1[2]:.4f} "
        f"RMSE={rmse_s1:.1f}mV"
    )

    # ── Stage-2: k_SEI 1D 최적화 ────────────────────────────────────────────
    t2 = time_module.time()
    V_obs_late = {c: V_obs_dict[c] for c in late_cycles}
    try:
        log10_k_opt, loss_s2, per_cycle_data = _stage2_kseiinverse_1d(
            x_s1,
            V_obs_late=V_obs_late,
            I_discharge=I_mean,
            bounds_log10=(THETA4_BOUNDS[3][0], THETA4_BOUNDS[3][1]),
            n_grid=stage2_n_grid,
        )
    except Exception as e:
        return {
            "cell_key": cell_key,
            "success": False,
            "error": f"stage2_failed: {e}",
            "elapsed_s": time_module.time() - t0,
            **cell_info,
        }

    elapsed_s2 = time_module.time() - t2
    k_sei_opt = float(10.0 ** log10_k_opt)
    logger.info(
        f"  Stage-2 완료: {elapsed_s2:.0f}s "
        f"k_SEI={k_sei_opt:.2e} (×{k_sei_opt/K_SEI_BASELINE:.2f} baseline)"
    )

    # ── per-cycle RMSE 집계 ─────────────────────────────────────────────────
    # Stage-1 cycle
    per_cycle_rmse: dict[str, float] = {str(earliest_cyc): rmse_s1}
    # Stage-2 cycles
    V_sim_store: dict[str, dict] = {}
    V_obs_store: dict[str, dict] = {}
    for cyc, data in per_cycle_data.items():
        per_cycle_rmse[str(cyc)] = data["rmse_mV"]
        if data.get("V_sim"):
            V_sim_store[str(cyc)] = data["V_sim"]
        V_obs_store[str(cyc)] = data["V_obs"]

    # Stage-1 V_sim도 재계산해서 저장 (플롯용)
    theta_final = np.array([x_s1[0], x_s1[1], x_s1[2], log10_k_opt])
    V_sim_s1 = multi_cycle_forward(
        theta_final, [earliest_cyc], I_discharge=I_mean, n_points=40
    )
    if V_sim_s1 and earliest_cyc in V_sim_s1:
        V_sim_store[str(earliest_cyc)] = {
            "time_s": V_sim_s1[earliest_cyc]["time_s"].tolist(),
            "voltage_V": V_sim_s1[earliest_cyc]["voltage_V"].tolist(),
        }
    # Stage-1 V_obs
    V_obs_store[str(earliest_cyc)] = {
        "time_s": V_obs_dict[earliest_cyc]["time_s"].tolist(),
        "voltage_V": V_obs_dict[earliest_cyc]["voltage_V"].tolist(),
    }

    avg_rmse = float(
        np.nanmean([v for v in per_cycle_rmse.values() if not np.isnan(v)])
        if per_cycle_rmse else float("nan")
    )
    success = avg_rmse < 60.0 and not np.isnan(avg_rmse)

    # theta_4 파라미터
    theta4_params = {
        "SoC_init": float(x_s1[0]),
        "eps_neg": float(x_s1[1]),
        "eps_pos": float(x_s1[2]),
        "log10_k_SEI": log10_k_opt,
        "k_SEI_m_s": k_sei_opt,
        "k_SEI_baseline": K_SEI_BASELINE,
        "k_SEI_relative": k_sei_opt / K_SEI_BASELINE,
    }

    total_elapsed = time_module.time() - t0

    output = {
        **cell_info,
        "target_cycles": target_cycles,
        "observed_cycles": sorted_cycles,
        "stage1_cycle": earliest_cyc,
        "stage2_cycles": late_cycles,
        "success": success,
        "avg_rmse_mV": avg_rmse,
        "per_cycle_rmse_mV": per_cycle_rmse,
        "stage1_elapsed_s": elapsed_s1,
        "stage2_elapsed_s": elapsed_s2,
        "elapsed_s": total_elapsed,
        "theta4": theta4_params,
        "bounds": {
            "SoC_init": list(THETA4_BOUNDS[0]),
            "eps_neg": list(THETA4_BOUNDS[1]),
            "eps_pos": list(THETA4_BOUNDS[2]),
            "log10_k_SEI": list(THETA4_BOUNDS[3]),
        },
        "V_obs": V_obs_store,
        "V_sim": V_sim_store,
    }

    logger.info(
        f"완료: {cell_key} "
        f"avg_rmse={avg_rmse:.1f}mV "
        f"k_SEI={k_sei_opt:.2e} (×{k_sei_opt/K_SEI_BASELINE:.2f}) "
        f"총 {total_elapsed:.0f}s"
    )
    return output


# ---------------------------------------------------------------------------
# BATT-H-02 가드 (식별성 확인)
# ---------------------------------------------------------------------------

def check_identifiability(results: list[dict]) -> dict:
    """BATT-H-02: theta_4 표본 분포 표준편차 확인.

    3개 셀에서 theta_4 추정값 분포가 지나치게 좁으면 경계 수렴 의심.
    """
    from src.battery_guards import check_phase1_identifiability, BattGuardViolation

    theta_arr = np.array([
        [
            r["theta4"]["SoC_init"],
            r["theta4"]["eps_neg"],
            r["theta4"]["eps_pos"],
            r["theta4"]["log10_k_SEI"],
        ]
        for r in results
        if r.get("success") and "theta4" in r
    ])

    if len(theta_arr) < 2:
        logger.warning("성공한 셀 <2개 → 식별성 가드 스킵")
        return {"status": "skipped", "reason": "insufficient_samples"}

    # theta_4의 k_SEI 차원을 포함한 4D 가드
    # 표준 가드는 3D(SoC, eps_neg, eps_pos) 기준 — 여기서는 log10_k_SEI 추가
    std_thresholds_4d = {
        "SoC_init": 0.001,
        "eps_neg": 0.005,
        "eps_pos": 0.005,
        "log10_k_SEI": 0.01,  # k_SEI는 log 스케일 → 더 민감
    }

    guard_result = {"status": "passed", "dimensions": []}
    violations = []

    dim_names = ["SoC_init", "eps_neg", "eps_pos", "log10_k_SEI"]
    for i, name in enumerate(dim_names):
        std = float(theta_arr[:, i].std())
        thr = std_thresholds_4d[name]
        passed = std >= thr
        guard_result["dimensions"].append({
            "name": name,
            "std": std,
            "threshold": thr,
            "pass": passed,
        })
        if not passed:
            violations.append(f"{name}: std={std:.5f} < {thr}")

    if violations:
        logger.warning(f"BATT-H-02 경고 (식별성): {violations}")
        guard_result["status"] = "warning"
        guard_result["violations"] = violations
    else:
        logger.info("BATT-H-02 통과: theta_4 모든 차원 식별 가능")

    return guard_result


# ---------------------------------------------------------------------------
# EOL 예측 비교
# ---------------------------------------------------------------------------

def predict_eol_from_theta4(theta4: dict, cycle_life_true: float) -> dict:
    """k_SEI로부터 EOL 예측.

    sqrt 법칙: SEI 성장 ∝ sqrt(t) → EOL ∝ 1 / k_SEI
    calibration baseline: k_SEI=5.3e-16 → EOL≈1046 사이클 (batch1_cell5 근사)
    """
    k_sei = theta4["k_SEI_m_s"]
    k_base = K_SEI_BASELINE
    eol_base = 1046.0  # degradation_config.yaml calibration

    # EOL 예측: sqrt-law 역산
    # SEI thickness ~ sqrt(k_SEI * t) → EOL ~ (eol_base * k_base) / k_sei
    eol_predicted = eol_base * (k_base / k_sei)

    mape = abs(eol_predicted - cycle_life_true) / max(cycle_life_true, 1) * 100.0

    return {
        "eol_predicted": float(eol_predicted),
        "eol_true": float(cycle_life_true),
        "mape_pct": float(mape),
        "k_SEI_used": k_sei,
        "eol_formula": "eol_pred = eol_base * (k_base / k_SEI)",
    }


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot_multi_cycle_fit(
    results: list[dict],
    save_path: Path,
) -> None:
    """3 셀 × 3 cycles V(t) 비교 그래프."""
    n_cells = len(results)
    target_cycles = sorted({
        int(c)
        for r in results
        for c in r.get("V_obs", {}).keys()
    })
    n_cycles = len(target_cycles)

    if n_cells == 0 or n_cycles == 0:
        logger.warning("플롯 데이터 없음")
        return

    fig, axes = plt.subplots(
        n_cells, n_cycles,
        figsize=(5 * n_cycles, 4 * n_cells),
        squeeze=False,
    )

    colors_obs = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    palette_sim = ["#d62728", "#9467bd", "#8c564b"]

    for row, res in enumerate(results):
        cell_key = res.get("cell_key", f"cell_{row}")
        theta4 = res.get("theta4", {})
        k_sei_str = f"k_SEI={theta4.get('k_SEI_m_s', float('nan')):.2e}"
        avg_rmse = res.get("avg_rmse_mV", float("nan"))

        for col, cyc in enumerate(target_cycles):
            ax = axes[row][col]
            cyc_str = str(cyc)

            obs = res.get("V_obs", {}).get(cyc_str)
            sim = res.get("V_sim", {}).get(cyc_str)

            if obs:
                t_obs = np.array(obs["time_s"]) / 60.0
                V_obs = np.array(obs["voltage_V"])
                ax.plot(t_obs, V_obs, "-", color=colors_obs[col % len(colors_obs)],
                        lw=2, label="Obs", alpha=0.9)

            if sim:
                t_sim = np.array(sim["time_s"]) / 60.0
                V_sim = np.array(sim["voltage_V"])
                ax.plot(t_sim, V_sim, "--", color=palette_sim[col % len(palette_sim)],
                        lw=1.5, label=f"Sim (θ_4)", alpha=0.85)

            rmse_cyc = res.get("per_cycle_rmse_mV", {}).get(cyc_str, float("nan"))
            ax.set_xlabel("Time [min]", fontsize=9)
            ax.set_ylabel("Voltage [V]", fontsize=9)
            ax.set_title(
                f"{cell_key}\nCycle {cyc} — RMSE={rmse_cyc:.1f}mV",
                fontsize=9,
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(1.95, 3.7)

        # 행 오른쪽에 파라미터 주석
        axes[row][-1].text(
            1.02, 0.5,
            f"{cell_key}\n{k_sei_str}\navg RMSE={avg_rmse:.1f}mV",
            transform=axes[row][-1].transAxes,
            fontsize=8,
            va="center",
            ha="left",
        )

    fig.suptitle(
        "P5-D Multi-Cycle Inverse Estimation — θ_4 = (SoC, ε_neg, ε_pos, k_SEI)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"플롯 저장: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 메인 배치 실행
# ---------------------------------------------------------------------------

def parse_cell_specs(cells_arg: str) -> list[dict]:
    """'batch1_cell5,batch1_cell9,hust_1-1' → cell_spec list."""
    specs = []
    for token in cells_arg.split(","):
        token = token.strip()
        if token.startswith("hust_"):
            cell_id = token[len("hust_"):]
            specs.append({
                "type": "hust",
                "cell_id": cell_id,
                "cell_key": token,
            })
        elif "_cell" in token:
            # e.g. batch1_cell5
            parts = token.rsplit("_cell", 1)
            batch_name = parts[0]
            cell_idx = int(parts[1])
            specs.append({
                "type": "tri",
                "batch": batch_name,
                "cell_idx": cell_idx,
                "cell_key": token,
            })
        else:
            logger.warning(f"셀 스펙 파싱 실패: {token}")
    return specs


def run_multi_cycle_batch(
    cell_specs: list[dict],
    target_cycles: list[int],
    maxiter: int = 50,
    popsize: int = 5,
    seed: int = 42,
    n_workers: int = 1,
    stage2_n_grid: int = 15,
    output_labels_dir: Optional[Path] = None,
    resume: bool = True,
) -> list[dict]:
    """여러 셀 multi-cycle 역산 배치 실행."""
    labels_dir = output_labels_dir or OUTPUT_LABELS_DIR
    labels_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for spec in cell_specs:
        cell_key = spec["cell_key"]
        out_path = labels_dir / f"{cell_key}_theta4_mc.json"
        if resume and out_path.exists():
            logger.info(f"Resume: {cell_key} 기존 결과 있음 — 스킵")
            continue
        tasks.append((spec, target_cycles, maxiter, popsize, seed, stage2_n_grid))

    if not tasks:
        logger.info("모든 셀 완료 (resume). 저장된 파일 로드.")
        results = []
        for spec in cell_specs:
            p = labels_dir / f"{spec['cell_key']}_theta4_mc.json"
            if p.exists():
                results.append(json.load(open(p)))
        return results

    logger.info(f"Multi-cycle 역산 시작: {len(tasks)}개 셀, n_workers={n_workers}")
    t_start = time_module.time()

    all_results = []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_single_cell_mc_inverse, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    res = future.result(timeout=3600)
                except Exception as e:
                    task = futures[future]
                    res = {
                        "cell_key": task[0]["cell_key"],
                        "success": False,
                        "error": str(e),
                    }
                all_results.append(res)
                cell_key = res.get("cell_key", "unknown")
                out_path = labels_dir / f"{cell_key}_theta4_mc.json"
                _save_result(res, out_path)
                logger.info(
                    f"저장: {out_path.name} "
                    f"success={res.get('success')} "
                    f"rmse={res.get('avg_rmse_mV', 'N/A'):.1f}mV"
                    if res.get("avg_rmse_mV") else f"저장: {out_path.name}"
                )
    else:
        for task in tasks:
            res = _run_single_cell_mc_inverse(task)
            all_results.append(res)
            cell_key = res.get("cell_key", "unknown")
            out_path = labels_dir / f"{cell_key}_theta4_mc.json"
            _save_result(res, out_path)
            rmse_str = f"{res['avg_rmse_mV']:.1f}mV" if res.get("avg_rmse_mV") else "N/A"
            logger.info(
                f"저장: {out_path.name} success={res.get('success')} rmse={rmse_str}"
            )

    # resume 경우 기저장 결과 합치기
    for spec in cell_specs:
        p = labels_dir / f"{spec['cell_key']}_theta4_mc.json"
        if p.exists() and not any(r.get("cell_key") == spec["cell_key"] for r in all_results):
            all_results.append(json.load(open(p)))

    elapsed = time_module.time() - t_start
    logger.info(f"배치 완료: {len(all_results)}개 셀, 총 {elapsed/60:.1f}분")
    return all_results


def _save_result(res: dict, path: Path) -> None:
    """결과 dict의 numpy 배열을 list로 변환 후 JSON 저장."""
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    safe = _convert(res)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="P5-D Multi-Cycle Inverse: cycle 10/100/200 V(t) 동시 fit으로 θ_4 추정"
    )
    parser.add_argument(
        "--cells",
        default="batch1_cell5,batch1_cell9,hust_1-1",
        help="역산할 셀 목록 (쉼표 구분). e.g. batch1_cell5,hust_1-1",
    )
    parser.add_argument(
        "--cycles",
        default="10,100,200",
        help="fit에 사용할 사이클 번호 (쉼표 구분)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=50,
        help="Stage-1 DE 최대 반복 수 (기본 50 — cycle 10 sim ~2.7s/eval 기준)",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=5,
        help="Stage-1 DE population 크기 (기본 5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--workers", type=int, default=1, help="병렬 셀 처리 수")
    parser.add_argument(
        "--stage2-grid",
        type=int,
        default=15,
        dest="stage2_grid",
        help="Stage-2 k_SEI 격자 탐색 포인트 수 (기본 15, ~cycle100sim×grid=150s)",
    )
    parser.add_argument("--no-resume", action="store_true", help="기존 결과 무시")
    parser.add_argument(
        "--skip-plot", action="store_true", help="V(t) 비교 플롯 생성 스킵"
    )
    args = parser.parse_args()

    cell_specs = parse_cell_specs(args.cells)
    target_cycles = [int(c.strip()) for c in args.cycles.split(",")]

    logger.info(f"P5-D 시작: {len(cell_specs)}개 셀, 사이클={target_cycles}")
    logger.info(
        f"DE: maxiter={args.maxiter}, popsize={args.popsize}, "
        f"workers={args.workers}, stage2_grid={args.stage2_grid}"
    )

    # 배치 역산
    results = run_multi_cycle_batch(
        cell_specs=cell_specs,
        target_cycles=target_cycles,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        n_workers=args.workers,
        stage2_n_grid=args.stage2_grid,
        resume=not args.no_resume,
    )

    successful = [r for r in results if r.get("success")]
    logger.info(f"성공: {len(successful)}/{len(results)} 셀")

    if not successful:
        logger.error("성공한 셀이 없음 — 보고 생성 스킵")
        return

    # BATT-H-02 식별성 가드
    identifiability = check_identifiability(successful)
    logger.info(f"BATT-H-02 식별성: {identifiability['status']}")

    # EOL 비교
    eol_rows = []
    for res in successful:
        if "theta4" not in res:
            continue
        eol = predict_eol_from_theta4(res["theta4"], res.get("cycle_life", float("nan")))
        eol_rows.append({
            "cell_key": res["cell_key"],
            "cycle_life_true": eol["eol_true"],
            "eol_predicted": eol["eol_predicted"],
            "mape_pct": eol["mape_pct"],
            "k_SEI": res["theta4"]["k_SEI_m_s"],
            "k_SEI_relative": res["theta4"]["k_SEI_relative"],
            "SoC_init": res["theta4"]["SoC_init"],
            "eps_neg": res["theta4"]["eps_neg"],
            "eps_pos": res["theta4"]["eps_pos"],
            "avg_rmse_mV": res.get("avg_rmse_mV", float("nan")),
            "per_cycle_rmse_mV": res.get("per_cycle_rmse_mV", {}),
        })

    # 결과 요약 저장
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_RESULTS_DIR / "multicycle_eol_comparison.json"
    summary = {
        "method": "P5-D multi-cycle inverse",
        "target_cycles": target_cycles,
        "n_cells": len(results),
        "n_success": len(successful),
        "identifiability_guard": identifiability,
        "eol_comparison": eol_rows,
    }
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"EOL 비교 저장: {comparison_path}")

    # 플롯
    if not args.skip_plot:
        plot_path = FIGURES_DIR / "p5d_multicycle_v_fit.png"
        plot_multi_cycle_fit(successful, plot_path)

    # 결과 출력
    print(f"\n{'='*65}")
    print("P5-D Multi-Cycle Inverse — 결과 요약")
    print(f"{'='*65}")
    print(f"{'셀':<20} {'k_SEI':>12} {'k_rel':>7} {'EOL_pred':>10} {'EOL_true':>10} {'MAPE':>7} {'RMSE':>8}")
    print(f"{'-'*65}")
    for row in eol_rows:
        print(
            f"{row['cell_key']:<20} "
            f"{row['k_SEI']:.2e} "
            f"{row['k_SEI_relative']:>7.2f}x "
            f"{row['eol_predicted']:>10.0f} "
            f"{row['cycle_life_true']:>10.0f} "
            f"{row['mape_pct']:>6.1f}% "
            f"{row['avg_rmse_mV']:>7.1f}mV"
        )
    if eol_rows:
        mean_mape = float(np.mean([r["mape_pct"] for r in eol_rows]))
        print(f"{'-'*65}")
        print(f"{'평균 MAPE':>55} {mean_mape:.1f}%")
    print(f"{'='*65}")
    print(f"\n산출물:")
    print(f"  labels : {OUTPUT_LABELS_DIR}/cell_*_theta4_mc.json")
    print(f"  results: {comparison_path}")
    if not args.skip_plot:
        print(f"  figure : {FIGURES_DIR}/p5d_multicycle_v_fit.png")


if __name__ == "__main__":
    main()
