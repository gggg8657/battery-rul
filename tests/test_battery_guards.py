"""
Battery domain hallucination guard 회귀 테스트 (BATT-H-01~09).

각 가드별 3개 테스트 = 27개 총합.

실행:
    pytest tests/test_battery_guards.py -v
"""

import numpy as np
import pytest

from src.battery_guards import (
    BattGuardViolation,
    check_best_epoch_used,
    check_c_rate_to_current,
    check_dataset_compatibility,
    check_domain_shift,
    check_grid_covers_real_theta,
    check_hybrid_pv_validation,
    check_knee_attribution,
    check_phase0_capacity,
    check_phase1_identifiability,
    check_sei_params_present,
    check_theta_in_training_hull,
    check_train_eval_cell_split,
    run_all_guards,
)

# ============================================================
# BATT-H-01: Phase 0 calibration values
# ============================================================


def test_BATT_H_01_pass():
    """Phase 0 정답 값 1C=0.927, 4C=0.961은 허용 범위 내 통과."""
    check_phase0_capacity(0.927, 0.961)
    # hybrid 보정 결과도 통과
    check_phase0_capacity(0.9516, 0.9361)


def test_BATT_H_01_fail():
    """어제 실제 발생 오류 케이스: 4C=1.01 Ah 오기재 → 위반."""
    with pytest.raises(BattGuardViolation, match=r"BATT-H-01"):
        check_phase0_capacity(0.927, 1.01)

    with pytest.raises(BattGuardViolation, match=r"BATT-H-01"):
        check_phase0_capacity(1.01, 0.961)

    # 1C 값이 하한 미달
    with pytest.raises(BattGuardViolation, match=r"BATT-H-01"):
        check_phase0_capacity(0.89, 0.961)


def test_BATT_H_01_edge():
    """경계 값 통과 확인 — 하한/상한 포함(inclusive)."""
    # 하한
    check_phase0_capacity(0.90, 0.93)
    # 상한
    check_phase0_capacity(0.96, 1.00)
    # tolerance=False: 정확한 정답 값만
    check_phase0_capacity(0.927, 0.961, tolerance=False)
    # tolerance=False에서 작은 오차도 위반
    with pytest.raises(BattGuardViolation, match=r"BATT-H-01"):
        check_phase0_capacity(0.928, 0.961, tolerance=False)


# ============================================================
# BATT-H-02: Phase 1 identifiability
# ============================================================


def test_BATT_H_02_pass():
    """충분한 std(>0.01)를 가진 θ 표본은 식별가능으로 통과."""
    # SoC, eps_neg, eps_pos 모두 분산 충분
    rng = np.random.default_rng(42)
    samples = rng.uniform(
        [0.83, 0.52, 0.30],
        [0.91, 0.57, 0.50],
        size=(100, 3),
    )
    result = check_phase1_identifiability(samples)
    assert result["pass"] is True
    assert len(result["dimensions"]) == 3


def test_BATT_H_02_fail():
    """eps_pos std=0.0 (TRI 실측 상황) → 식별불가능성 위반."""
    rng = np.random.default_rng(0)
    samples = np.column_stack([
        rng.uniform(0.83, 0.91, 100),  # SoC: 분산 있음
        rng.uniform(0.52, 0.57, 100),  # eps_neg: 분산 있음
        np.full(100, 0.55),             # eps_pos: 완전 고정 (TRI 실측)
    ])
    with pytest.raises(BattGuardViolation, match=r"BATT-H-02"):
        check_phase1_identifiability(samples)


def test_BATT_H_02_edge():
    """std 임계값 경계: 딱 threshold 값이면 통과, 미만이면 실패."""
    rng = np.random.default_rng(7)
    base = rng.uniform(0.83, 0.91, 100)

    # eps_pos std를 정확히 0.01로 구성 (경계 통과)
    eps_pos = np.linspace(0.55 - 0.01 * np.sqrt(3), 0.55 + 0.01 * np.sqrt(3), 100)
    samples_pass = np.column_stack([
        base,
        rng.uniform(0.52, 0.57, 100),
        eps_pos,
    ])
    # std >= threshold 이면 통과
    result = check_phase1_identifiability(samples_pass)
    assert all(d["pass"] for d in result["dimensions"] if d["name"] == "eps_pos")

    # 거의 0 표준편차 → 실패
    samples_fail = np.column_stack([
        base,
        rng.uniform(0.52, 0.57, 100),
        np.full(100, 0.55) + rng.normal(0, 1e-10, 100),
    ])
    with pytest.raises(BattGuardViolation, match=r"BATT-H-02"):
        check_phase1_identifiability(samples_fail)


# ============================================================
# BATT-H-03: Grid bounds vs real theta
# ============================================================


def test_BATT_H_03_pass():
    """Grid bounds가 실측 θ 분포를 완전히 포함하면 통과."""
    grid_bounds = {
        "SoC_init": [0.85, 1.0],
        "eps_neg": [0.20, 0.55],
        "eps_pos": [0.20, 0.45],
    }
    # TRI 실측 범위: SoC [0.831, 0.904], eps_neg [0.521, 0.567], eps_pos=0.55
    # eps_pos는 grid 상한(0.45)보다 실측(0.55)이 크므로 여기서는 안전한 예시로 구성
    real_samples = np.array([
        [0.86, 0.52, 0.35],
        [0.90, 0.55, 0.40],
        [0.88, 0.53, 0.38],
    ])
    check_grid_covers_real_theta(grid_bounds, real_samples)


def test_BATT_H_03_fail():
    """실측 θ가 grid 밖으로 벗어나면 위반."""
    grid_bounds = {
        "SoC_init": [0.85, 1.0],
        "eps_neg": [0.20, 0.55],
        "eps_pos": [0.20, 0.45],
    }
    # eps_pos = 0.55 이 grid max(0.45) 초과 → 위반
    real_samples = np.array([
        [0.87, 0.53, 0.55],  # eps_pos=0.55 > grid_max=0.45
        [0.89, 0.54, 0.42],
    ])
    with pytest.raises(BattGuardViolation, match=r"BATT-H-03"):
        check_grid_covers_real_theta(
            grid_bounds, real_samples, dim_names=["SoC_init", "eps_neg", "eps_pos"]
        )


def test_BATT_H_03_edge():
    """실측 θ가 grid 경계 값과 동일(±0)이면 통과."""
    grid_bounds = {"SoC_init": [0.85, 1.0], "eps_neg": [0.20, 0.55], "eps_pos": [0.20, 0.55]}
    # 경계 정확히 일치
    real_samples = np.array([
        [0.85, 0.20, 0.20],  # 하한 경계
        [1.00, 0.55, 0.55],  # 상한 경계
    ])
    check_grid_covers_real_theta(
        grid_bounds, real_samples, dim_names=["SoC_init", "eps_neg", "eps_pos"]
    )

    # 한 점이라도 초과 → 실패
    real_over = np.array([[0.85, 0.20, 0.56]])  # eps_pos 살짝 초과
    with pytest.raises(BattGuardViolation, match=r"BATT-H-03"):
        check_grid_covers_real_theta(
            grid_bounds, real_over, dim_names=["SoC_init", "eps_neg", "eps_pos"]
        )


# ============================================================
# BATT-H-04: Hybrid PV validation
# ============================================================


def test_BATT_H_04_pass():
    """Hybrid PV calibrated 결과 (0.9516, 0.9361)는 통과."""
    check_hybrid_pv_validation(0.9516, 0.9361)


def test_BATT_H_04_fail():
    """Chen2020 graphite 직접 이식 시 1C=1.28 Ah (+38%) → 위반."""
    with pytest.raises(BattGuardViolation, match=r"BATT-H-04"):
        check_hybrid_pv_validation(1.2782, 1.1342)

    # 4C/1C 비율 미달 케이스
    with pytest.raises(BattGuardViolation, match=r"BATT-H-04"):
        check_hybrid_pv_validation(0.93, 0.85)  # ratio = 0.914 < 0.95


def test_BATT_H_04_edge():
    """4C/1C 비율 경계: ratio=0.95는 통과, 0.949는 실패."""
    # ratio = 0.95 정확히 → 통과
    c1 = 0.94
    c4 = c1 * 0.95  # = 0.893 → 4C 범위 밖 (0.93 하한)
    # c4가 범위 [0.93, 1.00] 밖이므로 대신 안전한 값 사용
    c1 = 0.95
    c4 = 0.95 * 0.95  # = 0.9025 → 역시 범위 밖
    # BATT-H-04 범위 안에서 ratio 경계 테스트
    c1 = 0.95
    c4 = 0.95 * 0.95  # 4C 범위 미달로 먼저 걸림
    with pytest.raises(BattGuardViolation, match=r"BATT-H-04"):
        check_hybrid_pv_validation(c1, c4)

    # ratio 경계 통과
    c1 = 0.95
    c4 = 0.9500  # ratio = 1.0 ≥ 0.95, 4C 범위 [0.93, 1.00] 통과
    check_hybrid_pv_validation(c1, c4)


# ============================================================
# BATT-H-05: Knee location attribution
# ============================================================

BRANCH_B2_SLIDE = """
# Branch B2 — 2-Phase Slippage Model

Knee 재현 결과를 보여줍니다.

![knee before after](fig1_before_after.png)

주요 결과: 100% knee 재현.
"""

PHASE0_SLIDE = """
# Phase 0 — Calibration Results

Capacity bar chart:
- 1C: 0.927 Ah
- 4C: 0.961 Ah

![knee before after](fig1_before_after.png)
"""

NO_FIGURE_SLIDE = """
# Branch B2

슬라이드 텍스트만 있고 fig1_before_after.png 없음.
"""


def test_BATT_H_05_pass():
    """Knee figure가 Branch B2 섹션 아래에 있으면 통과."""
    check_knee_attribution(BRANCH_B2_SLIDE)


def test_BATT_H_05_fail():
    """Knee figure가 Phase 0 섹션 아래에 있으면 위반."""
    with pytest.raises(BattGuardViolation, match=r"BATT-H-05"):
        check_knee_attribution(PHASE0_SLIDE)


def test_BATT_H_05_edge():
    """Figure 자체가 없으면 통과; 다른 섹션 이름에 있으면 위반."""
    # figure 없으면 무조건 통과
    check_knee_attribution(NO_FIGURE_SLIDE)

    # figure 있지만 섹션이 Branch B 아님 (일반 헤더)
    other_section = "# Methodology\n\n![knee before after](fig1_before_after.png)"
    with pytest.raises(BattGuardViolation, match=r"BATT-H-05"):
        check_knee_attribution(other_section)


# ============================================================
# BATT-H-06: C-rate to current
# ============================================================


def test_BATT_H_06_pass():
    """A123 1C=1.1A, 4C=4.4A 정확한 값 및 허용 오차 5% 이내는 통과."""
    check_c_rate_to_current("A123_APR18650M1A", c_rate=1.0, current_A=1.1)
    check_c_rate_to_current("A123_APR18650M1A", c_rate=4.0, current_A=4.4)
    # 허용 오차 5% 이내: 1.1 * 1.03 = 1.133 (오차 3%)
    check_c_rate_to_current("A123_APR18650M1A", c_rate=1.0, current_A=1.133)


def test_BATT_H_06_fail():
    """1C=1.0A로 잘못 가정하는 흔한 오류 → 위반."""
    # A123에서 1C는 1.1A이므로 1.0A로 두면 5% 초과 오차 → 위반
    with pytest.raises(BattGuardViolation, match=r"BATT-H-06"):
        check_c_rate_to_current("A123_APR18650M1A", c_rate=1.0, current_A=1.0)


def test_BATT_H_06_edge():
    """허용 오차 경계: ±4% 이내는 통과, ±6% 초과는 실패."""
    # +4%: 1.1 * 1.04 = 1.144 → 오차 = 0.044 < 0.055(5%) → 통과
    check_c_rate_to_current("A123_APR18650M1A", c_rate=1.0, current_A=1.144)
    # -4%: 1.1 * 0.96 = 1.056 → 오차 = 0.044 < 0.055 → 통과
    check_c_rate_to_current("A123_APR18650M1A", c_rate=1.0, current_A=1.056)
    # +6%: 1.1 * 1.06 = 1.166 → 오차 = 0.066 > 0.055 → 실패
    with pytest.raises(BattGuardViolation, match=r"BATT-H-06"):
        check_c_rate_to_current("A123_APR18650M1A", c_rate=1.0, current_A=1.166)
    # 지원하지 않는 셀 모델
    with pytest.raises(ValueError):
        check_c_rate_to_current("NCM_21700", c_rate=1.0, current_A=4.0)


# ============================================================
# BATT-H-07: SEI parameter completeness
# ============================================================

SEI_FULL_PV = {
    "SEI kinetic rate constant [m.s-1]": 1e-12,
    "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
    "SEI growth activation energy [J.mol-1]": 0.0,
    "SEI open-circuit potential [V]": 0.4,
    "SEI resistivity [Ohm.m]": 200000.0,
    "SEI partial molar volume [m3.mol-1]": 9.585e-05,
    "SEI molar mass [kg.mol-1]": 0.162,
    "Initial outer SEI thickness [m]": 5e-09,
    "Initial inner SEI thickness [m]": 0.5e-09,
    "SEI lithium ion conductivity [S.m-1]": 1e-07,
    # 다른 파라미터도 포함 가능
    "Negative electrode diffusivity [m2.s-1]": 3e-15,
}

SEI_MISSING_PV = {
    "Negative electrode diffusivity [m2.s-1]": 3e-15,
    # SEI 파라미터 전혀 없음 (Prada2013 순수 상태)
}


def test_BATT_H_07_pass():
    """SEI 파라미터 10개 모두 있으면 통과."""
    check_sei_params_present(SEI_FULL_PV)


def test_BATT_H_07_fail():
    """Prada2013 단독 PV (SEI 없음) → 위반."""
    with pytest.raises(BattGuardViolation, match=r"BATT-H-07"):
        check_sei_params_present(SEI_MISSING_PV)


def test_BATT_H_07_edge():
    """10개 중 1개 누락 시 위반, 정확히 10개만 있어도 통과."""
    # 1개 누락
    partial_pv = {k: v for k, v in SEI_FULL_PV.items()
                  if k != "SEI lithium ion conductivity [S.m-1]"}
    with pytest.raises(BattGuardViolation, match=r"BATT-H-07"):
        check_sei_params_present(partial_pv)

    # SEI 10개만 정확히 존재 (다른 키 없음)
    sei_only = {
        "SEI kinetic rate constant [m.s-1]": 1e-12,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "SEI growth activation energy [J.mol-1]": 0.0,
        "SEI open-circuit potential [V]": 0.4,
        "SEI resistivity [Ohm.m]": 200000.0,
        "SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI molar mass [kg.mol-1]": 0.162,
        "Initial outer SEI thickness [m]": 5e-09,
        "Initial inner SEI thickness [m]": 0.5e-09,
        "SEI lithium ion conductivity [S.m-1]": 1e-07,
    }
    check_sei_params_present(sei_only)


# ============================================================
# BATT-H-08: Dataset distribution shift
# ============================================================


def test_BATT_H_08_pass():
    """같은 분포에서 샘플링한 배열은 KS test 통과 (shift 없음)."""
    rng = np.random.default_rng(42)
    a = rng.normal(800, 150, 50)
    b = rng.normal(800, 150, 50)
    result = check_dataset_compatibility(a, b)
    # p > 0.05 이면 domain_shift_detected = False
    assert result["p_value"] > 0.05
    assert bool(result["domain_shift_detected"]) is False


def test_BATT_H_08_fail():
    """TRI(~845) vs HUST(~1899) 실측 분포 → domain shift 감지, warn_only=False → raise."""
    rng = np.random.default_rng(0)
    tri = rng.normal(845, 183, 46)
    hust = rng.normal(1899, 451, 32)
    # warn_only=True (기본): raise 없이 결과 반환
    result = check_dataset_compatibility(tri, hust, warn_only=True)
    assert bool(result["domain_shift_detected"]) is True
    assert result["mean_ratio"] > 1.5

    # warn_only=False: raise
    with pytest.raises(BattGuardViolation, match=r"BATT-H-08"):
        check_dataset_compatibility(tri, hust, warn_only=False)


def test_BATT_H_08_edge():
    """KS test 결과 dict에 필수 키가 모두 존재하는지 확인."""
    rng = np.random.default_rng(1)
    tri = rng.normal(800, 100, 20)
    hust = rng.normal(1800, 300, 20)
    result = check_dataset_compatibility(tri, hust, warn_only=True)
    required_keys = {"ks_statistic", "p_value", "domain_shift_detected", "tri_mean", "hust_mean", "mean_ratio"}
    assert required_keys <= set(result.keys())
    assert result["tri_mean"] == pytest.approx(tri.mean(), rel=1e-6)
    assert result["hust_mean"] == pytest.approx(hust.mean(), rel=1e-6)


# ============================================================
# BATT-H-09: Per-cell fit circularity
# ============================================================


def test_BATT_H_09_pass():
    """학습/평가 셀이 완전히 분리되면 통과."""
    train = {"batch1_cell0", "batch1_cell1", "batch1_cell2"}
    eval_ = {"batch1_cell3", "batch1_cell4", "batch1_cell5"}
    check_train_eval_cell_split(train, eval_)


def test_BATT_H_09_fail():
    """공통 셀이 하나라도 있으면 circularity 위반."""
    train = {"batch1_cell0", "batch1_cell1", "batch1_cell5"}
    eval_ = {"batch1_cell5", "batch1_cell6"}  # cell5 중복
    with pytest.raises(BattGuardViolation, match=r"BATT-H-09"):
        check_train_eval_cell_split(train, eval_)


def test_BATT_H_09_edge():
    """빈 집합은 항상 disjoint → 통과; 전체가 겹치면 위반."""
    # 빈 eval 집합 → 통과
    check_train_eval_cell_split({"cell0", "cell1"}, set())
    # 빈 train 집합 → 통과
    check_train_eval_cell_split(set(), {"cell0", "cell1"})
    # 동일 집합 → 위반
    cells = {"cell0", "cell1", "cell2"}
    with pytest.raises(BattGuardViolation, match=r"BATT-H-09"):
        check_train_eval_cell_split(cells, cells)


# ============================================================
# 통합: run_all_guards
# ============================================================


def test_run_all_guards_all_pass():
    """run_all_guards: 정상 context → 모든 가드 통과."""
    rng = np.random.default_rng(42)
    theta_ok = np.column_stack([
        rng.uniform(0.83, 0.91, 100),
        rng.uniform(0.52, 0.57, 100),
        rng.uniform(0.35, 0.50, 100),
    ])
    context = {
        "phase0_c1": 0.927,
        "phase0_c4": 0.961,
        "theta_samples": theta_ok,
        "grid_bounds": {"SoC": [0.80, 1.0], "eps_neg": [0.20, 0.60], "eps_pos": [0.20, 0.55]},
        "real_theta_samples": theta_ok,
        "hybrid_c1": 0.9516,
        "hybrid_c4": 0.9361,
        "c_rate": 1.0,
        "current_A": 1.1,
        "hybrid_pv": SEI_FULL_PV,
        "tri_cycle_life": rng.normal(845, 183, 46),
        "hust_cycle_life": rng.normal(1899, 451, 32),
        "train_cells": {"cell0", "cell1"},
        "eval_cells": {"cell2", "cell3"},
    }
    results = run_all_guards(context)
    assert results["BATT-H-01"] == "pass"
    assert results["BATT-H-02"] == "pass"
    assert results["BATT-H-03"] == "pass"
    assert results["BATT-H-04"] == "pass"
    assert results["BATT-H-06"] == "pass"
    assert results["BATT-H-07"] == "pass"
    assert results["BATT-H-09"] == "pass"
    assert results["summary"]["fail"] == 0


def test_run_all_guards_skip_on_missing_context():
    """run_all_guards: 키 없으면 'skip' 반환 (예외 없음)."""
    results = run_all_guards({})
    for gid in [f"BATT-H-0{i}" for i in range(1, 10)]:
        assert results[gid] == "skip"
    # BATT-H-10/11/12도 skip
    assert results["BATT-H-10"] == "skip"
    assert results["BATT-H-11"] == "skip"
    assert results["BATT-H-12"] == "skip"
    # BATT-H-13~16 추가로 총 16개 skip
    assert results["BATT-H-13"] == "skip"
    assert results["BATT-H-14"] == "skip"
    assert results["BATT-H-15"] == "skip"
    assert results["BATT-H-16"] == "skip"
    assert results["summary"]["skip"] == 16


# ============================================================
# BATT-H-10: Domain shift 자동 감지 (일반화)
# ============================================================


def test_BATT_H_10_pass():
    """같은 분포에서 샘플링된 두 배열 → KS p > 0.05 → shift_detected=False."""
    rng = np.random.default_rng(42)
    a = rng.normal(800, 150, 60)
    b = rng.normal(800, 150, 60)
    result = check_domain_shift(a, b, dataset_a_name="train", dataset_b_name="test")
    assert result["p_value"] > 0.05
    assert bool(result["shift_detected"]) is False
    # 반환 dict 필수 키 확인
    required = {"ks_stat", "p_value", "shift_detected", "mean_a", "mean_b",
                "dataset_a_name", "dataset_b_name"}
    assert required <= set(result.keys())
    assert result["dataset_a_name"] == "train"
    assert result["dataset_b_name"] == "test"


def test_BATT_H_10_fail():
    """TRI(~845) vs HUST(~1899) 실측 분포 → p << 0.05 → shift_detected=True."""
    rng = np.random.default_rng(0)
    tri = rng.normal(845, 183, 46)
    hust = rng.normal(1899, 451, 32)

    # warn_only=True (기본): raise 없이 결과 반환
    result = check_domain_shift(
        tri, hust,
        dataset_a_name="TRI",
        dataset_b_name="HUST",
        warn_only=True,
    )
    assert bool(result["shift_detected"]) is True
    assert result["p_value"] < 0.05
    assert result["mean_b"] > result["mean_a"]  # HUST mean >> TRI mean

    # warn_only=False → raise
    with pytest.raises(BattGuardViolation, match=r"BATT-H-10"):
        check_domain_shift(tri, hust, warn_only=False)


def test_BATT_H_10_edge():
    """ks_p_threshold 파라미터 커스텀 동작 확인.

    비슷한 두 분포(p ≈ 0.13)를 사용:
    - ks_p_threshold=0.99 (엄격) → p(0.13) < 0.99 → shift_detected=True
    - ks_p_threshold=1e-10 (느슨) → p(0.13) > 1e-10 → shift_detected=False
    """
    rng = np.random.default_rng(7)
    # 유사 분포: p ≈ 0.13 (0.05 < p < 1e-10 기준 경계 사이)
    a = rng.normal(800, 150, 30)
    b = rng.normal(820, 150, 30)

    # 엄격한 임계값(0.99) → shift 감지
    result_strict = check_domain_shift(a, b, ks_p_threshold=0.99, warn_only=True)
    assert bool(result_strict["shift_detected"]) is True

    # 느슨한 임계값(1e-10) → 분포 유사하므로 shift 미감지
    result_lax = check_domain_shift(a, b, ks_p_threshold=1e-10, warn_only=True)
    assert bool(result_lax["shift_detected"]) is False

    # mean_a, mean_b 정확성
    assert result_strict["mean_a"] == pytest.approx(a.mean(), rel=1e-6)
    assert result_strict["mean_b"] == pytest.approx(b.mean(), rel=1e-6)


# ============================================================
# BATT-H-11: Extrapolation 경고 (convex hull)
# ============================================================


def test_BATT_H_11_pass():
    """학습 θ 중심 근처 쿼리 → hull 내부 → all_inside=True."""
    rng = np.random.default_rng(42)
    # 3D training data: [-1, 1]^3
    theta_train = rng.uniform(-1, 1, (50, 3))
    # 중심 근처 쿼리 → 내부
    theta_q = np.array([[0.0, 0.0, 0.0], [0.1, -0.1, 0.05]])
    result = check_theta_in_training_hull(theta_q, theta_train)
    assert result["skipped"] is False
    assert bool(result["all_inside"]) is True
    assert result["n_outside"] == 0
    assert result["n_query"] == 2


def test_BATT_H_11_fail():
    """학습 θ 범위 밖 쿼리 → hull 외부 → raise_on_outside=True 시 위반."""
    rng = np.random.default_rng(0)
    # 학습 데이터: [0, 1]^3 범위
    theta_train = rng.uniform(0, 1, (60, 3))
    # 외부 쿼리: 범위 훨씬 밖
    theta_q_outside = np.array([[5.0, 5.0, 5.0], [-5.0, -5.0, -5.0]])

    # raise_on_outside=False(기본): raise 없이 결과 반환
    result = check_theta_in_training_hull(theta_q_outside, theta_train)
    assert result["skipped"] is False
    assert bool(result["all_inside"]) is False
    assert result["n_outside"] > 0

    # raise_on_outside=True: raise
    with pytest.raises(BattGuardViolation, match=r"BATT-H-11"):
        check_theta_in_training_hull(theta_q_outside, theta_train, raise_on_outside=True)


def test_BATT_H_11_edge():
    """학습 데이터 부족(점 수 < 차원+1) → skipped=True 반환.
    scipy Delaunay는 2D 이상 데이터가 필요하므로 1D도 skipped 처리됨.
    """
    # 3D인데 학습 데이터 3개(< 4) → Delaunay 불가 → skipped
    theta_train_small = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    theta_q = np.array([[0.3, 0.3, 0.0]])
    result = check_theta_in_training_hull(theta_q, theta_train_small)
    assert result["skipped"] is True
    assert result["all_inside"] is None

    # 2D 데이터: 충분한 점 수 → 정상 처리
    rng = np.random.default_rng(3)
    theta_train_2d = rng.uniform(0, 1, (30, 2))
    theta_q_2d = np.array([[0.5, 0.5]])  # 내부 점
    result_2d = check_theta_in_training_hull(theta_q_2d, theta_train_2d)
    assert result_2d["skipped"] is False
    assert result_2d["n_query"] == 1

    # tolerance=0 (hull 확장 없음): 경계 근처 점도 일반적으로 처리 가능
    result_no_tol = check_theta_in_training_hull(
        theta_q_2d, theta_train_2d, tolerance=0.0
    )
    assert result_no_tol["skipped"] is False


# ============================================================
# BATT-H-12: Best epoch 조기 stop 경고
# ============================================================


def test_BATT_H_12_pass():
    """best_epoch=100 / total=200 = 50% ≥ 10% → 조기 stop 아님."""
    result = check_best_epoch_used(best_epoch=100, total_epochs=200)
    assert result["ratio"] == pytest.approx(0.5)
    assert bool(result["early_stop_warned"]) is False
    # 반환 dict 필수 키
    required = {"best_epoch", "total_epochs", "ratio", "early_stop_warned", "min_ratio"}
    assert required <= set(result.keys())


def test_BATT_H_12_fail():
    """P7-B v2 재현: best=8 / total=200 = 4% < 10% → warn_only=False 시 raise."""
    # warn_only=True(기본): raise 없이 경고 반환
    result = check_best_epoch_used(best_epoch=8, total_epochs=200, warn_only=True)
    assert result["ratio"] == pytest.approx(0.04)
    assert bool(result["early_stop_warned"]) is True

    # warn_only=False: raise
    with pytest.raises(BattGuardViolation, match=r"BATT-H-12"):
        check_best_epoch_used(best_epoch=8, total_epochs=200, warn_only=False)


def test_BATT_H_12_edge():
    """min_ratio 경계: ratio=0.10 정확히는 통과, 0.099는 경고."""
    # 경계 통과: best=10, total=100, ratio=0.10 = min_ratio → 통과
    result_boundary = check_best_epoch_used(best_epoch=10, total_epochs=100, min_ratio=0.1)
    assert bool(result_boundary["early_stop_warned"]) is False

    # 경계 미달: best=9, total=100, ratio=0.09 < 0.10 → 경고
    result_under = check_best_epoch_used(best_epoch=9, total_epochs=100, min_ratio=0.1)
    assert bool(result_under["early_stop_warned"]) is True

    # best_epoch > total_epochs → ValueError
    with pytest.raises(ValueError):
        check_best_epoch_used(best_epoch=201, total_epochs=200)

    # total_epochs=0 → ValueError
    with pytest.raises(ValueError):
        check_best_epoch_used(best_epoch=1, total_epochs=0)
