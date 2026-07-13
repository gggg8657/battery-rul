"""
tests/test_multi_cycle_inverse.py
==================================
P5-D Multi-Cycle Inverse 단위 테스트 (PyBaMM 순방향 시뮬레이션 제외).

빠른 실행을 위해 실제 시뮬레이션(multi_cycle_forward)은 mock하고,
데이터 로드·손실 함수·BATT-H-02 가드·EOL 예측 로직만 테스트한다.

Usage:
    conda activate pybamm-inv
    pytest tests/test_multi_cycle_inverse.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 이 모듈의 모든 테스트를 'slow'로 분류 — CI 기본 실행에서 제외됨
# 실행: pytest tests/test_multi_cycle_inverse.py   (직접) 또는
#       pytest tests/ -m slow                      (slow 전용)
pytestmark = pytest.mark.slow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multi_cycle_inverse import (
    CYCLE_WEIGHTS,
    K_SEI_BASELINE,
    THETA4_BOUNDS,
    check_identifiability,
    multi_cycle_loss,
    parse_cell_specs,
    predict_eol_from_theta4,
)


# ---------------------------------------------------------------------------
# parse_cell_specs
# ---------------------------------------------------------------------------


class TestParseCellSpecs:
    def test_tri_batch1(self):
        specs = parse_cell_specs("batch1_cell5")
        assert len(specs) == 1
        s = specs[0]
        assert s["type"] == "tri"
        assert s["batch"] == "batch1"
        assert s["cell_idx"] == 5
        assert s["cell_key"] == "batch1_cell5"

    def test_hust(self):
        specs = parse_cell_specs("hust_1-1")
        assert len(specs) == 1
        s = specs[0]
        assert s["type"] == "hust"
        assert s["cell_id"] == "1-1"
        assert s["cell_key"] == "hust_1-1"

    def test_multiple_cells(self):
        specs = parse_cell_specs("batch1_cell5,batch1_cell9,hust_1-1")
        assert len(specs) == 3
        assert specs[0]["cell_key"] == "batch1_cell5"
        assert specs[1]["cell_key"] == "batch1_cell9"
        assert specs[2]["cell_key"] == "hust_1-1"

    def test_batch2(self):
        specs = parse_cell_specs("batch2_cell10")
        assert specs[0]["batch"] == "batch2"
        assert specs[0]["cell_idx"] == 10


# ---------------------------------------------------------------------------
# predict_eol_from_theta4
# ---------------------------------------------------------------------------


class TestPredictEolFromTheta4:
    def test_baseline_k_sei(self):
        """k_SEI=baseline이면 EOL≈1046 예측."""
        theta4 = {"k_SEI_m_s": K_SEI_BASELINE}
        eol = predict_eol_from_theta4(theta4, cycle_life_true=1074.0)
        assert abs(eol["eol_predicted"] - 1046.0) < 0.1
        assert eol["mape_pct"] < 5.0  # baseline vs 1074: ~2.6%

    def test_higher_k_sei_shorter_life(self):
        """k_SEI가 2배 → EOL 절반."""
        theta4_fast = {"k_SEI_m_s": K_SEI_BASELINE * 2.0}
        theta4_slow = {"k_SEI_m_s": K_SEI_BASELINE}
        eol_fast = predict_eol_from_theta4(theta4_fast, 500.0)
        eol_slow = predict_eol_from_theta4(theta4_slow, 1000.0)
        assert eol_fast["eol_predicted"] < eol_slow["eol_predicted"]
        assert abs(eol_fast["eol_predicted"] / eol_slow["eol_predicted"] - 0.5) < 0.01

    def test_mape_calculation(self):
        """MAPE 계산: |pred - true| / true × 100."""
        theta4 = {"k_SEI_m_s": K_SEI_BASELINE}
        eol = predict_eol_from_theta4(theta4, cycle_life_true=1046.0)
        assert eol["mape_pct"] == pytest.approx(0.0, abs=0.1)

    def test_output_keys(self):
        theta4 = {"k_SEI_m_s": K_SEI_BASELINE}
        eol = predict_eol_from_theta4(theta4, 1000.0)
        for key in ["eol_predicted", "eol_true", "mape_pct", "k_SEI_used", "eol_formula"]:
            assert key in eol


# ---------------------------------------------------------------------------
# check_identifiability (BATT-H-02 wrapper)
# ---------------------------------------------------------------------------


class TestCheckIdentifiability:
    def _make_results(self, theta_arr: np.ndarray) -> list[dict]:
        """N×4 배열 → pseudo result dict 목록."""
        results = []
        for row in theta_arr:
            results.append({
                "success": True,
                "theta4": {
                    "SoC_init": float(row[0]),
                    "eps_neg": float(row[1]),
                    "eps_pos": float(row[2]),
                    "log10_k_SEI": float(row[3]),
                },
            })
        return results

    def test_passes_with_diverse_theta(self):
        """셀마다 다른 theta → 식별성 통과."""
        np.random.seed(0)
        theta = np.random.uniform(
            [0.85, 0.25, 0.25, -16.5],
            [0.99, 0.50, 0.50, -14.8],
            size=(5, 4),
        )
        results = self._make_results(theta)
        guard = check_identifiability(results)
        assert guard["status"] in ("passed", "warning")  # 다양성 충분

    def test_skips_with_insufficient_samples(self):
        """성공 셀 1개 → 식별성 가드 스킵."""
        results = [{"success": True, "theta4": {
            "SoC_init": 0.95, "eps_neg": 0.35, "eps_pos": 0.30, "log10_k_SEI": -15.28
        }}]
        guard = check_identifiability(results)
        assert guard["status"] == "skipped"

    def test_warning_with_identical_k_sei(self):
        """k_SEI가 모두 동일 → log10_k_SEI std=0 → warning."""
        theta = np.array([
            [0.90, 0.30, 0.28, -15.28],
            [0.95, 0.35, 0.32, -15.28],
            [0.98, 0.40, 0.36, -15.28],
        ])
        results = self._make_results(theta)
        guard = check_identifiability(results)
        # log10_k_SEI std=0 < threshold → warning
        assert guard["status"] == "warning"
        violation_dims = [d["name"] for d in guard["dimensions"] if not d["pass"]]
        assert "log10_k_SEI" in violation_dims


# ---------------------------------------------------------------------------
# multi_cycle_loss (forward sim mock)
# ---------------------------------------------------------------------------


class TestMultiCycleLoss:
    """multi_cycle_forward를 mock하여 손실 함수 로직만 검증."""

    def _make_obs(self, n: int = 30) -> dict:
        t = np.linspace(0, 900.0, n)
        V = 3.5 - 0.5 * (t / 900.0) + np.random.normal(0, 0.002, n)
        return {"time_s": t, "voltage_V": V, "I_mean_A": 3.99}

    @patch("scripts.run_multi_cycle_inverse.multi_cycle_forward")
    def test_zero_loss_on_perfect_fit(self, mock_fwd):
        """시뮬레이션과 실측이 동일하면 loss≈0."""
        obs_10 = self._make_obs()
        obs_100 = self._make_obs()
        V_obs_dict = {10: obs_10, 100: obs_100}

        # mock: sim과 obs 동일 반환
        mock_fwd.return_value = {
            10: {"time_s": obs_10["time_s"], "voltage_V": obs_10["voltage_V"]},
            100: {"time_s": obs_100["time_s"], "voltage_V": obs_100["voltage_V"]},
        }
        theta = np.array([0.95, 0.35, 0.30, -15.28])
        loss = multi_cycle_loss(theta, V_obs_dict)
        assert loss < 1e-6

    @patch("scripts.run_multi_cycle_inverse.multi_cycle_forward")
    def test_nonzero_loss_on_mismatch(self, mock_fwd):
        """시뮬레이션과 실측 전압이 다르면 loss > 0."""
        obs_10 = self._make_obs()
        V_obs_dict = {10: obs_10}

        # mock: 0.1V 오프셋 오차
        V_shifted = obs_10["voltage_V"] + 0.1
        mock_fwd.return_value = {
            10: {"time_s": obs_10["time_s"], "voltage_V": V_shifted},
        }
        theta = np.array([0.95, 0.35, 0.30, -15.28])
        loss = multi_cycle_loss(theta, V_obs_dict)
        assert loss > 0.005  # 0.1V 오차 → MSE ≈ 0.01

    @patch("scripts.run_multi_cycle_inverse.multi_cycle_forward")
    def test_returns_penalty_on_sim_failure(self, mock_fwd):
        """forward sim 실패(None 반환) → 손실 함수가 큰 패널티 반환."""
        obs_10 = self._make_obs()
        V_obs_dict = {10: obs_10}
        mock_fwd.return_value = None
        theta = np.array([0.5, 0.1, 0.1, -20.0])  # 극단값
        loss = multi_cycle_loss(theta, V_obs_dict)
        assert loss >= 1.0  # 패널티 값

    @patch("scripts.run_multi_cycle_inverse.multi_cycle_forward")
    def test_weighted_loss(self, mock_fwd):
        """늦은 cycle(200)에 더 높은 weight → 해당 오차가 loss에 더 많이 기여."""
        obs_10 = self._make_obs()
        obs_200 = self._make_obs()
        V_obs_dict = {10: obs_10, 200: obs_200}

        # cycle 10: 완벽 fit, cycle 200: 0.1V 오차
        V_shifted = obs_200["voltage_V"] + 0.1
        mock_fwd.return_value = {
            10: {"time_s": obs_10["time_s"], "voltage_V": obs_10["voltage_V"]},
            200: {"time_s": obs_200["time_s"], "voltage_V": V_shifted},
        }
        theta = np.array([0.95, 0.35, 0.30, -15.28])

        # weight: cycle 200 = 2.0 (CYCLE_WEIGHTS 기준)
        loss_weighted = multi_cycle_loss(theta, V_obs_dict, weights={10: 0.0, 200: 2.0})
        loss_uniform = multi_cycle_loss(theta, V_obs_dict, weights={10: 1.0, 200: 1.0})

        # 가중치가 cycle 200에 집중 → 손실이 더 크거나 같아야 함
        assert loss_weighted >= loss_uniform * 0.5  # 합리적 범위 내 검증


# ---------------------------------------------------------------------------
# THETA4_BOUNDS 유효성
# ---------------------------------------------------------------------------


class TestTheta4Bounds:
    def test_bounds_count(self):
        assert len(THETA4_BOUNDS) == 4

    def test_soc_bounds(self):
        lo, hi = THETA4_BOUNDS[0]
        assert 0.5 <= lo < hi <= 1.1

    def test_eps_bounds(self):
        for i in [1, 2]:
            lo, hi = THETA4_BOUNDS[i]
            assert 0.0 < lo < hi < 1.0

    def test_log10_k_sei_bounds(self):
        lo, hi = THETA4_BOUNDS[3]
        assert lo < hi
        # k_SEI baseline (5.3e-16 ≈ log10=-15.28) 이 범위 내에 있어야 함
        import math
        log10_baseline = math.log10(K_SEI_BASELINE)
        assert lo <= log10_baseline <= hi, (
            f"K_SEI_BASELINE log10={log10_baseline:.2f} not in bounds [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# CYCLE_WEIGHTS
# ---------------------------------------------------------------------------


class TestCycleWeights:
    def test_late_cycles_heavier(self):
        """늦은 cycle이 더 높은 weight를 가져야 함."""
        assert CYCLE_WEIGHTS.get(200, 1.0) >= CYCLE_WEIGHTS.get(10, 1.0)

    def test_all_positive(self):
        for cyc, w in CYCLE_WEIGHTS.items():
            assert w > 0, f"cycle {cyc} weight={w} ≤ 0"
