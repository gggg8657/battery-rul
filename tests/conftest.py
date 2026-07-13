"""
tests/conftest.py
=================
공통 pytest fixture 및 설정.

Usage:
    pytest tests/ -v
    pytest tests/ -m "not slow"   # 빠른 가드 테스트 (BATT-H 29개)만 실행
    pytest tests/ -m slow         # 무거운 multi-cycle 테스트만 실행
"""

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 프로젝트 경로
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    """레포 루트 Path 객체."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def lookup_table_path() -> Path:
    """배터리 가드 lookup table YAML 경로."""
    return PROJECT_ROOT / "configs" / "battery_guards_lookup.yaml"


# ---------------------------------------------------------------------------
# θ 표본 fixture (BATT-H-02, BATT-H-03 등에 재사용)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_theta_array() -> np.ndarray:
    """
    TRI 실측 범위 내 랜덤 θ 표본 (100×3).

    열 순서: [SoC_init, eps_neg, eps_pos]
    - SoC_init : U(0.83, 0.91)
    - eps_neg  : U(0.52, 0.57)
    - eps_pos  : U(0.35, 0.50)
    """
    rng = np.random.default_rng(42)
    return rng.uniform(
        [0.83, 0.52, 0.35],
        [0.91, 0.57, 0.50],
        size=(100, 3),
    )


@pytest.fixture
def sample_theta4_array() -> np.ndarray:
    """
    4-파라미터 θ 표본 (50×4): [SoC_init, eps_neg, eps_pos, log10_k_SEI].

    multi-cycle inverse 테스트에서 재사용.
    """
    rng = np.random.default_rng(7)
    return rng.uniform(
        [0.85, 0.25, 0.25, -16.5],
        [0.99, 0.50, 0.50, -14.8],
        size=(50, 4),
    )


# ---------------------------------------------------------------------------
# V(t) 더미 시계열
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_voltage_series() -> dict:
    """
    단일 싸이클 더미 전압 시계열.

    Returns
    -------
    dict with keys: time_s, voltage_V, I_mean_A
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, 900.0, 60)
    V = 3.5 - 0.5 * (t / 900.0) + rng.normal(0, 0.002, 60)
    return {"time_s": t, "voltage_V": V, "I_mean_A": 3.99}


@pytest.fixture
def dummy_voltage_dict(dummy_voltage_series) -> dict:
    """
    다중 사이클 더미 V(t) dict: {cycle_idx: {"time_s": ..., "voltage_V": ...}}

    BATT-H 가드 및 multi-cycle loss 테스트에서 재사용.
    """
    obs = dummy_voltage_series
    return {
        10: {"time_s": obs["time_s"], "voltage_V": obs["voltage_V"]},
        100: {"time_s": obs["time_s"], "voltage_V": obs["voltage_V"] - 0.02},
        200: {"time_s": obs["time_s"], "voltage_V": obs["voltage_V"] - 0.05},
    }


# ---------------------------------------------------------------------------
# SEI 파라미터 fixture (BATT-H-07 재사용)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sei_full_pv() -> dict:
    """
    Chen2020 / OKane2022 SEI 필수 10개 파라미터 완전 집합.
    """
    return {
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
        "Negative electrode diffusivity [m2.s-1]": 3e-15,
    }


# ---------------------------------------------------------------------------
# 환경 확인 (선택적 skip 지원)
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """커스텀 마커 설명 등록 (strict-markers 경고 방지 보조)."""
    # pytest.ini의 markers 선언으로 충분하지만, 동적 등록도 병행
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_pybamm: requires PyBaMM environment"
    )
    config.addinivalue_line(
        "markers", "requires_torch: requires PyTorch"
    )
