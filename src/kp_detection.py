"""
src/kp_detection.py
====================
개선된 Knee Point (KP) 탐지 모듈.

배경:
  Phase 4 P4-A 분석에서 기존 Bacon-Watts(BW) 구현이 시뮬레이션 궤적
  (Q_pred)에서 cycle=3 같은 비현실적 knee를 반환하는 문제가 발견됨.
  원인:
    1. 초기 knee guess = len(Q)/2 → EOL 정보를 무시
    2. bound 하한이 cycle[2]=3 → 극초반 수렴 허용
    3. 시뮬 Q(n)이 너무 완만해 BW 발산 → fallback 없음

개선 사항:
  - EOL 기반 knee search range 제한 (기본 [0.35, 0.70] * EOL)
  - slope_post > slope_pre 오탐 방지 (knee 물리적 의미 강제)
  - fit_rmse threshold 검사
  - 다중 초기값 재시도 (grid search over knee0)
  - fallback: max_curvature_knee (smoothed 2차 미분)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
KP_DETECTION_MIN_CYCLES: int = 20
EOL_THRESHOLD: float = 0.80          # 80% 용량 보존 = EoL
BW_MAX_RMSE: float = 0.05            # Bacon-Watts fit RMSE 허용 상한 (Ah)
BW_D_LOWER: float = 1.0              # tanh 전환 폭 하한 (cycle)
BW_D_UPPER: float = 1000.0           # tanh 전환 폭 상한 (cycle)
_N_KNEE_INIT_GRID: int = 5           # 초기값 grid 탐색 개수


# ===========================================================================
# 유틸리티
# ===========================================================================

def _estimate_eol_cycle(Q_n: np.ndarray, eol_threshold: float = EOL_THRESHOLD) -> Optional[int]:
    """Q(n) 배열에서 EOL cycle 추정.

    Q_n[0]을 기준 용량으로 삼아 threshold 아래로 내려가는 첫 사이클을 반환.
    도달하지 못하면 None.
    """
    if len(Q_n) < 2:
        return None
    q0 = Q_n[0]
    if q0 <= 0:
        return None
    below = np.where(Q_n < eol_threshold * q0)[0]
    if len(below) == 0:
        return None
    return int(below[0] + 1)  # 1-indexed cycle number


def _bacon_watts_model(
    x: np.ndarray,
    a: float,
    b: float,
    c: float,
    knee: float,
    d: float,
) -> np.ndarray:
    """Bacon-Watts 2-선형 접합 모델.

    y = a + b*(x - knee) + c*(x - knee)*tanh((x - knee)/d)
    """
    return a + b * (x - knee) + c * (x - knee) * np.tanh((x - knee) / (d + 1e-9))


# ===========================================================================
# 핵심 함수 1: bacon_watts_constrained
# ===========================================================================

def bacon_watts_constrained(
    Q_n: np.ndarray,
    eol_cycle: Optional[int] = None,
    knee_search_range: tuple[float, float] = (0.35, 0.70),
    cycles: Optional[np.ndarray] = None,
) -> dict:
    """EOL 기반 제약 Bacon-Watts KP 탐지.

    개선 포인트:
      - knee 초기 guess = EOL * 0.5 (기존: len(Q)/2)
      - knee bound = [EOL * knee_search_range[0], EOL * knee_search_range[1]]
      - slope_post > slope_pre 일 때 오탐으로 처리
      - fit_rmse > BW_MAX_RMSE 일 때 실패 처리
      - 초기값 grid 탐색으로 local minimum 회피

    Args:
        Q_n:               용량 궤적 1-D 배열 (Ah). cycle 1부터 순서대로.
        eol_cycle:         알려진 또는 추정된 EOL cycle 번호.
                           None이면 Q_n 내부에서 자동 추정.
        knee_search_range: EOL 대비 knee 탐색 구간 (기본 35%~70%).
        cycles:            Q_n에 대응하는 cycle 번호 배열.
                           None이면 1, 2, ..., len(Q_n)으로 자동 생성.

    Returns:
        dict with keys:
          'knee_cycle':  int  — 탐지된 knee cycle 번호
          'slope_pre':   float — knee 이전 선형 기울기 (Ah/cycle)
          'slope_post':  float — knee 이후 선형 기울기 (Ah/cycle)
          'fit_rmse':    float — Bacon-Watts fit RMSE (Ah)
          'success':     bool
          'method':      str  — 'bacon_watts' | 'max_curvature' | 'fallback_midpoint'
          'reason':      str  — 실패 이유 (success=False일 때)
    """
    if cycles is None:
        cycles = np.arange(1, len(Q_n) + 1, dtype=float)
    else:
        cycles = np.asarray(cycles, dtype=float)

    Q_n = np.asarray(Q_n, dtype=float)

    # --- 기본 검사 ---
    if len(Q_n) < KP_DETECTION_MIN_CYCLES:
        return _fallback_result(
            cycles, Q_n,
            reason=f"사이클 수 부족 ({len(Q_n)} < {KP_DETECTION_MIN_CYCLES})"
        )

    # --- EOL 추정 ---
    if eol_cycle is None:
        eol_cycle = _estimate_eol_cycle(Q_n)

    # EOL 없으면 시뮬이 아직 EoL에 도달 못 한 것 → len(Q_n) 전체를 EOL로 간주
    if eol_cycle is None:
        eol_cycle = int(cycles[-1])
        logger.debug(f"EOL 미도달 — 전체 궤적 길이({eol_cycle})를 EOL로 사용")

    # knee 탐색 구간
    knee_lo = float(eol_cycle) * knee_search_range[0]
    knee_hi = float(eol_cycle) * knee_search_range[1]

    # 탐색 구간이 data 범위를 벗어나는 경우 클리핑
    cycle_lo = float(cycles[min(5, len(cycles) - 1)])
    cycle_hi = float(cycles[max(-6, -len(cycles))])
    knee_lo = max(knee_lo, cycle_lo)
    knee_hi = min(knee_hi, cycle_hi)

    if knee_lo >= knee_hi:
        return _fallback_result(
            cycles, Q_n,
            reason=f"knee 탐색 구간 무효: [{knee_lo:.0f}, {knee_hi:.0f}]"
        )

    # --- 초기값 grid 탐색 ---
    knee_inits = np.linspace(knee_lo, knee_hi, _N_KNEE_INIT_GRID)
    best_result: Optional[dict] = None
    best_rmse = np.inf

    for knee_init in knee_inits:
        p0 = [float(Q_n[0]), -1e-4, -1e-4, float(knee_init), 50.0]
        bounds = (
            [-np.inf, -1.0, -1.0, knee_lo, BW_D_LOWER],
            [ np.inf,  0.0,  0.0, knee_hi, BW_D_UPPER],
        )
        try:
            popt, _ = curve_fit(
                _bacon_watts_model,
                cycles, Q_n,
                p0=p0,
                bounds=bounds,
                maxfev=10000,
            )
        except Exception:
            continue

        # fit quality
        residuals = Q_n - _bacon_watts_model(cycles, *popt)
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        # 물리적 타당성: slope_post는 slope_pre보다 더 가파르게 음수여야 함
        # BW 모델에서 knee 이후 기울기 = b + c (tanh→1), 이전 = b - c (tanh→-1)
        a_fit, b_fit, c_fit, knee_fit, d_fit = popt
        slope_pre = b_fit - c_fit    # tanh(-inf) = -1
        slope_post = b_fit + c_fit   # tanh(+inf) = +1

        # knee 의미 강제: post가 pre보다 더 빠른 감소
        # slope_post < slope_pre (둘 다 음수 기준, post가 더 음수)
        physical_valid = (slope_post < slope_pre) or (abs(slope_pre) < 1e-8)

        if physical_valid and rmse < best_rmse:
            best_rmse = rmse
            best_result = {
                "knee_cycle": int(round(knee_fit)),
                "slope_pre": float(slope_pre),
                "slope_post": float(slope_post),
                "fit_rmse": rmse,
                "popt": popt.tolist(),
            }

    # --- RMSE threshold 검사 ---
    if best_result is None:
        return _fallback_result(
            cycles, Q_n,
            reason="모든 BW 초기값 시도 실패 (curve_fit 미수렴)"
        )

    if best_rmse > BW_MAX_RMSE:
        return _fallback_result(
            cycles, Q_n,
            reason=f"BW fit RMSE({best_rmse:.4f}) > 허용 상한({BW_MAX_RMSE:.4f})"
        )

    best_result.update({
        "success": True,
        "method": "bacon_watts",
        "reason": "",
    })
    return best_result


# ===========================================================================
# 핵심 함수 2: max_curvature_knee (fallback)
# ===========================================================================

def max_curvature_knee(
    Q_n: np.ndarray,
    cycles: Optional[np.ndarray] = None,
    smooth_window: Optional[int] = None,
) -> Optional[int]:
    """최대 곡률(2차 미분) 기반 knee point 탐지 — BW fallback.

    Args:
        Q_n:           용량 궤적 1-D 배열
        cycles:        cycle 번호 배열 (None이면 1..N)
        smooth_window: 평활화 윈도우 크기 (None이면 자동)

    Returns:
        knee cycle 번호 (int) 또는 None
    """
    if cycles is None:
        cycles = np.arange(1, len(Q_n) + 1, dtype=float)
    else:
        cycles = np.asarray(cycles, dtype=float)

    Q_n = np.asarray(Q_n, dtype=float)

    if len(Q_n) < KP_DETECTION_MIN_CYCLES:
        return None

    if smooth_window is None:
        smooth_window = max(5, len(Q_n) // 20)

    smoothed = uniform_filter1d(Q_n, size=smooth_window)
    d1 = np.gradient(smoothed, cycles)
    d2 = np.gradient(d1, cycles)
    curvature = np.abs(d2) / (1.0 + d1 ** 2) ** 1.5

    # 양 끝 10% 마진 제외
    margin = max(int(len(Q_n) * 0.10), 5)
    interior = curvature[margin:-margin]
    if len(interior) == 0:
        return None

    peak_idx = int(np.argmax(interior)) + margin
    return int(cycles[peak_idx])


# ===========================================================================
# 내부 헬퍼
# ===========================================================================

def _fallback_result(
    cycles: np.ndarray,
    Q_n: np.ndarray,
    reason: str,
) -> dict:
    """BW 실패 시 max_curvature로 fallback."""
    mc = max_curvature_knee(Q_n, cycles)
    if mc is not None:
        logger.debug(f"BW fallback → max_curvature: knee={mc}  reason='{reason}'")
        return {
            "knee_cycle": mc,
            "slope_pre": None,
            "slope_post": None,
            "fit_rmse": None,
            "success": False,
            "method": "max_curvature",
            "reason": reason,
        }
    # 최후 fallback: 궤적 중간점
    midpoint = int(cycles[len(cycles) // 2])
    logger.warning(f"KP 탐지 완전 실패 → midpoint={midpoint}  reason='{reason}'")
    return {
        "knee_cycle": midpoint,
        "slope_pre": None,
        "slope_post": None,
        "fit_rmse": None,
        "success": False,
        "method": "fallback_midpoint",
        "reason": reason,
    }


# ===========================================================================
# 공개 인터페이스: detect_knee_point
# ===========================================================================

def detect_knee_point(
    Q_n: np.ndarray,
    eol_cycle: Optional[int] = None,
    cycles: Optional[np.ndarray] = None,
    knee_search_range: tuple[float, float] = (0.35, 0.70),
) -> dict:
    """통합 KP 탐지 — bacon_watts_constrained → max_curvature 순서.

    Args:
        Q_n:               용량 궤적
        eol_cycle:         EOL cycle (없으면 자동 추정)
        cycles:            cycle 번호 배열
        knee_search_range: BW 탐색 구간 [lo_frac, hi_frac] × EOL

    Returns:
        bacon_watts_constrained와 동일한 dict 구조
    """
    return bacon_watts_constrained(
        Q_n=Q_n,
        eol_cycle=eol_cycle,
        knee_search_range=knee_search_range,
        cycles=cycles,
    )
