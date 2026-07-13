"""
Battery domain hallucination guards (BATT-H-01~09).

9가지 도메인 환각 위험에 대한 assert 형 회귀 차단.
실제 개발 중 관찰된 오류 패턴에서 도출한 방어적 검증 규칙.

Usage example::

    from src.battery_guards import (
        check_phase0_capacity,
        check_phase1_identifiability,
        check_grid_covers_real_theta,
        check_hybrid_pv_validation,
        check_knee_attribution,
        check_c_rate_to_current,
        check_sei_params_present,
        check_dataset_compatibility,
        check_train_eval_cell_split,
        run_all_guards,
        BattGuardViolation,
    )
    import numpy as np

    # BATT-H-01: Phase 0 캘리브레이션 결과 검증
    check_phase0_capacity(c1=0.927, c4=0.961)

    # BATT-H-06: C-rate → 절대 전류 검증
    check_c_rate_to_current("A123_APR18650M1A", c_rate=1.0, current_A=1.1)

    # BATT-H-09: 학습/평가 셀 분리 검증
    check_train_eval_cell_split(
        train_cells={"cell0", "cell1"},
        eval_cells={"cell2", "cell3"},
    )
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOOKUP_PATH = PROJECT_ROOT / "configs" / "battery_guards_lookup.yaml"


# ---------------------------------------------------------------------------
# 예외 클래스
# ---------------------------------------------------------------------------


class BattGuardViolation(AssertionError):
    """BATT-H-* 가드 위반.

    guard_id 속성으로 어느 가드를 위반했는지 추적 가능.
    """

    def __init__(self, guard_id: str, message: str) -> None:
        self.guard_id = guard_id
        super().__init__(f"[{guard_id}] {message}")


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------


def _load_lookup() -> dict:
    """battery_guards_lookup.yaml을 파싱해 반환."""
    with open(LOOKUP_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# BATT-H-01: Phase 0 calibration values
# ---------------------------------------------------------------------------


def check_phase0_capacity(
    c1: float,
    c4: float,
    *,
    tolerance: bool = True,
) -> None:
    """1C/4C 방전 용량이 Phase 0 정답 범위 내인지 확인.

    BATT-H-01: Phase 0 figure/슬라이드에 '4C=1.01 Ah' 등 잘못된 값 오기재 방지.
    정답: 1C=0.927 Ah, 4C=0.961 Ah (configs/A123_Prada2013_hybrid.yaml, TRI cell5 cycle1).
    GATE-C: 1C ∈ [0.90, 0.96], 4C ∈ [0.93, 1.00].

    Args:
        c1: 1C 방전 용량 [Ah].
        c4: 4C 방전 용량 [Ah].
        tolerance: True(기본)이면 허용 범위 검사; False이면 정확한 정답 값 비교.

    Raises:
        BattGuardViolation: 용량 값이 허용 범위를 벗어난 경우.
    """
    lut = _load_lookup()["phase0_calibration"]

    if tolerance:
        lo1, hi1 = lut["tolerance"]["capacity_1C"]
        lo4, hi4 = lut["tolerance"]["capacity_4C"]
        if not (lo1 <= c1 <= hi1):
            raise BattGuardViolation(
                "BATT-H-01",
                f"1C 용량 {c1:.4f} Ah가 허용 범위 [{lo1}, {hi1}] 밖입니다.",
            )
        if not (lo4 <= c4 <= hi4):
            raise BattGuardViolation(
                "BATT-H-01",
                f"4C 용량 {c4:.4f} Ah가 허용 범위 [{lo4}, {hi4}] 밖입니다.",
            )
    else:
        target_c1 = lut["capacity_1C_Ah"]
        target_c4 = lut["capacity_4C_Ah"]
        tol = 1e-6
        if abs(c1 - target_c1) > tol:
            raise BattGuardViolation(
                "BATT-H-01",
                f"1C 용량 {c1} ≠ Phase 0 정답 {target_c1}",
            )
        if abs(c4 - target_c4) > tol:
            raise BattGuardViolation(
                "BATT-H-01",
                f"4C 용량 {c4} ≠ Phase 0 정답 {target_c4}",
            )


# ---------------------------------------------------------------------------
# BATT-H-02: Phase 1 identifiability
# ---------------------------------------------------------------------------


def check_phase1_identifiability(
    theta_samples: np.ndarray,
    *,
    dim_names: Optional[list] = None,
    std_thresholds: Optional[dict] = None,
) -> dict:
    """θ 표본의 각 차원 표준편차를 검사해 boundary 수렴 여부를 감지.

    BATT-H-02: TRI 188 샘플에서 ε_pos=0.55(boundary)로 100% 수렴 → 실질 2D 분포.
    '3-param 역산'이라 주장할 때 실제 자유도를 확인하는 가드.

    Args:
        theta_samples: shape (N, d) — 역산 결과 θ 표본 행렬.
        dim_names: 차원 이름 목록 (기본: ["SoC", "eps_neg", "eps_pos"]).
        std_thresholds: 차원별 최소 std 임계값 (기본: lookup yaml 값 사용).

    Returns:
        각 차원의 {'name': str, 'std': float, 'pass': bool} 목록을 담은 dict.

    Raises:
        BattGuardViolation: std < 임계값인 차원이 하나 이상 존재할 경우.
    """
    lut = _load_lookup()["phase1_identifiability"]
    default_names = ["SoC", "eps_neg", "eps_pos"]
    default_thresholds = {
        "eps_pos": lut["eps_pos_std_threshold"],
        "eps_neg": lut["eps_neg_std_threshold"],
        "SoC": lut["soc_std_threshold"],
    }

    names = dim_names or default_names
    thresholds = std_thresholds or default_thresholds

    if theta_samples.ndim != 2:
        raise ValueError(f"theta_samples는 2D 배열이어야 합니다. 현재 shape: {theta_samples.shape}")

    results: dict = {"dimensions": [], "pass": True}
    violations = []

    for i, name in enumerate(names):
        if i >= theta_samples.shape[1]:
            break
        std = float(theta_samples[:, i].std())
        threshold = thresholds.get(name, 0.01)
        passed = std >= threshold
        results["dimensions"].append({"name": name, "std": std, "pass": passed})
        if not passed:
            violations.append(f"dim='{name}' std={std:.6f} < {threshold}")

    if violations:
        results["pass"] = False
        raise BattGuardViolation(
            "BATT-H-02",
            "식별불가능성 의심 (boundary 수렴): " + "; ".join(violations),
        )

    return results


# ---------------------------------------------------------------------------
# BATT-H-03: Grid bounds vs real theta
# ---------------------------------------------------------------------------


def check_grid_covers_real_theta(
    grid_bounds: dict,
    real_theta_samples: np.ndarray,
    dim_names: Optional[list] = None,
) -> None:
    """Synthetic θ grid bounds가 실측 Phase 1 θ 분포를 완전히 포함하는지 확인.

    BATT-H-03: Grid bounds를 임의 설정하면 실측 TRI θ 분포와 mismatch →
    density estimator 무의미. P3-A grid 생성 작업 시작 전 반드시 호출.

    Args:
        grid_bounds: 차원별 [min, max] dict.
            예: {"SoC_init": [0.85, 1.0], "eps_neg": [0.2, 0.55], "eps_pos": [0.2, 0.45]}
        real_theta_samples: shape (N, d) — 실측 역산 θ 표본 행렬.
        dim_names: grid_bounds 키 순서와 동일한 차원 이름 목록.

    Raises:
        BattGuardViolation: grid가 실측 θ 분포를 포함하지 못할 경우.
    """
    names = dim_names or list(grid_bounds.keys())

    if real_theta_samples.ndim != 2:
        raise ValueError(f"real_theta_samples는 2D 배열이어야 합니다. shape: {real_theta_samples.shape}")

    violations = []
    for i, name in enumerate(names):
        if i >= real_theta_samples.shape[1]:
            break
        if name not in grid_bounds:
            continue
        g_min, g_max = grid_bounds[name]
        r_min = float(real_theta_samples[:, i].min())
        r_max = float(real_theta_samples[:, i].max())

        if r_min < g_min:
            violations.append(
                f"dim='{name}': 실측 min={r_min:.4f} < grid min={g_min:.4f}"
            )
        if r_max > g_max:
            violations.append(
                f"dim='{name}': 실측 max={r_max:.4f} > grid max={g_max:.4f}"
            )

    if violations:
        raise BattGuardViolation(
            "BATT-H-03",
            "Grid bounds가 실측 θ 분포를 포함하지 못합니다: " + "; ".join(violations),
        )


# ---------------------------------------------------------------------------
# BATT-H-04: Hybrid PV validation
# ---------------------------------------------------------------------------


def check_hybrid_pv_validation(c1: float, c4: float) -> None:
    """Mix-and-match 방식으로 구성된 Hybrid PV의 1C/4C 용량을 검증.

    BATT-H-04: Chen2020 graphite를 LFP Prada2013에 통째로 가져오면
    용량 +38% 과다 계산 (1.28 Ah). Mix-and-match 후 반드시 이 가드 호출.
    기준: 1C ∈ [0.90, 0.96], 4C ∈ [0.93, 1.00], ratio ≥ 0.95.

    Args:
        c1: Hybrid PV 1C 방전 용량 [Ah].
        c4: Hybrid PV 4C 방전 용량 [Ah].

    Raises:
        BattGuardViolation: 용량 또는 4C/1C 비율이 기준을 벗어날 경우.
    """
    lut = _load_lookup()["hybrid_pv_validation"]
    lo1, hi1 = lut["pass_range_1C"]
    lo4, hi4 = lut["pass_range_4C"]
    ratio_min = lut["ratio_4C_over_1C_min"]

    if not (lo1 <= c1 <= hi1):
        raise BattGuardViolation(
            "BATT-H-04",
            f"Hybrid PV 1C={c1:.4f} Ah가 통과 범위 [{lo1}, {hi1}] 밖입니다.",
        )
    if not (lo4 <= c4 <= hi4):
        raise BattGuardViolation(
            "BATT-H-04",
            f"Hybrid PV 4C={c4:.4f} Ah가 통과 범위 [{lo4}, {hi4}] 밖입니다.",
        )
    ratio = c4 / c1 if c1 > 0 else 0.0
    if ratio < ratio_min:
        raise BattGuardViolation(
            "BATT-H-04",
            f"Hybrid PV 4C/1C 비율 {ratio:.4f} < 최소 기준 {ratio_min}. "
            "kinetics 회복 미확인.",
        )


# ---------------------------------------------------------------------------
# BATT-H-05: Knee location attribution
# ---------------------------------------------------------------------------


def check_knee_attribution(
    slide_content: str,
    figure_name: str = "fig1_before_after.png",
) -> None:
    """Knee figure가 Phase 0이 아닌 Branch B/B2 섹션에 배치되어 있는지 확인.

    BATT-H-05: knee 재현 figure는 Branch B2(2-Phase slippage 모델)의 결과물.
    Phase 0 슬라이드에 knee figure가 잘못 삽입되는 것을 방지.
    Phase 0 결과는 capacity bar chart만 포함해야 함.

    Args:
        slide_content: 슬라이드 또는 마크다운 파일 전체 텍스트.
        figure_name: 검사할 figure 파일명 (기본: 'fig1_before_after.png').

    Raises:
        BattGuardViolation: figure_name이 발견되었지만 Branch B/B2 헤더 아래에
            없는 경우, 또는 Phase 0 섹션 아래에 있는 경우.
    """
    import re

    if figure_name not in slide_content:
        # figure 자체가 없으면 문제 없음
        return

    lines = slide_content.splitlines()
    figure_line_indices = [i for i, ln in enumerate(lines) if figure_name in ln]

    for fig_idx in figure_line_indices:
        # figure 위의 가장 가까운 헤더 찾기
        nearest_header = ""
        nearest_header_idx = -1
        for j in range(fig_idx - 1, -1, -1):
            ln = lines[j].strip()
            if re.match(r"^#{1,6}\s+", ln):
                nearest_header = ln
                nearest_header_idx = j
                break

        # Branch B/B2 헤더 아래인지 확인
        is_branch_b = bool(re.search(r"Branch\s+B", nearest_header, re.IGNORECASE))
        # Phase 0 헤더 아래인지 확인
        is_phase0 = bool(re.search(r"Phase\s*0", nearest_header, re.IGNORECASE))

        if is_phase0:
            raise BattGuardViolation(
                "BATT-H-05",
                f"'{figure_name}'이 Phase 0 섹션('{nearest_header}')에 배치되어 있습니다. "
                "Knee figure는 Branch B/B2 슬라이드에 있어야 합니다.",
            )
        if not is_branch_b and nearest_header:
            raise BattGuardViolation(
                "BATT-H-05",
                f"'{figure_name}'이 Branch B/B2가 아닌 섹션('{nearest_header}')에 있습니다. "
                "Knee 메커니즘은 Branch B2(slippage 모델)에서만 발생합니다.",
            )


# ---------------------------------------------------------------------------
# BATT-H-06: C-rate to current
# ---------------------------------------------------------------------------


def check_c_rate_to_current(
    cell_model: str,
    c_rate: float,
    current_A: float,
) -> None:
    """C-rate와 절대 전류 [A] 간 변환이 올바른지 확인.

    BATT-H-06: '1C = 1A'로 잘못 가정하는 오류 방지.
    A123 APR18650M1A: 공칭 1.1 Ah → 1C = 1.1 A, 4C = 4.4 A.

    Args:
        cell_model: 셀 모델명 (현재 'A123_APR18650M1A'만 hardcoded 지원).
        c_rate: C-rate 값 (예: 1.0, 4.0).
        current_A: 대응하는 절대 전류 [A].

    Raises:
        BattGuardViolation: 계산된 전류와 입력 전류의 오차가 허용치 초과.
        ValueError: 지원하지 않는 셀 모델.
    """
    lut = _load_lookup()["c_rate_a123"]

    if cell_model != lut["cell_model"]:
        raise ValueError(
            f"'{cell_model}'은 지원하지 않는 셀 모델입니다. "
            f"현재 지원: '{lut['cell_model']}'"
        )

    nominal_cap = lut["nominal_capacity_Ah"]
    expected_A = c_rate * nominal_cap
    tolerance_pct = lut["current_tolerance_pct"] / 100.0
    allowed_error = expected_A * tolerance_pct

    if abs(current_A - expected_A) > allowed_error:
        raise BattGuardViolation(
            "BATT-H-06",
            f"{c_rate}C 기대 전류 {expected_A:.3f} A (±{allowed_error:.3f} A), "
            f"입력값 {current_A:.3f} A. "
            f"A123 공칭 용량 {nominal_cap} Ah 기반.",
        )


# ---------------------------------------------------------------------------
# BATT-H-07: SEI parameter completeness
# ---------------------------------------------------------------------------


def check_sei_params_present(pv: dict) -> None:
    """Hybrid PV dict에 SEI 파라미터 10개가 모두 존재하는지 확인.

    BATT-H-07: Prada2013에는 SEI 파라미터가 없음. Hybrid PV에서 그대로 쓰면
    SEI growth 모델 작동 불가. Chen2020/OKane2022에서 10개 키 통째 import 필요.
    P1-B 문서 참조.

    Args:
        pv: PyBaMM parameter values dict (또는 key 목록을 담은 dict).

    Raises:
        BattGuardViolation: 필수 SEI 키가 하나 이상 누락된 경우.
    """
    lut = _load_lookup()
    required_keys = lut["required_sei_keys"]

    missing = [k for k in required_keys if k not in pv]
    if missing:
        raise BattGuardViolation(
            "BATT-H-07",
            f"Hybrid PV에 SEI 파라미터 {len(missing)}개 누락: {missing}. "
            "Chen2020 또는 OKane2022에서 SEI 파라미터 10개를 통째로 import하십시오.",
        )


# ---------------------------------------------------------------------------
# BATT-H-08: Dataset distribution shift
# ---------------------------------------------------------------------------


def check_dataset_compatibility(
    tri_cycle_life: np.ndarray,
    hust_cycle_life: np.ndarray,
    *,
    warn_only: bool = True,
) -> dict:
    """TRI vs HUST cycle_life 분포 KS test를 수행하고 결과를 반환.

    BATT-H-08: TRI 평균 666~845 vs HUST 평균 1899 (3배 이상 차이).
    같은 셀(A123 APR18650M1A)이지만 프로토콜 다양성 차이 → domain shift.
    두 데이터셋을 단순 합쳐 학습하면 분포 불일치 발생.

    Args:
        tri_cycle_life: TRI 셀들의 cycle_life 배열.
        hust_cycle_life: HUST 셀들의 cycle_life 배열.
        warn_only: True(기본)이면 경고 정보 반환만, False이면 p<0.05 시 raise.

    Returns:
        {'ks_statistic': float, 'p_value': float, 'domain_shift_detected': bool,
         'tri_mean': float, 'hust_mean': float, 'mean_ratio': float}

    Raises:
        BattGuardViolation: warn_only=False이고 p-value < 임계값인 경우.
    """
    from scipy import stats

    lut = _load_lookup()["dataset_distribution"]
    threshold = lut["ks_pvalue_threshold"]

    ks_stat, p_value = stats.ks_2samp(tri_cycle_life, hust_cycle_life)
    tri_mean = float(np.mean(tri_cycle_life))
    hust_mean = float(np.mean(hust_cycle_life))
    mean_ratio = hust_mean / tri_mean if tri_mean > 0 else float("inf")
    domain_shift = p_value < threshold

    result = {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "domain_shift_detected": domain_shift,
        "tri_mean": tri_mean,
        "hust_mean": hust_mean,
        "mean_ratio": mean_ratio,
    }

    if domain_shift and not warn_only:
        raise BattGuardViolation(
            "BATT-H-08",
            f"TRI(mean={tri_mean:.0f}) vs HUST(mean={hust_mean:.0f}) "
            f"KS p-value={p_value:.4f} < {threshold}. "
            "Domain shift 처리(source 컬럼 의무, 도메인 적응 모델) 없이 단순 합산 금지.",
        )

    return result


# ---------------------------------------------------------------------------
# BATT-H-09: Per-cell fit circularity
# ---------------------------------------------------------------------------


def check_train_eval_cell_split(
    train_cells: set,
    eval_cells: set,
) -> None:
    """학습 셀과 평가 셀이 cell_id 기준으로 disjoint인지 확인.

    BATT-H-09 (★ Critical #1): Per-cell DE fit → θ → forward sim → augmented data
    학습 시, θ를 만든 셀이 해당 셀의 evaluation set에 포함되면 순환 편향 발생.
    per-cell fit 기반 augmented data는 해당 셀의 eval set에 절대 포함 금지.

    Args:
        train_cells: 학습 셀 ID 집합.
        eval_cells: 평가 셀 ID 집합.

    Raises:
        BattGuardViolation: 두 집합에 공통 셀이 하나 이상 있을 경우.
    """
    overlap = train_cells & eval_cells
    if overlap:
        raise BattGuardViolation(
            "BATT-H-09",
            f"학습/평가 셀이 disjoint하지 않습니다. 중복 셀: {sorted(overlap)}. "
            "Per-cell fit 기반 augmented data를 해당 셀 eval에 사용하면 circularity 발생.",
        )


# ---------------------------------------------------------------------------
# BATT-H-10: Domain shift 자동 감지 (일반화)
# ---------------------------------------------------------------------------


def check_domain_shift(
    cycle_life_a: np.ndarray,
    cycle_life_b: np.ndarray,
    *,
    dataset_a_name: str = "A",
    dataset_b_name: str = "B",
    ks_p_threshold: float = 0.05,
    warn_only: bool = True,
) -> dict:
    """두 임의 분포 간 KS 2-sample test로 domain shift를 감지.

    BATT-H-08이 TRI vs HUST cycle_life에 특화된 것과 달리,
    BATT-H-10은 임의의 두 분포(θ_3 차원, V(t) 통계 등)를 비교 가능.
    새 데이터셋 통합 전 분포 차이를 자동으로 확인.

    Args:
        cycle_life_a: 배열 A (1D).
        cycle_life_b: 배열 B (1D).
        dataset_a_name: 배열 A의 레이블 (로그/오류 메시지용).
        dataset_b_name: 배열 B의 레이블 (로그/오류 메시지용).
        ks_p_threshold: KS test p-value 임계값 (기본: 0.05).
        warn_only: True(기본)이면 결과 반환만; False이면 shift 감지 시 raise.

    Returns:
        {
            "ks_stat": float,
            "p_value": float,
            "shift_detected": bool,
            "mean_a": float,
            "mean_b": float,
            "dataset_a_name": str,
            "dataset_b_name": str,
        }

    Raises:
        BattGuardViolation: warn_only=False이고 p-value < ks_p_threshold인 경우.
    """
    from scipy import stats

    a = np.asarray(cycle_life_a, dtype=float).ravel()
    b = np.asarray(cycle_life_b, dtype=float).ravel()

    ks_stat, p_value = stats.ks_2samp(a, b)
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    shift_detected = p_value < ks_p_threshold

    result: dict = {
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "shift_detected": shift_detected,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "dataset_a_name": dataset_a_name,
        "dataset_b_name": dataset_b_name,
    }

    if shift_detected and not warn_only:
        raise BattGuardViolation(
            "BATT-H-10",
            f"'{dataset_a_name}'(mean={mean_a:.1f}) vs '{dataset_b_name}'(mean={mean_b:.1f}) "
            f"KS p-value={p_value:.2e} < {ks_p_threshold}. "
            "Domain shift 감지 — 도메인 적응 처리 없이 단순 합산 금지.",
        )

    return result


# ---------------------------------------------------------------------------
# BATT-H-11: Extrapolation 경고 (convex hull)
# ---------------------------------------------------------------------------


def check_theta_in_training_hull(
    theta_query: np.ndarray,
    theta_training: np.ndarray,
    *,
    tolerance: float = 0.05,
    raise_on_outside: bool = False,
) -> dict:
    """새 θ 쿼리가 학습 데이터의 convex hull 내부인지 확인.

    외부이면 외삽(extrapolation) → 예측 정확도 하락 가능.
    P5-D smoke test에서 학습 θ 범위 밖 추론 시 성능 저하 발견.

    Implementation:
        - scipy.spatial.Delaunay로 학습 데이터 볼록 껍질 구성
        - tolerance만큼 hull을 무게중심 기준으로 확장 후 포함 여부 검사

    Args:
        theta_query: 쿼리 θ 배열 (shape: (d,) 또는 (N, d)).
        theta_training: 학습 θ 배열 (shape: (M, d)).
        tolerance: hull 확장 비율 (기본 0.05 = 5% 확장). 0이면 정확한 hull.
        raise_on_outside: True이면 하나라도 hull 외부 시 BattGuardViolation raise.

    Returns:
        {
            "all_inside": bool,
            "inside_mask": list[bool],
            "n_outside": int,
            "n_query": int,
            "skipped": bool,       # 학습 데이터 부족으로 hull 계산 불가 시 True
            "reason": str | None,  # skipped=True 시 사유
        }

    Raises:
        BattGuardViolation: raise_on_outside=True이고 hull 외부 점이 있을 경우.
    """
    from scipy.spatial import Delaunay

    theta_query = np.atleast_2d(np.asarray(theta_query, dtype=float))
    theta_training = np.atleast_2d(np.asarray(theta_training, dtype=float))

    n_pts, n_dim = theta_training.shape

    # Delaunay는 d+1개 이상의 비공면 점이 필요
    if n_pts < n_dim + 1:
        return {
            "all_inside": None,
            "inside_mask": None,
            "n_outside": None,
            "n_query": int(len(theta_query)),
            "skipped": True,
            "reason": (
                f"학습 데이터 포인트 수({n_pts})가 차원({n_dim})+1 미만. "
                "Convex hull 계산 불가 — skip."
            ),
        }

    # Tolerance 적용: hull을 무게중심 기준으로 (1+tolerance) 배 확장
    centroid = theta_training.mean(axis=0)
    theta_expanded = centroid + (1.0 + tolerance) * (theta_training - centroid)

    try:
        hull = Delaunay(theta_expanded)
    except Exception as exc:  # QhullError 등
        return {
            "all_inside": None,
            "inside_mask": None,
            "n_outside": None,
            "n_query": int(len(theta_query)),
            "skipped": True,
            "reason": f"Delaunay 삼각분할 실패: {exc}",
        }

    inside_mask: np.ndarray = hull.find_simplex(theta_query) >= 0
    n_outside = int((~inside_mask).sum())
    all_inside = bool(inside_mask.all())

    result: dict = {
        "all_inside": all_inside,
        "inside_mask": inside_mask.tolist(),
        "n_outside": n_outside,
        "n_query": int(len(theta_query)),
        "skipped": False,
        "reason": None,
    }

    if not all_inside and raise_on_outside:
        raise BattGuardViolation(
            "BATT-H-11",
            f"θ 쿼리 {n_outside}/{len(theta_query)}개가 학습 데이터 convex hull 외부. "
            f"외삽(extrapolation) 발생 — 예측 정확도 하락 가능. "
            f"tolerance={tolerance}",
        )

    return result


# ---------------------------------------------------------------------------
# BATT-H-12: Best epoch 조기 stop 경고
# ---------------------------------------------------------------------------


def check_best_epoch_used(
    best_epoch: int,
    total_epochs: int,
    *,
    min_ratio: float = 0.1,
    warn_only: bool = True,
) -> dict:
    """Best epoch이 total_epochs의 min_ratio 미만이면 조기 stop 경고.

    BATT-H-12: P7-B v2에서 best_epoch=8 / total_epochs=200 = 4% < 10%.
    학습이 충분히 진행되기 전에 최적 모델이 결정된 것으로, 하이퍼파라미터
    설정(lr, weight_decay 등) 재검토 필요.

    Args:
        best_epoch: 검증 손실이 가장 낮았던 epoch 번호 (1-indexed).
        total_epochs: 전체 학습 epoch 수.
        min_ratio: best_epoch / total_epochs 최소 비율 (기본 0.1 = 10%).
        warn_only: True(기본)이면 결과 반환만; False이면 비율 미달 시 raise.

    Returns:
        {
            "best_epoch": int,
            "total_epochs": int,
            "ratio": float,
            "early_stop_warned": bool,
            "min_ratio": float,
        }

    Raises:
        BattGuardViolation: warn_only=False이고 ratio < min_ratio인 경우.
    """
    if total_epochs <= 0:
        raise ValueError(f"total_epochs은 양의 정수여야 합니다. 현재: {total_epochs}")
    if best_epoch <= 0:
        raise ValueError(f"best_epoch은 양의 정수여야 합니다. 현재: {best_epoch}")
    if best_epoch > total_epochs:
        raise ValueError(
            f"best_epoch({best_epoch}) > total_epochs({total_epochs})."
        )

    ratio = best_epoch / total_epochs
    early_stop_warned = ratio < min_ratio

    result: dict = {
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "ratio": ratio,
        "early_stop_warned": early_stop_warned,
        "min_ratio": min_ratio,
    }

    if early_stop_warned and not warn_only:
        raise BattGuardViolation(
            "BATT-H-12",
            f"Best epoch={best_epoch} / total={total_epochs} = {ratio:.1%} < "
            f"min_ratio={min_ratio:.1%}. "
            "조기 수렴 의심 — lr, weight_decay 등 하이퍼파라미터 재검토 권장.",
        )

    return result


# ---------------------------------------------------------------------------
# BATT-H-13: SHAP value sign consistency
# ---------------------------------------------------------------------------


def check_shap_sign_consistency(
    shap_matrix_runs: list,
    *,
    max_variation: Optional[float] = None,
) -> dict:
    """여러 random seed 실행에서 SHAP value의 sign 일관성을 확인.

    BATT-H-13: XAI 해석에서 동일 셀에 대해 SHAP sign이 seed마다 뒤집히면
    모델 설명이 불안정 — feature importance 해석 신뢰 불가.
    기준: 음수/양수 비율 변동 < 10% (max_sign_ratio_variation).

    Args:
        shap_matrix_runs: shape (n_runs, n_samples, n_features) 에 해당하는
            numpy 배열 리스트. 각 원소는 (n_samples, n_features) 배열.
        max_variation: 허용 최대 sign 비율 변동 (기본: lookup yaml 값 0.10).

    Returns:
        {
            "pass": bool,
            "max_variation_observed": float,
            "max_variation_threshold": float,
            "per_feature_variation": list[float],
        }

    Raises:
        BattGuardViolation: 어느 feature에서든 sign 비율 변동이 임계값 초과.
        ValueError: shap_matrix_runs가 비어있거나 배열 shape 불일치.
    """
    if not shap_matrix_runs:
        raise ValueError("shap_matrix_runs는 비어있을 수 없습니다.")

    lut = _load_lookup()["batt_h_13_shap_sign"]
    threshold = max_variation if max_variation is not None else lut["max_sign_ratio_variation"]

    arrays = [np.asarray(m, dtype=float) for m in shap_matrix_runs]
    n_features = arrays[0].shape[-1]

    # 각 run별 feature별 양수 비율 계산
    # shape: (n_runs, n_features)
    pos_ratios = np.array([
        [(arr[:, f] > 0).mean() for f in range(n_features)]
        for arr in arrays
    ])  # shape (n_runs, n_features)

    # feature별 양수 비율 최대-최소 차이 = variation
    per_feature_variation = (pos_ratios.max(axis=0) - pos_ratios.min(axis=0)).tolist()
    max_variation_observed = float(max(per_feature_variation))

    result: dict = {
        "pass": max_variation_observed <= threshold,
        "max_variation_observed": max_variation_observed,
        "max_variation_threshold": threshold,
        "per_feature_variation": per_feature_variation,
    }

    if not result["pass"]:
        worst_feature = int(np.argmax(per_feature_variation))
        raise BattGuardViolation(
            "BATT-H-13",
            f"SHAP sign 비율 변동 {max_variation_observed:.4f} > 임계값 {threshold:.4f}. "
            f"가장 불안정한 feature index={worst_feature} "
            f"(variation={per_feature_variation[worst_feature]:.4f}). "
            "Random seed 변화에 따른 SHAP sign 불일관성 — 모델 재학습 또는 앙상블 SHAP 사용 권장.",
        )

    return result


# ---------------------------------------------------------------------------
# BATT-H-14: MMD threshold trigger
# ---------------------------------------------------------------------------


def check_mmd_da_trigger(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    tau: Optional[float] = None,
    kernel_bandwidth: float = 1.0,
) -> dict:
    """Two-sample Maximum Mean Discrepancy(MMD)를 계산하고 DA 트리거 여부를 반환.

    BATT-H-14: MMD >= τ(기본 0.1) 이면 domain adaptation이 필요한 상황.
    RBF kernel 기반 unbiased MMD 추정 (numpy만으로 구현, 외부 의존성 없음).

    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    where k(a,b) = exp(-||a-b||^2 / (2*h^2)), h=kernel_bandwidth.

    Args:
        source_features: 소스 도메인 feature 행렬 (n_s, d).
        target_features: 타겟 도메인 feature 행렬 (n_t, d).
        tau: MMD 임계값 (기본: lookup yaml 값 0.1).
        kernel_bandwidth: RBF kernel bandwidth h (기본: 1.0).

    Returns:
        {
            "mmd": float,
            "tau": float,
            "require_DA": bool,
            "n_source": int,
            "n_target": int,
        }

    Raises:
        BattGuardViolation: MMD >= tau인 경우 (DA 트리거 필요 상황 명시적 차단).
        ValueError: 입력 배열이 2D가 아니거나 feature 차원 불일치.
    """
    lut = _load_lookup()["batt_h_14_mmd"]
    tau_val = tau if tau is not None else lut["mmd_tau"]

    src = np.atleast_2d(np.asarray(source_features, dtype=float))
    tgt = np.atleast_2d(np.asarray(target_features, dtype=float))

    if src.ndim != 2 or tgt.ndim != 2:
        raise ValueError("source_features, target_features는 2D 배열이어야 합니다.")
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(
            f"feature 차원 불일치: source={src.shape[1]}, target={tgt.shape[1]}"
        )

    def _rbf_kernel_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """RBF(Gaussian) kernel matrix K[i,j] = exp(-||a_i - b_j||^2 / (2h^2))."""
        # pairwise squared distances via broadcasting
        diff = a[:, None, :] - b[None, :, :]  # (n_a, n_b, d)
        sq_dist = (diff ** 2).sum(axis=-1)      # (n_a, n_b)
        return np.exp(-sq_dist / (2.0 * kernel_bandwidth ** 2))

    kxx = _rbf_kernel_matrix(src, src)
    kyy = _rbf_kernel_matrix(tgt, tgt)
    kxy = _rbf_kernel_matrix(src, tgt)

    n_s = src.shape[0]
    n_t = tgt.shape[0]

    # unbiased MMD^2 추정 (대각선 제외)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)

    term_xx = kxx.sum() / (n_s * (n_s - 1)) if n_s > 1 else 0.0
    term_yy = kyy.sum() / (n_t * (n_t - 1)) if n_t > 1 else 0.0
    term_xy = kxy.mean()

    mmd2 = term_xx + term_yy - 2.0 * term_xy
    mmd = float(np.sqrt(max(mmd2, 0.0)))

    require_da = mmd >= tau_val

    result: dict = {
        "mmd": mmd,
        "tau": tau_val,
        "require_DA": require_da,
        "n_source": n_s,
        "n_target": n_t,
    }

    if require_da:
        raise BattGuardViolation(
            "BATT-H-14",
            f"MMD={mmd:.4f} >= τ={tau_val:.4f}. "
            "Domain adaptation(DA) 자동 트리거 — "
            "소스/타겟 feature 분포 불일치가 임계값 초과. "
            "DA 레이어 또는 도메인 정규화 없이 직접 적용 금지.",
        )

    return result


# ---------------------------------------------------------------------------
# BATT-H-15: Counterfactual monotonicity
# ---------------------------------------------------------------------------


def check_counterfactual_monotonicity(
    theta_deltas: np.ndarray,
    eol_deltas: np.ndarray,
    *,
    max_violation_ratio: Optional[float] = None,
) -> dict:
    """Counterfactual θ 변화량 vs EOL 변화량의 단조성(sign 일관성)을 검사.

    BATT-H-15: ∂EOL/∂θ_neg가 같은 방향이어야 하는데 sign이 뒤집히면
    counterfactual 설명이 물리적으로 모순. XAI/CF 모듈 품질 검사.
    기준: 부호 위반 비율 < max_violation_ratio(기본 0.10 = 10%).

    sign 일관성 기준:
        - (theta_delta > 0, eol_delta > 0) 또는 (theta_delta < 0, eol_delta < 0)
          → 같은 방향 (통과)
        - theta_delta ≈ 0 (|delta| < eps) → 무시
        - sign이 반대 → 위반

    Args:
        theta_deltas: θ 변화량 배열 (1D or 2D). 1D이면 단일 feature.
        eol_deltas: EOL 변화량 배열 (1D), theta_deltas와 같은 길이.
        max_violation_ratio: 허용 최대 sign 위반 비율 (기본: lookup yaml 0.10).

    Returns:
        {
            "pass": bool,
            "violation_ratio": float,
            "n_violations": int,
            "n_valid": int,
            "max_violation_threshold": float,
        }

    Raises:
        BattGuardViolation: sign 위반 비율이 임계값 초과.
        ValueError: 입력 배열 길이 불일치.
    """
    lut = _load_lookup()["batt_h_15_counterfactual"]
    threshold = (
        max_violation_ratio
        if max_violation_ratio is not None
        else lut["max_violation_ratio"]
    )

    theta_d = np.asarray(theta_deltas, dtype=float).ravel()
    eol_d = np.asarray(eol_deltas, dtype=float).ravel()

    if len(theta_d) != len(eol_d):
        raise ValueError(
            f"theta_deltas({len(theta_d)})와 eol_deltas({len(eol_d)}) 길이가 다릅니다."
        )

    eps = 1e-10
    # theta_delta가 거의 0인 경우 제외
    valid_mask = np.abs(theta_d) > eps
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        # 변화량이 없으면 검사 불필요 → pass
        return {
            "pass": True,
            "violation_ratio": 0.0,
            "n_violations": 0,
            "n_valid": 0,
            "max_violation_threshold": threshold,
        }

    theta_valid = theta_d[valid_mask]
    eol_valid = eol_d[valid_mask]

    # 같은 sign이면 단조성 유지, 다른 sign이면 위반
    same_sign = (np.sign(theta_valid) == np.sign(eol_valid))
    n_violations = int((~same_sign).sum())
    violation_ratio = n_violations / n_valid

    result: dict = {
        "pass": violation_ratio <= threshold,
        "violation_ratio": violation_ratio,
        "n_violations": n_violations,
        "n_valid": n_valid,
        "max_violation_threshold": threshold,
    }

    if not result["pass"]:
        raise BattGuardViolation(
            "BATT-H-15",
            f"Counterfactual sign 위반 비율 {violation_ratio:.4f} > 임계값 {threshold:.4f}. "
            f"위반 {n_violations}/{n_valid}개. "
            "∂EOL/∂θ sign 불일관성 — counterfactual 설명이 물리적으로 모순.",
        )

    return result


# ---------------------------------------------------------------------------
# BATT-H-16: TTT update stability
# ---------------------------------------------------------------------------


def check_ttt_update_stability(
    weights_before: np.ndarray,
    weights_after: np.ndarray,
    *,
    max_rel_change: Optional[float] = None,
) -> dict:
    """Test-time training(TTT) 전후 모델 가중치 L2 norm 상대 변화를 검사.

    BATT-H-16: TTT 후 가중치가 너무 크게 변하면 catastrophic forgetting 발생.
    기준: ||θ_after - θ_before||_2 / ||θ_before||_2 < 0.05 (5%).

    Args:
        weights_before: TTT 전 가중치 벡터 (1D 또는 임의 shape → 자동 flatten).
        weights_after: TTT 후 가중치 벡터 (같은 shape).
        max_rel_change: 허용 최대 상대 변화 (기본: lookup yaml 0.05).

    Returns:
        {
            "rel_change": float,
            "l2_diff": float,
            "l2_before": float,
            "max_rel_change": float,
            "pass": bool,
        }

    Raises:
        BattGuardViolation: rel_change >= max_rel_change인 경우.
        ValueError: 입력 배열 shape 불일치 또는 before norm이 0.
    """
    lut = _load_lookup()["batt_h_16_ttt_stability"]
    threshold = (
        max_rel_change if max_rel_change is not None else lut["max_rel_change"]
    )

    w_before = np.asarray(weights_before, dtype=float).ravel()
    w_after = np.asarray(weights_after, dtype=float).ravel()

    if w_before.shape != w_after.shape:
        raise ValueError(
            f"weights_before shape {w_before.shape}와 "
            f"weights_after shape {w_after.shape}가 다릅니다."
        )

    l2_before = float(np.linalg.norm(w_before))
    if l2_before < 1e-12:
        raise ValueError(
            f"||θ_before||_2 = {l2_before:.2e} ≈ 0. "
            "상대 변화 계산 불가 — 초기화되지 않은 가중치 입력 의심."
        )

    l2_diff = float(np.linalg.norm(w_after - w_before))
    rel_change = l2_diff / l2_before

    result: dict = {
        "rel_change": rel_change,
        "l2_diff": l2_diff,
        "l2_before": l2_before,
        "max_rel_change": threshold,
        "pass": rel_change < threshold,
    }

    if not result["pass"]:
        raise BattGuardViolation(
            "BATT-H-16",
            f"TTT 가중치 상대 변화 {rel_change:.4f} >= 임계값 {threshold:.4f}. "
            f"||θ_after-θ_before||_2={l2_diff:.4f}, ||θ_before||_2={l2_before:.4f}. "
            "Catastrophic forgetting 의심 — TTT lr 감소 또는 EWC/LoRA 적용 권장.",
        )

    return result


# ---------------------------------------------------------------------------
# run_all_guards: 전체 가드 일괄 실행
# ---------------------------------------------------------------------------


def run_all_guards(context: dict) -> dict:
    """가능한 가드를 모두 실행하고 통과 결과 dict를 반환.

    context 키 목록:
        - phase0_c1 (float): BATT-H-01 1C 용량
        - phase0_c4 (float): BATT-H-01 4C 용량
        - theta_samples (np.ndarray): BATT-H-02, shape (N,3)
        - grid_bounds (dict): BATT-H-03
        - real_theta_samples (np.ndarray): BATT-H-03, shape (N,3)
        - hybrid_c1 (float): BATT-H-04
        - hybrid_c4 (float): BATT-H-04
        - slide_content (str): BATT-H-05
        - c_rate (float): BATT-H-06
        - current_A (float): BATT-H-06
        - hybrid_pv (dict): BATT-H-07
        - tri_cycle_life (np.ndarray): BATT-H-08
        - hust_cycle_life (np.ndarray): BATT-H-08
        - train_cells (set): BATT-H-09
        - eval_cells (set): BATT-H-09
        - domain_shift_a (np.ndarray): BATT-H-10 배열 A
        - domain_shift_b (np.ndarray): BATT-H-10 배열 B
        - theta_query (np.ndarray): BATT-H-11 쿼리 θ
        - theta_training (np.ndarray): BATT-H-11 학습 θ
        - best_epoch (int): BATT-H-12
        - total_epochs (int): BATT-H-12
        - shap_matrix_runs (list[np.ndarray]): BATT-H-13
        - mmd_source (np.ndarray): BATT-H-14 소스 feature
        - mmd_target (np.ndarray): BATT-H-14 타겟 feature
        - cf_theta_deltas (np.ndarray): BATT-H-15 θ 변화량
        - cf_eol_deltas (np.ndarray): BATT-H-15 EOL 변화량
        - ttt_weights_before (np.ndarray): BATT-H-16 TTT 전 가중치
        - ttt_weights_after (np.ndarray): BATT-H-16 TTT 후 가중치

    Returns:
        {'BATT-H-01': 'pass'|'fail'|'skip', ..., 'BATT-H-16': ...,
         'summary': {'pass': N, 'fail': N, 'skip': N}}
    """
    results: dict = {}

    def _run(guard_id: str, fn, *args, **kwargs) -> str:
        try:
            fn(*args, **kwargs)
            return "pass"
        except BattGuardViolation:
            return "fail"
        except Exception:
            return "skip"

    # BATT-H-01
    if "phase0_c1" in context and "phase0_c4" in context:
        results["BATT-H-01"] = _run(
            "BATT-H-01",
            check_phase0_capacity,
            context["phase0_c1"],
            context["phase0_c4"],
        )
    else:
        results["BATT-H-01"] = "skip"

    # BATT-H-02
    if "theta_samples" in context:
        results["BATT-H-02"] = _run(
            "BATT-H-02",
            check_phase1_identifiability,
            context["theta_samples"],
        )
    else:
        results["BATT-H-02"] = "skip"

    # BATT-H-03
    if "grid_bounds" in context and "real_theta_samples" in context:
        results["BATT-H-03"] = _run(
            "BATT-H-03",
            check_grid_covers_real_theta,
            context["grid_bounds"],
            context["real_theta_samples"],
        )
    else:
        results["BATT-H-03"] = "skip"

    # BATT-H-04
    if "hybrid_c1" in context and "hybrid_c4" in context:
        results["BATT-H-04"] = _run(
            "BATT-H-04",
            check_hybrid_pv_validation,
            context["hybrid_c1"],
            context["hybrid_c4"],
        )
    else:
        results["BATT-H-04"] = "skip"

    # BATT-H-05
    if "slide_content" in context:
        results["BATT-H-05"] = _run(
            "BATT-H-05",
            check_knee_attribution,
            context["slide_content"],
        )
    else:
        results["BATT-H-05"] = "skip"

    # BATT-H-06
    if "c_rate" in context and "current_A" in context:
        results["BATT-H-06"] = _run(
            "BATT-H-06",
            check_c_rate_to_current,
            "A123_APR18650M1A",
            context["c_rate"],
            context["current_A"],
        )
    else:
        results["BATT-H-06"] = "skip"

    # BATT-H-07
    if "hybrid_pv" in context:
        results["BATT-H-07"] = _run(
            "BATT-H-07",
            check_sei_params_present,
            context["hybrid_pv"],
        )
    else:
        results["BATT-H-07"] = "skip"

    # BATT-H-08
    if "tri_cycle_life" in context and "hust_cycle_life" in context:
        results["BATT-H-08"] = _run(
            "BATT-H-08",
            check_dataset_compatibility,
            context["tri_cycle_life"],
            context["hust_cycle_life"],
            warn_only=True,
        )
    else:
        results["BATT-H-08"] = "skip"

    # BATT-H-09
    if "train_cells" in context and "eval_cells" in context:
        results["BATT-H-09"] = _run(
            "BATT-H-09",
            check_train_eval_cell_split,
            context["train_cells"],
            context["eval_cells"],
        )
    else:
        results["BATT-H-09"] = "skip"

    # BATT-H-10
    if "domain_shift_a" in context and "domain_shift_b" in context:
        results["BATT-H-10"] = _run(
            "BATT-H-10",
            check_domain_shift,
            context["domain_shift_a"],
            context["domain_shift_b"],
            warn_only=True,
        )
    else:
        results["BATT-H-10"] = "skip"

    # BATT-H-11
    if "theta_query" in context and "theta_training" in context:
        results["BATT-H-11"] = _run(
            "BATT-H-11",
            check_theta_in_training_hull,
            context["theta_query"],
            context["theta_training"],
        )
    else:
        results["BATT-H-11"] = "skip"

    # BATT-H-12
    if "best_epoch" in context and "total_epochs" in context:
        results["BATT-H-12"] = _run(
            "BATT-H-12",
            check_best_epoch_used,
            context["best_epoch"],
            context["total_epochs"],
            warn_only=True,
        )
    else:
        results["BATT-H-12"] = "skip"

    # BATT-H-13
    if "shap_matrix_runs" in context:
        results["BATT-H-13"] = _run(
            "BATT-H-13",
            check_shap_sign_consistency,
            context["shap_matrix_runs"],
        )
    else:
        results["BATT-H-13"] = "skip"

    # BATT-H-14
    if "mmd_source" in context and "mmd_target" in context:
        results["BATT-H-14"] = _run(
            "BATT-H-14",
            check_mmd_da_trigger,
            context["mmd_source"],
            context["mmd_target"],
        )
    else:
        results["BATT-H-14"] = "skip"

    # BATT-H-15
    if "cf_theta_deltas" in context and "cf_eol_deltas" in context:
        results["BATT-H-15"] = _run(
            "BATT-H-15",
            check_counterfactual_monotonicity,
            context["cf_theta_deltas"],
            context["cf_eol_deltas"],
        )
    else:
        results["BATT-H-15"] = "skip"

    # BATT-H-16
    if "ttt_weights_before" in context and "ttt_weights_after" in context:
        results["BATT-H-16"] = _run(
            "BATT-H-16",
            check_ttt_update_stability,
            context["ttt_weights_before"],
            context["ttt_weights_after"],
        )
    else:
        results["BATT-H-16"] = "skip"

    # 요약
    counts = {"pass": 0, "fail": 0, "skip": 0}
    for v in results.values():
        counts[v] = counts.get(v, 0) + 1
    results["summary"] = counts

    return results
