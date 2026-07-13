"""
GMM K=5 Density Estimator — θ_3 = (SoC, ε_neg, ε_pos) valid prior 학습.

사용 예:
    python scripts/fit_theta_gmm.py
    python scripts/fit_theta_gmm.py --K 5 --input-dir data/fixed_d_labels --output-dir results/density_estimator

출력:
    results/density_estimator/gmm_k5.pkl
    results/density_estimator/gmm_evaluation.json
    docs/figures/gmm_k5_prototype.png
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

# ──────────────────────────────────────────────
# Korean font registration
# ──────────────────────────────────────────────
_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
for _fp in _FONT_PATHS:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
THETA_KEYS = ["SoC", "Neg_AM_vol_frac", "Pos_AM_vol_frac"]
THETA_LABELS = ["SoC", "ε_neg", "ε_pos"]

# Physical bounds for OOD synthesis
THETA_BOUNDS = {
    "SoC": (0.5, 1.0),
    "Neg_AM_vol_frac": (0.3, 0.7),
    "Pos_AM_vol_frac": (0.3, 0.7),
}

RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════
def load_theta(input_dir: Path) -> tuple[np.ndarray, list[str]]:
    """JSON 파일에서 θ_3 추출 → (N, 3) array + 파일명 목록."""
    rows: list[list[float]] = []
    filenames: list[str] = []

    json_files = sorted(input_dir.glob("batch1_cell*_cycle*.json"))
    if not json_files:
        raise FileNotFoundError(f"JSON 파일 없음: {input_dir}")

    skipped = 0
    for fp in json_files:
        with open(fp) as f:
            d = json.load(f)
        params = d.get("params", {})
        try:
            row = [float(params[k]) for k in THETA_KEYS]
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        # NaN / Inf 검사
        if any(not np.isfinite(v) for v in row):
            skipped += 1
            continue
        rows.append(row)
        filenames.append(fp.name)

    theta = np.array(rows, dtype=np.float64)  # (N, 3)
    print(f"[데이터] 로드 완료: {len(theta)}개 (제외 {skipped}개)")
    return theta, filenames


# ══════════════════════════════════════════════
# 2. GMM 학습
# ══════════════════════════════════════════════
def fit_gmm(
    X_scaled: np.ndarray,
    K: int = 5,
    random_state: int = 42,
) -> GaussianMixture:
    """StandardScaler 적용 후 데이터에 GMM 학습."""
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=random_state,
        max_iter=500,
        n_init=10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm.fit(X_scaled)
    ll = gmm.score(X_scaled)  # 평균 log-likelihood
    print(f"[GMM K={K}] log-likelihood={ll:.4f}  BIC={gmm.bic(X_scaled):.2f}  AIC={gmm.aic(X_scaled):.2f}")
    return gmm


# ══════════════════════════════════════════════
# 3. K ablation (BIC/AIC 비교)
# ══════════════════════════════════════════════
def k_ablation(
    X_scaled: np.ndarray,
    k_list: Optional[list[int]] = None,
) -> dict[int, dict[str, float]]:
    """여러 K 값에 대해 GMM 학습 후 BIC/AIC/LL 비교."""
    if k_list is None:
        k_list = [1, 3, 5, 7, 10]

    results: dict[int, dict[str, float]] = {}
    for K in k_list:
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            random_state=42,
            max_iter=500,
            n_init=10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(X_scaled)
        results[K] = {
            "bic": gmm.bic(X_scaled),
            "aic": gmm.aic(X_scaled),
            "log_likelihood": gmm.score(X_scaled),
        }
    return results


# ══════════════════════════════════════════════
# 4. OOD 샘플 생성
# ══════════════════════════════════════════════
def generate_ood_samples(
    theta_valid: np.ndarray,
    n_uniform: int = 100,
    n_shuffled: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    invalid θ 3종 생성.
    (a) 범위 밖 극단값 — hardcoded 물리 위반 파라미터
    (b) uniform random (100개)
    (c) shuffled joint (100개)
    반환: (invalid_array, labels) — labels는 모두 0
    """
    # (a) 범위 밖 극단값 (4개)
    ood_extreme = np.array([
        [0.5,  0.05, 0.55],   # ε_neg 너무 낮음
        [0.5,  0.80, 0.55],   # ε_neg 너무 높음
        [0.3,  0.55, 0.20],   # SoC+ε_pos 낮음
        [1.05, 0.55, 0.55],   # SoC > 1 물리 위반
    ], dtype=np.float64)

    # (b) uniform random
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.5, 1.0, 1.0])
    ood_uniform = RNG.uniform(lo, hi, (n_uniform, 3))

    # (c) shuffled joint — 각 차원 독립 셔플
    ood_shuffled = theta_valid.copy()
    for dim in range(ood_shuffled.shape[1]):
        RNG.shuffle(ood_shuffled[:, dim])
    ood_shuffled = ood_shuffled[:n_shuffled]

    invalid = np.vstack([ood_extreme, ood_uniform, ood_shuffled])
    return invalid


# ══════════════════════════════════════════════
# 5. ROC + Coverage 평가
# ══════════════════════════════════════════════
def evaluate_roc_coverage(
    gmm: GaussianMixture,
    scaler: StandardScaler,
    theta_valid: np.ndarray,
    theta_invalid: np.ndarray,
    coverage_pct: float = 0.90,
) -> dict:
    """
    ROC AUC, coverage 계산.
    valid=1 / invalid=0 로 레이블링.
    threshold τ는 valid 분포의 (1 - coverage_pct) 분위수로 설정.
    """
    # log-likelihood 계산
    ll_valid   = gmm.score_samples(scaler.transform(theta_valid))
    ll_invalid = gmm.score_samples(scaler.transform(theta_invalid))

    # ROC
    all_ll     = np.concatenate([ll_valid, ll_invalid])
    all_labels = np.concatenate([
        np.ones(len(ll_valid), dtype=int),
        np.zeros(len(ll_invalid), dtype=int),
    ])
    auc = roc_auc_score(all_labels, all_ll)
    fpr, tpr, thresholds = roc_curve(all_labels, all_ll)

    # Threshold: valid 분포의 10th percentile
    tau = float(np.percentile(ll_valid, (1.0 - coverage_pct) * 100))
    coverage = float(np.mean(ll_valid >= tau))

    print(f"[평가] AUC={auc:.4f}  coverage@τ={coverage:.3f}  τ={tau:.4f}")
    return {
        "auc": float(auc),
        "coverage": coverage,
        "tau": tau,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "ll_valid": ll_valid.tolist(),
        "ll_invalid": ll_invalid.tolist(),
    }


# ══════════════════════════════════════════════
# 6. 시각화 (3-panel figure)
# ══════════════════════════════════════════════
def _draw_ellipse(ax, mean: np.ndarray, cov: np.ndarray, color: str, alpha: float = 0.25, n_std: float = 2.0) -> None:
    """2D 공분산 행렬로부터 ellipse 그리기."""
    from matplotlib.patches import Ellipse
    import scipy.linalg

    eigenvalues, eigenvectors = scipy.linalg.eigh(cov)
    # 가장 큰 고유값 기준 각도
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    width, height = 2 * n_std * np.sqrt(np.abs(eigenvalues))
    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        edgecolor=color, facecolor=color, alpha=alpha, linewidth=1.5,
    )
    ax.add_patch(ellipse)


def make_figure(
    theta_valid: np.ndarray,
    gmm: GaussianMixture,
    scaler: StandardScaler,
    eval_result: dict,
    ablation: dict[int, dict[str, float]],
    out_path: Path,
) -> None:
    """3-panel figure 생성."""
    COLORS = plt.cm.tab10(np.linspace(0, 0.9, 5))

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("GMM K=5 Prior Prototype — theta_3 = (SoC, ep_neg, ep_pos)", fontsize=14, fontweight="bold", y=1.01)

    # ── Panel (a): 2D scatter (SoC vs ε_neg) + cluster centers + ellipses ──
    ax_a = fig.add_subplot(1, 3, 1)
    ax_a.scatter(theta_valid[:, 0], theta_valid[:, 1],
                 s=18, alpha=0.55, color="#3a86ff", label="유효 theta (92)")

    # cluster centers in original space
    centers_orig = scaler.inverse_transform(gmm.means_)
    for k in range(gmm.n_components):
        # covariance를 원본 스케일로 변환: Σ_orig = S @ Σ_scaled @ S^T
        S = np.diag(scaler.scale_)
        cov_orig = S @ gmm.covariances_[k] @ S.T
        # SoC vs ε_neg 2D slice (dim 0, 1)
        _draw_ellipse(ax_a,
                      centers_orig[k, [0, 1]],
                      cov_orig[np.ix_([0, 1], [0, 1])],
                      color=COLORS[k], alpha=0.20)
        ax_a.scatter(centers_orig[k, 0], centers_orig[k, 1],
                     s=120, color=COLORS[k], marker="*",
                     edgecolors="black", linewidths=0.8, zorder=5,
                     label=f"C{k+1}")

    ax_a.set_xlabel("SoC (상태충전도)", fontsize=11)
    ax_a.set_ylabel("ep_neg (음극 활물질 분율)", fontsize=11)
    ax_a.set_title("(a) theta 분포 + GMM cluster", fontsize=12)
    ax_a.legend(fontsize=8, ncol=2)

    # ── Panel (b): log-likelihood histogram ──
    ax_b = fig.add_subplot(1, 3, 2)
    ll_valid   = np.array(eval_result["ll_valid"])
    ll_invalid = np.array(eval_result["ll_invalid"])
    tau        = eval_result["tau"]

    bins = np.linspace(
        min(ll_valid.min(), ll_invalid.min()),
        max(ll_valid.max(), ll_invalid.max()),
        40,
    )
    ax_b.hist(ll_valid,   bins=bins, alpha=0.7, color="#06d6a0", label="valid (188)")
    ax_b.hist(ll_invalid, bins=bins, alpha=0.7, color="#ef233c", label="invalid (OOD)")
    ax_b.axvline(tau, color="black", linestyle="--", linewidth=1.5, label=f"tau={tau:.2f}")
    ax_b.set_xlabel("log p(theta)", fontsize=11)
    ax_b.set_ylabel("빈도", fontsize=11)
    ax_b.set_title("(b) log-likelihood 분포", fontsize=12)
    ax_b.legend(fontsize=9)

    # ── Panel (c): ROC curve ──
    ax_c = fig.add_subplot(1, 3, 3)
    fpr = np.array(eval_result["fpr"])
    tpr = np.array(eval_result["tpr"])
    auc = eval_result["auc"]
    coverage = eval_result["coverage"]

    ax_c.plot(fpr, tpr, color="#7209b7", linewidth=2, label=f"AUC={auc:.3f}")
    ax_c.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)

    # τ 지점 표시
    thresholds = np.array(eval_result["thresholds"])
    tau_idx = np.argmin(np.abs(thresholds - tau))
    if tau_idx < len(fpr):
        ax_c.scatter(fpr[tau_idx], tpr[tau_idx], s=120, color="red",
                     zorder=6, label=f"tau 지점 (cov={coverage:.2f})")

    ax_c.set_xlabel("FPR", fontsize=11)
    ax_c.set_ylabel("TPR", fontsize=11)
    ax_c.set_title("(c) ROC curve + threshold", fontsize=12)
    ax_c.legend(fontsize=9)
    ax_c.set_xlim([-0.02, 1.02])
    ax_c.set_ylim([-0.02, 1.02])

    # BIC ablation 부제목
    bic_str = "  BIC: " + "  ".join(f"K{k}={v['bic']:.0f}" for k, v in sorted(ablation.items()))
    fig.text(0.5, -0.03, bic_str, ha="center", fontsize=9, color="#555555")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[그림] 저장: {out_path}")


# ══════════════════════════════════════════════
# 7. BIC elbow figure (별도 저장 생략, 콘솔 출력)
# ══════════════════════════════════════════════
def report_bic_elbow(ablation: dict[int, dict[str, float]]) -> int:
    """BIC 최소 K 반환 + 콘솔 출력."""
    ks  = sorted(ablation.keys())
    bics = [ablation[k]["bic"] for k in ks]
    best_k = ks[int(np.argmin(bics))]
    print("\n[BIC ablation]")
    print(f"{'K':>5}  {'BIC':>12}  {'AIC':>12}  {'LL':>10}")
    for k in ks:
        v = ablation[k]
        print(f"{k:>5}  {v['bic']:>12.2f}  {v['aic']:>12.2f}  {v['log_likelihood']:>10.4f}")
    print(f"→ BIC elbow / 최솟값: K={best_k}")
    return best_k


# ══════════════════════════════════════════════
# main
# ══════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="GMM θ_3 density estimator 학습 + 평가")
    parser.add_argument("--K", type=int, default=5, help="GMM components (default: 5)")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/fixed_d_labels",
        help="JSON 파일 디렉토리 (PROJECT_ROOT 기준 상대경로 또는 절대경로)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/density_estimator",
        help="출력 디렉토리",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir) if Path(args.input_dir).is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 데이터 로드 ──
    theta_valid, filenames = load_theta(input_dir)
    N = len(theta_valid)
    print(f"θ 범위: SoC=[{theta_valid[:,0].min():.3f}, {theta_valid[:,0].max():.3f}]  "
          f"ε_neg=[{theta_valid[:,1].min():.3f}, {theta_valid[:,1].max():.3f}]  "
          f"ε_pos=[{theta_valid[:,2].min():.3f}, {theta_valid[:,2].max():.3f}]")

    # ── 2. StandardScaler ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(theta_valid)

    # ── 3. K ablation ──
    K_LIST = [1, 3, 5, 7, 10]
    print("\n[K ablation 실행 중...]")
    ablation = k_ablation(X_scaled, k_list=K_LIST)
    best_k = report_bic_elbow(ablation)

    # ── 4. GMM K=5 학습 ──
    K = args.K
    print(f"\n[메인 GMM K={K} 학습]")
    gmm = fit_gmm(X_scaled, K=K)

    # ── 5. OOD 생성 + ROC/Coverage 평가 ──
    theta_invalid = generate_ood_samples(theta_valid)
    print(f"[OOD] invalid 샘플: {len(theta_invalid)}개")
    eval_result = evaluate_roc_coverage(gmm, scaler, theta_valid, theta_invalid)

    # ── 6. 시각화 ──
    fig_path = PROJECT_ROOT / "docs" / "figures" / "gmm_k5_prototype.png"
    make_figure(theta_valid, gmm, scaler, eval_result, ablation, out_path=fig_path)

    # ── 7. 모델 저장 ──
    model_path = output_dir / "gmm_k5.pkl"
    joblib.dump({"gmm": gmm, "scaler": scaler}, model_path)
    print(f"[저장] 모델: {model_path}")

    # evaluation JSON (fpr/tpr/thresholds 제외 — 크기 절약)
    eval_json = {
        "K": K,
        "N_valid": N,
        "N_invalid": len(theta_invalid),
        "auc": eval_result["auc"],
        "coverage": eval_result["coverage"],
        "tau": eval_result["tau"],
        "log_likelihood": float(gmm.score(X_scaled)),
        "bic_k5": float(gmm.bic(X_scaled)),
        "aic_k5": float(gmm.aic(X_scaled)),
        "bic_elbow_K": best_k,
        "ablation": {str(k): v for k, v in ablation.items()},
        "gmm_means_orig": scaler.inverse_transform(gmm.means_).tolist(),
        "gmm_weights": gmm.weights_.tolist(),
    }
    json_path = output_dir / "gmm_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(eval_json, f, indent=2)
    print(f"[저장] 평가 결과: {json_path}")

    print("\n─── 최종 요약 ───")
    print(f"  AUC       : {eval_result['auc']:.4f}  (목표 ≥ 0.85)")
    print(f"  Coverage  : {eval_result['coverage']:.3f}  (목표 0.80~0.95)")
    print(f"  BIC elbow : K={best_k}")
    print(f"  Figure    : {fig_path}")


if __name__ == "__main__":
    main()
