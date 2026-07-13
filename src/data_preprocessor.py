"""
Data Preprocessor: Toyota TRI → PyBOP Dataset format

Converts raw TRI cycle data (from .mat/.pkl) into the format
required by pybop.Dataset for FittingProblem.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pybop
except ImportError:
    pybop = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def extract_cycle(cell_data: dict, cycle_num: int = 10) -> dict:
    """
    단일 셀의 특정 사이클에서 (time, voltage, current) 추출.

    Args:
        cell_data: load_batch()에서 반환된 단일 셀 dict
        cycle_num: 추출할 사이클 번호

    Returns:
        dict with keys: time_s, voltage_V, current_A
    """
    cycle_key = str(cycle_num)
    if cycle_key not in cell_data["cycles"]:
        available = sorted(int(k) for k in cell_data["cycles"].keys() if k.isdigit())
        closest = min(available, key=lambda x: abs(x - cycle_num))
        logger.warning(f"Cycle {cycle_num} not found, using cycle {closest}")
        cycle_key = str(closest)

    cycle = cell_data["cycles"][cycle_key]
    t_raw = cycle["t"].astype(np.float64)
    V = cycle["V"].astype(np.float64)
    I = cycle["I"].astype(np.float64)

    # TRI data stores time in MINUTES; convert to SECONDS for PyBOP
    t_seconds = t_raw * 60.0
    t_offset = t_seconds - t_seconds[0]

    mask = np.isfinite(t_offset) & np.isfinite(V) & np.isfinite(I)
    return {
        "time_s": t_offset[mask],
        "voltage_V": V[mask],
        "current_A": I[mask],
    }


def extract_discharge_segment(cycle_data: dict, negate_current: bool = True) -> dict:
    """
    사이클 데이터에서 방전 구간만 추출.
    TRI 데이터는 방전 시 I < 0 (물리 관례),
    PyBAMM/PyBOP는 방전 시 I > 0을 기대하므로 부호 반전 필요.

    Args:
        negate_current: True면 전류 부호를 반전 (TRI→PyBAMM 관례)
    """
    I = cycle_data["current_A"]
    discharge_mask = I < -0.01

    if discharge_mask.sum() < 10:
        logger.warning("Discharge segment too short, using full cycle")
        return cycle_data

    first_idx = np.argmax(discharge_mask)
    last_idx = len(discharge_mask) - 1 - np.argmax(discharge_mask[::-1])

    segment = {
        "time_s": cycle_data["time_s"][first_idx : last_idx + 1],
        "voltage_V": cycle_data["voltage_V"][first_idx : last_idx + 1],
        "current_A": cycle_data["current_A"][first_idx : last_idx + 1],
    }

    if negate_current:
        segment["current_A"] = -segment["current_A"]
        logger.info("Current sign negated: TRI (neg=discharge) → PyBAMM (pos=discharge)")

    return segment


def trim_initial_ramp(
    cycle_data: dict,
    threshold_fraction: float = 0.80,
    margin_points: int = 2,
) -> dict:
    """
    전류 ramp-up 구간 제거. 안정적인 CC 구간부터 시작하도록 트리밍.
    TRI 실험에서 방전 시작 시 전류가 점진적으로 올라가는 과도 구간 존재.
    """
    I = np.abs(cycle_data["current_A"])
    if len(I) < 10:
        return cycle_data

    target = threshold_fraction * np.max(I)
    idx = int(np.argmax(I >= target))
    if I[idx] < target:
        return cycle_data

    start = max(idx - margin_points, 0)
    trimmed = {k: v[start:] for k, v in cycle_data.items()}
    trimmed["time_s"] = trimmed["time_s"] - trimmed["time_s"][0]
    logger.info(f"Trimmed initial ramp: removed {start} points, {len(trimmed['time_s'])} remaining")
    return trimmed


def downsample(cycle_data: dict, max_points: int = 500) -> dict:
    """시간 균등 다운샘플링. PyBOP solver가 너무 많은 포인트에서 느릴 수 있음."""
    n = len(cycle_data["time_s"])
    if n <= max_points:
        return cycle_data

    indices = np.linspace(0, n - 1, max_points, dtype=int)
    return {k: v[indices] for k, v in cycle_data.items()}


def to_pybop_dataset(cycle_data: dict):
    """
    dict → pybop.Dataset 변환.
    PyBOP는 {"Time [s]": ..., "Current [A]": ..., "Voltage [V]": ...} 형태 필요.
    """
    if pybop is None:
        raise ImportError("pybop is not installed")

    return pybop.Dataset(
        {
            "Time [s]": cycle_data["time_s"],
            "Current [A]": cycle_data["current_A"],
            "Voltage [V]": cycle_data["voltage_V"],
        }
    )


def to_dataframe(cycle_data: dict) -> pd.DataFrame:
    """dict → pandas DataFrame 변환."""
    return pd.DataFrame(cycle_data)


def prepare_fitting_data(
    cell_data: dict,
    cycle_num: int = 10,
    discharge_only: bool = True,
    max_points: int = 500,
    save_path: Optional[Path] = None,
) -> dict:
    """
    전체 전처리 파이프라인: 셀 데이터 → 피팅 준비 데이터.

    Steps:
        1. 지정 사이클 추출
        2. (선택) 방전 구간만 분리
        3. 다운샘플링
        4. (선택) CSV 저장
    """
    cycle_data = extract_cycle(cell_data, cycle_num)
    logger.info(
        f"Extracted cycle {cycle_num}: {len(cycle_data['time_s'])} points, "
        f"V=[{cycle_data['voltage_V'].min():.3f}, {cycle_data['voltage_V'].max():.3f}]"
    )

    if discharge_only:
        cycle_data = extract_discharge_segment(cycle_data)
        logger.info(f"Discharge segment: {len(cycle_data['time_s'])} points")

    cycle_data = downsample(cycle_data, max_points)
    logger.info(f"After downsample: {len(cycle_data['time_s'])} points")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        to_dataframe(cycle_data).to_csv(save_path, index=False)
        logger.info(f"Saved to {save_path}")

    return cycle_data


def prepare_batch_fitting_data(
    cells: dict,
    cell_keys: list,
    cycle_num: int = 10,
    discharge_only: bool = True,
    max_points: int = 500,
    output_dir: Optional[Path] = None,
) -> dict:
    """여러 셀에 대해 피팅 데이터 일괄 준비."""
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for key in cell_keys:
        if key not in cells:
            logger.warning(f"Cell {key} not found, skipping")
            continue
        try:
            save_path = output_dir / f"{key}_cycle{cycle_num}.csv"
            data = prepare_fitting_data(
                cells[key], cycle_num, discharge_only, max_points, save_path
            )
            results[key] = data
        except Exception as e:
            logger.error(f"Failed to prepare {key}: {e}")

    logger.info(f"Prepared {len(results)}/{len(cell_keys)} cells")
    return results
