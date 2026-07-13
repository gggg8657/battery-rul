"""
Toyota TRI Battery Cycling Dataset Downloader + Loader

Severson et al. 2019 (Nature Energy)
124 LFP/graphite cells, fast-charging, 150~2300 cycles

Data source: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
GitHub: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
"""
import pickle
import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MAT_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

DOWNLOAD_URLS = {
    "batch1": "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
    "batch2": "https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",
    "batch3": "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
}


def load_batch_from_mat(mat_path: Path, batch_name: str) -> dict:
    """
    HDF5 형식의 .mat 파일에서 배터리 사이클 데이터를 파싱.

    Returns dict of {cell_key: {cycle_life, charge_policy, summary, cycles}}
    """
    logger.info(f"Loading {batch_name} from {mat_path}")
    f = h5py.File(mat_path, "r")
    batch_struct = f["batch"]

    num_cells = batch_struct["summary"].shape[0]
    batch_dict = {}

    for i in range(num_cells):
        cell_key = f"{batch_name}_cell{i}"
        try:
            cl = f[batch_struct["cycle_life"][i, 0]][()].item()
            policy = "".join(
                chr(c) for c in f[batch_struct["policy_readable"][i, 0]][()].flatten()
            )

            # Summary data (per-cycle aggregates)
            # .mat file uses QDischarge/QCharge; we map to QD/QC for consistency
            summary_key_map = {
                "IR": "IR",
                "QC": "QCharge",
                "QD": "QDischarge",
                "Tavg": "Tavg",
                "Tmin": "Tmin",
                "Tmax": "Tmax",
                "chargetime": "chargetime",
                "cycle": "cycle",
            }
            summary = {}
            ref = batch_struct["summary"][i, 0]
            summary_obj = f[ref]
            for our_key, mat_key in summary_key_map.items():
                try:
                    data = summary_obj[mat_key][()].flatten()
                    summary[our_key] = data
                except Exception:
                    summary[our_key] = np.array([])

            # Cycle-level data (within-cycle measurements)
            cycles_struct = f[batch_struct["cycles"][i, 0]]
            num_cycles = cycles_struct["I"].shape[0]
            cycles = {}
            base_vars = ["I", "V", "t", "Qc", "Qd", "T"]
            derived_vars = ["Qdlin", "Tdlin", "discharge_dQdV"]
            for j in range(num_cycles):
                cycle_data = {}
                for var in base_vars + derived_vars:
                    try:
                        ref = cycles_struct[var][j, 0]
                        cycle_data[var] = f[ref][()].flatten()
                    except Exception:
                        pass
                cycles[str(j)] = cycle_data

            batch_dict[cell_key] = {
                "cycle_life": cl,
                "charge_policy": policy,
                "summary": summary,
                "cycles": cycles,
            }
        except Exception as e:
            logger.warning(f"Skipping {cell_key}: {e}")

    f.close()
    logger.info(f"{batch_name}: loaded {len(batch_dict)} cells")
    return batch_dict


def load_batch_from_pkl(pkl_path: Path, batch_name: str) -> dict:
    """pickle 파일에서 배치 데이터 로드."""
    logger.info(f"Loading {batch_name} from {pkl_path}")
    with open(pkl_path, "rb") as fp:
        batch_dict = pickle.load(fp)
    logger.info(f"{batch_name}: loaded {len(batch_dict)} cells")
    return batch_dict


def load_batch(batch_name: str, data_dir: Optional[Path] = None) -> dict:
    """
    .pkl 우선, 없으면 .mat 파일에서 로드.
    """
    data_dir = data_dir or RAW_DIR

    pkl_path = data_dir / f"{batch_name}.pkl"
    if pkl_path.exists():
        return load_batch_from_pkl(pkl_path, batch_name)

    mat_filename = MAT_FILES.get(batch_name)
    if mat_filename:
        mat_path = data_dir / mat_filename
        if mat_path.exists():
            batch_dict = load_batch_from_mat(mat_path, batch_name)
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pkl_path, "wb") as fp:
                pickle.dump(batch_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Cached parsed data to {pkl_path}")
            return batch_dict

    raise FileNotFoundError(
        f"No data found for {batch_name}.\n"
        f"Expected .pkl at: {pkl_path}\n"
        f"Or .mat at: {data_dir / (mat_filename or '???')}\n\n"
        f"Download from: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204\n"
        f"Place files in: {data_dir}/"
    )


def load_all_batches(data_dir: Optional[Path] = None) -> dict:
    """batch1, batch2, batch3를 모두 로드하여 하나의 dict로 합침."""
    all_cells = {}
    for batch_name in ["batch1", "batch2", "batch3"]:
        try:
            batch = load_batch(batch_name, data_dir)
            all_cells.update(batch)
        except FileNotFoundError as e:
            logger.warning(str(e))
    logger.info(f"Total cells loaded: {len(all_cells)}")
    return all_cells


def summarize_dataset(cells: dict) -> dict:
    """데이터셋 통계 요약."""
    cycle_lives = [c["cycle_life"] for c in cells.values()]
    info = {
        "total_cells": len(cells),
        "cycle_life_min": int(min(cycle_lives)) if cycle_lives else 0,
        "cycle_life_max": int(max(cycle_lives)) if cycle_lives else 0,
        "cycle_life_mean": float(np.mean(cycle_lives)) if cycle_lives else 0,
        "cycle_life_median": float(np.median(cycle_lives)) if cycle_lives else 0,
        "policies": list(set(c["charge_policy"] for c in cells.values())),
    }
    return info


def print_download_instructions():
    """수동 다운로드 안내 출력."""
    instructions = """
╔══════════════════════════════════════════════════════════════╗
║  Toyota TRI Battery Dataset Download Instructions           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Visit: https://data.matr.io/1/                          ║
║                                                              ║
║  2. Search for project:                                      ║
║     "Data-driven prediction of battery cycle life"           ║
║                                                              ║
║  3. Download the following .mat files:                       ║
║     - 2017-05-12_batchdata_updated_struct_errorcorrect.mat   ║
║     - 2017-06-30_batchdata_updated_struct_errorcorrect.mat   ║
║     - 2018-04-12_batchdata_updated_struct_errorcorrect.mat   ║
║                                                              ║
║  4. Place them in:                                           ║
║     PyBAMM_Inverse/data/raw/                                 ║
║                                                              ║
║  Alternative: Download pre-built pickle files if available   ║
║  (batch1.pkl, batch2.pkl, batch3.pkl)                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(instructions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Toyota TRI battery dataset")
    parser.add_argument("--batch", type=str, default=None, help="Specific batch (batch1/2/3)")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--info", action="store_true", help="Print download instructions")
    args = parser.parse_args()

    if args.info:
        print_download_instructions()
        exit(0)

    data_dir = Path(args.data_dir) if args.data_dir else RAW_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.batch:
            cells = load_batch(args.batch, data_dir)
        else:
            cells = load_all_batches(data_dir)

        info = summarize_dataset(cells)
        print(json.dumps(info, indent=2, ensure_ascii=False))

    except FileNotFoundError:
        print_download_instructions()
