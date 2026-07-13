import math

from src.eval_protocol import build_split_manifest, mape, rmse, rul_error


def test_v7_split_manifest_has_required_modes_and_disjoint_splits():
    manifest = build_split_manifest(seed=42)
    split_names = {split["name"] for split in manifest["splits"]}
    assert {
        "within_tri_severson",
        "within_hust_ma",
        "pooled_tri_hust",
        "tri_to_hust",
        "hust_to_tri",
        "leave_tri_out",
        "leave_hust_out",
        "calce_sanity",
    }.issubset(split_names)
    assert manifest["batt_h09_all_pass"] is True
    assert all(split["disjoint"] for split in manifest["splits"])
    assert manifest["datasets"]["calce_a123"]["n_cells"] >= 1


def test_calce_is_sanity_only():
    manifest = build_split_manifest(seed=42)
    calce = next(split for split in manifest["splits"] if split["name"] == "calce_sanity")
    assert calce["metric_scope"] == "sanity_only"
    assert calce["mode"] == "external_sanity_check"


def test_metric_helpers():
    assert mape([100, 200], [90, 220]) == 10.0
    assert rmse([1, 3], [1, 5]) == math.sqrt(2.0)
    assert rul_error([100, 200], [110, 190], observation_cycle=50) == [10.0, -10.0]
