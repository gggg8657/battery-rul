from src.uncertainty_layer import build_uncertainty_bundle, safe_corr


def test_safe_corr_returns_value_for_monotone_inputs():
    assert safe_corr([1, 2, 3], [2, 4, 6]) == 1.0


def test_uncertainty_bundle_schema():
    records = [
        {"dataset_shift_score": 0.0, "q_abs_error_pct": 1.0},
        {"dataset_shift_score": 1.0, "q_abs_error_pct": 5.0},
        {"dataset_shift_score": 1.0, "q_abs_error_pct": 6.0},
    ]
    bundle = build_uncertainty_bundle(records, unavailable={"ensemble_variance": "missing"})
    assert bundle["n_records"] == 3
    assert "dataset_shift_score" in bundle["signals"]
    assert bundle["signals"]["ensemble_variance"]["status"] == "not_available"
