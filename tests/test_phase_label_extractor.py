from src.phase_label_extractor import summarize_dataset


REQUIRED_LABEL_KEYS = {
    "dataset",
    "cell_key",
    "eol_cycle",
    "kp_cycle",
    "kp_over_eol",
    "early_slope",
    "plateau_duration",
    "post_kp_slope",
    "noise_estimate",
    "detection_status",
    "failure_reasons",
    "ambiguous_reasons",
}


def test_phase_label_schema_and_summary_keep_ambiguous_records():
    labels = [
        {
            "dataset": "TRI",
            "cell_key": "cell_success",
            "eol_cycle": 100,
            "kp_cycle": 60,
            "kp_over_eol": 0.6,
            "early_slope": -0.001,
            "plateau_duration": 59,
            "post_kp_slope": -0.003,
            "noise_estimate": 0.0001,
            "detection_status": "success",
            "failure_reasons": [],
            "ambiguous_reasons": [],
        },
        {
            "dataset": "TRI",
            "cell_key": "cell_ambiguous",
            "eol_cycle": 120,
            "kp_cycle": 72,
            "kp_over_eol": 0.6,
            "early_slope": -0.0012,
            "plateau_duration": 71,
            "post_kp_slope": -0.0035,
            "noise_estimate": 0.0002,
            "detection_status": "ambiguous",
            "failure_reasons": [],
            "ambiguous_reasons": ["KP shifts by >10% EOL under search-range sensitivity"],
        },
    ]

    for label in labels:
        assert REQUIRED_LABEL_KEYS.issubset(label)

    summary = summarize_dataset("TRI", labels)
    assert summary["n_cells"] == 2
    assert summary["counts"] == {"success": 1, "ambiguous": 1}
    assert summary["kp_over_eol"]["n"] == 2
    assert summary["plateau_duration"]["median"] == 65.0
