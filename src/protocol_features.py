"""Protocol metadata encoder for V7 protocol-conditioned models.

The encoder deliberately keeps physical protocol features separate from dataset
identity. EOL definitions, cycle life, and RUL are evaluation metadata and are
rejected if supplied as inputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INVENTORY_PATH = PROJECT_ROOT / "data" / "protocol_inventory.json"

LEAKAGE_KEYS = {
    "cycle_life",
    "eol",
    "eol_cycle",
    "eol_definition",
    "rul",
    "remaining_useful_life",
}

NUMERIC_FEATURES = [
    "charge_c_rate_stage1",
    "charge_soc_switch_fraction",
    "charge_c_rate_stage2",
    "discharge_c_rate_stage1",
    "discharge_c_rate_stage2",
    "discharge_c_rate_stage3",
    "temperature_C",
    "depth_of_discharge_fraction",
    "nominal_capacity_Ah",
    "voltage_min_V",
    "voltage_max_V",
]

CATEGORICAL_FEATURES = {
    "charge_rule": [
        "two_step_fast_charge_cc",
        "constant_current",
        "unknown",
    ],
    "discharge_rule": [
        "constant_current",
        "multi_stage_constant_current",
        "unknown",
    ],
    "form_factor": [
        "18650",
        "26650",
        "unknown",
    ],
}

DATASET_IDS = ["tri_severson", "hust_ma", "calce_a123"]


def load_protocol_inventory(path: str | Path = DEFAULT_INVENTORY_PATH) -> dict[str, Any]:
    """Load the JSON protocol inventory."""

    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def physical_feature_names() -> list[str]:
    """Return deterministic physical feature names, including known masks."""

    names: list[str] = []
    for name in NUMERIC_FEATURES:
        names.extend([name, f"{name}__known"])
    for field, values in CATEGORICAL_FEATURES.items():
        names.extend([f"{field}__{value}" for value in values])
    return names


def dataset_identity_feature_names() -> list[str]:
    """Return deterministic dataset identity one-hot feature names."""

    return [f"dataset__{dataset_id}" for dataset_id in DATASET_IDS]


def parse_tri_charge_policy(policy: str | None) -> dict[str, float | None]:
    """Parse TRI policy strings such as ``4.4C(55%)-6C``.

    Returns None values when parsing fails; callers keep the unknown mask at 0.
    """

    if not policy:
        return {
            "charge_c_rate_stage1": None,
            "charge_soc_switch_fraction": None,
            "charge_c_rate_stage2": None,
        }
    match = re.search(r"([\d.]+)C\(([\d.]+)%?\).*?([\d.]+)C", policy)
    if not match:
        return {
            "charge_c_rate_stage1": None,
            "charge_soc_switch_fraction": None,
            "charge_c_rate_stage2": None,
        }
    return {
        "charge_c_rate_stage1": float(match.group(1)),
        "charge_soc_switch_fraction": float(match.group(2)) / 100.0,
        "charge_c_rate_stage2": float(match.group(3)),
    }


def _reject_target_leakage(record: dict[str, Any]) -> None:
    leaked = sorted(LEAKAGE_KEYS.intersection(record))
    if leaked:
        raise ValueError(
            "Protocol feature records must not include target/evaluation fields: "
            + ", ".join(leaked)
        )


def _dataset_entry(inventory: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    try:
        return inventory["datasets"][dataset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset_id: {dataset_id}") from exc


def _value_with_known(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    return float(value), 1.0


def _base_physical_values(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "charge_c_rate_stage1": entry["charge"]["c_rate_stage1"]["value"],
        "charge_soc_switch_fraction": entry["charge"]["soc_switch_fraction"]["value"],
        "charge_c_rate_stage2": entry["charge"]["c_rate_stage2"]["value"],
        "discharge_c_rate_stage1": entry["discharge"]["c_rate_stage1"]["value"],
        "discharge_c_rate_stage2": entry["discharge"]["c_rate_stage2"]["value"],
        "discharge_c_rate_stage3": entry["discharge"]["c_rate_stage3"]["value"],
        "temperature_C": entry["temperature_C"]["value"],
        "depth_of_discharge_fraction": entry["depth_of_discharge_fraction"]["value"],
        "nominal_capacity_Ah": entry["nominal_capacity_Ah"]["value"],
        "voltage_min_V": entry["voltage_window_V"]["min"],
        "voltage_max_V": entry["voltage_window_V"]["max"],
        "charge_rule": entry["charge"]["rule"],
        "discharge_rule": entry["discharge"]["rule"],
        "form_factor": entry["form_factor"],
    }


def protocol_record_from_dataset(
    dataset_id: str,
    *,
    inventory: dict[str, Any] | None = None,
    charge_policy: str | None = None,
    discharge_rates: list[float] | tuple[float, ...] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an inspectable protocol record for one cell or dataset.

    ``charge_policy`` is used for TRI cell-specific charge rates.
    ``discharge_rates`` is used for HUST cell-specific multi-stage discharge
    rates. ``overrides`` can fill known physical metadata, but target/evaluation
    fields such as EOL and cycle_life are rejected.
    """

    inv = inventory or load_protocol_inventory()
    entry = _dataset_entry(inv, dataset_id)
    record = _base_physical_values(entry)
    record["dataset_id"] = dataset_id

    if dataset_id == "tri_severson" and charge_policy:
        record.update(parse_tri_charge_policy(charge_policy))
    if dataset_id == "hust_ma" and discharge_rates:
        padded = list(discharge_rates)[:3] + [None] * max(0, 3 - len(discharge_rates))
        record.update(
            {
                "discharge_c_rate_stage1": padded[0],
                "discharge_c_rate_stage2": padded[1],
                "discharge_c_rate_stage3": padded[2],
            }
        )
    if overrides:
        _reject_target_leakage(overrides)
        record.update(overrides)
    _reject_target_leakage(record)
    return record


def encode_physical_protocol(record: dict[str, Any]) -> list[float]:
    """Encode only physical protocol features, without dataset identity."""

    _reject_target_leakage(record)
    vector: list[float] = []
    for name in NUMERIC_FEATURES:
        value, known = _value_with_known(record.get(name))
        vector.extend([value, known])
    for field, values in CATEGORICAL_FEATURES.items():
        selected = record.get(field) or "unknown"
        if selected not in values:
            selected = "unknown"
        vector.extend([1.0 if selected == value else 0.0 for value in values])
    return vector


def encode_dataset_identity(dataset_id: str) -> list[float]:
    """Encode dataset identity separately from physical protocol metadata."""

    if dataset_id not in DATASET_IDS:
        raise KeyError(f"Unknown dataset_id: {dataset_id}")
    return [1.0 if dataset_id == value else 0.0 for value in DATASET_IDS]


def encode_protocol(
    dataset_id: str,
    *,
    inventory: dict[str, Any] | None = None,
    charge_policy: str | None = None,
    discharge_rates: list[float] | tuple[float, ...] | None = None,
    overrides: dict[str, Any] | None = None,
    include_dataset_identity: bool = False,
) -> list[float]:
    """Encode a protocol record deterministically.

    By default this returns only physical protocol features. Pass
    ``include_dataset_identity=True`` to append a separate dataset one-hot block.
    """

    record = protocol_record_from_dataset(
        dataset_id,
        inventory=inventory,
        charge_policy=charge_policy,
        discharge_rates=discharge_rates,
        overrides=overrides,
    )
    vector = encode_physical_protocol(record)
    if include_dataset_identity:
        vector = vector + encode_dataset_identity(dataset_id)
    return vector


def feature_names(include_dataset_identity: bool = False) -> list[str]:
    """Return feature names matching ``encode_protocol``."""

    names = physical_feature_names()
    if include_dataset_identity:
        names = names + dataset_identity_feature_names()
    return names
