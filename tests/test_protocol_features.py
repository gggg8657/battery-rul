import pytest

from src.protocol_features import (
    dataset_identity_feature_names,
    encode_dataset_identity,
    encode_protocol,
    feature_names,
    load_protocol_inventory,
    physical_feature_names,
    protocol_record_from_dataset,
)


def test_protocol_feature_vector_length_is_stable():
    inv = load_protocol_inventory()
    tri = encode_protocol(
        "tri_severson",
        inventory=inv,
        charge_policy="4.4C(55%)-6C",
        include_dataset_identity=True,
    )
    hust = encode_protocol(
        "hust_ma",
        inventory=inv,
        discharge_rates=[5, 1, 1],
        include_dataset_identity=True,
    )
    calce = encode_protocol("calce_a123", inventory=inv, include_dataset_identity=True)

    assert len(tri) == len(feature_names(include_dataset_identity=True))
    assert len(hust) == len(tri)
    assert len(calce) == len(tri)
    assert len(physical_feature_names()) + len(dataset_identity_feature_names()) == len(tri)


def test_dataset_identity_is_separate_from_physical_protocol_features():
    inv = load_protocol_inventory()
    physical = encode_protocol(
        "tri_severson",
        inventory=inv,
        charge_policy="4.4C(55%)-6C",
        include_dataset_identity=False,
    )
    combined = encode_protocol(
        "tri_severson",
        inventory=inv,
        charge_policy="4.4C(55%)-6C",
        include_dataset_identity=True,
    )

    assert combined[: len(physical)] == physical
    assert combined[len(physical):] == encode_dataset_identity("tri_severson")


def test_unknown_metadata_uses_zero_value_and_zero_known_mask():
    inv = load_protocol_inventory()
    record = protocol_record_from_dataset("hust_ma", inventory=inv, discharge_rates=[5, 1, 1])
    vector = encode_protocol("hust_ma", inventory=inv, discharge_rates=[5, 1, 1])
    names = feature_names()

    assert record["charge_c_rate_stage1"] is None
    value_idx = names.index("charge_c_rate_stage1")
    mask_idx = names.index("charge_c_rate_stage1__known")
    assert vector[value_idx] == 0.0
    assert vector[mask_idx] == 0.0


def test_target_leakage_fields_are_rejected():
    with pytest.raises(ValueError, match="cycle_life"):
        protocol_record_from_dataset("tri_severson", overrides={"cycle_life": 1000})

    with pytest.raises(ValueError, match="eol_definition"):
        encode_protocol("calce_a123", overrides={"eol_definition": "80% capacity"})
