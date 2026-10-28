import numpy as np
import pandas as pd
import pytest

from rotor_sail_performance.data_qc import (
    QCConfig,
    clean_operational_data,
    frozen_mask,
    generate_synthetic_operations,
    hampel_mask,
)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_synthetic_operations(days=30, seed=7)


def test_generation_is_reproducible_with_seed():
    a = generate_synthetic_operations(days=10, seed=123)
    b = generate_synthetic_operations(days=10, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_data():
    a = generate_synthetic_operations(days=10, seed=1)
    b = generate_synthetic_operations(days=10, seed=2)
    assert not a["shaft_power_kw"].equals(b["shaft_power_kw"])


def test_dataset_shape_and_faults(dataset):
    assert len(dataset) == 30 * 96
    assert dataset["timestamp"].diff().dropna().eq(pd.Timedelta("15min")).all()
    assert dataset["shaft_power_kw"].isna().any()
    assert dataset["injected_fault"].str.contains("frozen").any()
    assert dataset["injected_fault"].str.contains("outlier").any()


def test_clean_data_has_no_faults_and_no_gaps(dataset):
    result = clean_operational_data(dataset)
    assert 0 < len(result.clean) < len(dataset)
    assert (result.clean["injected_fault"] == "").all()
    sensor_cols = ["ship_speed_ms", "true_wind_speed_ms", "shaft_power_kw", "rotor_power_kw"]
    assert result.clean[sensor_cols].notna().all().all()
    assert (result.clean["mode"] == "sea").all()


def test_qc_log_accounts_for_every_removed_row(dataset):
    result = clean_operational_data(dataset)
    removed = (result.flagged["qc_flag"] != "").sum()
    assert result.log["n_newly_removed"].sum() == removed
    assert set(result.log["step"]) == {"missing", "range", "frozen", "outlier", "steady"}


def test_frozen_mask_detects_runs_and_ignores_zeros():
    s = pd.Series([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    mask = frozen_mask(s, min_run=4)
    assert mask.tolist() == [False, False, True, True, True, True, False, False, False, False, False, False]


def test_hampel_flags_spike_not_trend():
    s = pd.Series(np.linspace(1000.0, 2000.0, 60))
    s.iloc[30] = 5000.0
    mask = hampel_mask(s, window=11, threshold=4.0, min_scale=10.0)
    assert mask.iloc[30]
    assert mask.sum() == 1


def test_clean_synthetic_without_faults_keeps_more(dataset):
    clean_input = generate_synthetic_operations(days=30, seed=7, inject_faults=False)
    with_faults = clean_operational_data(dataset)
    without = clean_operational_data(clean_input, QCConfig())
    assert len(without.clean) > len(with_faults.clean)
