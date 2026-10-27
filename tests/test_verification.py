import numpy as np
import pandas as pd
import pytest

from rotor_sail_performance import predict_savings
from rotor_sail_performance.data_qc import generate_synthetic_operations
from rotor_sail_performance.verification import (
    InputUncertainty,
    bootstrap_mean_ci,
    compare,
    compare_by_awa,
    measured_net_saving_kw,
    monte_carlo_savings,
    sanity_check,
)


def test_compare_known_values():
    m = compare([1.0, 2.0, 3.0], [2.0, 2.0, 5.0])
    assert m.n == 3
    assert m.bias == pytest.approx(1.0)
    assert m.rmse == pytest.approx(np.sqrt(5.0 / 3.0))
    assert m.mae == pytest.approx(1.0)


def test_compare_ignores_nan():
    m = compare([1.0, np.nan, 3.0], [1.0, 2.0, np.nan])
    assert m.n == 1
    assert m.bias == 0.0


def test_bootstrap_ci_contains_true_mean_iid():
    rng = np.random.default_rng(11)
    sample = rng.normal(250.0, 80.0, 2000)
    ci = bootstrap_mean_ci(sample, n_boot=1000, seed=3)
    assert ci.lower < 250.0 < ci.upper
    assert ci.lower < ci.mean < ci.upper


def test_bootstrap_ci_contains_true_mean_on_clean_synthetic_data(ship):
    """With no injected faults, the block-bootstrap CI of the measured saving
    must contain the mean of the hidden true saving."""
    df = generate_synthetic_operations(days=60, seed=5, inject_faults=False)
    sea = df[df["mode"] == "sea"]
    measured = measured_net_saving_kw(sea, ship)
    ci = bootstrap_mean_ci(measured, n_boot=1000, block_length=96, seed=0)
    true_mean = sea["true_net_saving_kw"].mean()
    assert ci.lower < true_mean < ci.upper


def test_block_bootstrap_is_wider_for_autocorrelated_data():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=3000)) * 0.1 + rng.normal(size=3000)
    iid = bootstrap_mean_ci(x, n_boot=500, seed=1)
    block = bootstrap_mean_ci(x, n_boot=500, block_length=100, seed=1)
    assert (block.upper - block.lower) > (iid.upper - iid.lower)


def test_compare_by_awa_bins():
    df = pd.DataFrame({
        "apparent_wind_angle_deg": [10.0, -20.0, 95.0, -100.0, 170.0],
        "predicted_net_saving_kw": [0.0, 0.0, 100.0, 100.0, 50.0],
        "measured_net_saving_kw": [10.0, -10.0, 120.0, 80.0, 50.0],
    })
    binned = compare_by_awa(df)
    assert list(binned["n"]) == [2, 0, 0, 2, 0, 1]
    assert binned.loc[0, "bias"] == pytest.approx(0.0)
    assert binned.loc[3, "rmse"] == pytest.approx(20.0)


def test_monte_carlo_spread_and_centre(rotor, ship, coeffs):
    df = generate_synthetic_operations(days=5, seed=3, inject_faults=False)
    df = df[df["mode"] == "sea"]
    nominal = predict_savings(df["ship_speed_ms"], df["heading_deg"], df["true_wind_speed_ms"],
                              df["true_wind_dir_deg"], rotor, ship, coeffs)["net_saving_kw"].mean()
    mc = monte_carlo_savings(df, rotor, ship, coeffs, n_samples=200, seed=4)
    assert len(mc) == 200
    assert mc["mean_net_saving_kw"].std() > 0
    assert mc["mean_net_saving_kw"].median() == pytest.approx(nominal, rel=0.1)
    no_unc = monte_carlo_savings(df, rotor, ship, coeffs, InputUncertainty(0.0, 0.0, 0.0), n_samples=3)
    np.testing.assert_allclose(no_unc["mean_net_saving_kw"], nominal)


def test_sanity_check_passes_reasonable_predictions(rotor, ship, coeffs):
    pred = predict_savings([6.0, 6.0], [0.0, 0.0], [10.0, 8.0], [90.0, 120.0], rotor, ship, coeffs)
    assert sanity_check(pred, coeffs) == []


def test_sanity_check_flags_unreasonable_results(rotor, ship, coeffs):
    pred = predict_savings([6.0, 6.0], [0.0, 0.0], [10.0, 8.0], [90.0, 120.0], rotor, ship, coeffs)
    pred.loc[0, "net_saving_kw"] = 1e6
    pred.loc[1, "rotor_power_kw"] = -5.0
    pred.loc[1, "spin_ratio"] = 9.0
    checks = {f.check for f in sanity_check(pred, coeffs)}
    assert {"saving_exceeds_baseline", "rotor_power", "spin_ratio_range", "percent_saving"} <= checks
