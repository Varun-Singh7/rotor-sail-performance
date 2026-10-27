import numpy as np
import pandas as pd
import pytest

from rotor_sail_performance import RotorConfig, ShipConfig, estimate_voyage, predict_savings
from rotor_sail_performance.savings import (
    co2_saving_kg,
    fuel_saving_kg,
    net_power_saving_kw,
    plot_savings_polar,
    savings_polar,
)


def test_net_power_saving_formula():
    # 100 kN at 6 m/s with eta 0.75 -> 800 kW at the shaft, minus 50 kW rotor power.
    assert net_power_saving_kw(100e3, 6.0, 50e3, 0.75) == pytest.approx(750.0)


def test_fuel_and_co2_conversion():
    fuel = fuel_saving_kg(1000.0, 2.0, 180.0)  # 1 MW for 2 h at 180 g/kWh
    assert fuel == pytest.approx(360.0)
    assert co2_saving_kg(fuel, 3.114) == pytest.approx(360.0 * 3.114)


def _route(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ship_speed_ms": rng.uniform(5.5, 6.5, n),
        "heading_deg": rng.uniform(0, 360, n),
        "true_wind_speed_ms": rng.uniform(3, 15, n),
        "true_wind_dir_deg": rng.uniform(0, 360, n),
    })


def test_savings_scale_linearly_with_number_of_rotors(ship, coeffs):
    route = _route()
    results = {n: estimate_voyage(route, RotorConfig(n_rotors=n), ship, coeffs, allow_switch_off=False)
               for n in (1, 2, 4)}
    for n in (2, 4):
        assert results[n].fuel_saved_t == pytest.approx(n * results[1].fuel_saved_t)
        assert results[n].co2_saved_t == pytest.approx(n * results[1].co2_saved_t)
        assert results[n].percent_saving == pytest.approx(n * results[1].percent_saving)
    # Baseline does not depend on the rotors.
    assert results[4].baseline_fuel_t == pytest.approx(results[1].baseline_fuel_t)


def test_switch_off_never_worse_than_always_on(rotor, ship, coeffs):
    route = _route(200)
    smart = estimate_voyage(route, rotor, ship, coeffs, allow_switch_off=True)
    always = estimate_voyage(route, rotor, ship, coeffs, allow_switch_off=False)
    assert np.all(smart.timeseries["net_saving_kw"] >= always.timeseries["net_saving_kw"] - 1e-9)


def test_head_wind_leads_to_rotors_off(rotor, ship, coeffs):
    pred = predict_savings(6.0, 0.0, 10.0, 0.0, rotor, ship, coeffs)
    assert not pred["rotors_on"].iloc[0]
    assert pred["net_saving_kw"].iloc[0] < 0  # a stopped rotor still adds drag


def test_voyage_totals_consistent(rotor, ship, coeffs):
    route = _route()
    res = estimate_voyage(route, rotor, ship, coeffs, timestep_h=0.5)
    assert res.duration_h == pytest.approx(25.0)
    assert res.fuel_saved_t == pytest.approx(res.saved_energy_mwh * ship.sfoc_g_per_kwh / 1000.0)
    assert res.co2_saved_t == pytest.approx(res.fuel_saved_t * ship.co2_factor)
    assert res.percent_saving == pytest.approx(100 * res.fuel_saved_t / res.baseline_fuel_t)


def test_baseline_power_curve():
    ship = ShipConfig(reference_power_kw=8000.0, reference_speed_ms=6.0, speed_power_exponent=3.0)
    assert ship.baseline_power_kw(6.0) == pytest.approx(8000.0)
    assert ship.baseline_power_kw(3.0) == pytest.approx(1000.0)


def test_polar_table_and_plot(rotor, ship, coeffs):
    polar = savings_polar(6.0, [5.0, 10.0], rotor, ship, coeffs, apparent_wind_angles_deg=[-90, 0, 90])
    assert len(polar) == 6
    beam = polar[(polar["apparent_wind_angle_deg"].abs() == 90) & (polar["apparent_wind_speed_ms"] == 10.0)]
    assert beam["net_saving_kw"].iloc[0] == pytest.approx(beam["net_saving_kw"].iloc[1])
    fig = plot_savings_polar(polar)
    assert fig.axes
