"""From rotor thrust to power, fuel and CO2 savings.

Model
-----
The rotor thrust T unloads the propeller. At ship speed V_s the propeller
would otherwise have had to deliver the effective power T * V_s, which at
the shaft costs T * V_s / eta_D, with eta_D the (quasi-)propulsive
efficiency. The rotors themselves consume electrical power P_rotor::

    P_net = T * V_s / eta_D - P_rotor                                   [W]
    fuel  = P_net[kW] * SFOC[g/kWh] / 1000 * dt[h]                      [kg]
    CO2   = fuel * C_F                                                  [kg]

The calm-water baseline propulsion power (used for percentage savings) is a
simple power law in speed (an "admiralty"-style speed-power curve)::

    P_base(V_s) = P_ref * (V_s / V_ref) ** n

Assumptions
-----------
* eta_D is constant: it does not change when the propeller is unloaded.
* One SFOC applies to both the propulsion saving and the rotor power, i.e.
  the rotors are fed from generators with the same specific consumption as
  the main engine and SFOC does not vary with engine load.
* The side force is ignored: in reality it causes leeway and rudder angle,
  both of which add resistance.
* The rotor controller runs at the target spin ratio and stops the rotors
  whenever stopping gives the larger net saving (e.g. in head winds).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from .aero import RHO_AIR, RotorCoefficients, RotorConfig, apparent_wind, rotor_forces

KNOT = 0.514444
"""One knot in m/s."""

CO2_FACTOR_HFO = 3.114
"""t CO2 per t fuel for heavy fuel oil (IMO carbon factor C_F)."""


@dataclass(frozen=True)
class ShipConfig:
    """Ship-side parameters of the savings model.

    The defaults describe a generic mid-size bulk carrier; they are
    illustrative, not taken from a particular vessel.
    """

    propulsive_efficiency: float = 0.70
    sfoc_g_per_kwh: float = 175.0
    co2_factor: float = CO2_FACTOR_HFO
    reference_power_kw: float = 7000.0
    reference_speed_ms: float = 12.0 * KNOT
    speed_power_exponent: float = 3.0

    def baseline_power_kw(self, ship_speed: ArrayLike) -> NDArray[np.float64]:
        """Calm-water shaft power without rotors, P_ref * (V/V_ref)^n [kW]."""
        v = np.clip(np.asarray(ship_speed, dtype=float), 0.0, None)
        return self.reference_power_kw * (v / self.reference_speed_ms) ** self.speed_power_exponent


def net_power_saving_kw(
    thrust_n: ArrayLike,
    ship_speed: ArrayLike,
    rotor_power_w: ArrayLike,
    propulsive_efficiency: float,
) -> NDArray[np.float64]:
    """Net shaft-power saving P_net = T * V_s / eta_D - P_rotor, returned in kW."""
    thrust = np.asarray(thrust_n, dtype=float)
    vs = np.asarray(ship_speed, dtype=float)
    return (thrust * vs / propulsive_efficiency - np.asarray(rotor_power_w, dtype=float)) / 1000.0


def fuel_saving_kg(net_power_kw: ArrayLike, duration_h: ArrayLike, sfoc_g_per_kwh: float) -> NDArray[np.float64]:
    """Fuel saved = P_net [kW] * dt [h] * SFOC [g/kWh] / 1000 [kg]."""
    return np.asarray(net_power_kw, dtype=float) * np.asarray(duration_h, dtype=float) * sfoc_g_per_kwh / 1000.0


def co2_saving_kg(fuel_kg: ArrayLike, co2_factor: float) -> NDArray[np.float64]:
    """CO2 avoided = fuel saved * C_F (t CO2 per t fuel, so kg in -> kg out)."""
    return np.asarray(fuel_kg, dtype=float) * co2_factor


def evaluate_apparent_wind(
    apparent_wind_speed: ArrayLike,
    apparent_wind_angle_deg: ArrayLike,
    ship_speed: ArrayLike,
    rotor: RotorConfig,
    ship: ShipConfig,
    coefficients: RotorCoefficients,
    rho_air: float = RHO_AIR,
    allow_switch_off: bool = True,
) -> pd.DataFrame:
    """Net saving for given apparent wind conditions, choosing the rotor state.

    Both states are evaluated: rotors spinning at the controlled spin ratio,
    and rotors stopped (SR = 0: no power, but still drag). If
    ``allow_switch_off`` the state with the larger net saving is selected,
    otherwise the rotors always spin.

    Returns:
        DataFrame with columns ``apparent_wind_speed_ms, apparent_wind_angle_deg,
        rotors_on, spin_ratio, rotor_rpm, thrust_n, side_force_n, rotor_power_kw,
        net_saving_kw``.
    """
    aws, awa, vs = np.broadcast_arrays(
        np.asarray(apparent_wind_speed, dtype=float),
        np.asarray(apparent_wind_angle_deg, dtype=float),
        np.asarray(ship_speed, dtype=float),
    )
    on = rotor_forces(aws, awa, rotor, coefficients, rho_air, spinning=True)
    net_on = net_power_saving_kw(on.thrust_n, vs, on.power_w, ship.propulsive_efficiency)
    if allow_switch_off:
        off = rotor_forces(aws, awa, rotor, coefficients, rho_air, spinning=False)
        net_off = net_power_saving_kw(off.thrust_n, vs, off.power_w, ship.propulsive_efficiency)
        use_on = net_on >= net_off
    else:
        off, net_off = on, net_on
        use_on = np.ones_like(aws, dtype=bool)

    def pick(a: NDArray, b: NDArray) -> NDArray:
        return np.where(use_on, a, b).ravel()

    return pd.DataFrame(
        {
            "apparent_wind_speed_ms": aws.ravel(),
            "apparent_wind_angle_deg": awa.ravel(),
            "rotors_on": use_on.ravel(),
            "spin_ratio": pick(on.spin_ratio, off.spin_ratio),
            "rotor_rpm": pick(on.rpm, off.rpm),
            "thrust_n": pick(on.thrust_n, off.thrust_n),
            "side_force_n": pick(on.side_force_n, off.side_force_n),
            "rotor_power_kw": pick(on.power_w, off.power_w) / 1000.0,
            "net_saving_kw": pick(net_on, net_off),
        }
    )


def predict_savings(
    ship_speed: ArrayLike,
    ship_heading_deg: ArrayLike,
    true_wind_speed: ArrayLike,
    true_wind_direction_deg: ArrayLike,
    rotor: RotorConfig,
    ship: ShipConfig,
    coefficients: RotorCoefficients,
    rho_air: float = RHO_AIR,
    allow_switch_off: bool = True,
) -> pd.DataFrame:
    """Predict rotor performance from ship motion and true wind.

    Solves the wind triangle and then calls :func:`evaluate_apparent_wind`.
    Also adds ``baseline_power_kw`` from the ship's speed-power curve.
    """
    aws, awa = apparent_wind(true_wind_speed, true_wind_direction_deg, ship_speed, ship_heading_deg)
    result = evaluate_apparent_wind(aws, awa, ship_speed, rotor, ship, coefficients, rho_air, allow_switch_off)
    result["baseline_power_kw"] = ship.baseline_power_kw(np.broadcast_to(ship_speed, aws.shape)).ravel()
    return result


@dataclass(frozen=True)
class VoyageResult:
    """Totals for a voyage plus the per-time-step predictions."""

    timeseries: pd.DataFrame
    duration_h: float
    baseline_energy_mwh: float
    saved_energy_mwh: float
    baseline_fuel_t: float
    fuel_saved_t: float
    co2_saved_t: float

    @property
    def percent_saving(self) -> float:
        """Net energy (= fuel) saving as a percentage of baseline propulsion energy."""
        if self.baseline_energy_mwh <= 0.0:
            return float("nan")
        return 100.0 * self.saved_energy_mwh / self.baseline_energy_mwh


def estimate_voyage(
    route: pd.DataFrame,
    rotor: RotorConfig,
    ship: ShipConfig,
    coefficients: RotorCoefficients,
    timestep_h: float | ArrayLike = 0.25,
    rho_air: float = RHO_AIR,
    allow_switch_off: bool = True,
) -> VoyageResult:
    """Integrate savings over a route given as a time series.

    Args:
        route: DataFrame with columns ``ship_speed_ms, heading_deg,
            true_wind_speed_ms, true_wind_dir_deg``. One row per time step.
        timestep_h: Duration represented by each row [h], scalar or per row.
            Totals cover only the rows supplied (gaps are not filled).

    Returns:
        :class:`VoyageResult`. Percentage saving is relative to the calm-water
        baseline, so weather-induced added resistance is not in the denominator.
    """
    pred = predict_savings(
        route["ship_speed_ms"].to_numpy(),
        route["heading_deg"].to_numpy(),
        route["true_wind_speed_ms"].to_numpy(),
        route["true_wind_dir_deg"].to_numpy(),
        rotor,
        ship,
        coefficients,
        rho_air,
        allow_switch_off,
    )
    pred.index = route.index
    dt = np.broadcast_to(np.asarray(timestep_h, dtype=float), (len(route),))
    pred["duration_h"] = dt
    pred["fuel_saved_kg"] = fuel_saving_kg(pred["net_saving_kw"], dt, ship.sfoc_g_per_kwh)
    pred["co2_saved_kg"] = co2_saving_kg(pred["fuel_saved_kg"], ship.co2_factor)

    baseline_kwh = float(np.sum(pred["baseline_power_kw"] * dt))
    saved_kwh = float(np.sum(pred["net_saving_kw"] * dt))
    return VoyageResult(
        timeseries=pred,
        duration_h=float(np.sum(dt)),
        baseline_energy_mwh=baseline_kwh / 1000.0,
        saved_energy_mwh=saved_kwh / 1000.0,
        baseline_fuel_t=float(fuel_saving_kg(baseline_kwh, 1.0, ship.sfoc_g_per_kwh)) / 1000.0,
        fuel_saved_t=float(pred["fuel_saved_kg"].sum()) / 1000.0,
        co2_saved_t=float(pred["co2_saved_kg"].sum()) / 1000.0,
    )


def savings_polar(
    ship_speed: float,
    apparent_wind_speeds: ArrayLike,
    rotor: RotorConfig,
    ship: ShipConfig,
    coefficients: RotorCoefficients,
    apparent_wind_angles_deg: ArrayLike | None = None,
    rho_air: float = RHO_AIR,
) -> pd.DataFrame:
    """Tidy table of thrust and net saving over an AWA x AWS grid at fixed ship speed."""
    if apparent_wind_angles_deg is None:
        apparent_wind_angles_deg = np.arange(-180.0, 181.0, 5.0)
    awa_grid, aws_grid = np.meshgrid(np.asarray(apparent_wind_angles_deg, float), np.asarray(apparent_wind_speeds, float))
    table = evaluate_apparent_wind(aws_grid, awa_grid, ship_speed, rotor, ship, coefficients, rho_air)
    table.insert(0, "ship_speed_ms", ship_speed)
    return table


def plot_savings_polar(polar: pd.DataFrame, quantity: str = "net_saving_kw", ax: plt.Axes | None = None) -> Figure:
    """Polar plot of ``quantity`` vs apparent wind angle, one line per apparent wind speed.

    Angle 0 (head wind) is at the top and angles increase clockwise, so the
    plot reads like a view of the ship from above with the bow pointing up.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    else:
        fig = ax.figure
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    speeds = np.sort(polar["apparent_wind_speed_ms"].unique())
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(speeds)))
    for speed, colour in zip(speeds, colours):
        rows = polar[polar["apparent_wind_speed_ms"] == speed].sort_values("apparent_wind_angle_deg")
        values = rows[quantity].to_numpy()
        if quantity == "thrust_n":
            values = values / 1000.0
        ax.plot(np.radians(rows["apparent_wind_angle_deg"]), values, color=colour, label=f"{speed:.0f} m/s")

    unit = "kN" if quantity == "thrust_n" else "kW"
    label = "Thrust" if quantity == "thrust_n" else "Net power saving"
    v_kn = polar["ship_speed_ms"].iloc[0] / KNOT
    ax.set_title(f"{label} [{unit}] vs apparent wind angle\nship speed {v_kn:.1f} kn", pad=20)
    ax.legend(title="Apparent wind", loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize="small")
    fig.tight_layout()
    return fig
