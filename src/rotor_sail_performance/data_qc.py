"""Synthetic operational data and its quality control.

Synthetic data
--------------
:func:`generate_synthetic_operations` simulates a ship shuttling between two
ports with a rotor installation. Everything is driven by one
``numpy.random.Generator`` seeded with ``seed`` so the dataset is exactly
reproducible.

* Operating profile: sea legs of 4-7 days alternating between headings of
  about 250 deg and 70 deg, port stays of 0.75-1.5 days, 2 h manoeuvring
  ramps, slow course drift and occasional 15-30 deg course alterations.
* Wind: a prevailing westerly plus a 2-D first-order autoregressive (AR(1))
  fluctuation with an 18 h correlation time, which gives Rayleigh-like
  wind speeds and slowly veering directions.
* "True" rotor physics deliberately differ from the model: the real rotors
  produce 8 % less lift and 5 % more drag than the coefficient table says.
  This gives the verification step a known bias to find.
* Shaft power = calm-water baseline * (1 + weather term) - rotor contribution,
  where the weather term is an AR(1) process (6 h correlation, 4 % std)
  standing in for waves, fouling and other unmodelled effects.
* Sensor noise: wind speed 3 % + 0.3 m/s, wind direction 3 deg, shaft power
  1 %, rotor power 2 %, speed 0.05 m/s, heading 0.5 deg.

Columns starting with ``true_`` (and ``injected_fault``) hold the hidden
truth. They would not exist in a real dataset and the QC and verification
code never uses them except in tests and for reporting.

Quality control
---------------
:func:`clean_operational_data` applies, in order:

1. Missing values in required columns.
2. Range checks against physical limits.
3. Frozen-sensor detection: a value repeated exactly for at least
   ``frozen_min_samples`` consecutive samples. Exact zeros are exempt
   because a stopped engine or rotor legitimately reads zero.
4. Outliers: Hampel filter (rolling median +- k * robust std), with the
   robust std floored at an absolute and a relative minimum. Run before
   the steady-state filter because the gaps it leaves skew rolling medians.
5. Steady-state filter: speed above a minimum and, within a centred
   rolling window, low speed variation and little heading change.

Each removed row gets the reason of the *first* step that flagged it, and a
log of counts per step and column is returned and written to ``logging``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .aero import RotorCoefficients, RotorConfig, wrap_angle_deg
from .savings import ShipConfig, predict_savings

logger = logging.getLogger(__name__)

SENSOR_COLUMNS = [
    "ship_speed_ms",
    "heading_deg",
    "true_wind_speed_ms",
    "true_wind_dir_deg",
    "rotor_rpm",
    "rotor_power_kw",
    "shaft_power_kw",
]


def _ar1(rng: np.random.Generator, n: int, dt_h: float, tau_h: float, sigma: float,
         smooth_h: float = 2.0) -> np.ndarray:
    """Stationary AR(1) series x_t = phi x_{t-1} + sqrt(1 - phi^2) sigma e_t, phi = exp(-dt/tau).

    A plain AR(1) jumps too much from one 15-min sample to the next, so the
    series is smoothed with a centred moving average over ``smooth_h`` hours
    and rescaled back to standard deviation ``sigma``.
    """
    phi = np.exp(-dt_h / tau_h)
    shocks = rng.normal(0.0, sigma * np.sqrt(1.0 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + shocks[i]
    window = max(1, int(round(smooth_h / dt_h)))
    if window > 1 and n > window:
        x = pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_numpy()
        x *= sigma / x.std()
    return x


def _operating_profile(rng: np.random.Generator, n: int, dt_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ship speed [m/s], heading [deg] and mode labels for n time steps."""
    speed = np.zeros(n)
    heading = np.zeros(n)
    mode = np.empty(n, dtype=object)
    ramp = int(round(2.0 / dt_h))
    i = 0
    outbound = True
    while i < n:
        # Port stay.
        port = int(rng.uniform(0.75, 1.5) * 24 / dt_h)
        base_heading = 250.0 if outbound else 70.0
        j = min(i + port, n)
        speed[i:j] = 0.0
        heading[i:j] = base_heading + rng.normal(0.0, 20.0)
        mode[i:j] = "port"
        i = j
        if i >= n:
            break
        # Sea leg with ramps at both ends.
        leg = int(rng.uniform(4.0, 7.0) * 24 / dt_h)
        j = min(i + leg, n)
        m = j - i
        leg_speed = rng.uniform(11.0, 13.0) * 0.514444
        v = leg_speed + _ar1(rng, m, dt_h, tau_h=6.0, sigma=0.12)
        envelope = np.ones(m)
        up = min(ramp, m)
        envelope[:up] = np.linspace(0.15, 1.0, up)
        down = min(ramp, m - up)
        if down > 0:
            envelope[m - down :] = np.linspace(1.0, 0.15, down)
        speed[i:j] = v * envelope

        drift = np.cumsum(rng.normal(0.0, 0.15, m))
        steps = np.zeros(m)
        for _ in range(rng.integers(1, 4)):
            k = rng.integers(ramp, max(ramp + 1, m - ramp))
            steps[k:] += rng.choice([-1.0, 1.0]) * rng.uniform(15.0, 30.0)
        heading[i:j] = base_heading + drift + steps
        mode[i:j] = "sea"
        mode[i : i + up] = "manoeuvring"
        if down > 0:
            mode[j - down : j] = "manoeuvring"
        i = j
        outbound = not outbound
    return speed, heading % 360.0, mode


def generate_synthetic_operations(
    days: float = 90.0,
    interval_min: int = 15,
    seed: int = 42,
    rotor: RotorConfig | None = None,
    ship: ShipConfig | None = None,
    coefficients: RotorCoefficients | None = None,
    inject_faults: bool = True,
    start: str = "2025-01-01",
) -> pd.DataFrame:
    """Generate a reproducible synthetic operational dataset.

    Args:
        days: Length of the record.
        interval_min: Sampling interval in minutes.
        seed: Seed for ``numpy.random.default_rng``; same seed -> identical data.
        rotor, ship, coefficients: Installation used to simulate the *true*
            performance (coefficients are then perturbed, see module docstring).
        inject_faults: Add dropouts, frozen values and outliers.
        start: First timestamp (UTC).

    Returns:
        DataFrame with ``timestamp`` (UTC, tz-naive), the sensor columns in
        :data:`SENSOR_COLUMNS`, and hidden-truth columns ``mode``,
        ``true_net_saving_kw`` and ``injected_fault``.
    """
    rotor = rotor or RotorConfig()
    ship = ship or ShipConfig()
    coefficients = coefficients or RotorCoefficients.from_csv()
    rng = np.random.default_rng(seed)

    dt_h = interval_min / 60.0
    n = int(days * 24 / dt_h)
    timestamps = pd.date_range(start, periods=n, freq=f"{interval_min}min")

    speed, heading, mode = _operating_profile(rng, n, dt_h)
    in_port = mode == "port"

    # Wind as a vector: prevailing westerly (blowing towards east) + AR(1) fluctuations.
    wind_east = 3.0 + _ar1(rng, n, dt_h, tau_h=18.0, sigma=5.0)
    wind_north = _ar1(rng, n, dt_h, tau_h=18.0, sigma=5.0)
    tws = np.hypot(wind_east, wind_north)
    twd = np.degrees(np.arctan2(-wind_east, -wind_north)) % 360.0

    # "True" physics differ from the model on purpose.
    true_coeffs = coefficients.scaled(cl_factor=0.92, cd_factor=1.05)
    truth = predict_savings(speed, heading, tws, twd, rotor, ship, true_coeffs)
    # Rotors are parked in port.
    truth.loc[in_port, "rotors_on"] = False
    truth.loc[in_port, ["rotor_rpm", "rotor_power_kw", "net_saving_kw", "thrust_n"]] = 0.0

    weather = _ar1(rng, n, dt_h, tau_h=6.0, sigma=0.04)
    baseline = ship.baseline_power_kw(speed)
    propeller_relief_kw = truth["thrust_n"].to_numpy() * speed / ship.propulsive_efficiency / 1000.0
    shaft_true = np.clip(baseline * (1.0 + weather) - propeller_relief_kw, 0.0, None)
    shaft_true[in_port] = 0.0

    rotor_power_true = truth["rotor_power_kw"].to_numpy()
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ship_speed_ms": np.clip(speed + rng.normal(0.0, 0.05, n), 0.0, None),
            "heading_deg": (heading + rng.normal(0.0, 0.5, n)) % 360.0,
            "true_wind_speed_ms": np.clip(tws * (1.0 + rng.normal(0.0, 0.03, n)) + rng.normal(0.0, 0.3, n), 0.0, None),
            "true_wind_dir_deg": (twd + rng.normal(0.0, 3.0, n)) % 360.0,
            "rotor_rpm": truth["rotor_rpm"].to_numpy(),
            "rotor_power_kw": rotor_power_true * (1.0 + rng.normal(0.0, 0.02, n)),
            "shaft_power_kw": shaft_true * (1.0 + rng.normal(0.0, 0.01, n)),
            "mode": mode.astype(str),
            "true_net_saving_kw": truth["net_saving_kw"].to_numpy(),
            "injected_fault": "",
        }
    )
    # Rounding mimics a logger's resolution and makes frozen values exact repeats.
    df = df.round({"ship_speed_ms": 3, "heading_deg": 2, "true_wind_speed_ms": 2, "true_wind_dir_deg": 1,
                   "rotor_rpm": 1, "rotor_power_kw": 2, "shaft_power_kw": 1})

    if inject_faults:
        _inject_faults(df, rng)
    return df


def _inject_faults(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Corrupt sensor columns in place and record what was done in ``injected_fault``."""
    n = len(df)

    def mark(idx: np.ndarray | slice, label: str) -> None:
        current = df.loc[df.index[idx], "injected_fault"]
        df.loc[df.index[idx], "injected_fault"] = np.where(current == "", label, current + ";" + label)

    # Short dropouts in individual sensors.
    for col in SENSOR_COLUMNS:
        for _ in range(12):
            s = rng.integers(0, n - 16)
            length = rng.integers(1, 17)
            df.loc[df.index[s : s + length], col] = np.nan
            mark(slice(s, s + length), f"dropout:{col}")
    # One whole-logger outage of a day.
    s = rng.integers(0, n - 96)
    df.loc[df.index[s : s + 96], SENSOR_COLUMNS] = np.nan
    mark(slice(s, s + 96), "dropout:logger")

    # Frozen sensors: last good value repeated.
    for col, episodes, (lo, hi) in [("true_wind_speed_ms", 3, (24, 97)), ("shaft_power_kw", 2, (16, 49)),
                                    ("true_wind_dir_deg", 1, (24, 73))]:
        for _ in range(episodes):
            s = rng.integers(1, n - hi)
            length = rng.integers(lo, hi)
            df.loc[df.index[s : s + length], col] = df[col].iloc[s - 1]
            mark(slice(s, s + length), f"frozen:{col}")

    # Outliers: spikes and impossible values.
    def spikes(col: str, frac: float, make: Callable[[np.ndarray, int], np.ndarray]) -> None:
        idx = rng.choice(n, size=int(frac * n), replace=False)
        df.loc[df.index[idx], col] = make(df[col].iloc[idx].to_numpy(), len(idx))
        mark(idx, f"outlier:{col}")

    spikes("shaft_power_kw", 0.004, lambda v, k: v * rng.uniform(1.4, 2.5, k))
    spikes("rotor_power_kw", 0.002, lambda v, k: v + rng.uniform(300.0, 800.0, k))
    spikes("ship_speed_ms", 0.001, lambda v, k: rng.uniform(25.0, 60.0, k))
    spikes("true_wind_speed_ms", 0.001, lambda v, k: rng.uniform(70.0, 120.0, k))
    spikes("heading_deg", 0.0005, lambda v, k: rng.uniform(361.0, 999.0, k))


@dataclass(frozen=True)
class QCConfig:
    """Thresholds for :func:`clean_operational_data`."""

    ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "ship_speed_ms": (0.0, 15.0),
            "heading_deg": (0.0, 360.0),
            "true_wind_speed_ms": (0.0, 50.0),
            "true_wind_dir_deg": (0.0, 360.0),
            "rotor_rpm": (0.0, 250.0),
            "rotor_power_kw": (0.0, 1000.0),
            "shaft_power_kw": (0.0, 20000.0),
        }
    )
    frozen_columns: tuple[str, ...] = (
        "ship_speed_ms", "true_wind_speed_ms", "true_wind_dir_deg", "rotor_power_kw", "shaft_power_kw",
    )
    frozen_min_samples: int = 8
    steady_window: int = 5
    min_speed_ms: float = 4.0
    max_speed_std_ms: float = 0.25
    max_heading_change_deg: float = 5.0
    outlier_columns: tuple[str, ...] = ("shaft_power_kw", "rotor_power_kw")
    hampel_window: int = 17
    hampel_threshold: float = 4.0
    hampel_min_scale: float = 30.0
    hampel_min_rel_scale: float = 0.03


@dataclass(frozen=True)
class QCResult:
    """Output of :func:`clean_operational_data`."""

    clean: pd.DataFrame
    flagged: pd.DataFrame
    log: pd.DataFrame


def frozen_mask(values: pd.Series, min_run: int, exempt: tuple[float, ...] = (0.0,)) -> pd.Series:
    """True where a value is part of a run of >= ``min_run`` identical consecutive samples."""
    v = values.to_numpy()
    same_as_previous = np.r_[False, v[1:] == v[:-1]]
    run_id = np.cumsum(~same_as_previous)
    run_length = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    mask = (run_length >= min_run) & values.notna().to_numpy() & ~np.isin(v, exempt)
    return pd.Series(mask, index=values.index)


def hampel_mask(values: pd.Series, window: int, threshold: float, min_scale: float,
                min_rel_scale: float = 0.0) -> pd.Series:
    """Hampel filter: flag |x - rolling median| > threshold * scale.

    scale = max(1.4826 * MAD, min_scale, min_rel_scale * |median|). The
    floors stop the filter from rejecting genuine short-term variability
    when the signal is locally very smooth (MAD close to zero).
    """
    med = values.rolling(window, center=True, min_periods=window // 2).median()
    mad = (values - med).abs().rolling(window, center=True, min_periods=window // 2).median()
    scale = np.maximum(np.maximum(1.4826 * mad, min_scale), min_rel_scale * med.abs())
    return ((values - med).abs() > threshold * scale).fillna(False)


def steady_state_mask(df: pd.DataFrame, cfg: QCConfig) -> pd.Series:
    """True where the ship sails steadily: fast enough, constant speed and course."""
    w = cfg.steady_window
    speed = df["ship_speed_ms"]
    heading_step = pd.Series(np.abs(wrap_angle_deg(df["heading_deg"].diff())), index=df.index)
    speed_ok = speed.rolling(w, center=True, min_periods=w).min() >= cfg.min_speed_ms
    std_ok = speed.rolling(w, center=True, min_periods=w).std() <= cfg.max_speed_std_ms
    turn_ok = heading_step.rolling(w, center=True, min_periods=w - 1).sum() <= cfg.max_heading_change_deg
    return speed_ok & std_ok & turn_ok


def clean_operational_data(df: pd.DataFrame, cfg: QCConfig | None = None) -> QCResult:
    """Run the QC chain described in the module docstring.

    Flagged values are set to NaN as the chain progresses so that later steps
    (rolling windows) are not influenced by already-rejected data.

    Returns:
        :class:`QCResult` with the clean rows, all rows with a ``qc_flag``
        column (empty string = kept), and a log DataFrame with columns
        ``step, column, n_flagged, n_newly_removed, reason``.
    """
    cfg = cfg or QCConfig()
    work = df.copy()
    flag = pd.Series("", index=df.index, dtype=object)
    log_rows: list[dict[str, object]] = []

    def record(step: str, column: str, mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False).astype(bool)
        new = mask & (flag == "")
        flag[new] = f"{step}:{column}"
        if column in work.columns:
            work.loc[mask, column] = np.nan
        log_rows.append({"step": step, "column": column, "n_flagged": int(mask.sum()),
                         "n_newly_removed": int(new.sum()), "reason": reason})
        logger.info("QC %-12s %-20s flagged %5d, newly removed %5d (%s)", step, column, mask.sum(), new.sum(), reason)

    for col in SENSOR_COLUMNS:
        record("missing", col, work[col].isna(), "no value recorded (dropout)")
    for col, (lo, hi) in cfg.ranges.items():
        values = work[col]
        record("range", col, (values < lo) | (values > hi), f"outside physical range [{lo}, {hi}]")
    for col in cfg.frozen_columns:
        record("frozen", col, frozen_mask(work[col], cfg.frozen_min_samples),
               f"identical non-zero value for >= {cfg.frozen_min_samples} consecutive samples")
    for col in cfg.outlier_columns:
        record("outlier", col, hampel_mask(work[col], cfg.hampel_window, cfg.hampel_threshold,
                                                   cfg.hampel_min_scale, cfg.hampel_min_rel_scale),
               f"Hampel filter, |x - median| > {cfg.hampel_threshold} robust std over {cfg.hampel_window} samples")
    record("steady", "ship_speed_ms/heading_deg", ~steady_state_mask(work, cfg),
           f"not steady: speed < {cfg.min_speed_ms} m/s, speed std > {cfg.max_speed_std_ms} m/s "
           f"or heading change > {cfg.max_heading_change_deg} deg in {cfg.steady_window} samples")

    flagged = df.copy()
    flagged["qc_flag"] = flag
    clean = df.loc[flag == ""].copy()
    log = pd.DataFrame(log_rows)
    logger.info("QC kept %d of %d rows (%.1f %%)", len(clean), len(df), 100.0 * len(clean) / max(len(df), 1))
    return QCResult(clean=clean, flagged=flagged, log=log)
