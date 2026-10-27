"""Verification of predicted savings against (synthetic) measurements.

"Measured" saving
-----------------
Real rotor savings cannot be measured directly; they are inferred. Here the
measured net saving is the shortfall of measured shaft power below the
calm-water speed-power curve, minus the measured rotor power::

    P_net,meas = P_base(V_s) - P_shaft,meas - P_rotor,meas

Anything the speed-power curve does not capture (waves, fouling, draught,
current) ends up in this quantity as noise or bias. Proper trials use
rotors-on / rotors-off comparisons or a calibrated baseline model instead
(see the ITTC and IMO references in the README).

Statistics
----------
* residual r = measured - predicted; bias = mean(r), RMSE = sqrt(mean(r^2)).
* Bootstrap: percentile confidence interval of the mean. Consecutive
  15-min samples are strongly autocorrelated, so a moving-block bootstrap
  (``block_length`` > 1) is used to avoid over-confident intervals.
* Monte Carlo: systematic input errors (wind speed scale, CL scale,
  propulsive efficiency) are drawn once per realisation and propagated
  through the full savings model. They are treated as *systematic* because
  random per-sample errors average out over thousands of samples, while a
  mis-calibrated anemometer or wrong CL table does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .aero import RHO_AIR, RotorCoefficients, RotorConfig
from .savings import ShipConfig, predict_savings

AWA_BINS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)


def measured_net_saving_kw(df: pd.DataFrame, ship: ShipConfig) -> pd.Series:
    """P_base(V_s) - P_shaft - P_rotor from the measured columns [kW]."""
    baseline = ship.baseline_power_kw(df["ship_speed_ms"].to_numpy())
    return pd.Series(baseline - df["shaft_power_kw"] - df["rotor_power_kw"], index=df.index)


@dataclass(frozen=True)
class ComparisonMetrics:
    """Agreement statistics between predicted and measured values."""

    n: int
    mean_predicted: float
    mean_measured: float
    bias: float
    rmse: float
    mae: float
    r2: float


def compare(predicted: ArrayLike, measured: ArrayLike) -> ComparisonMetrics:
    """Residual statistics with residual = measured - predicted (NaN pairs dropped)."""
    p = np.asarray(predicted, dtype=float)
    m = np.asarray(measured, dtype=float)
    ok = np.isfinite(p) & np.isfinite(m)
    p, m = p[ok], m[ok]
    if p.size == 0:
        nan = float("nan")
        return ComparisonMetrics(0, nan, nan, nan, nan, nan, nan)
    r = m - p
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    return ComparisonMetrics(
        n=int(p.size),
        mean_predicted=float(p.mean()),
        mean_measured=float(m.mean()),
        bias=float(r.mean()),
        rmse=float(np.sqrt(np.mean(r**2))),
        mae=float(np.mean(np.abs(r))),
        r2=1.0 - float(np.sum(r**2)) / ss_tot if ss_tot > 0 else float("nan"),
    )


def compare_by_awa(
    df: pd.DataFrame,
    predicted_col: str = "predicted_net_saving_kw",
    measured_col: str = "measured_net_saving_kw",
    awa_col: str = "apparent_wind_angle_deg",
    bins: tuple[float, ...] = AWA_BINS_DEG,
) -> pd.DataFrame:
    """:func:`compare` within bins of |AWA| (port and starboard pooled)."""
    labels = [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(bins[:-1], bins[1:])]
    binned = pd.cut(df[awa_col].abs(), bins=list(bins), labels=labels, include_lowest=True)
    rows = []
    for label, lo, hi in zip(labels, bins[:-1], bins[1:]):
        part = df[binned == label]
        metrics = compare(part[predicted_col], part[measured_col])
        rows.append({"awa_bin": label, "awa_bin_low_deg": lo, "awa_bin_high_deg": hi, **metrics.__dict__})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate and percentile confidence interval of the mean."""

    mean: float
    lower: float
    upper: float
    confidence: float
    n_boot: int
    block_length: int


def bootstrap_mean_ci(
    values: ArrayLike,
    n_boot: int = 2000,
    confidence: float = 0.95,
    block_length: int = 1,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap CI of the mean, optionally as a moving-block bootstrap.

    With ``block_length = L``, each resample is built from ceil(n / L) blocks
    of L consecutive values with random start points, then trimmed to n.
    L = 1 is the ordinary i.i.d. bootstrap.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        raise ValueError("no finite values to bootstrap")
    block_length = int(max(1, min(block_length, n)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_length + 1, n_blocks)
        idx = (starts[:, None] + offsets).ravel()[:n]
        means[b] = x[idx].mean()
    alpha = 1.0 - confidence
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(float(x.mean()), float(lower), float(upper), confidence, n_boot, block_length)


@dataclass(frozen=True)
class InputUncertainty:
    """1-sigma systematic uncertainties for Monte Carlo propagation.

    Defaults are assumptions for illustration, not derived from a sensor spec.
    """

    wind_speed_rel: float = 0.05
    lift_coefficient_rel: float = 0.10
    propulsive_efficiency_abs: float = 0.03


def monte_carlo_savings(
    df: pd.DataFrame,
    rotor: RotorConfig,
    ship: ShipConfig,
    coefficients: RotorCoefficients,
    uncertainty: InputUncertainty | None = None,
    n_samples: int = 500,
    seed: int = 1,
    rho_air: float = RHO_AIR,
) -> pd.DataFrame:
    """Propagate input uncertainties through the savings model.

    For each realisation k::

        V_tw,k  = V_tw * (1 + e_w),   e_w  ~ N(0, wind_speed_rel)
        CL_k    = CL   * (1 + e_L),   e_L  ~ N(0, lift_coefficient_rel)
        eta_D,k = eta_D + e_eta,      e_eta ~ N(0, propulsive_efficiency_abs)

    and the mean net saving and percentage saving over ``df`` are recomputed.

    Returns:
        One row per realisation with the sampled factors and
        ``mean_net_saving_kw`` and ``percent_saving``.
    """
    unc = uncertainty or InputUncertainty()
    rng = np.random.default_rng(seed)
    wind_factor = 1.0 + rng.normal(0.0, unc.wind_speed_rel, n_samples)
    cl_factor = 1.0 + rng.normal(0.0, unc.lift_coefficient_rel, n_samples)
    eta = np.clip(ship.propulsive_efficiency + rng.normal(0.0, unc.propulsive_efficiency_abs, n_samples), 0.3, 0.9)

    speed = df["ship_speed_ms"].to_numpy()
    heading = df["heading_deg"].to_numpy()
    tws = df["true_wind_speed_ms"].to_numpy()
    twd = df["true_wind_dir_deg"].to_numpy()
    baseline_mean = float(np.mean(ship.baseline_power_kw(speed)))

    rows = []
    for k in range(n_samples):
        ship_k = ShipConfig(**{**ship.__dict__, "propulsive_efficiency": float(eta[k])})
        pred = predict_savings(speed, heading, tws * wind_factor[k], twd, rotor, ship_k,
                               coefficients.scaled(cl_factor=float(cl_factor[k])), rho_air)
        mean_net = float(pred["net_saving_kw"].mean())
        rows.append({
            "realisation": k,
            "wind_speed_factor": wind_factor[k],
            "cl_factor": cl_factor[k],
            "propulsive_efficiency": eta[k],
            "mean_net_saving_kw": mean_net,
            "percent_saving": 100.0 * mean_net / baseline_mean,
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class SanityFlag:
    """A physically suspicious result found by :func:`sanity_check`."""

    severity: str  # "warning" or "error"
    check: str
    message: str
    n_affected: int


def sanity_check(
    predictions: pd.DataFrame,
    coefficients: RotorCoefficients,
    max_plausible_percent: float = 30.0,
) -> list[SanityFlag]:
    """Flag results that are physically unreasonable or outside model validity.

    ``predictions`` is the output of :func:`~rotor_sail_performance.savings.predict_savings`
    (must contain ``baseline_power_kw``). Checks:

    * non-finite values in key outputs (error);
    * negative rotor power (error);
    * net saving larger than the baseline propulsion power, i.e. the rotors
      would push the ship faster than its own propeller - the constant
      eta_D model is invalid there (error);
    * spin ratio above the coefficient table (silent clamping) (warning);
    * overall percentage saving above ``max_plausible_percent`` - published
      full-scale rotor results are mostly single-digit to low-double-digit
      percentages (warning);
    * overall net saving negative, suggesting a wrong control strategy (warning).
    """
    flags: list[SanityFlag] = []
    key = predictions[["net_saving_kw", "thrust_n", "rotor_power_kw"]]
    n_bad = int((~np.isfinite(key.to_numpy())).any(axis=1).sum())
    if n_bad:
        flags.append(SanityFlag("error", "finite", "non-finite thrust, power or saving", n_bad))
    n_neg = int((predictions["rotor_power_kw"] < 0).sum())
    if n_neg:
        flags.append(SanityFlag("error", "rotor_power", "negative rotor power", n_neg))
    moving = predictions["baseline_power_kw"] > 0
    n_over = int((moving & (predictions["net_saving_kw"] > predictions["baseline_power_kw"])).sum())
    if n_over:
        flags.append(SanityFlag("error", "saving_exceeds_baseline",
                                "net saving exceeds baseline propulsion power", n_over))
    n_extrap = int((predictions["spin_ratio"] > coefficients.max_spin_ratio + 1e-9).sum())
    if n_extrap:
        flags.append(SanityFlag("warning", "spin_ratio_range",
                                f"spin ratio above table maximum {coefficients.max_spin_ratio}", n_extrap))
    total_base = float(predictions.loc[moving, "baseline_power_kw"].sum())
    if total_base > 0:
        percent = 100.0 * float(predictions.loc[moving, "net_saving_kw"].sum()) / total_base
        if percent > max_plausible_percent:
            flags.append(SanityFlag("warning", "percent_saving",
                                    f"overall saving {percent:.1f} % exceeds plausible {max_plausible_percent:.0f} %",
                                    int(moving.sum())))
        if percent < 0:
            flags.append(SanityFlag("warning", "negative_saving",
                                    f"overall saving is negative ({percent:.1f} %)", int(moving.sum())))
    return flags


def plot_predicted_vs_measured(df: pd.DataFrame, predicted_col: str = "predicted_net_saving_kw",
                               measured_col: str = "measured_net_saving_kw") -> Figure:
    """Scatter of measured vs predicted net saving with the 1:1 line."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(df[predicted_col], df[measured_col], s=4, alpha=0.25, color="tab:blue", rasterized=True)
    lo = float(np.nanmin([df[predicted_col].min(), df[measured_col].min()]))
    hi = float(np.nanmax([df[predicted_col].max(), df[measured_col].max()]))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, label="1:1")
    m = compare(df[predicted_col], df[measured_col])
    ax.set_xlabel("Predicted net saving [kW]")
    ax.set_ylabel("Measured net saving [kW]")
    ax.set_title(f"Predicted vs measured (n={m.n}, bias={m.bias:.0f} kW, RMSE={m.rmse:.0f} kW)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals_by_awa(df: pd.DataFrame, binned: pd.DataFrame, predicted_col: str = "predicted_net_saving_kw",
                          measured_col: str = "measured_net_saving_kw",
                          awa_col: str = "apparent_wind_angle_deg") -> Figure:
    """Residuals vs |AWA| with per-bin bias and RMSE."""
    fig, ax = plt.subplots(figsize=(8, 5))
    residual = df[measured_col] - df[predicted_col]
    ax.scatter(df[awa_col].abs(), residual, s=4, alpha=0.2, color="tab:gray", rasterized=True, label="samples")
    centres = 0.5 * (binned["awa_bin_low_deg"] + binned["awa_bin_high_deg"])
    ax.errorbar(centres, binned["bias"], yerr=binned["rmse"], fmt="o", color="tab:red", capsize=4,
                label="bin bias ± RMSE")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xlabel("|Apparent wind angle| [deg]")
    ax.set_ylabel("Measured - predicted [kW]")
    ax.set_xticks(AWA_BINS_DEG)
    ax.set_title("Residuals binned by apparent wind angle")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monte_carlo(samples: pd.DataFrame, measured_mean_ci: BootstrapResult | None = None) -> Figure:
    """Histogram of Monte Carlo mean savings, optionally with the measured bootstrap CI."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(samples["mean_net_saving_kw"], bins=40, color="tab:blue", alpha=0.7, label="model (Monte Carlo)")
    lo, hi = np.quantile(samples["mean_net_saving_kw"], [0.025, 0.975])
    ax.axvspan(lo, hi, color="tab:blue", alpha=0.1, label="model 95 % interval")
    if measured_mean_ci is not None:
        ax.axvline(measured_mean_ci.mean, color="tab:red", lw=2, label="measured mean")
        ax.axvspan(measured_mean_ci.lower, measured_mean_ci.upper, color="tab:red", alpha=0.2,
                   label=f"measured {measured_mean_ci.confidence:.0%} bootstrap CI")
    ax.set_xlabel("Mean net saving over clean samples [kW]")
    ax.set_ylabel("Realisations")
    ax.set_title("Uncertainty of mean net saving")
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig
