"""End-to-end rotor sail performance assessment on synthetic data.

Steps: generate synthetic operations -> quality control -> predict savings
-> compare with "measured" savings -> uncertainty -> plots, Markdown report
and Power BI-ready CSV files.

Usage::

    python examples/run_assessment.py --out outputs
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rotor_sail_performance import KNOT, RotorCoefficients, RotorConfig, ShipConfig, estimate_voyage  # noqa: E402
from rotor_sail_performance.data_qc import clean_operational_data, generate_synthetic_operations  # noqa: E402
from rotor_sail_performance.savings import plot_savings_polar, predict_savings, savings_polar  # noqa: E402
from rotor_sail_performance.verification import (  # noqa: E402
    AWA_BINS_DEG,
    InputUncertainty,
    bootstrap_mean_ci,
    compare,
    compare_by_awa,
    measured_net_saving_kw,
    monte_carlo_savings,
    plot_monte_carlo,
    plot_predicted_vs_measured,
    plot_residuals_by_awa,
    sanity_check,
)

BLOCK_LENGTH = 96  # one day of 15-min samples for the block bootstrap


def plot_qc_overview(flagged: pd.DataFrame) -> plt.Figure:
    """Shaft power and wind speed over time, kept vs rejected samples."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    kept = flagged["qc_flag"] == ""
    for ax, col, label in [(axes[0], "shaft_power_kw", "Shaft power [kW]"),
                           (axes[1], "true_wind_speed_ms", "True wind speed [m/s]")]:
        ax.plot(flagged["timestamp"], flagged[col], color="lightgray", lw=0.6, label="raw")
        ax.scatter(flagged.loc[kept, "timestamp"], flagged.loc[kept, col], s=1.5, color="tab:blue", label="kept")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(bottom=0, top=float(np.nanpercentile(flagged["shaft_power_kw"], 99.9)) * 1.2)
    axes[1].set_ylim(0, 35)
    axes[0].legend(loc="upper right", markerscale=6)
    axes[0].set_title("Synthetic operational data: raw vs samples kept by QC")
    fig.tight_layout()
    return fig


def plot_qc_summary(log: pd.DataFrame) -> plt.Figure:
    """Rows removed per QC step (first reason only)."""
    by_step = log.groupby("step", sort=False)["n_newly_removed"].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(by_step.index[::-1], by_step.values[::-1], color="tab:orange")
    for y, v in enumerate(by_step.values[::-1]):
        ax.text(v, y, f" {v}", va="center")
    ax.set_xlabel("Rows removed (first failing check)")
    ax.set_title("Quality-control removals by step")
    fig.tight_layout()
    return fig


def markdown_table(df: pd.DataFrame, floatfmt: str = ".0f") -> str:
    """Render a DataFrame as a GitHub Markdown table (avoids a tabulate dependency)."""
    def cell(v: object) -> str:
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument("--days", type=float, default=90.0, help="length of synthetic record")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the synthetic data")
    parser.add_argument("--mc-samples", type=int, default=500, help="Monte Carlo realisations")
    parser.add_argument("--coefficients", type=Path, default=None, help="CSV with spin_ratio,cl,cd,cp")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fig_dir = args.out / "figures"
    bi_dir = args.out / "powerbi"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bi_dir.mkdir(parents=True, exist_ok=True)

    rotor = RotorConfig()
    ship = ShipConfig()
    coeffs = RotorCoefficients.from_csv(args.coefficients)
    uncertainty = InputUncertainty()

    # 1. Data and quality control.
    raw = generate_synthetic_operations(days=args.days, seed=args.seed, rotor=rotor, ship=ship, coefficients=coeffs)
    qc = clean_operational_data(raw)
    clean = qc.clean

    # 2. Predicted vs measured on the clean samples.
    pred = predict_savings(clean["ship_speed_ms"], clean["heading_deg"], clean["true_wind_speed_ms"],
                           clean["true_wind_dir_deg"], rotor, ship, coeffs)
    pred.index = clean.index
    comp = pd.DataFrame({
        "apparent_wind_angle_deg": pred["apparent_wind_angle_deg"],
        "predicted_net_saving_kw": pred["net_saving_kw"],
        "measured_net_saving_kw": measured_net_saving_kw(clean, ship),
    })
    overall = compare(comp["predicted_net_saving_kw"], comp["measured_net_saving_kw"])
    by_awa = compare_by_awa(comp)
    residual = comp["measured_net_saving_kw"] - comp["predicted_net_saving_kw"]

    # 3. Uncertainty.
    ci_meas = bootstrap_mean_ci(comp["measured_net_saving_kw"], block_length=BLOCK_LENGTH, seed=1)
    ci_pred = bootstrap_mean_ci(comp["predicted_net_saving_kw"], block_length=BLOCK_LENGTH, seed=2)
    ci_bias = bootstrap_mean_ci(residual, block_length=BLOCK_LENGTH, seed=3)
    mc = monte_carlo_savings(clean, rotor, ship, coeffs, uncertainty, n_samples=args.mc_samples, seed=4)
    mc_lo, mc_med, mc_hi = np.quantile(mc["mean_net_saving_kw"], [0.025, 0.5, 0.975])
    mcp_lo, mcp_med, mcp_hi = np.quantile(mc["percent_saving"], [0.025, 0.5, 0.975])

    # 4. Voyage totals over the clean (steady sailing) samples.
    voyage = estimate_voyage(clean, rotor, ship, coeffs, timestep_h=0.25)
    baseline_mean_kw = float(pred["baseline_power_kw"].mean())
    meas_pct = 100.0 * ci_meas.mean / baseline_mean_kw
    meas_pct_lo, meas_pct_hi = 100.0 * ci_meas.lower / baseline_mean_kw, 100.0 * ci_meas.upper / baseline_mean_kw
    flags = sanity_check(pred, coeffs)
    true_mean = float(clean["true_net_saving_kw"].mean())

    # 5. Figures.
    polar = savings_polar(12.0 * KNOT, [6.0, 9.0, 12.0, 15.0, 18.0], rotor, ship, coeffs)
    save(plot_savings_polar(polar, "net_saving_kw"), fig_dir / "polar_net_saving.png")
    save(plot_savings_polar(polar, "thrust_n"), fig_dir / "polar_thrust.png")
    save(plot_predicted_vs_measured(comp), fig_dir / "predicted_vs_measured.png")
    save(plot_residuals_by_awa(comp, by_awa), fig_dir / "residuals_by_awa.png")
    save(plot_monte_carlo(mc, ci_meas), fig_dir / "uncertainty.png")
    save(plot_qc_overview(qc.flagged), fig_dir / "qc_overview.png")
    save(plot_qc_summary(qc.log), fig_dir / "qc_summary.png")

    # 6. Power BI exports (tidy: one row per observation, one column per variable, units in names).
    fact = qc.flagged.rename(columns={"true_net_saving_kw": "synthetic_truth_net_saving_kw",
                                      "mode": "synthetic_truth_mode"})
    fact["qc_kept"] = fact["qc_flag"] == ""
    fact["qc_step"] = fact["qc_flag"].str.split(":").str[0]
    fact = fact.join(pred.add_prefix("predicted_").rename(columns={
        "predicted_apparent_wind_speed_ms": "apparent_wind_speed_ms",
        "predicted_apparent_wind_angle_deg": "apparent_wind_angle_deg",
        "predicted_baseline_power_kw": "baseline_power_kw"}))
    fact["measured_net_saving_kw"] = comp["measured_net_saving_kw"]
    fact["residual_kw"] = residual
    labels = [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(AWA_BINS_DEG[:-1], AWA_BINS_DEG[1:])]
    fact["awa_bin"] = pd.cut(fact["apparent_wind_angle_deg"].abs(), list(AWA_BINS_DEG), labels=labels,
                             include_lowest=True).astype(str).replace("nan", "")
    fact.to_csv(bi_dir / "fact_operations.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S")

    qc.log.to_csv(bi_dir / "qc_log.csv", index=False)
    by_awa.to_csv(bi_dir / "metrics_by_awa_bin.csv", index=False)
    mc.to_csv(bi_dir / "monte_carlo_samples.csv", index=False)
    polar.to_csv(bi_dir / "polar.csv", index=False)

    summary_rows = [
        ("mean_measured_net_saving", ci_meas.mean, ci_meas.lower, ci_meas.upper, "kW", "block bootstrap 95% CI"),
        ("mean_predicted_net_saving", ci_pred.mean, ci_pred.lower, ci_pred.upper, "kW", "block bootstrap 95% CI"),
        ("mean_bias_measured_minus_predicted", ci_bias.mean, ci_bias.lower, ci_bias.upper, "kW",
         "block bootstrap 95% CI"),
        ("rmse", overall.rmse, np.nan, np.nan, "kW", "all clean samples"),
        ("r2", overall.r2, np.nan, np.nan, "-", "all clean samples"),
        ("model_mean_net_saving_mc", mc_med, mc_lo, mc_hi, "kW", "Monte Carlo 2.5/50/97.5 percentiles"),
        ("model_percent_saving_mc", mcp_med, mcp_lo, mcp_hi, "%", "Monte Carlo 2.5/50/97.5 percentiles"),
        ("measured_percent_saving", meas_pct, meas_pct_lo, meas_pct_hi, "%", "bootstrap CI / mean baseline"),
        ("voyage_fuel_saved", voyage.fuel_saved_t, np.nan, np.nan, "t", "model, clean samples only"),
        ("voyage_co2_saved", voyage.co2_saved_t, np.nan, np.nan, "t", "model, clean samples only"),
        ("voyage_percent_saving", voyage.percent_saving, np.nan, np.nan, "%", "model, clean samples only"),
        ("clean_hours", voyage.duration_h, np.nan, np.nan, "h", "steady sailing after QC"),
        ("synthetic_truth_mean_net_saving", true_mean, np.nan, np.nan, "kW", "hidden truth (synthetic only)"),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value", "lower", "upper", "unit", "method"]).to_csv(
        bi_dir / "summary_metrics.csv", index=False)

    # 7. Markdown report.
    def fmt_flags() -> str:
        if not flags:
            return "No physically unreasonable results were flagged."
        return "\n".join(f"- **{f.severity}** `{f.check}`: {f.message} ({f.n_affected} samples)" for f in flags)

    qc_table = markdown_table(qc.log.groupby("step", sort=False)["n_newly_removed"].sum()
                              .rename("rows removed").rename_axis("QC step").reset_index())
    awa_table = markdown_table(by_awa[["awa_bin", "n", "mean_predicted", "mean_measured", "bias", "rmse"]]
                               .rename(columns={"awa_bin": "abs(AWA) bin [deg]", "mean_predicted": "predicted [kW]",
                                                "mean_measured": "measured [kW]", "bias": "bias [kW]",
                                                "rmse": "RMSE [kW]"}))
    true_bias = true_mean - ci_pred.mean
    detected = "excludes" if (ci_bias.upper < 0 or ci_bias.lower > 0) else "includes"
    covers_truth = "contains" if ci_bias.lower <= true_bias <= ci_bias.upper else "does not contain"

    report = f"""# Rotor sail performance assessment

*Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `examples/run_assessment.py`
(seed {args.seed}, {args.days:.0f} days, {args.mc_samples} Monte Carlo realisations).*

> **All data in this report are synthetic and the rotor coefficients are an
> illustrative placeholder, not measurements.** The numbers demonstrate the
> method; they say nothing about any real ship or rotor product.

## Installation assessed

| Item | Value |
|---|---|
| Rotors | {rotor.n_rotors} x {rotor.height_m:.0f} m x {rotor.diameter_m:.0f} m, target spin ratio {rotor.target_spin_ratio}, max {rotor.max_rpm:.0f} rpm |
| Coefficients | {coeffs.source} |
| Propulsive efficiency | {ship.propulsive_efficiency} |
| SFOC | {ship.sfoc_g_per_kwh:.0f} g/kWh |
| CO2 factor | {ship.co2_factor} t CO2 / t fuel |
| Baseline | {ship.reference_power_kw:.0f} kW at {ship.reference_speed_ms / KNOT:.1f} kn, P ~ V^{ship.speed_power_exponent:g} |

## Method

1. **Data**: {len(raw)} samples at 15-min intervals with injected dropouts, frozen sensors and outliers.
2. **Quality control**: missing values, range checks, frozen-sensor detection, Hampel outlier filter and a
   steady-state filter. {len(clean)} samples ({100 * len(clean) / len(raw):.1f} %) were kept.

{qc_table}

3. **Prediction**: wind triangle -> spin ratio -> CL, CD, Cp lookup -> thrust along the heading ->
   P_net = T V / eta_D - P_rotor. Rotors are stopped whenever that gives a larger net saving.
4. **"Measured" saving**: P_baseline(V) - P_shaft - P_rotor, i.e. the shortfall of shaft power below the
   calm-water speed-power curve, minus rotor consumption.
5. **Verification**: residual statistics overall and per apparent wind angle bin; moving-block bootstrap
   (block = {BLOCK_LENGTH} samples = 1 day) for confidence intervals of means; Monte Carlo propagation of
   systematic input uncertainties (wind speed {100 * uncertainty.wind_speed_rel:.0f} %, CL
   {100 * uncertainty.lift_coefficient_rel:.0f} %, eta_D +-{uncertainty.propulsive_efficiency_abs}, all 1 sigma).

## Results

| Quantity | Estimate | 95 % interval |
|---|---|---|
| Mean measured net saving | {ci_meas.mean:.0f} kW | {ci_meas.lower:.0f} to {ci_meas.upper:.0f} kW (bootstrap) |
| Mean predicted net saving | {ci_pred.mean:.0f} kW | {mc_lo:.0f} to {mc_hi:.0f} kW (Monte Carlo, input uncertainty) |
| Bias (measured - predicted) | {ci_bias.mean:.0f} kW | {ci_bias.lower:.0f} to {ci_bias.upper:.0f} kW (bootstrap) |
| RMSE per 15-min sample | {overall.rmse:.0f} kW | |
| Measured saving, % of baseline propulsion power | {meas_pct:.1f} % | {meas_pct_lo:.1f} to {meas_pct_hi:.1f} % |
| Predicted saving, % of baseline propulsion power | {voyage.percent_saving:.1f} % | {mcp_lo:.1f} to {mcp_hi:.1f} % (Monte Carlo) |
| Fuel saved over {voyage.duration_h:.0f} h of clean steady sailing | {voyage.fuel_saved_t:.0f} t | |
| CO2 avoided over the same period | {voyage.co2_saved_t:.0f} t | |

Because the data are synthetic, the hidden true mean saving is known: **{true_mean:.0f} kW**. The synthetic
rotors were built with 8 % less lift and 5 % more drag than the coefficient table, so the model really
over-predicts, by {-true_bias:.0f} kW on average. The bootstrap interval of the measured bias {detected} zero
and {covers_truth} the true bias of {true_bias:.0f} kW. With measured savings inferred from shaft power
against a calm-water baseline, the per-sample noise (RMSE {overall.rmse:.0f} kW) is large compared with the
saving itself, so a bias of this size is hard to resolve from {args.days:.0f} days of data.

### By apparent wind angle

{awa_table}

![Predicted vs measured](figures/predicted_vs_measured.png)
![Residuals by AWA](figures/residuals_by_awa.png)
![Uncertainty](figures/uncertainty.png)
![Net saving polar](figures/polar_net_saving.png)

### Sanity checks

{fmt_flags()}

## Key assumptions

- Placeholder CL/CD/Cp table; rotor-rotor interaction, wind shear and ship motions ignored.
- Constant propulsive efficiency and SFOC; side-force-induced leeway and rudder drag ignored.
- Calm-water power law used as the baseline; weather, fouling and draught effects end up in the residual.
- Monte Carlo errors are treated as systematic (one draw per realisation) and independent of each other.

## Limitations

- Synthetic data cannot validate the physics: agreement here only shows the pipeline is self-consistent.
- The "measured" saving depends entirely on the baseline speed-power curve; a real campaign needs a calibrated
  baseline or rotors-on/off comparisons (ITTC / IMO MEPC.1/Circ.896 approach).
- Clean-sample totals cover only steady sailing and are not annualised.
"""
    (args.out / "performance_report.md").write_text(report, encoding="utf-8")
    logging.info("wrote report, %d figures and Power BI CSVs to %s", len(list(fig_dir.glob("*.png"))), args.out)


if __name__ == "__main__":
    main()
