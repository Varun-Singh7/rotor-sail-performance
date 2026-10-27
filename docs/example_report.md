# Rotor sail performance assessment

*Generated 2026-10-27 19:06 UTC by `examples/run_assessment.py`
(seed 42, 90 days, 500 Monte Carlo realisations).*

> **All data in this report are synthetic and the rotor coefficients are an
> illustrative placeholder, not measurements.** The numbers demonstrate the
> method; they say nothing about any real ship or rotor product.

## Installation assessed

| Item | Value |
|---|---|
| Rotors | 2 x 30 m x 5 m, target spin ratio 3.0, max 180 rpm |
| Coefficients | bundled placeholder (rotor_coefficients_illustrative.csv) - illustrative, not measured |
| Propulsive efficiency | 0.7 |
| SFOC | 175 g/kWh |
| CO2 factor | 3.114 t CO2 / t fuel |
| Baseline | 7000 kW at 12.0 kn, P ~ V^3 |

## Method

1. **Data**: 8640 samples at 15-min intervals with injected dropouts, frozen sensors and outliers.
2. **Quality control**: missing values, range checks, frozen-sensor detection, Hampel outlier filter and a
   steady-state filter. 5603 samples (64.8 %) were kept.

| QC step | rows removed |
|---|---|
| missing | 706 |
| range | 19 |
| frozen | 189 |
| outlier | 79 |
| steady | 2044 |

3. **Prediction**: wind triangle -> spin ratio -> CL, CD, Cp lookup -> thrust along the heading ->
   P_net = T V / eta_D - P_rotor. Rotors are stopped whenever that gives a larger net saving.
4. **"Measured" saving**: P_baseline(V) - P_shaft - P_rotor, i.e. the shortfall of shaft power below the
   calm-water speed-power curve, minus rotor consumption.
5. **Verification**: residual statistics overall and per apparent wind angle bin; moving-block bootstrap
   (block = 96 samples = 1 day) for confidence intervals of means; Monte Carlo propagation of
   systematic input uncertainties (wind speed 5 %, CL
   10 %, eta_D +-0.03, all 1 sigma).

## Results

| Quantity | Estimate | 95 % interval |
|---|---|---|
| Mean measured net saving | 277 kW | 188 to 365 kW (bootstrap) |
| Mean predicted net saving | 287 kW | 176 to 415 kW (Monte Carlo, input uncertainty) |
| Bias (measured - predicted) | -10 kW | -57 to 30 kW (bootstrap) |
| RMSE per 15-min sample | 344 kW | |
| Measured saving, % of baseline propulsion power | 4.0 % | 2.7 to 5.3 % |
| Predicted saving, % of baseline propulsion power | 4.1 % | 2.5 to 6.0 % (Monte Carlo) |
| Fuel saved over 1401 h of clean steady sailing | 70 t | |
| CO2 avoided over the same period | 219 t | |

Because the data are synthetic, the hidden true mean saving is known: **240 kW**. The synthetic
rotors were built with 8 % less lift and 5 % more drag than the coefficient table, so the model really
over-predicts, by 47 kW on average. The bootstrap interval of the measured bias includes zero
and contains the true bias of -47 kW. With measured savings inferred from shaft power
against a calm-water baseline, the per-sample noise (RMSE 344 kW) is large compared with the
saving itself, so a bias of this size is hard to resolve from 90 days of data.

### By apparent wind angle

| abs(AWA) bin [deg] | n | predicted [kW] | measured [kW] | bias [kW] | RMSE [kW] |
|---|---|---|---|---|---|
| 0-30 | 2464 | 3 | 23 | 20 | 350 |
| 30-60 | 1412 | 496 | 462 | -35 | 332 |
| 60-90 | 660 | 578 | 581 | 3 | 379 |
| 90-120 | 543 | 584 | 517 | -67 | 338 |
| 120-150 | 322 | 503 | 431 | -72 | 316 |
| 150-180 | 202 | 209 | 201 | -9 | 282 |

![Predicted vs measured](images/predicted_vs_measured.png)
![Residuals by AWA](images/residuals_by_awa.png)
![Uncertainty](images/uncertainty.png)
![Net saving polar](images/polar_net_saving.png)

### Sanity checks

No physically unreasonable results were flagged.

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
