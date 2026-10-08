# GridOpt

**Reproducible day-ahead load forecasting + constrained battery dispatch optimization** for commercial electricity demand. A portfolio research prototype illustrating how grid data can inform battery-storage decisions.

> **Disclosure:** The default load profile and electricity tariff are **synthetic**, not measured grid/utility data. Results are an illustrative simulation, **not** verified savings, tariff advice, or a deployable battery control system. No claims of utility partnerships or actual battery deployments are made.

## What it does

1. **Validate** hourly metered loads and optional weather series (or generate a seeded 120-day synthetic example).
2. **Forecast demand** for each 24-hour interval with a histogram gradient-boosted regression model trained chronologically on historical lag, rolling-mean, calendar, and temperature features. No future observed load is used at day-ahead prediction time.
3. **Measure forecast quality** on a held-out 30-day period with MAE, RMSE, and a yesterday-at-the-same-hour seasonal-naive baseline.
4. **Optimize battery dispatch** each morning with SciPy HiGHS linear programming. Decision variables are charge/discharge (kW) and the billed grid-demand peak. Constraints include round-trip efficiency, energy capacity, charge/discharge limits, 10%-90% SOC, non-export, and a daily terminal SOC requirement.
5. **Replay against realized load**, adding a simple feedback rule to discharge against surprise load spikes where energy is available, tracking the actual grid peak, and never exporting.
6. **Compare costs** against a no-storage baseline: TOU energy ($/kWh) + a 30-day demand charge ($/kW), less simplified battery throughput wear.
7. **Export evidence** to `metrics.json`, `dispatch_timeseries.csv`, and `overview.png`.

## Quickstart

```bash
python -m pip install -e '.[dev]'
python -m gridopt --output-dir outputs
python -m pytest -q
```

### Reproducible synthetic demonstration (seed 42)

After a chronological 90-day training period and 30-day day-ahead evaluation:

| Measure | Result on *synthetic* load + illustrative tariff |
| --- | ---: |
| Day-ahead forecast MAE | 14.70 kW |
| Yesterday-same-hour baseline MAE | 29.05 kW |
| Baseline 30-day bill | $65,361.76 |
| Simulated bill after battery | $62,519.70 |
| Battery throughput wear proxy | $124.98 |
| **Net modeled bill savings** | **$2,717.07 (4.16%)** |
| **Modeled peak-demand reduction** | **7.65%** |

These figures are simulation outputs only. They are not commercial customer savings or independently validated grid performance.

## Bring your own interval meter data

Create a CSV with at least **65 complete consecutive 24-hour days**:

```csv
timestamp,load_kw,temperature_c
2026-01-01 00:00:00,371.2,-2.0
2026-01-01 01:00:00,360.8,-2.3
```

Every timestamp must be continuous, one hour apart, and timezone-naive; normalize DST and timezones outside this demo. Temperature is optional (zero is supplied as a placeholder if omitted). For a real day-ahead operational interpretation, temperatures on forecast days must be *predicted weather*, not observed future weather. The default utility tariff is **still synthetic** when loading measured demand; replace it with the actual site's rate schedule before interpreting costs.

```bash
python -m gridopt --input-csv meter.csv --output-dir outputs/meter \
  --capacity-kwh 600 --power-kw 250 --demand-charge 24
```

### Model and financial assumptions

| Assumption | Demo value |
| --- | --- |
| Synthetic data | 120 days (90 training / 30 testing) |
| Forecast cadence | 24-hour day-ahead, hourly resolution |
| Forecast features | 1h/24h/168h lags, 24h rolling mean, time/calendar, temperature |
| Battery | 600 kWh, 250 kW, 90% round-trip efficiency |
| Allowed SOC | 10% to 90%; begin and end each day at 50% |
| TOU electricity rate | $0.08 overnight, $0.14 other, $0.28 weekday peak |
| Demand charge | $24/kW over the 30-day test window |
| Wear proxy | $0.01/kWh, applied to half of charge + discharge throughput |
| Financial reporting | Simplified monthly bill savings after throughput wear, before battery capex/O&M |

### Optimization math

For each day, minimize `sum(price[t] * (load_forecast[t] + charge[t] - discharge[t])) + demand_charge * peak + wear_cost` subject to:

- `grid_import[t] = forecast_load[t] + charge[t] - discharge[t]`, `0 <= grid_import[t] <= peak`.
- `SOC[t+1] = SOC[t] + sqrt(eta_rt) * charge[t] - discharge[t] / sqrt(eta_rt)` (1-hour timestep).
- `SOC_min <= SOC[t] <= SOC_max`, `0 <= charge[t], discharge[t] <= power_kw`.
- `peak >= highest_realized_grid_import_so_far_in_this_cycle`.
- `SOC[end_of_day] = SOC[start_of_day] = 50% capacity`.

The objective includes the historical monthly peak as a lower bound, so a new day's forecast cannot erase already incurred demand charges. Linear programming does not enforce an explicit binary no-simultaneous-charge/discharge condition, but with this model's nonnegative prices, losses, and positive wear cost, simultaneous operation is economically dominated. This would require additional controls for real-world deployment.

## Limitations and next experiments

- The default 30-day period is **one illustrative billing cycle**, not a specific utility's calendar billing period, ratchet, demand-response, or export tariff.
- `temperature_c` is treated as a known exogenous 24-hour weather forecast. Synthetic data uses its realized series as an idealized forecast, so realistic weather forecast error is not modeled.
- Hourly LP replay is not a live controller. Inverters, ramp-rate, degradation physics, state-of-health, emissions, tariff tax/fees, outages, and electricity market bids are excluded.
- Dispatch is re-planned once per day with a basic hour-by-hour peak-clipping feedback rule. Simulation tracks **realized** load and avoids grid export, but correcting for forecast errors can cause a nonzero end-of-day SOC deviation; full model predictive control is a logical improvement.
- Validate on open real utility demand, temperature forecasts, and published TOU/demand tariffs; then add rolling calibration, uncertainty-aware scheduling, and multi-site scenario evaluation.
- The prototype makes no claim of optimal operating profit or measured reductions from a real facility.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the leakage policy, reproducibility details, and evaluation caveats.

## License

MIT. Author: Adam Christley.
