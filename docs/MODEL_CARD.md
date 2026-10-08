# GridOpt: Methodology and model card

**Purpose:** Demonstrate reproducible analysis of hourly electricity demand, a 24-hour-ahead forecast model, and linear-program battery dispatch under illustrative tariff constraints.

**Data:** Default generator uses pseudorandom industrial-like load with operating-hour effects, temperature sensitivity, and occasional demand spikes. No customer data, meter data, proprietary grid information, or actual savings are contained in the repository.

**Split:** Chronological first N-30 days for fitting; last 30 days for evaluation. The first seven training days are lag warmup and not used as labeled train examples. Day-ahead prediction operates at the start of each 24-hour block and recursively uses predictions for within-day load lags. Lagged loads from previous days are observed before the forecast is issued. Model fitting never sees held-out load. The synthetic generated *temperature* for future hours is an idealized exogenous forecast with no error and must not be mistaken for genuinely available future measured temperatures.

**Metrics:** MAE, RMSE, seasonal-naive MAE, baseline versus storage bill, realized peak demand (kW), percent reduction, and a simplified wear-adjusted bill savings calculation. Cost figures depend entirely on assumed battery and tariffs, and are not estimates of actual business savings.

**Optimizer:** Daily 24-hour scipy.optimize.linprog (HiGHS). SOC updates account for split round-trip efficiency. A single billed peak variable per day has a lower bound equal to the peak incurred to date, and no-export constraints apply to forecast. Decisions are replayed on *realized* load with a simple load-spike peak clipping rule, discharge limited to prevent grid export, and power/energy bounds checked.

**Leakage prevention:** No random data split. Training inputs at time t use strictly past load readings, timestamps, and the exogenous weather assumption. Recursive day-ahead lag handling never reads observed same-day load.

**Failure modes:** Price structure or actual load distributions may change; lagged/temperature features may not generalize; forecast errors can erase peak savings. Model depends on assumptions about no exports, linear rates, fixed battery efficiency, no simultaneous charge/discharge integer constraint, battery wear, and perfect hourly response. No system for production dispatch or regulated market operations is included.

**Reproduce:** `python -m pip install -e '.[dev]' && python -m pytest -q && python -m gridopt --seed 42 --output-dir outputs`. Outputs folder is ignored by Git. See README for option flags.
