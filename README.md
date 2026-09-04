# F1 Strategy Simulator 2026

A Formula 1 race strategy analysis tool using Bayesian tire modeling, closed-form pit-window optimization, and Monte Carlo simulation, updated for the 2026 regulation era. Built with Dash and Plotly.

A strategy is a compound sequence; the simulator solves the optimal pit lap for each sequence, reports the window of laps statistically equivalent to it, and compares strategies at their own optima.

## Methodology

### Tire Degradation Model

Tire performance follows a quadratic degradation form:

```
mu = alpha + beta * lap + gamma * lap^2
```

The quadratic term captures two regimes of tire degradation: linear wear that dominates early in a stint, and accelerating cliff degradation as tire age increases. Parameters {alpha, beta, gamma, sigma, rho} are compound-specific.

When trained posterior samples exist for a circuit-compound pair (from MCMC inference on historical stint data), parameters are drawn from the posterior distribution. Otherwise, parameters are drawn from informative priors calibrated against historical compound degradation ranges. Both modes use the same model structure; the prior represents the state of belief before observing circuit-specific data.

Parameters are drawn **once per simulation per compound**. This correctly separates epistemic uncertainty (we don't know the true degradation rate; different simulations explore different parameter values) from aleatoric uncertainty (inherent lap-to-lap randomness within a single realization of the race).

### Noise Structure

Lap-to-lap variability follows an AR(1) process:

```
epsilon_1 ~ Normal(0, sigma)
epsilon_t ~ Normal(rho * epsilon_{t-1}, sigma * sqrt(1 - rho^2))
```

The autocorrelation parameter rho captures the empirical observation that consecutive lap times are not independent: a slow lap (dirty air, lockup, suboptimal exit) tends to be followed by another slightly slow lap rather than an immediate reversion to mean pace.

### Pit Window Optimization

A strategy is a compound sequence, and the pit laps are solved rather than supplied. Expected race time is linear in the tire parameters, and the AR(1) noise, base pace perturbation, and fuel term either average out or depend only on the race lap index, so none of them can move the optimal stop. The expected-time surface over pit laps is therefore closed-form, evaluated at posterior means in posterior mode and distribution means in prior mode:

```
E[T] = sum over stints of [ alpha * len + beta * sum(k) + gamma * sum(k^2) ] + stops * pit_loss
```

where the per-stint sums run over the effective lap indices, offset by tire age when a used set is assigned. The optimizer scans the feasible stint-length partitions and takes the minimum directly, with no simulation in the loop. A minimum stint length and the tire-life caps below define feasibility. Monte Carlo then runs at the chosen optimum to produce the distribution and risk metrics.

The reported window is the set of pit laps whose expected time falls within a fraction of the race-time standard deviation of the best, given as a lap range per stop. A wide window means the timing is forgiving; a narrow one means the stop must land on a specific lap. The width is itself the result.

### Tire Set Assignment

When custom tire allocation is enabled, each stint runs on a specific physical set with its own age, and the assignment of sets to stints is solved jointly with the pit laps. Within a compound the marginal cost of tire age rises with stint length, so assigning the freshest set to the longest stint minimizes total time. This is exact rather than heuristic (it follows from the rearrangement inequality on a supermodular cost), and it decomposes per compound since a set can only fill a stint of its own compound.

### Tire Life and Extrapolation Limits

Each compound's stint length is bounded by the evidence available for it. In posterior mode the bound comes from the longest run observed in practice for that compound, extended by a modest margin; in every mode it is also capped by a nominal race life per compound. The quadratic curve is a reliable local description within the range it was fit on, and the bound keeps the optimizer from extrapolating it into stint lengths that no data supports.

This matters most for the soft compound, whose practice running is short and low-fuel, which otherwise makes long soft stints look more viable than they are. Any solved stint that reaches or exceeds the observed range or the nominal life is flagged as extrapolation-limited in the results, so the result is disclosed rather than hidden. Per-circuit tire life is derived from historical race stint lengths (see below), with global defaults applied until that derivation is run.

### Fuel Correction

The tire model is fit on fuel-corrected data, so alpha represents a zero-fuel lap time. The simulator reconstructs on-track pace by adding the fuel mass carried on each lap back in:

```
laptime = model_laptime + (total_laps - current_lap + 1) * fuel_per_lap * weight_effect
```

Early laps carry the most fuel and are correctly the slowest for a given tire state, with the penalty falling to near zero on the final lap. Because the fuel term depends only on the race lap index, it is identical across every strategy over the same distance and does not affect strategy ranking, win rates, or pit windows.

Fuel load is not regulated as a starting mass in 2026. The limit is energy-based: a 3000 MJ/h flow cap replaces the former 100 kg/h mass-flow rule, with no mandated race fuel quantity. The simulator uses a representative observed start-of-race load of 92.5 kg (89.5 kg usable after 3 kg reserve). The 2026 sustainable fuel has lower energy density (38-41 MJ/kg), so cars carry more mass for a given race energy than the early 70 kg projections suggested. Weight effect fixed at 0.03 s/kg/lap.

### Base Pace

In posterior mode the fitted alpha carries absolute pace per compound, so base pace is not used for a fitted compound. It anchors two cases. On circuits with no fit it is the zero-fuel reference lap for a fresh medium. On circuits with a partial fit, a compound that was not fit is anchored to a base pace derived from the fitted alphas, so all compounds share one scale rather than mixing the fitted zero-fuel scale with a with-fuel prior.

Per-circuit values are held as representative race laps and converted to a zero-fuel reference by removing one full fuel load of lap time (89.5 kg times 0.03 s/kg, which is 2.685 s, uniform across circuits). A `base_pace.json` produced from fuel-corrected historical race pace overrides these with directly derived values.

### Monte Carlo Engine

Each simulation draws:
- A base pace perturbation from Normal(0, 0.4) representing session-level variation
- One parameter set per compound from the posterior or prior
- AR(1) noise per lap within each stint

Simulation runs at the optimized pit split for each strategy. Results are aggregated across simulations to produce empirical distributions for percentile-based risk metrics, median performance comparisons, and head-to-head win rate calculations.

### Sensitivity Analysis

After the base simulation, a sensitivity analysis tests whether the ranking is robust to input uncertainty. Two parameters are swept, independently and jointly: pit loss (default plus or minus 4s in 0.5s steps) and a degradation multiplier (0.70x to 1.30x in 0.05 steps) that scales beta and gamma together while preserving their ratio. Every point in the sweep re-solves the optimal pit split for each strategy, so strategies are compared at their own optima throughout rather than at a fixed split, and because the optimization is closed-form this adds little cost.

The pit loss and degradation tabs show median race time with the crossover points where the optimal strategy changes. The Optimal Pit Lap tab shows how each strategy's solved pit lap and its window respond to degradation. Pit loss changes which stop count is fastest overall, which appears as strategies trading rank rather than as pit laps moving, since the pit lap within a fixed sequence is degradation-driven. A coarser 2D sweep maps which strategy is optimal across the joint parameter space.

## 2026 Regulations

- Energy-based fuel limit: 3000 MJ/h flow cap (no regulated starting fuel mass), sustainable fuel with lower energy density (38-41 MJ/kg)
- Pirelli C1-C5 compounds (C6 removed), narrower construction
- 768 kg minimum car weight, per published technical regulations (down from 798 kg)
- ~15-30% downforce reduction, ~55% drag reduction
- Active aerodynamics replacing DRS
- 50/50 ICE/electric power split, MGU-H removed, MGU-K tripled to 350 kW
- 24 races: Imola removed, Madrid added

## Features

- All 24 circuits on the 2026 calendar with circuit-specific base pace and pit loss defaults
- 11 compound sequences (1-stop and 2-stop) with pit laps solved by closed-form optimization per circuit
- Optimal pit window per strategy, reported as a lap range per stop, with an extrapolation-limited flag when a stint exceeds the supported range
- Custom strategy editor to override the solved split per stint
- Custom tire allocation with age tracking and exact freshest-to-longest set assignment
- Performance distribution, box plot, CDF, and comparison visualizations
- Summary statistics, risk analysis, and head-to-head win rate tables
- Sensitivity analysis with per-point re-optimization, pit-loss and degradation sweeps, crossover detection, an optimal pit lap view, and a 2D heatmap
- CSV export for raw simulation data and summary statistics

## Deployment

Built with Dash and deployed on Railway with Gunicorn.

```
pip install -r requirements.txt
gunicorn app:server --bind 0.0.0.0:$PORT
```

Optional data files are loaded at startup when present: `tire_life.json` (per-circuit nominal tire life) and `base_pace.json` (per-circuit zero-fuel base pace). The app runs without them, falling back to global nominal-life defaults and the fuel-load base-pace conversion.

## Fitting Models from Practice Data

A separate pipeline script fits Bayesian tire models from FP2 (or other session) data. This is run locally after practice sessions, and the resulting pickle files are committed to `prebuilt_models/` for the deployed app to load.

### Setup

```
pip install -r requirements-pipeline.txt
```

### Usage

```
python fit_models.py australia --year 2026 --session FP2
```

This will:
1. Load the FP2 session via FastF1
2. Extract stint data, filter outlaps/inlaps/slow laps, require minimum 4-lap stints
3. Fuel-correct lap times (assumes ~60% of a 92.5 kg load for FP2 runs)
4. Fit the quadratic Bayesian model per compound via MCMC (NUTS sampler)
5. Estimate AR(1) autocorrelation from residuals
6. Record the fitted stint-lap support (min and max observed stint lap) per compound
7. Save posterior samples to `prebuilt_models/{circuit}_models.pkl`

The support boundary recorded in step 6 is what the app uses to cap and flag stint lengths beyond the range the model was fit on.

### Options

```
--year          Season year (default: 2026)
--session       FP1, FP2, FP3, or R (default: FP2)
--output-dir    Output directory (default: prebuilt_models)
--cache-dir     FastF1 cache (default: .f1_cache)
--warmup        MCMC warmup samples (default: 500)
--samples       MCMC posterior samples (default: 1000)
```

### Race Weekend Workflow

1. FP2 ends Friday afternoon
2. Run `python fit_models.py {circuit} --year 2026`
3. Verify output: check printed diagnostics for reasonable alpha/beta/gamma/sigma values
4. Commit the updated pickle file and redeploy
5. App switches from "Prior model" to "Posterior model" for that circuit

### Deriving Tire Life and Base Pace

A second pipeline script derives per-circuit values from historical race data and writes them to JSON files the app loads at startup. This is run occasionally rather than per weekend, since the values are stable across a season.

```
python compute_tire_life.py
```

This pools race sessions across the given seasons and, for each circuit, computes a nominal tire life per compound from a high quantile of observed race stint lengths (written to `tire_life.json`) and a zero-fuel base pace from fuel-corrected low-tire-age race laps (written to `base_pace.json`). Run it locally rather than on a GitHub runner, which is IP-blocked from the livetiming API.

```
--start-year    First season to pool (default: 2019)
--end-year      Last season to pool (default: 2025)
--quantile      Stint-length quantile for nominal life (default: 0.90)
--min-stint     Minimum stint length to count (default: 5)
--exclude-2022  Skip 2022 (default), --include-2022 to keep it
--base-pace-age Max tire age for base-pace laps (default: 6)
```

Restrict to specific circuits by passing their keys, for example `python compute_tire_life.py hungary monaco`. The global nominal-life defaults and the fuel-load base-pace conversion apply for any circuit not present in the JSON files.

### Automated Updates via GitHub Actions

A workflow in `.github/workflows/update-models.yml` automates the model-fitting pipeline. Go to the Actions tab, select "Update Tire Models," choose the circuit and session from the dropdowns, and run. The workflow installs pipeline dependencies, runs `fit_models.py`, commits the updated model files, and pushes. Railway redeploys automatically on the new commit.

Requires write permissions for Actions: Settings -> Actions -> General -> Workflow permissions -> "Read and write permissions."

GitHub-hosted runners are periodically IP-blocked by the livetiming API. When a run fails within seconds of starting, run `fit_models.py` locally and commit the resulting pickle by hand.

### Available circuits

```
australia, china, japan, bahrain, saudi_arabia, miami, canada,
monaco, barcelona_catalunya, austria, britain, belgium, hungary,
netherlands, italy, madrid, azerbaijan, singapore, united_states,
mexico, brazil, las_vegas, qatar, abu_dhabi
```

### Fuel Correction Assumptions

FP2 fuel loads are unknown. The pipeline assumes teams start long runs at approximately 60% of race fuel (about 55 kg, against a 92.5 kg representative race load). This is a reasonable midpoint; actual loads vary by team and run plan. The intercept parameter (alpha) absorbs most fuel-related baseline shift, so moderate errors in assumed fuel load affect the absolute lap time level but not the degradation slope (beta, gamma) which is what the simulator primarily uses.

## Configuration Constants

Set at the top of `app.py`:

- `MIN_STINT_LAPS` (5): shortest stint the optimizer will consider
- `SUPPORT_CAP_MULT` (1.5): margin applied to the fitted long-run support when capping stint length
- `EXTRAP_FLAG_FRAC` (1.0): fraction of the support or nominal life at which a stint is flagged extrapolation-limited
- `WINDOW_TOL_FRAC` (0.25): fraction of race-time standard deviation defining the pit window
- `COMPOUND_NOMINAL_LIFE`: global fallback tire life per compound, overridden per circuit by `tire_life.json`

## Known Limitations

- Quadratic model is more expressive than linear but still a parametric approximation; real degradation has thermal and compound-chemistry dynamics not captured by lap count alone
- Weight effect coefficient (0.03 s/kg) is fixed across circuits; in practice it varies with corner speed profile
- Prior parameters are informed by historical ranges but not formally fit to pre-2026 data via hierarchical pooling (planned for Phase 2 after 2026 race data becomes available)
- The optimizer minimizes expected race time using posterior or prior means, so it targets the mean-optimal stop rather than a median or risk-adjusted one
- The window tolerance is a fixed fraction of race-time dispersion rather than a formal significance threshold
- Nominal tire life defaults are domain priors until per-circuit values are derived from historical stint lengths
- No modeling of track evolution, safety car, traffic, or weather
- AR(1) autocorrelation parameter rho is estimated post-hoc from model residuals rather than jointly within the MCMC; a fully Bayesian treatment would include rho in the generative model
- Base pace sigma (0.4s) represents inter-simulation variation but is not estimated from historical session-to-session variance
- The 92.5 kg fuel load is a representative observed value, not a regulated or per-circuit figure; real loads vary by circuit, team, and run plan

## Changelog

- Pit window optimization. Strategies are now compound sequences with the pit laps solved by closed-form minimization of expected race time rather than supplied or lap-scaled. Adds an optimal pit window per strategy, exact freshest-to-longest tire-set assignment, and tire-life caps with an extrapolation-limited flag that guards against long soft stints unsupported by practice data. Sensitivity re-optimizes at every sweep point so strategies are compared at their own optima, and adds an Optimal Pit Lap view. A second pipeline script (`compute_tire_life.py`) derives per-circuit tire life and zero-fuel base pace from historical race data.

- Fuel correction sign. The simulator now adds the carried fuel mass to the zero-fuel model lap time, so heavy early laps are correctly the slowest, correcting an earlier convention that subtracted it. Base pace is redefined as a zero-fuel reference to match. As with the fuel-load change, the fuel term is identical across strategies, so this shifts absolute race times by a constant but leaves rankings, win rates, and pit windows unchanged.

- Fuel load revised from 70 kg to a representative observed start-of-race load of 92.5 kg. The 2026 regulations impose no mandated starting fuel mass; the limit is energy-based (3000 MJ/h flow cap), and the lower energy density of the sustainable fuel means real loads run higher than the early 70 kg projections. Revised following practitioner feedback on real-world fuel loads. The fuel correction applies an identical per-lap offset to every strategy, so this affects absolute race times but not relative strategy comparisons, win rates, or risk metrics.

## Contact

jessica.5t3313@gmail.com
