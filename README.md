# F1 Strategy Simulator 2026

A Formula 1 race strategy analysis tool using Bayesian tire modeling and Monte Carlo simulation, updated for the 2026 regulation era. Built with Dash and Plotly.

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

### Fuel Correction

```
correction = (total_laps - current_lap) * fuel_per_lap * weight_effect
```

Fuel load: 70 kg race allowance (67 kg usable after 3 kg reserve), reflecting 2026 sustainable fuel regulations with lower energy density (38-41 MJ/kg). Weight effect fixed at 0.03 s/kg/lap.

### Monte Carlo Engine

Each simulation draws:
- A base pace perturbation from Normal(0, 0.4) representing session-level variation
- One parameter set per compound from the posterior or prior
- AR(1) noise per lap within each stint

Results are aggregated across simulations to produce empirical distributions for percentile-based risk metrics, median performance comparisons, and head-to-head win rate calculations.

## 2026 Regulations

- 70 kg fuel load, sustainable fuel with lower energy density
- Pirelli C1-C5 compounds (C6 removed), narrower construction
- 768 kg minimum car weight, per published technical regulations (down from 798 kg)
- ~15-30% downforce reduction, ~55% drag reduction
- Active aerodynamics replacing DRS
- 50/50 ICE/electric power split, MGU-H removed, MGU-K tripled to 350 kW
- 24 races: Imola removed, Madrid added

## Features

- All 24 circuits on the 2026 calendar with circuit-specific base pace and pit loss defaults
- 11 pre-defined strategies (1-stop and 2-stop) with automatic lap scaling per circuit
- Custom strategy editor for per-stint compound and lap count adjustment
- Custom tire allocation with age tracking
- Performance distribution, box plot, CDF, and comparison visualizations
- Summary statistics, risk analysis, and head-to-head win rate tables
- CSV export for raw simulation data and summary statistics

## Deployment

Built with Dash and deployed on Railway with Gunicorn.

```
pip install -r requirements.txt
gunicorn app:server --bind 0.0.0.0:$PORT
```

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
3. Fuel-correct lap times (assumes ~60% fuel load for FP2 runs)
4. Fit the quadratic Bayesian model per compound via MCMC (NUTS sampler)
5. Estimate AR(1) autocorrelation from residuals
6. Save posterior samples to `prebuilt_models/{circuit}_models.pkl`

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

### Automated Updates via GitHub Actions

A workflow in `.github/workflows/update-models.yml` automates the pipeline. Go to the Actions tab, select "Update Tire Models," choose the circuit and session from the dropdowns, and run. The workflow installs pipeline dependencies, runs `fit_models.py`, commits the updated model files, and pushes. Railway redeploys automatically on the new commit.

Requires write permissions for Actions: Settings → Actions → General → Workflow permissions → "Read and write permissions."

### Available circuits

```
australia, china, japan, bahrain, saudi_arabia, miami, canada,
monaco, barcelona_catalunya, austria, britain, belgium, hungary,
netherlands, italy, madrid, azerbaijan, singapore, united_states,
mexico, brazil, las_vegas, qatar, abu_dhabi
```

### Fuel Correction Assumptions

FP2 fuel loads are unknown. The pipeline assumes teams start long runs at approximately 60% of race fuel (42 kg). This is a reasonable midpoint; actual loads vary by team and run plan. The intercept parameter (alpha) absorbs most fuel-related baseline shift, so moderate errors in assumed fuel load affect the absolute lap time level but not the degradation slope (beta, gamma) which is what the simulator primarily uses.

## Known Limitations

- Quadratic model is more expressive than linear but still a parametric approximation; real degradation has thermal and compound-chemistry dynamics not captured by lap count alone
- Weight effect coefficient (0.03 s/kg) is fixed across circuits; in practice it varies with corner speed profile
- Prior parameters are informed by historical ranges but not formally fit to pre-2026 data via hierarchical pooling (planned for Phase 2 after 2026 race data becomes available)
- No modeling of track evolution, safety car, traffic, or weather
- AR(1) autocorrelation parameter rho is estimated post-hoc from model residuals rather than jointly within the MCMC; a fully Bayesian treatment would include rho in the generative model
- Base pace sigma (0.4s) represents inter-simulation variation but is not estimated from historical session-to-session variance

## Contact

jessica.5t3313@gmail.com
