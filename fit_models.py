import argparse
import pickle
import sys
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

jax.config.update("jax_platform_name", "cpu")
numpyro.set_host_device_count(1)

CIRCUIT_LOOKUP = {
    "australia": "Australia",
    "china": "China",
    "japan": "Japan",
    "bahrain": "Bahrain",
    "saudi_arabia": "Saudi Arabia",
    "miami": "Miami",
    "canada": "Canada",
    "monaco": "Monaco",
    "barcelona_catalunya": "Barcelona-Catalunya",
    "austria": "Austria",
    "britain": "Britain",
    "belgium": "Belgium",
    "hungary": "Hungary",
    "netherlands": "Netherlands",
    "italy": "Italy",
    "madrid": "Madrid",
    "azerbaijan": "Azerbaijan",
    "singapore": "Singapore",
    "united_states": "United States",
    "mexico": "Mexico",
    "brazil": "Brazil",
    "las_vegas": "Las Vegas",
    "qatar": "Qatar",
    "abu_dhabi": "Abu Dhabi",
}

CIRCUIT_LAPS = {
    "Australia": 58, "China": 56, "Japan": 53, "Bahrain": 57,
    "Saudi Arabia": 50, "Miami": 57, "Canada": 70, "Monaco": 78,
    "Barcelona-Catalunya": 66, "Austria": 71, "Britain": 52,
    "Belgium": 44, "Hungary": 70, "Netherlands": 72, "Italy": 53,
    "Madrid": 57, "Azerbaijan": 51, "Singapore": 62,
    "United States": 56, "Mexico": 71, "Brazil": 71,
    "Las Vegas": 50, "Qatar": 57, "Abu Dhabi": 58,
}

COMPOUND_PRIORS = {
    "SOFT": {
        "alpha_mu": 85.0, "alpha_sigma": 8.0,
        "beta_mu": 0.07, "beta_sigma": 0.02,
        "gamma_mu": 0.0015, "gamma_sigma": 0.0008,
        "sigma_scale": 0.4,
    },
    "MEDIUM": {
        "alpha_mu": 86.0, "alpha_sigma": 8.0,
        "beta_mu": 0.04, "beta_sigma": 0.015,
        "gamma_mu": 0.0008, "gamma_sigma": 0.0005,
        "sigma_scale": 0.35,
    },
    "HARD": {
        "alpha_mu": 86.5, "alpha_sigma": 8.0,
        "beta_mu": 0.02, "beta_sigma": 0.008,
        "gamma_mu": 0.0004, "gamma_sigma": 0.0003,
        "sigma_scale": 0.3,
    },
}

FUEL_LOAD_KG = 70.0
FUEL_RESERVE_KG = 3.0
WEIGHT_EFFECT = 0.03
FP2_ESTIMATED_FUEL_FRACTION = 0.6


def load_session(year, circuit_name, session_type="FP2", cache_dir=".f1_cache"):
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, circuit_name, session_type)
    session.load()
    return session


def extract_stints(session, circuit_name):
    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()].copy()
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps = laps[laps["LapTime_s"] > 0].copy()

    laps = laps[~laps["Compound"].isin(["INTERMEDIATE", "WET", "UNKNOWN"])].copy()
    laps = laps[laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()

    if len(laps) == 0:
        return pd.DataFrame()

    laps = laps[~laps["IsPersonalBest"].isna()].copy()
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    stints = []
    for driver, driver_laps in laps.groupby("Driver"):
        driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)

        stint_id = 0
        prev_compound = None
        prev_lap = None

        for idx, row in driver_laps.iterrows():
            compound = row["Compound"]
            lap_num = row["LapNumber"]

            if prev_compound is None:
                stint_id = 0
            elif compound != prev_compound or (prev_lap and lap_num - prev_lap > 2):
                stint_id += 1

            stints.append({
                "Driver": driver,
                "LapNumber": lap_num,
                "LapTime_s": row["LapTime_s"],
                "Compound": compound,
                "StintID": f"{driver}_{stint_id}",
                "TyreLife": row.get("TyreLife", np.nan),
            })

            prev_compound = compound
            prev_lap = lap_num

    df = pd.DataFrame(stints)

    if len(df) == 0:
        return df

    for stint_id, group in df.groupby("StintID"):
        if len(group) < 2:
            continue
        stint_laps = group.sort_values("LapNumber")
        df.loc[stint_laps.index, "StintLap"] = range(1, len(stint_laps) + 1)

    df["StintLap"] = df["StintLap"].fillna(1).astype(int)

    return df


def filter_laps(df, outlap_threshold=1.07, min_stint_length=4):
    if len(df) == 0:
        return df

    filtered = df.copy()

    first_laps = []
    for stint_id, group in filtered.groupby("StintID"):
        if len(group) > 0:
            first_laps.append(group.sort_values("LapNumber").index[0])
    filtered = filtered.drop(first_laps, errors="ignore")

    for compound in filtered["Compound"].unique():
        mask = filtered["Compound"] == compound
        compound_laps = filtered.loc[mask, "LapTime_s"]
        if len(compound_laps) < 3:
            continue
        median_time = compound_laps.median()
        threshold = median_time * outlap_threshold
        filtered = filtered[~(mask & (filtered["LapTime_s"] > threshold))]

    for compound in filtered["Compound"].unique():
        mask = filtered["Compound"] == compound
        compound_laps = filtered.loc[mask, "LapTime_s"]
        if len(compound_laps) < 5:
            continue
        q1 = compound_laps.quantile(0.05)
        too_fast = compound_laps < q1 * 0.95
        filtered = filtered[~(mask & too_fast)]

    valid_stints = (
        filtered.groupby("StintID")
        .size()
        .reset_index(name="count")
    )
    valid_stints = valid_stints[valid_stints["count"] >= min_stint_length]
    filtered = filtered[filtered["StintID"].isin(valid_stints["StintID"])]

    return filtered.reset_index(drop=True)


def fuel_correct(df, circuit_name, session_type="FP2"):
    total_race_laps = CIRCUIT_LAPS.get(circuit_name, 58)
    usable_fuel = FUEL_LOAD_KG - FUEL_RESERVE_KG
    fuel_per_lap = usable_fuel / total_race_laps

    corrected = df.copy()

    if session_type == "FP2":
        estimated_start_fuel = FUEL_LOAD_KG * FP2_ESTIMATED_FUEL_FRACTION
    else:
        estimated_start_fuel = usable_fuel

    for stint_id, group in corrected.groupby("StintID"):
        group_sorted = group.sort_values("StintLap")
        for idx, row in group_sorted.iterrows():
            laps_into_stint = row["StintLap"] - 1
            remaining_fuel = max(0, estimated_start_fuel - laps_into_stint * fuel_per_lap)
            fuel_correction = remaining_fuel * WEIGHT_EFFECT
            corrected.loc[idx, "LapTime_fc"] = row["LapTime_s"] - fuel_correction

    return corrected


def build_tire_model(prior):
    def model(lap, laptime=None):
        alpha = numpyro.sample(
            "alpha",
            dist.Normal(prior["alpha_mu"], prior["alpha_sigma"]),
        )
        beta = numpyro.sample(
            "beta",
            dist.TruncatedNormal(
                prior["beta_mu"], prior["beta_sigma"], low=0.0,
            ),
        )
        gamma = numpyro.sample(
            "gamma",
            dist.TruncatedNormal(
                prior["gamma_mu"], prior["gamma_sigma"], low=0.0,
            ),
        )
        sigma = numpyro.sample(
            "sigma",
            dist.HalfNormal(prior["sigma_scale"]),
        )

        mu = alpha + beta * lap + gamma * jnp.square(lap)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=laptime)

    return model


def estimate_rho(residuals_by_stint):
    numerator = 0.0
    denominator = 0.0
    for resids in residuals_by_stint:
        if len(resids) < 3:
            continue
        for t in range(1, len(resids)):
            numerator += resids[t] * resids[t - 1]
            denominator += resids[t - 1] ** 2
    if denominator < 1e-10:
        return 0.35
    rho = numerator / denominator
    return float(np.clip(rho, 0.0, 0.9))


def fit_compound(df_compound, compound, circuit_name, num_warmup=500, num_samples=1000):
    if len(df_compound) < 6:
        print(f"    {compound}: insufficient data ({len(df_compound)} laps), skipping")
        return None

    laps = jnp.array(df_compound["StintLap"].values, dtype=jnp.float32)
    times = jnp.array(df_compound["LapTime_fc"].values, dtype=jnp.float32)

    prior = COMPOUND_PRIORS[compound]
    tire_model = build_tire_model(prior)

    kernel = NUTS(tire_model, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1)

    rng_key = jax.random.PRNGKey(np.random.randint(0, 10000))
    mcmc.run(rng_key, laps, times)

    samples = mcmc.get_samples()

    alpha_mean = float(jnp.mean(samples["alpha"]))
    beta_mean = float(jnp.mean(samples["beta"]))
    gamma_mean = float(jnp.mean(samples["gamma"]))

    residuals_by_stint = []
    for stint_id, group in df_compound.groupby("StintID"):
        group = group.sort_values("StintLap")
        stint_laps = group["StintLap"].values
        stint_times = group["LapTime_fc"].values
        predicted = alpha_mean + beta_mean * stint_laps + gamma_mean * stint_laps ** 2
        residuals_by_stint.append(stint_times - predicted)

    rho = estimate_rho(residuals_by_stint)

    rho_samples = np.full(num_samples, rho)
    rho_samples += np.random.normal(0, 0.05, num_samples)
    rho_samples = np.clip(rho_samples, 0.0, 0.9)

    result = {
        "samples": {
            "alpha": np.array(samples["alpha"]),
            "beta": np.array(samples["beta"]),
            "gamma": np.array(samples["gamma"]),
            "sigma": np.array(samples["sigma"]),
            "rho": rho_samples,
        },
        "n_laps": len(df_compound),
        "n_stints": df_compound["StintID"].nunique(),
        "rho_point_estimate": rho,
    }

    alpha_m = float(jnp.mean(samples["alpha"]))
    beta_m = float(jnp.mean(samples["beta"]))
    gamma_m = float(jnp.mean(samples["gamma"]))
    sigma_m = float(jnp.mean(samples["sigma"]))

    print(f"    {compound}: {len(df_compound)} laps, "
          f"{df_compound['StintID'].nunique()} stints")
    print(f"      alpha={alpha_m:.2f}  beta={beta_m:.4f}  "
          f"gamma={gamma_m:.6f}  sigma={sigma_m:.3f}  rho={rho:.3f}")

    return result


def run_pipeline(year, circuit_key, session_type="FP2", output_dir="prebuilt_models",
                 cache_dir=".f1_cache", num_warmup=500, num_samples=1000):
    circuit_name = CIRCUIT_LOOKUP.get(circuit_key)
    if circuit_name is None:
        available = ", ".join(sorted(CIRCUIT_LOOKUP.keys()))
        print(f"Unknown circuit: {circuit_key}")
        print(f"Available: {available}")
        return False

    print(f"Loading {session_type} data for {circuit_name} {year}")
    try:
        session = load_session(year, circuit_name, session_type, cache_dir)
    except Exception as e:
        print(f"Failed to load session: {e}")
        return False

    print("Extracting stints")
    stints = extract_stints(session, circuit_name)
    if len(stints) == 0:
        print("No valid stint data found")
        return False

    print(f"  Raw laps: {len(stints)}")
    for compound in stints["Compound"].unique():
        n = len(stints[stints["Compound"] == compound])
        print(f"    {compound}: {n} laps")

    print("Filtering")
    filtered = filter_laps(stints)
    print(f"  After filtering: {len(filtered)} laps")

    if len(filtered) == 0:
        print("No laps survived filtering")
        return False

    print("Fuel correcting")
    corrected = fuel_correct(filtered, circuit_name, session_type)

    print("Fitting models")
    models = {}
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        df_c = corrected[corrected["Compound"] == compound]
        if len(df_c) == 0:
            print(f"    {compound}: no data")
            continue
        result = fit_compound(df_c, compound, circuit_name, num_warmup, num_samples)
        if result is not None:
            models[compound] = result

    if not models:
        print("No models fitted")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    slug = circuit_name.lower().replace(" ", "_").replace("-", "_")
    filepath = output_path / f"{slug}_models.pkl"

    output = {
        "circuit": circuit_name,
        "year": year,
        "session": session_type,
        "models": models,
    }

    with open(filepath, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved to {filepath}")
    print(f"Compounds fitted: {list(models.keys())}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fit Bayesian tire models from F1 practice session data"
    )
    parser.add_argument(
        "circuit",
        help="Circuit key (e.g., australia, monaco, las_vegas)",
    )
    parser.add_argument(
        "--year", type=int, default=2026,
        help="Season year (default: 2026)",
    )
    parser.add_argument(
        "--session", default="FP2",
        choices=["FP1", "FP2", "FP3", "R"],
        help="Session type (default: FP2)",
    )
    parser.add_argument(
        "--output-dir", default="prebuilt_models",
        help="Output directory for model files (default: prebuilt_models)",
    )
    parser.add_argument(
        "--cache-dir", default=".f1_cache",
        help="FastF1 cache directory (default: .f1_cache)",
    )
    parser.add_argument(
        "--warmup", type=int, default=500,
        help="MCMC warmup samples (default: 500)",
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="MCMC posterior samples (default: 1000)",
    )

    args = parser.parse_args()

    success = run_pipeline(
        year=args.year,
        circuit_key=args.circuit,
        session_type=args.session,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        num_warmup=args.warmup,
        num_samples=args.samples,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
