import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

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

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

CIRCUIT_LAPS = {
    "Australia": 58, "China": 56, "Japan": 53, "Bahrain": 57,
    "Saudi Arabia": 50, "Miami": 57, "Canada": 70, "Monaco": 78,
    "Barcelona-Catalunya": 66, "Austria": 71, "Britain": 52,
    "Belgium": 44, "Hungary": 70, "Netherlands": 72, "Italy": 53,
    "Madrid": 57, "Azerbaijan": 51, "Singapore": 62,
    "United States": 56, "Mexico": 71, "Brazil": 71,
    "Las Vegas": 50, "Qatar": 57, "Abu Dhabi": 58,
}

FUEL_LOAD_KG = 92.5
FUEL_RESERVE_KG = 3.0
WEIGHT_EFFECT = 0.03
REFERENCE_COMPOUND = "MEDIUM"
REFERENCE_OFFSET = {"SOFT": -1.0, "MEDIUM": 0.0, "HARD": 0.7}


def load_race(year, circuit_name, cache_dir):
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, circuit_name, "R")
    session.load()
    return session


def stint_lengths(session, min_stint):
    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()].copy()
    laps = laps[laps["Compound"].isin(DRY_COMPOUNDS)].copy()
    if len(laps) == 0:
        return {}

    if "Stint" not in laps.columns:
        return {}

    lengths = defaultdict(list)
    for (driver, stint), group in laps.groupby(["Driver", "Stint"]):
        compound = group["Compound"].mode()
        if len(compound) == 0:
            continue
        compound = compound.iloc[0]
        if compound not in DRY_COMPOUNDS:
            continue
        n = int(group["LapNumber"].nunique())
        if n >= min_stint:
            lengths[compound].append(n)
    return lengths


def base_pace_samples(session, circuit_name, age_threshold):
    total_laps = CIRCUIT_LAPS.get(circuit_name, 58)
    fpl = (FUEL_LOAD_KG - FUEL_RESERVE_KG) / total_laps

    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()].copy()
    laps = laps[laps["Compound"].isin(DRY_COMPOUNDS)].copy()
    if len(laps) == 0:
        return {}

    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps = laps[laps["LapTime_s"] > 0].copy()
    if "TyreLife" in laps.columns:
        laps = laps[laps["TyreLife"] <= age_threshold].copy()
    else:
        return {}

    remaining = (total_laps - laps["LapNumber"] + 1).clip(lower=0)
    laps["LapTime_zf"] = laps["LapTime_s"] - remaining * fpl * WEIGHT_EFFECT

    out = defaultdict(list)
    for compound, group in laps.groupby("Compound"):
        vals = group["LapTime_zf"]
        if len(vals) < 3:
            continue
        median = vals.median()
        clean = vals[vals < median * 1.07]
        out[compound].extend(clean.tolist())
    return out


def derive_base_pace(circuit_key, years, age_threshold, cache_dir, exclude_2022):
    circuit_name = CIRCUIT_LOOKUP[circuit_key]
    pooled = defaultdict(list)
    for year in years:
        if exclude_2022 and year == 2022:
            continue
        try:
            session = load_race(year, circuit_name, cache_dir)
        except Exception:
            continue
        for compound, vals in base_pace_samples(session, circuit_name, age_threshold).items():
            pooled[compound].extend(vals)

    estimates = []
    for compound, vals in pooled.items():
        if len(vals) < 5:
            continue
        estimates.append(float(np.median(vals)) - REFERENCE_OFFSET[compound])
    if not estimates:
        return circuit_name, None
    return circuit_name, round(float(np.median(estimates)), 1)


def derive_circuit(circuit_key, years, quantile, min_stint, cache_dir,
                   exclude_2022):
    circuit_name = CIRCUIT_LOOKUP[circuit_key]
    pooled = defaultdict(list)
    for year in years:
        if exclude_2022 and year == 2022:
            continue
        try:
            session = load_race(year, circuit_name, cache_dir)
        except Exception as e:
            print(f"  {year}: skipped ({e})")
            continue
        lengths = stint_lengths(session, min_stint)
        for compound, vals in lengths.items():
            pooled[compound].extend(vals)
        counts = {c: len(v) for c, v in lengths.items()}
        print(f"  {year}: {counts}")

    result = {}
    for compound in DRY_COMPOUNDS:
        vals = pooled.get(compound, [])
        if len(vals) < 3:
            continue
        result[compound] = int(round(float(np.quantile(vals, quantile))))
    return circuit_name, result, {c: len(v) for c, v in pooled.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Derive per-circuit nominal tire life from historical races"
    )
    parser.add_argument("circuits", nargs="*", default=list(CIRCUIT_LOOKUP.keys()))
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--min-stint", type=int, default=5)
    parser.add_argument("--exclude-2022", action="store_true", default=True)
    parser.add_argument("--include-2022", dest="exclude_2022", action="store_false")
    parser.add_argument("--output", default="tire_life.json")
    parser.add_argument("--base-pace-output", default="base_pace.json")
    parser.add_argument("--base-pace-age", type=int, default=6)
    parser.add_argument("--cache-dir", default=".f1_cache")
    args = parser.parse_args()

    circuits = args.circuits or list(CIRCUIT_LOOKUP.keys())
    for c in circuits:
        if c not in CIRCUIT_LOOKUP:
            print(f"Unknown circuit: {c}")
            print(f"Available: {', '.join(sorted(CIRCUIT_LOOKUP.keys()))}")
            sys.exit(1)

    years = list(range(args.start_year, args.end_year + 1))

    existing = {}
    out_path = Path(args.output)
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = {}

    base_existing = {}
    base_path = Path(args.base_pace_output)
    if base_path.exists():
        try:
            base_existing = json.loads(base_path.read_text())
        except Exception:
            base_existing = {}

    for circuit_key in circuits:
        print(f"Deriving {CIRCUIT_LOOKUP[circuit_key]}")
        name, life, n = derive_circuit(
            circuit_key, years, args.quantile, args.min_stint,
            args.cache_dir, args.exclude_2022,
        )
        if life:
            existing[name] = life
            print(f"  nominal life (q{args.quantile:.2f}): {life}  from n={n}")
        else:
            print("  insufficient life data, leaving unchanged")

        _, base = derive_base_pace(
            circuit_key, years, args.base_pace_age, args.cache_dir,
            args.exclude_2022,
        )
        if base:
            base_existing[name] = base
            print(f"  zero-fuel base pace: {base}s")
        else:
            print("  insufficient base-pace data, leaving unchanged")

    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    base_path.write_text(json.dumps(base_existing, indent=2, sort_keys=True))
    print(f"Wrote {out_path} and {base_path}")


if __name__ == "__main__":
    main()
