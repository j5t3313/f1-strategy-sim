import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import pickle
import os
from pathlib import Path


CIRCUIT_DATA = [
    ("Australia", {"laps": 58, "distance_km": 5.278, "gp_name": "Australian Grand Prix"}),
    ("China", {"laps": 56, "distance_km": 5.451, "gp_name": "Chinese Grand Prix"}),
    ("Japan", {"laps": 53, "distance_km": 5.807, "gp_name": "Japanese Grand Prix"}),
    ("Bahrain", {"laps": 57, "distance_km": 5.412, "gp_name": "Bahrain Grand Prix"}),
    ("Saudi Arabia", {"laps": 50, "distance_km": 6.174, "gp_name": "Saudi Arabian Grand Prix"}),
    ("Miami", {"laps": 57, "distance_km": 5.410, "gp_name": "Miami Grand Prix"}),
    ("Canada", {"laps": 70, "distance_km": 4.361, "gp_name": "Canadian Grand Prix"}),
    ("Monaco", {"laps": 78, "distance_km": 3.337, "gp_name": "Monaco Grand Prix"}),
    ("Barcelona-Catalunya", {"laps": 66, "distance_km": 4.675, "gp_name": "Barcelona-Catalunya Grand Prix"}),
    ("Austria", {"laps": 71, "distance_km": 4.318, "gp_name": "Austrian Grand Prix"}),
    ("Britain", {"laps": 52, "distance_km": 5.891, "gp_name": "British Grand Prix"}),
    ("Belgium", {"laps": 44, "distance_km": 7.004, "gp_name": "Belgian Grand Prix"}),
    ("Hungary", {"laps": 70, "distance_km": 4.381, "gp_name": "Hungarian Grand Prix"}),
    ("Netherlands", {"laps": 72, "distance_km": 4.259, "gp_name": "Dutch Grand Prix"}),
    ("Italy", {"laps": 53, "distance_km": 5.793, "gp_name": "Italian Grand Prix"}),
    ("Madrid", {"laps": 57, "distance_km": 5.416, "gp_name": "Spanish Grand Prix"}),
    ("Azerbaijan", {"laps": 51, "distance_km": 6.003, "gp_name": "Azerbaijan Grand Prix"}),
    ("Singapore", {"laps": 62, "distance_km": 4.940, "gp_name": "Singapore Grand Prix"}),
    ("United States", {"laps": 56, "distance_km": 5.513, "gp_name": "United States Grand Prix"}),
    ("Mexico", {"laps": 71, "distance_km": 4.304, "gp_name": "Mexico City Grand Prix"}),
    ("Brazil", {"laps": 71, "distance_km": 4.309, "gp_name": "Sao Paulo Grand Prix"}),
    ("Las Vegas", {"laps": 50, "distance_km": 6.201, "gp_name": "Las Vegas Grand Prix"}),
    ("Qatar", {"laps": 57, "distance_km": 5.380, "gp_name": "Qatar Grand Prix"}),
    ("Abu Dhabi", {"laps": 58, "distance_km": 5.281, "gp_name": "Abu Dhabi Grand Prix"}),
]

ALL_STRATEGIES = {
    "1-Stop: M-H": ["MEDIUM", "HARD"],
    "1-Stop: S-H": ["SOFT", "HARD"],
    "1-Stop: H-M": ["HARD", "MEDIUM"],
    "1-Stop: H-S": ["HARD", "SOFT"],
    "2-Stop: S-M-H": ["SOFT", "MEDIUM", "HARD"],
    "2-Stop: M-M-H": ["MEDIUM", "MEDIUM", "HARD"],
    "2-Stop: M-M-S": ["MEDIUM", "MEDIUM", "SOFT"],
    "2-Stop: H-M-S": ["HARD", "MEDIUM", "SOFT"],
    "2-Stop: M-H-M": ["MEDIUM", "HARD", "MEDIUM"],
    "2-Stop: M-H-H": ["MEDIUM", "HARD", "HARD"],
    "2-Stop: S-H-M": ["SOFT", "HARD", "MEDIUM"],
}

CIRCUIT_RACE_PACES = {
    "Australia": 84.0, "China": 97.0, "Japan": 94.0, "Bahrain": 95.0,
    "Saudi Arabia": 91.0, "Miami": 93.0, "Canada": 77.0, "Monaco": 78.0,
    "Barcelona-Catalunya": 80.0, "Austria": 69.0, "Britain": 90.0,
    "Belgium": 108.0, "Hungary": 81.0, "Netherlands": 74.0, "Italy": 84.0,
    "Madrid": 93.0, "Azerbaijan": 106.0, "Singapore": 98.0,
    "United States": 98.0, "Mexico": 80.0, "Brazil": 74.0,
    "Las Vegas": 86.0, "Qatar": 86.0, "Abu Dhabi": 89.0,
}

CIRCUIT_PIT_LOSSES = {
    "Australia": 21.5, "China": 20.8, "Japan": 20.2, "Bahrain": 19.8,
    "Saudi Arabia": 22.1, "Miami": 18.5, "Canada": 15.8, "Monaco": 16.2,
    "Barcelona-Catalunya": 21.4, "Austria": 18.9, "Britain": 20.5,
    "Belgium": 23.2, "Hungary": 22.8, "Netherlands": 16.5, "Italy": 23.7,
    "Madrid": 21.0, "Azerbaijan": 19.8, "Singapore": 22.5,
    "United States": 20.3, "Mexico": 21.1, "Brazil": 19.4,
    "Las Vegas": 19.6, "Qatar": 20.7, "Abu Dhabi": 21.3,
}

COMPOUND_PRIORS = {
    "SOFT": {
        "alpha_offset": {"mu": -1.0, "sigma": 0.3},
        "beta": {"mu": 0.07, "sigma": 0.02},
        "gamma": {"mu": 0.0015, "sigma": 0.0008},
        "sigma": {"mu": 0.25, "sigma": 0.08},
        "rho": {"mu": 0.35, "sigma": 0.10},
    },
    "MEDIUM": {
        "alpha_offset": {"mu": 0.0, "sigma": 0.2},
        "beta": {"mu": 0.04, "sigma": 0.015},
        "gamma": {"mu": 0.0008, "sigma": 0.0005},
        "sigma": {"mu": 0.22, "sigma": 0.06},
        "rho": {"mu": 0.35, "sigma": 0.10},
    },
    "HARD": {
        "alpha_offset": {"mu": 0.7, "sigma": 0.25},
        "beta": {"mu": 0.02, "sigma": 0.008},
        "gamma": {"mu": 0.0004, "sigma": 0.0003},
        "sigma": {"mu": 0.20, "sigma": 0.05},
        "rho": {"mu": 0.35, "sigma": 0.10},
    },
}

FUEL_LOAD_KG = 92.5
FUEL_RESERVE_KG = 3.0
WEIGHT_EFFECT_S_PER_KG = 0.03
PACE_SIGMA = 0.4

MIN_STINT_LAPS = 5
SUPPORT_CAP_MULT = 1.5
EXTRAP_FLAG_FRAC = 1.0
WINDOW_TOL_FRAC = 0.25
DEFAULT_OBJECTIVE = "mean"
MAX_PARTITION_EVALS = 400000

COMPOUND_NOMINAL_LIFE = {"SOFT": 22, "MEDIUM": 45, "HARD": 60}

FULL_FUEL_EFFECT_S = (FUEL_LOAD_KG - FUEL_RESERVE_KG) * WEIGHT_EFFECT_S_PER_KG

BASE_PACE_FILE = Path("base_pace.json")


def load_base_paces():
    if not BASE_PACE_FILE.exists():
        return {}
    try:
        with open(BASE_PACE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def build_base_paces():
    override = load_base_paces()
    out = {}
    for circuit, race_pace in CIRCUIT_RACE_PACES.items():
        if circuit in override:
            out[circuit] = float(override[circuit])
        else:
            out[circuit] = round(race_pace - FULL_FUEL_EFFECT_S, 1)
    return out


CIRCUIT_BASE_PACES = build_base_paces()

STRATEGY_COLORS = ["#e10600", "#0090ff", "#22c55e", "#ff8700", "#a855f7"]
COMPOUND_COLORS = {"SOFT": "#dc2626", "MEDIUM": "#ca8a04", "HARD": "#6b7280"}

UPDATES_FILE = Path("updates.json")
NOMINAL_LIFE_FILE = Path("tire_life.json")


def load_updates():
    if not UPDATES_FILE.exists():
        return []
    try:
        with open(UPDATES_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def load_nominal_life():
    if not NOMINAL_LIFE_FILE.exists():
        return {}
    try:
        with open(NOMINAL_LIFE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


CIRCUIT_NOMINAL_LIFE = load_nominal_life()


def _sum1(n):
    return n * (n + 1) / 2.0


def _sum2(n):
    return n * (n + 1) * (2 * n + 1) / 6.0


def _range_sum1(lo, hi):
    return _sum1(hi) - _sum1(lo - 1)


def _range_sum2(lo, hi):
    return _sum2(hi) - _sum2(lo - 1)


def _iter_partitions(total, caps, min_stint):
    n = len(caps)

    def recurse(remaining, idx):
        if idx == n - 1:
            if min_stint <= remaining <= caps[idx]:
                yield (remaining,)
            return
        floor = min_stint
        ceil = min(caps[idx], remaining - min_stint * (n - idx - 1))
        for length in range(floor, ceil + 1):
            for tail in recurse(remaining - length, idx + 1):
                yield (length,) + tail

    yield from recurse(total, 0)


class F1StrategySimulator:

    def __init__(self, models_dir="prebuilt_models"):
        self.models_dir = Path(models_dir)
        self.circuits = {name: data for name, data in CIRCUIT_DATA}
        self.posterior_models = {}
        self._base_alpha_cache = {}
        self.has_posteriors = self._load_posterior_models()

    def _load_posterior_models(self):
        if not self.models_dir.exists():
            return False
        for circuit_name, _ in CIRCUIT_DATA:
            slug = circuit_name.lower().replace(" ", "_").replace("-", "_")
            model_file = self.models_dir / f"{slug}_models.pkl"
            if model_file.exists():
                try:
                    with open(model_file, "rb") as f:
                        data = pickle.load(f)
                    for compound, model_data in data["models"].items():
                        self.posterior_models[f"{circuit_name}_{compound}"] = model_data
                except Exception:
                    continue
        return len(self.posterior_models) > 0

    def has_posterior(self, circuit_name):
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            if f"{circuit_name}_{compound}" in self.posterior_models:
                return True
        return False

    def _circuit_base_alpha(self, circuit_name):
        if circuit_name in self._base_alpha_cache:
            return self._base_alpha_cache[circuit_name]
        vals = []
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            key = f"{circuit_name}_{compound}"
            if key in self.posterior_models:
                a = float(np.mean(self.posterior_models[key]["samples"]["alpha"]))
                vals.append(a - COMPOUND_PRIORS[compound]["alpha_offset"]["mu"])
        base = float(np.mean(vals)) if vals else CIRCUIT_BASE_PACES.get(circuit_name, 80.0)
        self._base_alpha_cache[circuit_name] = base
        return base

    def _draw_compound_params(self, compound, circuit_name, deg_multiplier=1.0):
        model_key = f"{circuit_name}_{compound}"

        if self.has_posteriors and model_key in self.posterior_models:
            samples = self.posterior_models[model_key]["samples"]
            idx = np.random.choice(len(samples["alpha"]))

            alpha = float(samples["alpha"][idx])
            beta = float(samples["beta"][idx]) * deg_multiplier

            if "gamma" in samples and len(samples["gamma"]) > 0:
                gamma = float(samples["gamma"][min(idx, len(samples["gamma"]) - 1)]) * deg_multiplier
            else:
                p = COMPOUND_PRIORS[compound]["gamma"]
                gamma = abs(np.random.normal(p["mu"], p["sigma"])) * deg_multiplier

            sigma = float(samples["sigma"][idx])

            if "rho" in samples and len(samples["rho"]) > 0:
                rho = float(samples["rho"][min(idx, len(samples["rho"]) - 1)])
            else:
                p = COMPOUND_PRIORS[compound]["rho"]
                rho = np.random.normal(p["mu"], p["sigma"])

            return {
                "mode": "posterior",
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "sigma": max(0.01, sigma),
                "rho": np.clip(rho, 0.0, 0.9),
            }

        prior = COMPOUND_PRIORS[compound]

        if self.has_posteriors and self.has_posterior(circuit_name):
            base_alpha = self._circuit_base_alpha(circuit_name)
            offset = np.random.normal(
                prior["alpha_offset"]["mu"], prior["alpha_offset"]["sigma"]
            )
            return {
                "mode": "posterior",
                "alpha": base_alpha + offset,
                "beta": max(0.001, np.random.normal(
                    prior["beta"]["mu"], prior["beta"]["sigma"]
                )) * deg_multiplier,
                "gamma": abs(np.random.normal(
                    prior["gamma"]["mu"], prior["gamma"]["sigma"]
                )) * deg_multiplier,
                "sigma": max(0.01, abs(np.random.normal(
                    prior["sigma"]["mu"], prior["sigma"]["sigma"]
                ))),
                "rho": np.clip(
                    np.random.normal(prior["rho"]["mu"], prior["rho"]["sigma"]),
                    0.0, 0.9,
                ),
            }

        return {
            "mode": "prior",
            "alpha_offset": np.random.normal(
                prior["alpha_offset"]["mu"], prior["alpha_offset"]["sigma"]
            ),
            "beta": max(0.001, np.random.normal(
                prior["beta"]["mu"], prior["beta"]["sigma"]
            )) * deg_multiplier,
            "gamma": abs(np.random.normal(
                prior["gamma"]["mu"], prior["gamma"]["sigma"]
            )) * deg_multiplier,
            "sigma": max(0.01, abs(np.random.normal(
                prior["sigma"]["mu"], prior["sigma"]["sigma"]
            ))),
            "rho": np.clip(
                np.random.normal(prior["rho"]["mu"], prior["rho"]["sigma"]),
                0.0, 0.9,
            ),
        }

    def fuel_per_lap(self, circuit):
        return (FUEL_LOAD_KG - FUEL_RESERVE_KG) / self.circuits[circuit]["laps"]

    def assign_tires(self, strategy, tire_allocation):
        if not tire_allocation:
            return [
                {"compound": s["compound"], "laps": s["laps"], "tire_age": 0}
                for s in strategy
            ]
        sets = {c: [] for c in ["SOFT", "MEDIUM", "HARD"]}
        for t in tire_allocation:
            sets[t["compound"]].append(t)
        for c in sets:
            sets[c].sort(key=lambda x: x["age_laps"])
        result = []
        for stint in strategy:
            c = stint["compound"]
            if not sets[c]:
                raise ValueError(f"No {c} sets available")
            tire = sets[c].pop(0)
            result.append({
                "compound": c,
                "laps": stint["laps"],
                "tire_age": tire["age_laps"],
            })
        return result

    def validate_allocation(self, strategy, tire_allocation):
        if not tire_allocation:
            return True, ""
        required = {}
        for stint in strategy:
            c = stint["compound"]
            required[c] = required.get(c, 0) + 1
        for c, needed in required.items():
            available = len([t for t in tire_allocation if t["compound"] == c])
            if available < needed:
                return False, f"Need {needed} {c} sets, have {available}"
        return True, ""

    def simulate(self, circuit, strategy, tire_allocation=None,
                 base_pace=80.0, pit_loss=22.0, num_sims=1000,
                 deg_multiplier=1.0, pre_resolved=False):
        num_sims = int(num_sims)
        total_laps = self.circuits[circuit]["laps"]
        fpl = self.fuel_per_lap(circuit)

        if pre_resolved:
            enhanced = [
                {"compound": s["compound"], "laps": s["laps"],
                 "tire_age": s.get("tire_age", 0)}
                for s in strategy
            ]
        else:
            valid, msg = self.validate_allocation(strategy, tire_allocation)
            if not valid:
                raise ValueError(msg)
            enhanced = self.assign_tires(strategy, tire_allocation)

        compounds_used = list({s["compound"] for s in enhanced})
        results = np.zeros(num_sims)

        for sim in range(num_sims):
            sim_pace = base_pace + np.random.normal(0, PACE_SIGMA)
            compound_params = {
                c: self._draw_compound_params(c, circuit, deg_multiplier)
                for c in compounds_used
            }

            race_time = 0.0
            current_lap = 1

            for stint_idx, stint in enumerate(enhanced):
                compound = stint["compound"]
                stint_length = min(stint["laps"], total_laps - current_lap + 1)
                tire_age = stint["tire_age"]
                params = compound_params[compound]

                innovation_sigma = params["sigma"] * np.sqrt(
                    max(1e-10, 1.0 - params["rho"] ** 2)
                )
                epsilon = 0.0

                for stint_lap in range(1, stint_length + 1):
                    if current_lap > total_laps:
                        break

                    effective_lap = stint_lap + tire_age

                    if params["mode"] == "posterior":
                        mu = (
                            params["alpha"]
                            + params["beta"] * effective_lap
                            + params["gamma"] * effective_lap ** 2
                        )
                    else:
                        mu = (
                            sim_pace
                            + params["alpha_offset"]
                            + params["beta"] * effective_lap
                            + params["gamma"] * effective_lap ** 2
                        )

                    if stint_lap == 1:
                        epsilon = np.random.normal(0, params["sigma"])
                    else:
                        epsilon = (
                            params["rho"] * epsilon
                            + np.random.normal(0, innovation_sigma)
                        )

                    fuel_correction = (
                        (total_laps - current_lap + 1) * fpl * WEIGHT_EFFECT_S_PER_KG
                    )
                    race_time += mu + epsilon + fuel_correction
                    current_lap += 1

                if stint_idx < len(enhanced) - 1:
                    race_time += pit_loss

            results[sim] = race_time

        return results

    def _available_sets(self, tire_allocation):
        if not tire_allocation:
            return None
        sets = {}
        for t in tire_allocation:
            sets.setdefault(t["compound"], []).append(int(t["age_laps"]))
        for c in sets:
            sets[c].sort()
        return sets

    def _resolve_ages(self, sequence, lengths, avail):
        if avail is None:
            return [0] * len(sequence)
        pools = {c: list(ages) for c, ages in avail.items()}
        ages = [None] * len(sequence)
        by_compound = {}
        for i, c in enumerate(sequence):
            by_compound.setdefault(c, []).append(i)
        for c, idxs in by_compound.items():
            pool = pools.get(c, [])
            if len(pool) < len(idxs):
                raise ValueError(f"Need {len(idxs)} {c} sets, have {len(pool)}")
            order = sorted(idxs, key=lambda i: (-lengths[i], i))
            for rank, i in enumerate(order):
                ages[i] = pool[rank]
        return ages

    def _fitted_support_max(self, compound, circuit):
        key = f"{circuit}_{compound}"
        if self.has_posteriors and key in self.posterior_models:
            support = self.posterior_models[key].get("support")
            if support and support.get("max_lap"):
                return int(support["max_lap"])
        return None

    def _nominal_life(self, compound, circuit):
        override = CIRCUIT_NOMINAL_LIFE.get(circuit, {})
        if compound in override:
            return int(override[compound])
        return int(COMPOUND_NOMINAL_LIFE[compound])

    def _compound_cap(self, compound, circuit, min_stint, total_laps):
        cap = min(total_laps, self._nominal_life(compound, circuit))
        fitted = self._fitted_support_max(compound, circuit)
        if fitted is not None:
            cap = min(cap, int(round(SUPPORT_CAP_MULT * fitted)))
        return max(min_stint, cap)

    def _mean_params(self, compound, circuit, base_pace, deg_multiplier):
        key = f"{circuit}_{compound}"
        if self.has_posteriors and key in self.posterior_models:
            s = self.posterior_models[key]["samples"]
            a = float(np.mean(s["alpha"]))
            b = float(np.mean(s["beta"])) * deg_multiplier
            g = float(np.mean(s["gamma"])) * deg_multiplier
            return a, b, g, "posterior"
        p = COMPOUND_PRIORS[compound]
        if self.has_posteriors and self.has_posterior(circuit):
            a = self._circuit_base_alpha(circuit) + p["alpha_offset"]["mu"]
        else:
            a = base_pace + p["alpha_offset"]["mu"]
        b = p["beta"]["mu"] * deg_multiplier
        g = p["gamma"]["mu"] * deg_multiplier
        return a, b, g, "prior"

    def expected_race_time(self, sequence, lengths, ages, mean_params, pit_loss):
        total = 0.0
        for i, compound in enumerate(sequence):
            a, b, g, _ = mean_params[compound]
            lo = ages[i] + 1
            hi = ages[i] + lengths[i]
            total += (
                a * lengths[i]
                + b * _range_sum1(lo, hi)
                + g * _range_sum2(lo, hi)
            )
        total += pit_loss * (len(sequence) - 1)
        return total

    def _extrapolation_flags(self, sequence, lengths, circuit):
        flags = []
        for i, compound in enumerate(sequence):
            nominal = self._nominal_life(compound, circuit)
            fitted = self._fitted_support_max(compound, circuit)
            ref = fitted if fitted is not None else nominal
            flagged = lengths[i] >= EXTRAP_FLAG_FRAC * ref or lengths[i] >= nominal
            flags.append({
                "compound": compound,
                "length": int(lengths[i]),
                "max_lap": fitted,
                "nominal_life": nominal,
                "flagged": bool(flagged),
            })
        return flags

    def resolve_fixed(self, circuit, sequence, lengths, tire_allocation=None):
        avail = self._available_sets(tire_allocation)
        ages = self._resolve_ages(sequence, lengths, avail)
        resolved = [
            {"compound": c, "laps": int(lengths[i]), "tire_age": int(ages[i])}
            for i, c in enumerate(sequence)
        ]
        pit_laps = [int(x) for x in np.cumsum(lengths)[:-1]]
        return {
            "sequence": list(sequence),
            "lengths": [int(x) for x in lengths],
            "ages": [int(x) for x in ages],
            "resolved": resolved,
            "pit_laps": pit_laps,
            "extrapolation": self._extrapolation_flags(sequence, lengths, circuit),
        }

    def optimize(self, circuit, sequence, tire_allocation=None, base_pace=80.0,
                 pit_loss=22.0, deg_multiplier=1.0, objective=DEFAULT_OBJECTIVE,
                 min_stint=MIN_STINT_LAPS):
        total_laps = self.circuits[circuit]["laps"]
        n = len(sequence)
        avail = self._available_sets(tire_allocation)
        if avail is not None:
            need = {}
            for c in sequence:
                need[c] = need.get(c, 0) + 1
            for c, k in need.items():
                have = len(avail.get(c, []))
                if have < k:
                    raise ValueError(f"Need {k} {c} sets, have {have}")

        caps = [
            self._compound_cap(c, circuit, min_stint, total_laps) for c in sequence
        ]
        if sum(caps) < total_laps or min_stint * n > total_laps:
            raise ValueError("no feasible pit split for this sequence")

        mean_params = {
            c: self._mean_params(c, circuit, base_pace, deg_multiplier)
            for c in set(sequence)
        }

        best = None
        surface = []
        evals = 0
        for lengths in _iter_partitions(total_laps, caps, min_stint):
            evals += 1
            if evals > MAX_PARTITION_EVALS:
                raise ValueError("partition space too large to optimize")
            ages = self._resolve_ages(sequence, lengths, avail)
            t = self.expected_race_time(
                sequence, lengths, ages, mean_params, pit_loss
            )
            pit_laps = tuple(int(x) for x in np.cumsum(lengths)[:-1])
            surface.append((pit_laps, t))
            if best is None or t < best["mean"]:
                best = {"lengths": list(lengths), "ages": ages,
                        "pit_laps": list(pit_laps), "mean": t}

        resolved = [
            {"compound": c, "laps": int(best["lengths"][i]),
             "tire_age": int(best["ages"][i])}
            for i, c in enumerate(sequence)
        ]
        return {
            "sequence": list(sequence),
            "lengths": [int(x) for x in best["lengths"]],
            "ages": [int(x) for x in best["ages"]],
            "resolved": resolved,
            "pit_laps": [int(x) for x in best["pit_laps"]],
            "closed_form_mean": float(best["mean"]),
            "mode": mean_params[sequence[0]][3],
            "extrapolation": self._extrapolation_flags(
                sequence, best["lengths"], circuit
            ),
            "surface": surface,
        }


def pit_window_from_surface(surface, best_mean, tol):
    eligible = [pt for pt, m in surface if m <= best_mean + tol]
    if not eligible:
        return None, 0
    n_pits = len(eligible[0]) if eligible[0] else 0
    box = []
    for i in range(n_pits):
        vals = [pt[i] for pt in eligible]
        box.append([int(min(vals)), int(max(vals))])
    return box, len(eligible)


def default_partition(sequence, circuit):
    circuit_laps = simulator.circuits[circuit]["laps"]
    try:
        opt = simulator.optimize(
            circuit, sequence,
            base_pace=CIRCUIT_BASE_PACES.get(circuit, 80.0),
            pit_loss=CIRCUIT_PIT_LOSSES.get(circuit, 22.0),
        )
        return [{"compound": c, "laps": opt["lengths"][i]}
                for i, c in enumerate(sequence)]
    except Exception:
        n = len(sequence)
        base = circuit_laps // n
        lengths = [base] * n
        lengths[-1] = circuit_laps - base * (n - 1)
        return [{"compound": c, "laps": lengths[i]}
                for i, c in enumerate(sequence)]


def make_stint_block(compound, laps):
    return html.Span(
        f"{laps}{compound[0]}",
        className=f"stint-block stint-block-{compound.lower()}",
    )


def make_stint_sequence(strategy):
    elements = []
    for i, stint in enumerate(strategy):
        if i > 0:
            elements.append(html.Span("\u2192", className="stint-arrow"))
        elements.append(make_stint_block(stint["compound"], stint["laps"]))
    return html.Div(elements, className="stint-sequence")


def format_pit_laps(pit_laps):
    if not pit_laps:
        return "no stop"
    return " / ".join(str(int(p)) for p in pit_laps)


def format_window(window):
    if not window:
        return "\u2014"
    parts = []
    for lo, hi in window:
        parts.append(str(lo) if lo == hi else f"{lo}\u2013{hi}")
    return " / ".join(parts)


def chart_layout(title=""):
    return dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#374151", family="DM Sans", size=12),
        title=dict(
            text=title,
            font=dict(family="Barlow Semi Condensed", size=16, color="#111827"),
            x=0.01, xanchor="left",
        ),
        xaxis=dict(
            gridcolor="#e5e7eb", zerolinecolor="#d1d5db",
            tickfont=dict(family="JetBrains Mono", size=11),
        ),
        yaxis=dict(
            gridcolor="#e5e7eb", zerolinecolor="#d1d5db",
            tickfont=dict(family="JetBrains Mono", size=11),
        ),
        legend=dict(
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
        ),
        margin=dict(l=60, r=30, t=50, b=50),
    )


TABLE_HEADER = {
    "backgroundColor": "#f3f4f6", "color": "#111827",
    "fontFamily": "Barlow Semi Condensed", "fontWeight": "600",
    "fontSize": "12px", "letterSpacing": "0.5px",
    "border": "1px solid #e5e7eb", "textAlign": "center",
}

TABLE_CELL = {
    "backgroundColor": "#ffffff", "color": "#374151",
    "fontFamily": "JetBrains Mono", "fontSize": "12px",
    "border": "1px solid #e5e7eb", "textAlign": "center", "padding": "8px",
}

TABLE_CONDITIONAL = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#f9fafb"},
]


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Semi+Condensed:wght@300;400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap",
    ],
    title="F1 Strategy Simulator | 2026",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)

server = app.server
simulator = F1StrategySimulator()

header = html.Div(
    [
        html.Div(
            [
                html.Span("F1 STRATEGY SIMULATOR", className="app-title"),
                html.Span("2026", className="app-title-year"),
            ],
            style={"display": "flex", "alignItems": "baseline"},
        ),
        html.Span("@formulasteele", className="app-watermark"),
    ],
    className="app-header",
)

sidebar = html.Div(
    [
        html.Div("RACE CONFIGURATION", className="sidebar-section-label"),
        html.Hr(className="sidebar-divider"),
        html.Label("Circuit", className="form-label"),
        dcc.Dropdown(
            id="circuit-dropdown",
            options=[{"label": d["gp_name"], "value": n} for n, d in CIRCUIT_DATA],
            placeholder="Select a circuit",
            clearable=True,
            className="mb-2",
        ),
        html.Div(id="circuit-info"),
        html.Hr(className="sidebar-divider"),
        html.Label("Strategies", className="form-label"),
        dcc.Dropdown(
            id="strategy-dropdown",
            options=[{"label": k, "value": k} for k in ALL_STRATEGIES],
            multi=True,
            placeholder="Select strategies to compare",
            className="mb-2",
        ),
        html.Hr(className="sidebar-divider"),
        html.Label("Base Pace (s)", className="form-label"),
        dcc.Input(
            id="base-pace-input", type="number",
            value=80.0, min=50, max=130, step=0.1, className="mb-2",
        ),
        html.Label("Pit Loss (s)", className="form-label"),
        dcc.Input(
            id="pit-loss-input", type="number",
            value=22.0, min=10, max=40, step=0.1, className="mb-2",
        ),
        html.Label("Simulations", className="form-label"),
        dcc.Slider(
            id="sims-slider", min=100, max=2000, step=100, value=1000,
            marks={
                i: {"label": str(i), "style": {"fontSize": "10px"}}
                for i in range(100, 2100, 300)
            },
            className="mb-3",
        ),
        html.Hr(className="sidebar-divider"),
        dbc.Checklist(
            id="custom-tires-toggle",
            options=[{"label": " Custom Tire Allocation", "value": "on"}],
            value=[], className="mb-2", style={"fontSize": "13px"},
        ),
        html.Div(id="tire-allocation-section"),
        dcc.Store(id="tire-allocation-store"),
        html.Hr(className="sidebar-divider"),
        dbc.Checklist(
            id="editor-toggle",
            options=[{"label": " Custom Strategy Editor", "value": "on"}],
            value=[], className="mb-2", style={"fontSize": "13px"},
        ),
        html.Hr(className="sidebar-divider"),
        dbc.Button(
            "RUN ANALYSIS", id="run-button",
            className="run-button", disabled=True,
        ),
        html.Div(id="run-status", className="mt-2"),
        dbc.Button(
            "RESET", id="reset-button",
            className="reset-button mt-2",
        ),
        html.Hr(className="sidebar-divider"),
        html.Div("SENSITIVITY ANALYSIS", className="sidebar-section-label"),
        dbc.Button(
            "RUN SENSITIVITY", id="sensitivity-button",
            className="sensitivity-button", disabled=True,
        ),
        dcc.Loading(
            html.Div(id="sensitivity-status", className="mt-2"),
            type="circle", color="#e10600",
        ),
    ],
    className="sidebar",
)

def build_updates_section():
    updates = load_updates()
    if not updates:
        return html.Div(style={"display": "none"})
    items = []
    for entry in reversed(updates):
        date_str = entry.get("date", "")
        text = entry.get("text", "")
        entry_type = entry.get("type", "manual")
        badge_class = (
            "update-badge-model" if entry_type == "model"
            else "update-badge-info"
        )
        badge_text = "MODEL" if entry_type == "model" else "INFO"
        items.append(
            html.Div(
                [
                    html.Span(badge_text, className=f"update-badge {badge_class}"),
                    html.Span(date_str, className="update-date"),
                    html.Span(text, className="update-text"),
                ],
                className="update-item",
            )
        )
    return html.Div(
        [
            html.Div("Updates", className="methodology-heading"),
            html.Div(items, className="updates-list"),
        ],
        className="updates-section",
    )


welcome_content = html.Div(
    [
        html.Div("F1 Strategy Simulator", className="welcome-title"),
        html.P(
            "Solve, simulate, and compare Formula 1 pit stop strategies using "
            "Bayesian tire modeling, closed-form pit-window optimization, and "
            "Monte Carlo simulation, updated for the 2026 regulation era.",
            className="welcome-text",
        ),
        html.P(
            "Select a circuit, choose the compound sequences to compare, and run "
            "the analysis. The simulator solves the optimal pit lap for each "
            "sequence, reports the window of laps statistically equivalent to it, "
            "and compares strategies at their own optima.",
            className="welcome-text",
        ),
        build_updates_section(),
        html.Div(
            [
                html.Div("2026 Regulations", className="methodology-heading"),
                html.P(
                    "Energy-based fuel limit (3000 MJ/h flow cap replacing the former "
                    "100 kg/h mass-flow rule, with no regulated starting fuel mass), "
                    "sustainable fuel with lower energy density (38\u201341 MJ/kg), "
                    "Pirelli C1\u2013C5 compounds (C6 removed, narrower construction), "
                    "768 kg minimum car weight (per published technical regulations), "
                    "~15\u201330% downforce reduction, active aerodynamics, "
                    "50/50 ICE/electric power split.",
                    className="methodology-text",
                ),
                html.Div("Tire Degradation Model", className="methodology-heading"),
                html.P(
                    "Lap times are modeled as \u03bc = \u03b1 + \u03b2\u00b7lap + "
                    "\u03b3\u00b7lap\u00b2, where \u03b1 is the baseline lap time for "
                    "a given compound, \u03b2 captures linear degradation, and \u03b3 "
                    "captures the accelerating degradation (\"cliff\") observed in longer "
                    "stints. The linear term dominates early in a stint while the quadratic "
                    "term becomes significant as tire age increases, producing the nonlinear "
                    "performance falloff that teams observe when tires exceed their thermal "
                    "operating window.",
                    className="methodology-text",
                ),
                html.Div(
                    "Bayesian Inference & Prior/Posterior Modes",
                    className="methodology-heading",
                ),
                html.P([
                    "When FP2 (or other session) data is available, the model parameters "
                    "{\u03b1, \u03b2, \u03b3, \u03c3} are estimated via Markov Chain Monte "
                    "Carlo (MCMC) using the No-U-Turn Sampler (NUTS). This produces a "
                    "posterior distribution over parameters that reflects both the data and "
                    "the prior beliefs about plausible degradation rates. When no session "
                    "data is available, the simulator draws from ",
                    html.Em("informative priors"),
                    " calibrated against historical compound-specific degradation ranges "
                    "(e.g., soft tires degrade faster than hards). This is a single unified "
                    "model structure \u2014 there is no separate \"physics fallback.\" Sparse "
                    "data partially updates the prior; abundant data dominates it. This is "
                    "standard Bayesian behavior and means sprint weekends (FP1-only, shorter "
                    "stints) produce wider but still useful posteriors.",
                ], className="methodology-text"),
                html.Div(
                    "Epistemic vs. Aleatoric Uncertainty",
                    className="methodology-heading",
                ),
                html.P(
                    "The simulator separates two types of uncertainty. Epistemic uncertainty "
                    "(we don't know the true degradation rate) is handled by drawing one set "
                    "of parameters {\u03b1, \u03b2, \u03b3, \u03c3, \u03c1} per compound per "
                    "simulation. Different simulations explore different plausible degradation "
                    "curves. Aleatoric uncertainty (inherent lap-to-lap randomness even if the "
                    "true rate were known) is handled by an AR(1) noise process within each "
                    "simulation: \u03b5\u2081 ~ Normal(0, \u03c3) and \u03b5\u209c ~ "
                    "Normal(\u03c1\u00b7\u03b5\u209c\u208b\u2081, \u03c3\u221a(1 \u2212 "
                    "\u03c1\u00b2)). The autocorrelation parameter \u03c1 captures the "
                    "empirical observation that a slow lap tends to be followed by another "
                    "slow lap. The AR(1) process resets at stint boundaries (tire changes). "
                    "This separation matters: collapsing both into a single noise term (as "
                    "many simpler models do) produces artificially wide distributions that "
                    "overstate strategy differentiation uncertainty.",
                    className="methodology-text",
                ),
                html.Div("Fuel Correction", className="methodology-heading"),
                html.P(
                    "The tire model is fit on fuel-corrected data, so alpha represents a "
                    "zero-fuel lap time. The simulator reconstructs on-track pace by adding "
                    "the fuel mass carried on each lap back in: Laptime = model_laptime + "
                    "(Total_Laps \u2212 Current_Lap + 1) \u00d7 Fuel_Per_Lap \u00d7 Weight_Effect. "
                    "Early laps carry the most fuel and are correctly the slowest for a given "
                    "tire state, with the penalty falling to near zero on the final lap. The "
                    "2026 regulations limit fuel by energy through a 3000 MJ/h flow cap rather "
                    "than mandating a starting fuel mass, so there is no regulated race fuel "
                    "quantity. The simulator uses a representative observed start-of-race load "
                    "of 92.5 kg with a 3 kg reserve, giving 89.5 kg usable distributed across "
                    "race laps. The lower energy density of the 2026 sustainable fuel means "
                    "cars carry more mass for a given race energy than the early 70 kg "
                    "projections suggested. The weight effect is fixed at 0.03 s/kg/lap, a "
                    "documented simplification, since the true value varies with circuit corner "
                    "profile and downforce level. Because the fuel term depends only on the "
                    "race lap index, it is identical across every strategy and does not affect "
                    "strategy ranking, win rates, or pit windows.",
                    className="methodology-text",
                ),
                html.Div("Base Pace", className="methodology-heading"),
                html.P(
                    "In posterior mode the fitted alpha carries absolute pace per compound, so "
                    "base pace is not used for a fitted compound. It anchors two cases: circuits "
                    "with no fit, where it is the zero-fuel reference lap for a fresh medium, and "
                    "circuits with a partial fit, where an unfit compound is anchored to a base "
                    "pace derived from the fitted alphas so all compounds share one scale. "
                    "Per-circuit values are held as representative race laps and converted to a "
                    "zero-fuel reference by removing one full fuel load of lap time; a "
                    "base_pace.json produced from fuel-corrected historical race pace overrides "
                    "these with directly derived values.",
                    className="methodology-text",
                ),
                html.Div("Monte Carlo Simulation", className="methodology-heading"),
                html.P(
                    "Each simulation run draws a base pace perturbation ~ Normal(0, 0.4s) "
                    "representing session-level variation (setup, conditions), then one "
                    "parameter set per compound from the posterior (or prior). For each lap, "
                    "the tire model produces a predicted time, fuel correction is applied, and "
                    "AR(1) noise is added. Pit stops incur a circuit-specific time penalty. "
                    "Running 100\u20132000 simulations per strategy produces empirical "
                    "distributions of total race time, from which percentile-based risk "
                    "metrics, median performance comparisons, and head-to-head win rate "
                    "calculations are derived.",
                    className="methodology-text",
                ),
                html.Div("Pit Window Optimization", className="methodology-heading"),
                html.P(
                    "A strategy is a compound sequence; the pit laps are solved rather "
                    "than supplied. Expected race time is linear in the tire parameters, "
                    "and the AR(1) noise, base pace perturbation, and fuel term either "
                    "average out or depend only on the race lap index, so none of them "
                    "can move the optimal stop. The expected-time surface over pit laps "
                    "is therefore closed-form, and the optimizer scans the feasible "
                    "stint-length partitions to find the minimum directly, with no "
                    "simulation in the loop. Monte Carlo then runs at that optimum to "
                    "produce the distribution and risk metrics. The reported window is "
                    "the set of pit laps whose expected time falls within a fraction of "
                    "the race-time standard deviation of the best, given as a lap range "
                    "per stop. A wide window means the timing is forgiving; a narrow one "
                    "means the stop must land on a specific lap.",
                    className="methodology-text",
                ),
                html.Div("Tire Set Assignment", className="methodology-heading"),
                html.P(
                    "When custom tire allocation is enabled, each stint runs on a "
                    "specific physical set with its own age, and the assignment of sets "
                    "to stints is solved jointly with the pit laps. Within a compound the "
                    "marginal cost of tire age rises with stint length, so assigning the "
                    "freshest set to the longest stint minimizes total time. This is exact "
                    "rather than heuristic, and it decomposes per compound since a set can "
                    "only fill a stint of its own compound.",
                    className="methodology-text",
                ),
                html.Div(
                    "Tire Life and Extrapolation Limits",
                    className="methodology-heading",
                ),
                html.P(
                    "Each compound's stint length is bounded by the evidence available "
                    "for it. In posterior mode the bound comes from the longest run "
                    "observed in practice for that compound, extended by a modest margin, "
                    "and in every mode it is also capped by a nominal race life per "
                    "compound. The quadratic degradation curve is a reliable local "
                    "description within the range it was fit on, and the bound keeps the "
                    "optimizer from extrapolating it into stint lengths that no data "
                    "supports. This matters most for the soft compound, whose practice "
                    "running is short and low-fuel, which otherwise makes long soft "
                    "stints look more viable than they are. Any solved stint that reaches "
                    "or exceeds the observed range or the nominal life is flagged as "
                    "extrapolation-limited, so the result is disclosed rather than hidden. "
                    "Per-circuit tire life is derived from historical race stint lengths, "
                    "with global defaults applied until that derivation is run.",
                    className="methodology-text",
                ),
                html.Div("Data Pipeline", className="methodology-heading"),
                html.P(
                    "A separate pipeline script (fit_models.py) ingests practice session data "
                    "via the FastF1 API, extracts stints, filters outlaps and anomalous laps, "
                    "fuel-corrects the times, and fits the quadratic Bayesian model per "
                    "compound via MCMC. The AR(1) autocorrelation parameter \u03c1 is estimated "
                    "post-hoc from model residuals. Posterior samples are serialized and loaded "
                    "by the web app at startup. The intended workflow is: practice session ends "
                    "\u2192 run pipeline \u2192 commit updated model files \u2192 redeploy.",
                    className="methodology-text",
                ),
                html.Div("Sensitivity Analysis", className="methodology-heading"),
                html.P(
                    "After the base simulation, a sensitivity analysis tests whether the "
                    "ranking is robust to input uncertainty. Two parameters are swept, "
                    "independently and jointly: pit loss (default \u00b14s in 0.5s steps) "
                    "and a degradation multiplier (0.70x to 1.30x in 0.05 steps) that "
                    "scales \u03b2 and \u03b3 together while preserving their ratio. Every "
                    "point in the sweep re-solves the optimal pit split for each strategy, "
                    "so strategies are compared at their own optima throughout rather than "
                    "at a fixed split, and because the optimization is closed-form this "
                    "adds little cost. The pit loss and degradation tabs show median race "
                    "time with the crossover points where the optimal strategy changes; "
                    "the Optimal Pit Lap tab shows how each strategy's solved pit lap and "
                    "its window respond to degradation. Pit loss changes which stop count "
                    "is fastest overall, which appears as strategies trading rank rather "
                    "than as pit laps moving, since the pit lap within a fixed sequence is "
                    "degradation-driven. A coarser 2D sweep maps which strategy is optimal "
                    "across the joint parameter space. This may take a minute or longer to "
                    "run, depending on the number of strategies.",
                    className="methodology-text",
                ),
                html.Div("Known Limitations", className="methodology-heading"),
                html.P(
                    "The quadratic degradation form is parametric and does not capture "
                    "complex thermal or chemical dynamics. The weight effect (0.03 "
                    "s/kg/lap) is fixed across circuits. Prior parameters are informed by "
                    "historical ranges but not hierarchically fit across circuits. \u03c1 "
                    "is estimated from residuals rather than jointly within the MCMC. Base "
                    "pace variance (0.4s) is not empirically derived from session data. "
                    "The optimizer minimizes expected race time using posterior or prior "
                    "means, so it targets the mean-optimal stop rather than a median or "
                    "risk-adjusted one. The window tolerance is a fixed fraction of "
                    "race-time dispersion rather than a formal significance threshold. "
                    "Nominal tire life defaults are domain priors until per-circuit values "
                    "are derived from historical stint lengths. No modeling of track "
                    "evolution, traffic, weather, safety cars, or driver-specific "
                    "performance.",
                    className="methodology-text",
                ),
            ],
            className="welcome-detail",
        ),
    ],
    id="welcome-section",
    className="welcome-container",
)

main_content = html.Div(
    [
        welcome_content,
        html.Div(id="strategy-editor-container"),
        html.Div(id="strategy-display"),
        dcc.Loading(
            html.Div(id="results-section"),
            type="circle", color="#e10600",
        ),
        dcc.Store(id="results-store"),
        dcc.Store(id="custom-strategy-store"),
        dcc.Store(id="sensitivity-store"),
        dcc.Loading(
            html.Div(id="sensitivity-section"),
            type="circle", color="#e10600",
        ),
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-summary"),
    ],
    className="main-content",
)

app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(sidebar, lg=3, md=4, sm=12, className="p-0"),
                dbc.Col(main_content, lg=9, md=8, sm=12, className="p-0"),
            ],
            className="g-0",
        ),
        html.Div("@formulasteele", className="app-footer"),
    ],
    fluid=True, className="p-0",
)


@app.callback(
    Output("circuit-info", "children"),
    Input("circuit-dropdown", "value"),
)
def update_circuit_info(circuit):
    if not circuit:
        return []
    info = simulator.circuits[circuit]
    fpl = simulator.fuel_per_lap(circuit)
    has_post = simulator.has_posterior(circuit)
    badge_class = (
        "model-badge model-badge-bayesian" if has_post
        else "model-badge model-badge-prior"
    )
    badge_text = "Posterior model" if has_post else "Prior model"
    return html.Div(
        [
            html.Div(info["gp_name"], className="info-card-title"),
            html.Div(
                [
                    html.Div(f"Laps: {info['laps']}"),
                    html.Div(f"Distance: {info['distance_km']:.3f} km/lap"),
                    html.Div(f"Fuel/lap: {fpl:.2f} kg"),
                ],
                className="info-card-detail",
            ),
            html.Span(badge_text, className=badge_class),
        ],
        className="info-card",
    )


@app.callback(
    Output("base-pace-input", "value"),
    Output("pit-loss-input", "value"),
    Input("circuit-dropdown", "value"),
)
def update_defaults(circuit):
    if not circuit:
        return 80.0, 22.0
    return CIRCUIT_BASE_PACES.get(circuit, 80.0), CIRCUIT_PIT_LOSSES.get(circuit, 22.0)


@app.callback(
    Output("tire-allocation-section", "children"),
    Input("custom-tires-toggle", "value"),
)
def render_tire_sets(toggle):
    if "on" not in (toggle or []):
        return []
    rows = []
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        html.Span(compound, style={
                            "color": COMPOUND_COLORS[compound],
                            "fontSize": "12px",
                            "fontFamily": "JetBrains Mono",
                            "fontWeight": "500",
                        }),
                        width=4, className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        dcc.Input(
                            id=f"tire-sets-{compound.lower()}",
                            type="number",
                            value=2, min=0, max=5, step=1,
                            style={"fontSize": "12px"},
                        ),
                        width=8,
                    ),
                ],
                className="mb-1 g-1",
            )
        )
    return html.Div(
        [
            html.Div(
                "Sets per compound",
                style={"fontSize": "10px", "color": "#9ca3af", "marginBottom": "4px"},
            ),
            *rows,
            html.Hr(style={
                "borderColor": "#e5e7eb", "margin": "10px 0", "opacity": "0.5",
            }),
            html.Div(id="tire-ages-section"),
        ],
        className="mt-2",
    )


@app.callback(
    Output("tire-ages-section", "children"),
    Input("tire-sets-soft", "value"),
    Input("tire-sets-medium", "value"),
    Input("tire-sets-hard", "value"),
    prevent_initial_call=True,
)
def render_tire_ages(soft_sets, medium_sets, hard_sets):
    counts = {
        "SOFT": int(soft_sets or 0),
        "MEDIUM": int(medium_sets or 0),
        "HARD": int(hard_sets or 0),
    }
    sections = []
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        n = counts[compound]
        if n == 0:
            continue
        sections.append(
            html.Div(
                f"{compound} age per set (laps)",
                style={
                    "color": COMPOUND_COLORS[compound],
                    "fontSize": "10px",
                    "fontFamily": "JetBrains Mono",
                    "marginTop": "6px",
                    "marginBottom": "4px",
                },
            )
        )
        for i in range(n):
            sections.append(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Span(
                                f"Set {i + 1}",
                                style={
                                    "fontSize": "10px",
                                    "color": "#9ca3af",
                                    "paddingLeft": "6px",
                                },
                            ),
                            width=3, className="d-flex align-items-center",
                        ),
                        dbc.Col(
                            dcc.Input(
                                id={
                                    "type": "tire-age",
                                    "compound": compound,
                                    "set": i,
                                },
                                type="number",
                                value=0, min=0, max=50, step=1,
                                style={"fontSize": "12px"},
                            ),
                            width=9,
                        ),
                    ],
                    className="mb-1 g-1",
                )
            )
    if not sections:
        return html.Div(
            "No tire sets allocated",
            style={"fontSize": "11px", "color": "#9ca3af"},
        )
    return html.Div(sections)


@app.callback(
    Output("tire-allocation-store", "data"),
    Input({"type": "tire-age", "compound": ALL, "set": ALL}, "value"),
    State({"type": "tire-age", "compound": ALL, "set": ALL}, "id"),
)
def sync_tire_allocation(ages, ids):
    if not ages or not ids:
        return None
    allocation = []
    for age_val, id_dict in zip(ages, ids):
        allocation.append({
            "compound": id_dict["compound"],
            "age_laps": int(age_val or 0),
        })
    return allocation


@app.callback(
    Output("run-button", "disabled"),
    Input("circuit-dropdown", "value"),
    Input("strategy-dropdown", "value"),
)
def toggle_run_button(circuit, strategies):
    return not (circuit and strategies)


@app.callback(
    Output("strategy-editor-container", "children"),
    Input("editor-toggle", "value"),
    Input("circuit-dropdown", "value"),
    Input("strategy-dropdown", "value"),
)
def render_strategy_editor(toggle, circuit, strategies):
    if "on" not in (toggle or []) or not circuit or not strategies:
        return []

    circuit_laps = simulator.circuits[circuit]["laps"]
    cards = []

    for s_idx, name in enumerate(strategies):
        scaled = default_partition(ALL_STRATEGIES[name], circuit)

        stint_rows = []
        for t_idx, stint in enumerate(scaled):
            idx_key = f"{s_idx}_{t_idx}"
            stint_rows.append(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Span(
                                f"Stint {t_idx + 1}",
                                style={
                                    "fontSize": "11px",
                                    "color": "#6b7280",
                                    "fontFamily": "Barlow Semi Condensed",
                                    "fontWeight": "500",
                                },
                            ),
                            width=2, className="d-flex align-items-center",
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id={"type": "stint-compound", "idx": idx_key},
                                options=[
                                    {"label": "SOFT", "value": "SOFT"},
                                    {"label": "MEDIUM", "value": "MEDIUM"},
                                    {"label": "HARD", "value": "HARD"},
                                ],
                                value=stint["compound"],
                                clearable=False,
                                style={"fontSize": "12px"},
                            ),
                            width=5,
                        ),
                        dbc.Col(
                            dcc.Input(
                                id={"type": "stint-laps", "idx": idx_key},
                                type="number",
                                value=stint["laps"],
                                min=1, max=circuit_laps,
                                style={"fontSize": "12px"},
                            ),
                            width=3,
                        ),
                        dbc.Col(
                            html.Span(
                                "laps",
                                style={"fontSize": "10px", "color": "#9ca3af"},
                            ),
                            width=2, className="d-flex align-items-center",
                        ),
                    ],
                    className="mb-2 g-2",
                )
            )

        cards.append(
            html.Div(
                [
                    html.Div(name, className="strategy-card-name"),
                    *stint_rows,
                    html.Div(
                        id={"type": "stint-total", "idx": str(s_idx)},
                        className="editor-total",
                    ),
                ],
                className="editor-card",
            )
        )

    return html.Div(
        [
            html.Div("STRATEGY EDITOR", className="sidebar-section-label"),
            *cards,
        ],
        className="mb-3",
    )


@app.callback(
    Output({"type": "stint-total", "idx": ALL}, "children"),
    Output({"type": "stint-total", "idx": ALL}, "className"),
    Input({"type": "stint-laps", "idx": ALL}, "value"),
    State({"type": "stint-laps", "idx": ALL}, "id"),
    State({"type": "stint-total", "idx": ALL}, "id"),
    State("circuit-dropdown", "value"),
)
def update_stint_totals(lap_values, lap_ids, total_ids, circuit):
    if not circuit or not lap_values or not total_ids:
        return (
            ["" for _ in (total_ids or [])],
            ["editor-total" for _ in (total_ids or [])],
        )

    circuit_laps = simulator.circuits[circuit]["laps"]
    strategy_laps = {}
    for val, id_dict in zip(lap_values, lap_ids):
        s_idx = id_dict["idx"].split("_")[0]
        strategy_laps.setdefault(s_idx, 0)
        strategy_laps[s_idx] += int(val or 0)

    texts = []
    classes = []
    for total_id in total_ids:
        s_idx = total_id["idx"]
        total = strategy_laps.get(s_idx, 0)
        if total == circuit_laps:
            texts.append(f"{total} / {circuit_laps} laps")
            classes.append("editor-total editor-total-valid")
        else:
            diff = total - circuit_laps
            sign = "+" if diff > 0 else ""
            texts.append(f"{total} / {circuit_laps} laps ({sign}{diff})")
            classes.append("editor-total editor-total-invalid")

    return texts, classes


@app.callback(
    Output("custom-strategy-store", "data"),
    Input({"type": "stint-compound", "idx": ALL}, "value"),
    Input({"type": "stint-laps", "idx": ALL}, "value"),
    State({"type": "stint-compound", "idx": ALL}, "id"),
    State({"type": "stint-laps", "idx": ALL}, "id"),
    State("strategy-dropdown", "value"),
)
def sync_custom_strategies(compounds, laps, compound_ids, lap_ids, strategies):
    if not strategies or not compounds or not laps:
        return None

    stints_by_strategy = {}
    for comp_val, comp_id in zip(compounds, compound_ids):
        parts = comp_id["idx"].split("_")
        s_idx, t_idx = int(parts[0]), int(parts[1])
        stints_by_strategy.setdefault(s_idx, {}).setdefault(t_idx, {})
        stints_by_strategy[s_idx][t_idx]["compound"] = comp_val

    for lap_val, lap_id in zip(laps, lap_ids):
        parts = lap_id["idx"].split("_")
        s_idx, t_idx = int(parts[0]), int(parts[1])
        stints_by_strategy.setdefault(s_idx, {}).setdefault(t_idx, {})
        stints_by_strategy[s_idx][t_idx]["laps"] = int(lap_val or 1)

    result = {}
    for s_idx in sorted(stints_by_strategy.keys()):
        if s_idx < len(strategies):
            name = strategies[s_idx]
            stint_dict = stints_by_strategy[s_idx]
            result[name] = [
                stint_dict[t_idx] for t_idx in sorted(stint_dict.keys())
            ]

    return result


@app.callback(
    Output("strategy-display", "children"),
    Input("circuit-dropdown", "value"),
    Input("strategy-dropdown", "value"),
    Input("editor-toggle", "value"),
    Input("custom-strategy-store", "data"),
)
def update_strategy_display(circuit, strategies, editor_toggle, custom_data):
    if not circuit or not strategies:
        return []

    circuit_laps = simulator.circuits[circuit]["laps"]
    use_custom = "on" in (editor_toggle or []) and custom_data
    cards = []

    for name in strategies:
        if use_custom and name in custom_data:
            scaled = custom_data[name]
            compounds = "-".join(s["compound"][0] for s in scaled)
            stops = len(scaled) - 1
            label = f"{stops}-Stop: {compounds}"
        else:
            scaled = default_partition(ALL_STRATEGIES[name], circuit)
            label = name
        cards.append(
            html.Div(
                [
                    html.Div(label, className="strategy-card-name"),
                    make_stint_sequence(scaled),
                ],
                className="strategy-card",
            )
        )

    return html.Div(
        [html.Div("STRATEGIES", className="sidebar-section-label"), *cards],
        className="mb-3",
    )


@app.callback(
    Output("results-store", "data"),
    Output("run-status", "children"),
    Input("run-button", "n_clicks"),
    State("circuit-dropdown", "value"),
    State("strategy-dropdown", "value"),
    State("base-pace-input", "value"),
    State("pit-loss-input", "value"),
    State("sims-slider", "value"),
    State("custom-tires-toggle", "value"),
    State("tire-allocation-store", "data"),
    State("editor-toggle", "value"),
    State("custom-strategy-store", "data"),
    prevent_initial_call=True,
)
def run_simulation(n_clicks, circuit, strategies, pace, pit, sims,
                   tire_toggle, tire_data, editor_toggle, custom_data):
    if not circuit or not strategies:
        return None, ""

    pace = float(pace or 80.0)
    pit = float(pit or 22.0)
    sims = int(sims or 1000)

    tire_allocation = None
    if "on" in (tire_toggle or []) and tire_data:
        tire_allocation = tire_data

    circuit_laps = simulator.circuits[circuit]["laps"]
    use_custom = "on" in (editor_toggle or []) and custom_data
    results = {}
    opt_meta = {}
    errors = []

    for name in strategies:
        custom_here = use_custom and name in custom_data
        if custom_here:
            sequence = [s["compound"] for s in custom_data[name]]
            lengths = [int(s["laps"]) for s in custom_data[name]]
            compounds = "-".join(c[0] for c in sequence)
            stops = len(sequence) - 1
            label = f"{stops}-Stop: {compounds}"
            if sum(lengths) != circuit_laps:
                errors.append(
                    f"{label}: total laps ({sum(lengths)}) != circuit ({circuit_laps})"
                )
                continue
        else:
            sequence = ALL_STRATEGIES[name]
            label = name

        try:
            if custom_here:
                res = simulator.resolve_fixed(
                    circuit, sequence, lengths, tire_allocation,
                )
                resolved = res["resolved"]
                window, window_n, cf_mean = None, None, None
                pit_laps = res["pit_laps"]
                extrap = res["extrapolation"]
            else:
                opt = simulator.optimize(
                    circuit, sequence, tire_allocation, pace, pit,
                )
                resolved = opt["resolved"]
                pit_laps = opt["pit_laps"]
                cf_mean = opt["closed_form_mean"]
                extrap = opt["extrapolation"]

            times = simulator.simulate(
                circuit, resolved, None, pace, pit, sims, pre_resolved=True,
            )
            results[label] = times.tolist()

            if not custom_here:
                tol = WINDOW_TOL_FRAC * float(np.std(times))
                window, window_n = pit_window_from_surface(
                    opt["surface"], cf_mean, tol,
                )

            opt_meta[label] = {
                "sequence": sequence,
                "lengths": [s["laps"] for s in resolved],
                "ages": [s["tire_age"] for s in resolved],
                "pit_laps": pit_laps,
                "window": window,
                "window_n": window_n,
                "closed_form_mean": cf_mean,
                "extrapolation": extrap,
                "custom": custom_here,
            }
        except Exception as e:
            errors.append(f"{label}: {str(e)}")

    if errors:
        status = html.Div(
            [html.Div(e, style={"color": "#ef4444", "fontSize": "12px"}) for e in errors]
        )
    else:
        model_type = (
            "Posterior" if simulator.has_posteriors and simulator.has_posterior(circuit)
            else "Prior"
        )
        status = html.Div(
            f"Complete ({model_type} model)",
            style={
                "color": "#22c55e", "fontSize": "12px",
                "fontFamily": "JetBrains Mono",
            },
        )

    if not results:
        return None, status

    return {"results": results, "opt": opt_meta, "circuit": circuit}, status


@app.callback(
    Output("results-section", "children"),
    Output("strategy-display", "style"),
    Output("strategy-editor-container", "style"),
    Input("results-store", "data"),
)
def display_results(data):
    if not data:
        return [], {"display": "block"}, {"display": "block"}

    hide = {"display": "none"}
    results = {k: np.array(v) for k, v in data["results"].items()}
    circuit = data["circuit"]
    names = list(results.keys())
    n = len(names)

    dist_fig = go.Figure()
    for i, name in enumerate(names):
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        times = results[name]
        dist_fig.add_trace(go.Histogram(
            x=times, name=name, marker_color=color, opacity=0.6,
            histnorm="probability density", nbinsx=40,
        ))
        med = np.median(times)
        dist_fig.add_vline(
            x=med, line_dash="dash", line_color=color, line_width=2,
        )
    dist_fig.update_layout(
        **chart_layout("Performance Distribution"),
        barmode="overlay",
        xaxis_title="Race Time (s)", yaxis_title="Density",
    )

    box_fig = go.Figure()
    for i, name in enumerate(names):
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        box_fig.add_trace(go.Box(
            y=results[name], name=name,
            marker_color=color, line_color=color,
            fillcolor=f"rgba({r},{g},{b},0.3)",
        ))
    box_fig.update_layout(
        **chart_layout("Performance Spread"),
        yaxis_title="Race Time (s)", showlegend=False,
    )

    cdf_fig = go.Figure()
    for i, name in enumerate(names):
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        st = np.sort(results[name])
        probs = np.arange(1, len(st) + 1) / len(st)
        cdf_fig.add_trace(go.Scatter(
            x=st, y=probs, name=name,
            line=dict(color=color, width=2), mode="lines",
        ))
    cdf_fig.update_layout(
        **chart_layout("Cumulative Distribution"),
        xaxis_title="Race Time (s)", yaxis_title="Cumulative Probability",
    )

    medians = [np.median(results[s]) for s in names]
    p5s = [np.percentile(results[s], 5) for s in names]
    p95s = [np.percentile(results[s], 95) for s in names]
    colors = [STRATEGY_COLORS[i % len(STRATEGY_COLORS)] for i in range(n)]

    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(
        x=names, y=medians, marker_color=colors, opacity=0.85,
        error_y=dict(
            type="data", symmetric=False,
            array=[p95 - med for p95, med in zip(p95s, medians)],
            arrayminus=[med - p5 for p5, med in zip(p5s, medians)],
            color="#9ca3af", thickness=1.5,
        ),
    ))
    comp_fig.update_layout(
        **chart_layout("Median Race Time (5th\u201395th percentile)"),
        yaxis_title="Race Time (s)", showlegend=False,
    )

    summary = []
    for name in names:
        t = results[name]
        summary.append({
            "Strategy": name,
            "Median": f"{np.median(t):.1f}",
            "Mean": f"{np.mean(t):.1f}",
            "Std Dev": f"{np.std(t):.1f}",
            "5th Pctl": f"{np.percentile(t, 5):.1f}",
            "95th Pctl": f"{np.percentile(t, 95):.1f}",
            "Range (5-95)": f"{np.percentile(t, 95) - np.percentile(t, 5):.1f}",
        })
    summary_df = pd.DataFrame(summary).sort_values("Median")

    best_median = float(summary_df["Median"].min())
    risk = []
    for _, row in summary_df.iterrows():
        penalty = float(row["Median"]) - best_median
        r = float(row["Range (5-95)"]) / 2
        risk.append({
            "Strategy": row["Strategy"],
            "Time Penalty": f"+{penalty:.1f}s",
            "Risk": f"\u00b1{r:.1f}s",
        })
    risk_df = pd.DataFrame(risk)

    h2h = []
    for s1 in names:
        row = {"Strategy": s1}
        for s2 in names:
            row[s2] = (
                "\u2014" if s1 == s2
                else f"{np.mean(results[s1] < results[s2]):.0%}"
            )
        h2h.append(row)
    h2h_df = pd.DataFrame(h2h)

    best_name = summary_df.iloc[0]["Strategy"]
    best_time = summary_df.iloc[0]["Median"]
    cons_idx = summary_df["Std Dev"].astype(float).idxmin()
    cons_name = summary_df.loc[cons_idx, "Strategy"]
    cons_std = summary_df.loc[cons_idx, "Std Dev"]

    opt = data.get("opt", {})
    window_rows = []
    any_flag = False
    for name in summary_df["Strategy"]:
        meta = opt.get(name, {})
        pit_laps = meta.get("pit_laps") or []
        extrap = meta.get("extrapolation") or []
        flagged = [e for e in extrap if e.get("flagged")]
        if flagged:
            any_flag = True
        note = ""
        if meta.get("custom"):
            note = "manual split"
        elif flagged:
            note = "extrapolation-limited"
        window_rows.append({
            "Strategy": name,
            "Stops": len(pit_laps),
            "Optimal Pit Lap(s)": format_pit_laps(pit_laps),
            "Window": format_window(meta.get("window")),
            "Note": note or "\u2014",
        })
    window_df = pd.DataFrame(window_rows)

    window_section = [
        html.Div("OPTIMAL PIT WINDOWS", className="sidebar-section-label"),
        dash_table.DataTable(
            data=window_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in window_df.columns],
            style_header=TABLE_HEADER, style_cell=TABLE_CELL,
            style_data_conditional=TABLE_CONDITIONAL,
            style_table={"overflowX": "auto"},
        ),
    ]
    if any_flag:
        window_section.append(
            html.Div(
                "Extrapolation-limited strategies place a stint at or beyond the "
                "longest run observed in practice for that compound. The optimum "
                "reflects out-of-sample degradation extrapolation and should be "
                "read with caution.",
                style={
                    "fontSize": "11px", "color": "#854d0e",
                    "fontFamily": "JetBrains Mono", "marginTop": "8px",
                    "lineHeight": "1.6", "maxWidth": "720px",
                },
            )
        )

    return html.Div([
        html.Div(
            [
                dbc.Col(html.Div([
                    html.Div("FASTEST STRATEGY", className="metric-label"),
                    html.Div(best_name, className="metric-value"),
                    html.Div(f"{best_time}s median", className="metric-sub"),
                ], className="metric-card")),
                dbc.Col(html.Div([
                    html.Div("MOST CONSISTENT", className="metric-label"),
                    html.Div(cons_name, className="metric-value"),
                    html.Div(
                        f"\u00b1{cons_std}s std dev", className="metric-sub",
                    ),
                ], className="metric-card")),
            ],
            className="d-flex gap-3 mb-4",
        ),
        html.Div(window_section, className="mb-4"),
        html.Div([
            dbc.Button(
                "Export Raw Data", id="export-raw-btn",
                className="export-btn", n_clicks=0,
            ),
            dbc.Button(
                "Export Summary", id="export-summary-btn",
                className="export-btn", n_clicks=0,
            ),
        ], className="mb-3"),
        dbc.Tabs([
            dbc.Tab(
                dcc.Graph(figure=dist_fig, config={"displayModeBar": False}),
                label="Distribution",
            ),
            dbc.Tab(
                dcc.Graph(figure=box_fig, config={"displayModeBar": False}),
                label="Spread",
            ),
            dbc.Tab(
                dcc.Graph(figure=comp_fig, config={"displayModeBar": False}),
                label="Comparison",
            ),
            dbc.Tab(
                dcc.Graph(figure=cdf_fig, config={"displayModeBar": False}),
                label="CDF",
            ),
        ], className="mb-4"),
        html.Div("SUMMARY", className="sidebar-section-label"),
        dash_table.DataTable(
            data=summary_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in summary_df.columns],
            style_header=TABLE_HEADER, style_cell=TABLE_CELL,
            style_data_conditional=TABLE_CONDITIONAL,
            style_table={"overflowX": "auto"},
        ),
        html.Div("RISK ANALYSIS", className="sidebar-section-label mt-4"),
        dash_table.DataTable(
            data=risk_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in risk_df.columns],
            style_header=TABLE_HEADER, style_cell=TABLE_CELL,
            style_data_conditional=TABLE_CONDITIONAL,
            style_table={"overflowX": "auto"},
        ),
        html.Div("HEAD-TO-HEAD WIN RATES", className="sidebar-section-label mt-4"),
        dash_table.DataTable(
            data=h2h_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in h2h_df.columns],
            style_header=TABLE_HEADER, style_cell=TABLE_CELL,
            style_data_conditional=TABLE_CONDITIONAL,
            style_table={"overflowX": "auto"},
        ),
    ]), hide, hide


@app.callback(
    Output("download-csv", "data"),
    Input("export-raw-btn", "n_clicks"),
    State("results-store", "data"),
    prevent_initial_call=True,
)
def export_raw(n_clicks, data):
    if not data or not n_clicks:
        return None
    rows = []
    for name, times in data["results"].items():
        for i, t in enumerate(times):
            rows.append({
                "Strategy": name, "Simulation": i + 1,
                "Race_Time_s": round(t, 3), "Circuit": data["circuit"],
            })
    slug = data["circuit"].lower().replace(" ", "_").replace("-", "_")
    return dcc.send_data_frame(
        pd.DataFrame(rows).to_csv,
        f"f1_raw_{slug}.csv", index=False,
    )


@app.callback(
    Output("download-summary", "data"),
    Input("export-summary-btn", "n_clicks"),
    State("results-store", "data"),
    prevent_initial_call=True,
)
def export_summary(n_clicks, data):
    if not data or not n_clicks:
        return None
    rows = []
    for name, times in data["results"].items():
        t = np.array(times)
        rows.append({
            "Strategy": name,
            "Median_s": round(float(np.median(t)), 1),
            "Mean_s": round(float(np.mean(t)), 1),
            "Std_Dev_s": round(float(np.std(t)), 1),
            "P5_s": round(float(np.percentile(t, 5)), 1),
            "P95_s": round(float(np.percentile(t, 95)), 1),
            "Circuit": data["circuit"],
        })
    slug = data["circuit"].lower().replace(" ", "_").replace("-", "_")
    return dcc.send_data_frame(
        pd.DataFrame(rows).sort_values("Median_s").to_csv,
        f"f1_summary_{slug}.csv", index=False,
    )


@app.callback(
    Output("welcome-section", "style"),
    Input("circuit-dropdown", "value"),
    Input("strategy-dropdown", "value"),
)
def toggle_welcome(circuit, strategies):
    if not circuit and not strategies:
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("circuit-dropdown", "value", allow_duplicate=True),
    Output("strategy-dropdown", "value", allow_duplicate=True),
    Output("results-store", "data", allow_duplicate=True),
    Output("run-status", "children", allow_duplicate=True),
    Output("custom-tires-toggle", "value", allow_duplicate=True),
    Output("editor-toggle", "value", allow_duplicate=True),
    Output("sensitivity-store", "data", allow_duplicate=True),
    Output("sensitivity-status", "children", allow_duplicate=True),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_simulator(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return None, None, None, "", [], [], None, ""


@app.callback(
    Output("sensitivity-button", "disabled"),
    Input("results-store", "data"),
)
def toggle_sensitivity_button(data):
    return data is None


def _resolve_for_point(circuit, spec, base_pace, pit_loss, deg_multiplier,
                       tire_allocation):
    if spec["fixed"]:
        res = simulator.resolve_fixed(
            circuit, spec["sequence"], spec["lengths"], tire_allocation,
        )
        return res["resolved"], res["pit_laps"], None
    opt = simulator.optimize(
        circuit, spec["sequence"], tire_allocation, base_pace,
        pit_loss, deg_multiplier,
    )
    return opt["resolved"], opt["pit_laps"], opt


def _point_window(opt, times):
    if opt is None:
        return None
    tol = WINDOW_TOL_FRAC * float(np.std(times))
    window, _ = pit_window_from_surface(
        opt["surface"], opt["closed_form_mean"], tol,
    )
    return window


def run_pit_loss_sweep(circuit, specs, base_pace, default_pit,
                       tire_allocation, sims_per_point=300):
    pit_losses = np.arange(
        max(12.0, default_pit - 4.0),
        default_pit + 4.5,
        0.5,
    )
    results = {name: [] for name in specs}
    for pl in pit_losses:
        for name, spec in specs.items():
            resolved, pit_laps, opt = _resolve_for_point(
                circuit, spec, base_pace, float(pl), 1.0, tire_allocation,
            )
            times = simulator.simulate(
                circuit, resolved, None, base_pace, float(pl),
                sims_per_point, pre_resolved=True,
            )
            results[name].append({
                "pit_loss": float(pl),
                "median": float(np.median(times)),
                "p5": float(np.percentile(times, 5)),
                "p95": float(np.percentile(times, 95)),
                "pit_laps": pit_laps,
                "window": _point_window(opt, times),
            })
    return results, pit_losses.tolist()


def run_deg_sweep(circuit, specs, base_pace, pit_loss,
                  tire_allocation, sims_per_point=300):
    deg_multipliers = np.arange(0.70, 1.35, 0.05)
    results = {name: [] for name in specs}
    for dm in deg_multipliers:
        for name, spec in specs.items():
            resolved, pit_laps, opt = _resolve_for_point(
                circuit, spec, base_pace, pit_loss, float(dm), tire_allocation,
            )
            times = simulator.simulate(
                circuit, resolved, None, base_pace, pit_loss,
                sims_per_point, deg_multiplier=float(dm), pre_resolved=True,
            )
            results[name].append({
                "deg_multiplier": float(dm),
                "median": float(np.median(times)),
                "p5": float(np.percentile(times, 5)),
                "p95": float(np.percentile(times, 95)),
                "pit_laps": pit_laps,
                "window": _point_window(opt, times),
            })
    return results, deg_multipliers.tolist()


def run_2d_sweep(circuit, specs, base_pace, default_pit,
                 tire_allocation, sims_per_point=200):
    pit_losses = np.arange(
        max(12.0, default_pit - 4.0),
        default_pit + 4.5,
        1.0,
    )
    deg_multipliers = np.arange(0.70, 1.35, 0.10)
    grid = {}
    for dm in deg_multipliers:
        for pl in pit_losses:
            medians = {}
            for name, spec in specs.items():
                resolved, _, _ = _resolve_for_point(
                    circuit, spec, base_pace, float(pl), float(dm),
                    tire_allocation,
                )
                times = simulator.simulate(
                    circuit, resolved, None, base_pace, float(pl),
                    sims_per_point, deg_multiplier=float(dm), pre_resolved=True,
                )
                medians[name] = float(np.median(times))
            grid[f"{pl:.1f}_{dm:.2f}"] = medians
    return grid, pit_losses.tolist(), deg_multipliers.tolist()


def find_crossovers(sweep_results, param_key, strategy_names):
    crossovers = []
    param_vals = [r[param_key] for r in sweep_results[strategy_names[0]]]
    for i in range(len(param_vals) - 1):
        medians_a = {n: sweep_results[n][i]["median"] for n in strategy_names}
        medians_b = {n: sweep_results[n][i + 1]["median"] for n in strategy_names}
        best_a = min(medians_a, key=medians_a.get)
        best_b = min(medians_b, key=medians_b.get)
        if best_a != best_b:
            val_a, val_b = param_vals[i], param_vals[i + 1]
            gap_a = medians_a[best_b] - medians_a[best_a]
            gap_b = medians_b[best_b] - medians_b[best_a]
            if abs(gap_a - gap_b) > 1e-6:
                frac = gap_a / (gap_a - gap_b)
                crossover_val = val_a + frac * (val_b - val_a)
            else:
                crossover_val = (val_a + val_b) / 2
            crossovers.append({
                "param_value": round(crossover_val, 2),
                "from_strategy": best_a,
                "to_strategy": best_b,
            })
    return crossovers


def build_sweep_figure(sweep_results, param_key, param_label, default_val,
                       crossovers, strategy_names, circuit_name):
    fig = go.Figure()
    for i, name in enumerate(strategy_names):
        data = sweep_results[name]
        x = [d[param_key] for d in data]
        y = [d["median"] for d in data]
        p5 = [d["p5"] for d in data]
        p95 = [d["p95"] for d in data]
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=p95 + p5[::-1],
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.08)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode="lines",
            line=dict(color=color, width=2),
        ))
    fig.add_vline(
        x=default_val, line_dash="dash", line_color="#9ca3af", line_width=1,
    )
    for cx in crossovers:
        fig.add_vline(
            x=cx["param_value"], line_dash="dot",
            line_color="#d47264", line_width=1,
        )
    fig.update_layout(
        **chart_layout(f"Strategy Sensitivity to {param_label} \u2014 {circuit_name}"),
        xaxis_title=param_label, yaxis_title="Median Race Time (s)",
    )
    return fig


def build_relative_figure(sweep_results, param_key, param_label, default_val,
                          strategy_names, circuit_name):
    fig = go.Figure()
    n_points = len(sweep_results[strategy_names[0]])
    best_at_point = []
    for j in range(n_points):
        medians = {n: sweep_results[n][j]["median"] for n in strategy_names}
        best_at_point.append(min(medians.values()))
    for i, name in enumerate(strategy_names):
        data = sweep_results[name]
        x = [d[param_key] for d in data]
        y = [d["median"] - best_at_point[j] for j, d in enumerate(data)]
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode="lines",
            line=dict(color=color, width=2),
        ))
    fig.add_vline(
        x=default_val, line_dash="dash", line_color="#9ca3af", line_width=1,
    )
    fig.add_hline(y=0, line_color="#111827", line_width=0.8, opacity=0.3)
    fig.update_layout(
        **chart_layout(f"Delta to Fastest \u2014 {param_label} Sweep \u2014 {circuit_name}"),
        xaxis_title=param_label, yaxis_title="Delta to Fastest (s)",
    )
    return fig


def build_pitlap_figure(sweep_results, param_key, param_label, default_val,
                        strategy_names, circuit_name):
    fig = go.Figure()
    for i, name in enumerate(strategy_names):
        data = sweep_results[name]
        x = [d[param_key] for d in data]
        pit_series = [d.get("pit_laps") or [] for d in data]
        windows = [d.get("window") for d in data]
        n_stops = max((len(p) for p in pit_series), default=0)
        if n_stops == 0:
            continue
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        for stop_idx in range(n_stops):
            xb, lob, hib = [], [], []
            for xv, w in zip(x, windows):
                if w and stop_idx < len(w):
                    xb.append(xv)
                    lob.append(w[stop_idx][0])
                    hib.append(w[stop_idx][1])
            if xb:
                fig.add_trace(go.Scatter(
                    x=xb + xb[::-1], y=hib + lob[::-1],
                    fill="toself", fillcolor=f"rgba({r},{g},{b},0.10)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))

            y = [p[stop_idx] if stop_idx < len(p) else None for p in pit_series]
            trace_name = name if stop_idx == 0 else f"{name} \u00b7 stop {stop_idx + 1}"
            fig.add_trace(go.Scatter(
                x=x, y=y, name=trace_name, mode="lines",
                line=dict(
                    color=color, width=2,
                    dash="solid" if stop_idx == 0 else "dot",
                ),
            ))

    fig.add_vline(
        x=default_val, line_dash="dash", line_color="#9ca3af", line_width=1,
    )
    fig.update_layout(
        **chart_layout(f"Optimal Pit Lap vs {param_label} \u2014 {circuit_name}"),
        xaxis_title=param_label, yaxis_title="Optimal Pit Lap (shaded: window)",
    )
    return fig


def build_heatmap_figure(grid, pit_losses, deg_multipliers, strategy_names,
                         default_pit, circuit_name):
    baseline_medians = {}
    for key in grid:
        pl_str, dm_str = key.split("_")
        if abs(float(dm_str) - 1.0) < 0.05:
            closest_pl = min(pit_losses, key=lambda x: abs(x - default_pit))
            if abs(float(pl_str) - closest_pl) < 0.5:
                baseline_medians = grid[key]
                break

    if baseline_medians:
        sorted_strats = sorted(
            baseline_medians.keys(), key=lambda k: baseline_medians[k],
        )
        competitive = sorted_strats[:4]
    else:
        competitive = strategy_names[:4]

    strat_to_idx = {name: i for i, name in enumerate(competitive)}
    palette = ["#2066a8", "#8ec1da", "#f6d6c2", "#d47264"]

    z = []
    for dm in deg_multipliers:
        row = []
        for pl in pit_losses:
            key = f"{pl:.1f}_{dm:.2f}"
            if key in grid:
                medians = {k: grid[key][k] for k in competitive if k in grid[key]}
                if medians:
                    best = min(medians, key=medians.get)
                    row.append(strat_to_idx.get(best, 0))
                else:
                    row.append(0)
            else:
                row.append(0)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{pl:.1f}" for pl in pit_losses],
        y=[f"{dm:.2f}" for dm in deg_multipliers],
        colorscale=[
            [i / (len(competitive) - 1), palette[i]]
            for i in range(len(competitive))
        ],
        zmin=0, zmax=len(competitive) - 1,
        colorbar=dict(
            tickvals=list(range(len(competitive))),
            ticktext=competitive,
            tickfont=dict(family="JetBrains Mono", size=10),
        ),
        hovertemplate="Pit Loss: %{x}s<br>Deg: %{y}x<br><extra></extra>",
    ))
    fig.update_layout(
        **chart_layout(f"Optimal Strategy \u2014 {circuit_name}"),
        xaxis_title="Pit Loss (s)", yaxis_title="Degradation Multiplier",
    )
    return fig


@app.callback(
    Output("sensitivity-store", "data"),
    Output("sensitivity-status", "children"),
    Input("sensitivity-button", "n_clicks"),
    State("results-store", "data"),
    State("circuit-dropdown", "value"),
    State("strategy-dropdown", "value"),
    State("base-pace-input", "value"),
    State("pit-loss-input", "value"),
    State("custom-tires-toggle", "value"),
    State("tire-allocation-store", "data"),
    State("editor-toggle", "value"),
    State("custom-strategy-store", "data"),
    prevent_initial_call=True,
)
def run_sensitivity(n_clicks, results_data, circuit, strategies, pace, pit,
                    tire_toggle, tire_data, editor_toggle, custom_data):
    if not results_data or not circuit or not strategies:
        return None, ""

    pace = float(pace or 80.0)
    pit = float(pit or 22.0)

    tire_allocation = None
    if "on" in (tire_toggle or []) and tire_data:
        tire_allocation = tire_data

    circuit_laps = simulator.circuits[circuit]["laps"]
    use_custom = "on" in (editor_toggle or []) and custom_data
    gp_name = simulator.circuits[circuit]["gp_name"]

    specs = {}
    for name in strategies:
        if use_custom and name in custom_data:
            sequence = [s["compound"] for s in custom_data[name]]
            lengths = [int(s["laps"]) for s in custom_data[name]]
            compounds = "-".join(c[0] for c in sequence)
            stops = len(sequence) - 1
            label = f"{stops}-Stop: {compounds}"
            if sum(lengths) != circuit_laps:
                continue
            specs[label] = {
                "sequence": sequence, "fixed": True, "lengths": lengths,
            }
        else:
            specs[name] = {
                "sequence": ALL_STRATEGIES[name], "fixed": False, "lengths": None,
            }

    if not specs:
        return None, html.Div(
            "No valid strategies",
            style={"color": "#ef4444", "fontSize": "12px"},
        )

    pit_results, pit_vals = run_pit_loss_sweep(
        circuit, specs, pace, pit, tire_allocation, 300,
    )
    deg_results, deg_vals = run_deg_sweep(
        circuit, specs, pace, pit, tire_allocation, 300,
    )
    grid, grid_pit, grid_deg = run_2d_sweep(
        circuit, specs, pace, pit, tire_allocation, 200,
    )

    strategy_names = list(specs.keys())
    pit_crossovers = find_crossovers(pit_results, "pit_loss", strategy_names)
    deg_crossovers = find_crossovers(deg_results, "deg_multiplier", strategy_names)

    status = html.Div(
        "Sensitivity complete",
        style={
            "color": "#22c55e", "fontSize": "12px",
            "fontFamily": "JetBrains Mono",
        },
    )

    return {
        "pit_results": pit_results,
        "deg_results": deg_results,
        "grid": grid,
        "pit_vals": pit_vals,
        "deg_vals": deg_vals,
        "grid_pit": grid_pit,
        "grid_deg": grid_deg,
        "pit_crossovers": pit_crossovers,
        "deg_crossovers": deg_crossovers,
        "strategy_names": strategy_names,
        "default_pit": pit,
        "circuit": circuit,
        "gp_name": gp_name,
    }, status


@app.callback(
    Output("sensitivity-section", "children"),
    Input("sensitivity-store", "data"),
)
def display_sensitivity(data):
    if not data:
        return []

    strategy_names = data["strategy_names"]
    default_pit = data["default_pit"]
    gp_name = data["gp_name"]
    pit_results = data["pit_results"]
    deg_results = data["deg_results"]
    grid = data["grid"]
    pit_crossovers = data["pit_crossovers"]
    deg_crossovers = data["deg_crossovers"]

    pit_fig = build_sweep_figure(
        pit_results, "pit_loss", "Pit Loss (s)", default_pit,
        pit_crossovers, strategy_names, gp_name,
    )
    pit_rel_fig = build_relative_figure(
        pit_results, "pit_loss", "Pit Loss (s)", default_pit,
        strategy_names, gp_name,
    )
    deg_fig = build_sweep_figure(
        deg_results, "deg_multiplier", "Degradation Multiplier", 1.0,
        deg_crossovers, strategy_names, gp_name,
    )
    deg_rel_fig = build_relative_figure(
        deg_results, "deg_multiplier", "Degradation Multiplier", 1.0,
        strategy_names, gp_name,
    )
    heatmap_fig = build_heatmap_figure(
        grid, data["grid_pit"], data["grid_deg"],
        strategy_names, default_pit, gp_name,
    )
    deg_lap_fig = build_pitlap_figure(
        deg_results, "deg_multiplier", "Degradation Multiplier", 1.0,
        strategy_names, gp_name,
    )

    crossover_rows = []
    for cx in pit_crossovers:
        crossover_rows.append({
            "Parameter": "Pit Loss",
            "Value": f"{cx['param_value']}s",
            "From": cx["from_strategy"],
            "To": cx["to_strategy"],
        })
    for cx in deg_crossovers:
        crossover_rows.append({
            "Parameter": "Degradation",
            "Value": f"{cx['param_value']}x",
            "From": cx["from_strategy"],
            "To": cx["to_strategy"],
        })

    crossover_section = []
    if crossover_rows:
        crossover_df = pd.DataFrame(crossover_rows)
        crossover_section = [
            html.Div("CROSSOVER POINTS", className="sidebar-section-label mt-4"),
            dash_table.DataTable(
                data=crossover_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in crossover_df.columns],
                style_header=TABLE_HEADER, style_cell=TABLE_CELL,
                style_data_conditional=TABLE_CONDITIONAL,
                style_table={"overflowX": "auto"},
            ),
        ]
    else:
        crossover_section = [
            html.Div("CROSSOVER POINTS", className="sidebar-section-label mt-4"),
            html.Div(
                "No crossover points found. "
                "The optimal strategy is stable across the full parameter range.",
                className="sensitivity-stable-msg",
            ),
        ]

    return html.Div([
        html.Div("SENSITIVITY ANALYSIS", className="sidebar-section-label"),
        html.Div([
            dbc.Col(html.Div([
                html.Div("PIT LOSS RANGE", className="metric-label"),
                html.Div(
                    f"{data['pit_vals'][0]:.0f}s \u2013 {data['pit_vals'][-1]:.0f}s",
                    className="metric-value",
                ),
                html.Div(f"default: {default_pit}s", className="metric-sub"),
            ], className="metric-card")),
            dbc.Col(html.Div([
                html.Div("DEGRADATION RANGE", className="metric-label"),
                html.Div(
                    f"{data['deg_vals'][0]:.2f}x \u2013 {data['deg_vals'][-1]:.2f}x",
                    className="metric-value",
                ),
                html.Div("default: 1.00x", className="metric-sub"),
            ], className="metric-card")),
        ], className="d-flex gap-3 mb-4"),
        dbc.Tabs([
            dbc.Tab(
                dcc.Graph(figure=pit_fig, config={"displayModeBar": False}),
                label="Pit Loss",
            ),
            dbc.Tab(
                dcc.Graph(figure=pit_rel_fig, config={"displayModeBar": False}),
                label="Pit Loss (Relative)",
            ),
            dbc.Tab(
                dcc.Graph(figure=deg_fig, config={"displayModeBar": False}),
                label="Degradation",
            ),
            dbc.Tab(
                dcc.Graph(figure=deg_rel_fig, config={"displayModeBar": False}),
                label="Degradation (Relative)",
            ),
            dbc.Tab(
                dcc.Graph(figure=deg_lap_fig, config={"displayModeBar": False}),
                label="Optimal Pit Lap",
            ),
            dbc.Tab(
                dcc.Graph(figure=heatmap_fig, config={"displayModeBar": False}),
                label="2D Heatmap",
            ),
        ], className="mb-4"),
        *crossover_section,
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)