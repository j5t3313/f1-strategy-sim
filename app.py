import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
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
    "1-Stop: M-H": [
        {"compound": "MEDIUM", "laps": 20},
        {"compound": "HARD", "laps": 24},
    ],
    "1-Stop: S-H": [
        {"compound": "SOFT", "laps": 15},
        {"compound": "HARD", "laps": 29},
    ],
    "1-Stop: H-M": [
        {"compound": "HARD", "laps": 25},
        {"compound": "MEDIUM", "laps": 19},
    ],
    "1-Stop: H-S": [
        {"compound": "HARD", "laps": 31},
        {"compound": "SOFT", "laps": 13},
    ],
    "2-Stop: S-M-H": [
        {"compound": "SOFT", "laps": 12},
        {"compound": "MEDIUM", "laps": 16},
        {"compound": "HARD", "laps": 16},
    ],
    "2-Stop: M-M-H": [
        {"compound": "MEDIUM", "laps": 15},
        {"compound": "MEDIUM", "laps": 15},
        {"compound": "HARD", "laps": 14},
    ],
    "2-Stop: M-M-S": [
        {"compound": "MEDIUM", "laps": 16},
        {"compound": "MEDIUM", "laps": 16},
        {"compound": "SOFT", "laps": 12},
    ],
    "2-Stop: H-M-S": [
        {"compound": "HARD", "laps": 16},
        {"compound": "MEDIUM", "laps": 16},
        {"compound": "SOFT", "laps": 12},
    ],
    "2-Stop: M-H-M": [
        {"compound": "MEDIUM", "laps": 12},
        {"compound": "HARD", "laps": 18},
        {"compound": "MEDIUM", "laps": 14},
    ],
    "2-Stop: M-H-H": [
        {"compound": "MEDIUM", "laps": 18},
        {"compound": "HARD", "laps": 30},
        {"compound": "HARD", "laps": 24},
    ],
    "2-Stop: S-H-M": [
        {"compound": "SOFT", "laps": 15},
        {"compound": "HARD", "laps": 35},
        {"compound": "MEDIUM", "laps": 22},
    ],
}

CIRCUIT_BASE_PACES = {
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

FUEL_LOAD_KG = 70.0
FUEL_RESERVE_KG = 3.0
WEIGHT_EFFECT_S_PER_KG = 0.03
PACE_SIGMA = 0.4

STRATEGY_COLORS = ["#e10600", "#0090ff", "#22c55e", "#ff8700", "#a855f7"]
COMPOUND_COLORS = {"SOFT": "#dc2626", "MEDIUM": "#ca8a04", "HARD": "#6b7280"}


class F1StrategySimulator:

    def __init__(self, models_dir="prebuilt_models"):
        self.models_dir = Path(models_dir)
        self.circuits = {name: data for name, data in CIRCUIT_DATA}
        self.posterior_models = {}
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

    def _draw_compound_params(self, compound, circuit_name):
        model_key = f"{circuit_name}_{compound}"

        if self.has_posteriors and model_key in self.posterior_models:
            samples = self.posterior_models[model_key]["samples"]
            idx = np.random.choice(len(samples["alpha"]))

            alpha = float(samples["alpha"][idx])
            beta = float(samples["beta"][idx])

            if "gamma" in samples and len(samples["gamma"]) > 0:
                gamma = float(samples["gamma"][min(idx, len(samples["gamma"]) - 1)])
            else:
                p = COMPOUND_PRIORS[compound]["gamma"]
                gamma = abs(np.random.normal(p["mu"], p["sigma"]))

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
        return {
            "mode": "prior",
            "alpha_offset": np.random.normal(
                prior["alpha_offset"]["mu"], prior["alpha_offset"]["sigma"]
            ),
            "beta": max(0.001, np.random.normal(
                prior["beta"]["mu"], prior["beta"]["sigma"]
            )),
            "gamma": abs(np.random.normal(
                prior["gamma"]["mu"], prior["gamma"]["sigma"]
            )),
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
                 base_pace=80.0, pit_loss=22.0, num_sims=1000):
        num_sims = int(num_sims)
        total_laps = self.circuits[circuit]["laps"]
        fpl = self.fuel_per_lap(circuit)

        valid, msg = self.validate_allocation(strategy, tire_allocation)
        if not valid:
            raise ValueError(msg)

        enhanced = self.assign_tires(strategy, tire_allocation)
        compounds_used = list({s["compound"] for s in enhanced})
        results = np.zeros(num_sims)

        for sim in range(num_sims):
            sim_pace = base_pace + np.random.normal(0, PACE_SIGMA)
            compound_params = {
                c: self._draw_compound_params(c, circuit) for c in compounds_used
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
                        (total_laps - current_lap) * fpl * WEIGHT_EFFECT_S_PER_KG
                    )
                    race_time += mu + epsilon - fuel_correction
                    current_lap += 1

                if stint_idx < len(enhanced) - 1:
                    race_time += pit_loss

            results[sim] = race_time

        return results


def scale_strategy(strategy, circuit_laps):
    total_original = sum(s["laps"] for s in strategy)
    factor = circuit_laps / total_original
    scaled = []
    remaining = circuit_laps
    for i, stint in enumerate(strategy):
        if i == len(strategy) - 1:
            laps = remaining
        else:
            laps = max(1, round(stint["laps"] * factor))
            laps = min(laps, remaining - (len(strategy) - i - 1))
            remaining -= laps
        scaled.append({"compound": stint["compound"], "laps": laps})
    return scaled


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
    ],
    className="sidebar",
)

welcome_content = html.Div(
    [
        html.Div("F1 Strategy Simulator", className="welcome-title"),
        html.P(
            "Simulate and compare Formula 1 pit stop strategies using Bayesian "
            "tire modeling and Monte Carlo simulation, updated for the 2026 "
            "regulation era.",
            className="welcome-text",
        ),
        html.P(
            "Select a circuit from the sidebar, choose strategies to compare, "
            "adjust simulation parameters, and run the analysis.",
            className="welcome-text",
        ),
        html.Div(
            [
                html.P(
                    "2026 regulation updates: 70 kg fuel load (sustainable fuel, "
                    "lower energy density), Pirelli C1\u2013C5 compounds (C6 removed, "
                    "narrower construction), 768 kg minimum car weight, ~30% downforce "
                    "reduction, active aerodynamics, 50/50 ICE/electric power split.",
                    style={"marginBottom": "8px"},
                ),
                html.P(
                    "Tire degradation uses a quadratic model (\u03b1 + \u03b2\u00b7lap "
                    "+ \u03b3\u00b7lap\u00b2) with compound-specific informative priors. "
                    "When trained posterior samples are available, parameters are drawn "
                    "from the posterior; otherwise from priors informed by historical "
                    "degradation ranges. Parameters are drawn once per simulation per "
                    "compound (epistemic uncertainty), while lap-to-lap noise follows "
                    "an AR(1) process with autocorrelation parameter \u03c1 (aleatoric "
                    "uncertainty).",
                ),
            ],
            className="welcome-detail",
        ),
    ],
    id="welcome-section",
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
    prevent_initial_call=True,
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
        base = ALL_STRATEGIES[name]
        scaled = scale_strategy(base, circuit_laps)

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
    prevent_initial_call=True,
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
    prevent_initial_call=True,
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
        else:
            scaled = scale_strategy(ALL_STRATEGIES[name], circuit_laps)
        cards.append(
            html.Div(
                [
                    html.Div(name, className="strategy-card-name"),
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
    errors = []

    for name in strategies:
        if use_custom and name in custom_data:
            strategy = custom_data[name]
        else:
            strategy = scale_strategy(ALL_STRATEGIES[name], circuit_laps)

        total_strat_laps = sum(s["laps"] for s in strategy)
        if total_strat_laps != circuit_laps:
            errors.append(
                f"{name}: total laps ({total_strat_laps}) != circuit ({circuit_laps})"
            )
            continue

        try:
            times = simulator.simulate(
                circuit, strategy, tire_allocation, pace, pit, sims,
            )
            results[name] = times.tolist()
        except Exception as e:
            errors.append(f"{name}: {str(e)}")

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

    return {"results": results, "circuit": circuit}, status


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
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_simulator(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return None, None, None, "", [], []


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)