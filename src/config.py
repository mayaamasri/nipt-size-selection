"""Load the study configuration (config/config.yaml), optionally for a robustness scenario (Tasks 07, 08).

load_config() returns config.yaml unchanged: the main analysis. load_config(scenario=NAME) returns a copy with
  config["scenario"] = {"name": NAME, "kind": "fetal_profile" | "short_noise"}
and, by kind,
  fetal_profile  (robustness.fetal_profiles, decision #17): config["fetal_profile"] = {"name", "shift_bp",
                 "blend_maternal"}, read by src.simulate.lengths.load_length_distributions
  short_noise    (robustness.short_noise, decision #18): config["short_noise"] = {"name", "window_bp", "k"}, read
                 by src.simulate.patients (extra noise on the chr21 cells inside window_bp)
and paths.results set to <results>/robustness/NAME, so a scenario never writes to results/tables or
results/figures. Scripts that accept a scenario add the shared flag with add_scenario_argument(parser) and call
load_config(scenario=args.scenario); none of them changes the config itself.

Short-noise scenarios run only the noise levels with sigma > 0 (there is nothing to scale at sigma = 0). The grid
list itself is never changed, because patient seeds use the index of the noise level in grid.noise_sigma: every
loop over noise levels uses scenario_noise_levels(config) instead.
"""

import copy
from pathlib import Path

import yaml

# Repository root: src/config.py -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
SCENARIO_KINDS = ("fetal_profiles", "short_noise")      # robustness.<kind> -> config key without the plural
KIND_KEY = {"fetal_profiles": "fetal_profile", "short_noise": "short_noise"}


def load_config(path=CONFIG_PATH, scenario=None):
    """Return config.yaml as a dict; with scenario, the copy for that robustness scenario."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return config if scenario is None else apply_scenario(config, scenario)


def _scenarios(config):
    """{name: (kind, settings)} over every robustness scenario kind, in config order; names must be unique."""
    out = {}
    for kind in SCENARIO_KINDS:
        for name, settings in config.get("robustness", {}).get(kind, {}).items():
            if name in out:
                raise ValueError(f"scenario name {name!r} is used twice in robustness")
            out[name] = (kind, settings)
    return out


def scenario_names(config, kind=None):
    """Names of the robustness scenarios in the config, in config order (only kind, e.g. "short_noise", if given)."""
    return [n for n, (k, _) in _scenarios(config).items() if kind is None or k == kind]


def apply_scenario(config, name):
    """Copy of a main-analysis config set up for scenario name (see the module docstring)."""
    if "scenario" in config:
        raise ValueError(f"config is already set up for scenario {config['scenario']['name']!r}")
    scenarios = _scenarios(config)
    if name not in scenarios:
        raise ValueError(f"unknown scenario {name!r}; robustness has {list(scenarios)}")
    kind, settings = scenarios[name]
    out = copy.deepcopy(config)
    out["scenario"] = {"name": name, "kind": KIND_KEY[kind]}
    out[KIND_KEY[kind]] = {"name": name, **copy.deepcopy(settings)}
    out["paths"]["results"] = f"{config['paths']['results']}/robustness/{name}"
    return out


def scenario_name(config):
    """Name of the config's robustness scenario (any kind), or None in the main analysis."""
    return config.get("scenario", {}).get("name")


def fetal_profile_name(config):
    """Name of the config's fetal-profile scenario, or None (main analysis and short-noise scenarios use G1)."""
    return config.get("fetal_profile", {}).get("name")


def short_noise(config):
    """The short-noise settings {"name", "window_bp", "k"} of the config, or None."""
    return config.get("short_noise")


def scenario_noise_levels(config):
    """Noise levels (names as in grid.noise_sigma) that this config runs: all of them, except that short-noise
    scenarios skip the levels with sigma = 0. The grid list itself is never changed (seeds use its index)."""
    grid = config["grid"]
    levels = list(grid["noise_sigma"])
    if short_noise(config) is None:
        return levels
    return [n for n, s in zip(levels, grid["noise_sigma_values"]) if s > 0]


def base_config(config):
    """The main-analysis (G1 profile, no short noise) version of a config: no scenario, main results path."""
    if scenario_name(config) is None:
        return config
    out = copy.deepcopy(config)
    for key in ("scenario", "fetal_profile", "short_noise"):
        out.pop(key, None)
    out["paths"]["results"] = load_config()["paths"]["results"]
    return out


def add_scenario_argument(parser):
    """Shared --scenario flag (choices from robustness; default: the main analysis)."""
    parser.add_argument("--scenario", choices=scenario_names(load_config()), default=None,
                        help="robustness scenario (Tasks 07, 08); results go to results/robustness/<NAME>")
    return parser


def repo_path(relative):
    """Turn a path from the config (relative to the repo root) into an absolute Path."""
    return REPO_ROOT / relative
