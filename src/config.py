"""Load the study configuration (config/config.yaml), optionally for a robustness scenario (Task 07).

load_config() returns config.yaml unchanged: the main analysis. load_config(scenario=NAME) returns a copy with
  config["fetal_profile"] = {"name": NAME, "shift_bp": ..., "blend_maternal": ...}
from robustness.fetal_profiles[NAME] (decision #17), read by src.simulate.lengths.load_length_distributions, and
paths.results set to <results>/robustness/NAME, so a scenario never writes to results/tables or results/figures.
Scripts that accept a scenario add the shared flag with add_scenario_argument(parser) and call
load_config(scenario=args.scenario); none of them changes the config itself.
"""

import copy
from pathlib import Path

import yaml

# Repository root: src/config.py -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path=CONFIG_PATH, scenario=None):
    """Return config.yaml as a dict; with scenario, the copy for that fetal-profile scenario."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return config if scenario is None else apply_scenario(config, scenario)


def scenario_names(config):
    """Names of the fetal-profile scenarios in the config (robustness.fetal_profiles), in config order."""
    return list(config.get("robustness", {}).get("fetal_profiles", {}))


def apply_scenario(config, name):
    """Copy of a main-analysis config set up for scenario name (see the module docstring)."""
    if "fetal_profile" in config:
        raise ValueError(f"config is already set up for scenario {config['fetal_profile']['name']!r}")
    if name not in scenario_names(config):
        raise ValueError(f"unknown scenario {name!r}; robustness.fetal_profiles has {scenario_names(config)}")
    out = copy.deepcopy(config)
    out["fetal_profile"] = {"name": name, **out["robustness"]["fetal_profiles"][name]}
    out["paths"]["results"] = f"{config['paths']['results']}/robustness/{name}"
    return out


def scenario_name(config):
    """Name of the config's scenario, or None in the main analysis."""
    return config.get("fetal_profile", {}).get("name")


def base_config(config):
    """The main-analysis (G1 profile) version of a config: no fetal_profile, main results path."""
    if scenario_name(config) is None:
        return config
    out = copy.deepcopy(config)
    del out["fetal_profile"]
    out["paths"]["results"] = load_config()["paths"]["results"]
    return out


def add_scenario_argument(parser):
    """Shared --scenario flag (choices from robustness.fetal_profiles; default: the main analysis)."""
    parser.add_argument("--scenario", choices=scenario_names(load_config()), default=None,
                        help="fetal-profile scenario (Task 07); results go to results/robustness/<NAME>")
    return parser


def repo_path(relative):
    """Turn a path from the config (relative to the repo root) into an absolute Path."""
    return REPO_ROOT / relative
