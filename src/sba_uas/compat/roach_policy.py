"""Read-only access to CARLA-Roach policy classes.

SBA-UAS must keep its actor/policy checkpoint contract identical to Roach.
This module only adjusts sys.path and re-exports the upstream classes.
"""

from importlib import import_module
from pathlib import Path
import sys


def ensure_roach_on_path(repo_root=None):
    """Add the read-only carla-roach directory to sys.path."""
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[3]
    roach_dir = root / "carla-roach"
    if not roach_dir.is_dir():
        raise FileNotFoundError("Cannot find carla-roach directory at {}".format(roach_dir))

    roach_path = str(roach_dir)
    if roach_path not in sys.path:
        sys.path.insert(0, roach_path)
    return roach_dir


_ROACH_EXPORTS = {
    "BetaDistribution": "agents.rl_birdview.models.distributions:BetaDistribution",
    "PpoPolicy": "agents.rl_birdview.models.ppo_policy:PpoPolicy",
    "XtMaCNN": "agents.rl_birdview.models.torch_layers:XtMaCNN",
}


def load_roach_attr(export_name, repo_root=None):
    """Load one upstream Roach symbol after placing Roach on ``sys.path``."""

    if export_name not in _ROACH_EXPORTS:
        raise AttributeError("unknown Roach export '{}'".format(export_name))
    ensure_roach_on_path(repo_root=repo_root)
    module_name, attr_name = _ROACH_EXPORTS[export_name].split(":")
    module = import_module(module_name)
    return getattr(module, attr_name)


def __getattr__(name):
    if name in _ROACH_EXPORTS:
        value = load_roach_attr(name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "BetaDistribution",
    "PpoPolicy",
    "XtMaCNN",
    "ensure_roach_on_path",
    "load_roach_attr",
]
