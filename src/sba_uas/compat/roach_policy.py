"""Read-only access to CARLA-Roach policy classes.

SBA-UAS must keep its actor/policy checkpoint contract identical to Roach.
This module only adjusts sys.path and re-exports the upstream classes.
"""

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


ensure_roach_on_path()

from agents.rl_birdview.models.distributions import BetaDistribution  # noqa: E402
from agents.rl_birdview.models.ppo_policy import PpoPolicy  # noqa: E402
from agents.rl_birdview.models.torch_layers import XtMaCNN  # noqa: E402


__all__ = ["BetaDistribution", "PpoPolicy", "XtMaCNN", "ensure_roach_on_path"]
