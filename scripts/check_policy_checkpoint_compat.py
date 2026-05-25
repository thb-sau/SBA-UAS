"""Check whether a policy checkpoint follows the CARLA-Roach policy contract."""

from pathlib import Path
import argparse
import sys


EXPECTED_POLICY = {
    "policy_head_arch": [256, 256],
    "value_head_arch": [256, 256],
    "features_extractor_entry_point": "agents.rl_birdview.models.torch_layers:XtMaCNN",
    "features_extractor_kwargs": {"states_neurons": [256, 256]},
    "distribution_entry_point": "agents.rl_birdview.models.distributions:BetaDistribution",
    "distribution_kwargs": {"dist_init": None},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "carla-roach"))

    import torch  # noqa: WPS433

    payload = torch.load(str(args.checkpoint), map_location="cpu")
    init_kwargs = payload.get("policy_init_kwargs")
    if init_kwargs is None:
        raise SystemExit("checkpoint is missing policy_init_kwargs")
    if "policy_state_dict" not in payload:
        raise SystemExit("checkpoint is missing policy_state_dict")

    mismatches = []
    for key, expected in EXPECTED_POLICY.items():
        actual = init_kwargs.get(key)
        if actual != expected:
            mismatches.append((key, expected, actual))

    if mismatches:
        for key, expected, actual in mismatches:
            print("Mismatch {}: expected={!r}, actual={!r}".format(key, expected, actual))
        raise SystemExit(1)

    print("OK: checkpoint follows the CARLA-Roach PpoPolicy contract.")


if __name__ == "__main__":
    main()
