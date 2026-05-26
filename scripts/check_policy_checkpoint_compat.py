"""Check whether SBA-UAS preserves the CARLA-Roach policy contract."""

from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the SBA-UAS policy YAML and, optionally, a saved "
            "Roach-compatible policy checkpoint."
        )
    )
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/sba_uas/policy_compatibility.yaml"),
    )
    parser.add_argument(
        "--roach-policy-config",
        type=Path,
        default=Path("carla-roach/config/agent/ppo/policy/xtma_beta.yaml"),
    )
    parser.add_argument(
        "--load-with-roach",
        action="store_true",
        help="Also call Roach PpoPolicy.load(checkpoint). Requires CARLA deps.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "carla-roach"))

    contract = _load_yaml(root / args.contract)
    roach_policy = _load_yaml(root / args.roach_policy_config)
    expected_kwargs = _check_policy_yaml_contract(contract, roach_policy)
    print("OK: SBA-UAS policy YAML matches Roach xtma_beta policy config.")

    if args.checkpoint is None:
        return

    import torch  # noqa: WPS433

    payload = torch.load(str(root / args.checkpoint), map_location="cpu")
    init_kwargs = payload.get("policy_init_kwargs")
    if init_kwargs is None:
        raise SystemExit("checkpoint is missing policy_init_kwargs")
    if "policy_state_dict" not in payload:
        raise SystemExit("checkpoint is missing policy_state_dict")

    mismatches = []
    for key, expected in expected_kwargs.items():
        actual = init_kwargs.get(key)
        if actual != expected:
            mismatches.append((key, expected, actual))

    if mismatches:
        for key, expected, actual in mismatches:
            print("Mismatch {}: expected={!r}, actual={!r}".format(key, expected, actual))
        raise SystemExit(1)

    if args.load_with_roach:
        from sba_uas.compat.roach_policy import load_roach_attr  # noqa: WPS433

        ppo_policy = load_roach_attr("PpoPolicy", repo_root=root)
        ppo_policy.load(root / args.checkpoint)
        print("OK: checkpoint loaded with upstream Roach PpoPolicy.load().")

    print("OK: checkpoint follows the CARLA-Roach PpoPolicy contract.")


def _load_yaml(path: Path):
    try:
        import yaml  # noqa: WPS433
    except ImportError as exc:
        raise SystemExit("PyYAML is required for policy YAML checks.") from exc

    if not path.exists():
        raise SystemExit("YAML file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _check_policy_yaml_contract(contract, roach_policy):
    contract_entry = contract.get("entry_point")
    roach_entry = roach_policy.get("entry_point")
    if contract_entry != roach_entry:
        raise SystemExit(
            "policy entry_point mismatch: expected {!r}, actual {!r}".format(
                roach_entry,
                contract_entry,
            )
        )

    contract_kwargs = contract.get("kwargs") or {}
    roach_kwargs = roach_policy.get("kwargs") or {}
    mismatches = []
    for key, expected in roach_kwargs.items():
        actual = contract_kwargs.get(key)
        if actual != expected:
            mismatches.append((key, expected, actual))
    extra_keys = sorted(set(contract_kwargs) - set(roach_kwargs))
    if mismatches or extra_keys:
        for key, expected, actual in mismatches:
            print(
                "Policy YAML mismatch {}: expected={!r}, actual={!r}".format(
                    key,
                    expected,
                    actual,
                )
            )
        if extra_keys:
            print("Policy YAML has unexpected kwargs: {}".format(extra_keys))
        raise SystemExit(1)
    return roach_kwargs


if __name__ == "__main__":
    main()
