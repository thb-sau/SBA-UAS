#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/carla-roach:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${ROOT_DIR}"

POLICY_CKPT="${1:-checkpoints/policy.pth}"
if [[ $# -gt 0 ]]; then
  shift
fi
SUITE="${SBA_UAS_TEST_SUITE:-nocrash_dense}"
SEED="${SBA_UAS_SEED:-2021}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "driveadapter" ]]; then
  echo "warning: expected conda env 'driveadapter', current='${CONDA_DEFAULT_ENV:-none}'" >&2
fi

if [[ -z "${CARLA_ROOT:-}" ]]; then
  echo "CARLA_ROOT must point to a Linux CARLA install containing CarlaUE4.sh." >&2
  echo "Example: CARLA_ROOT=/opt/carla-0.9.11 scripts/run_benchmark_sba_uas.sh checkpoints/policy.pth" >&2
  exit 2
fi

if [[ ! -f "${POLICY_CKPT}" ]]; then
  echo "policy checkpoint not found: ${POLICY_CKPT}" >&2
  exit 2
fi

python scripts/check_policy_checkpoint_compat.py "${POLICY_CKPT}"

python -u carla-roach/benchmark.py \
  "resume=${SBA_UAS_BENCHMARK_RESUME:-true}" \
  "log_video=${SBA_UAS_LOG_VIDEO:-false}" \
  "no_rendering=${SBA_UAS_NO_RENDERING:-true}" \
  "wb_project=${SBA_UAS_WB_PROJECT:-sba_uas_benchmark}" \
  "wb_group=${SBA_UAS_WB_GROUP:-SBA-UAS}" \
  "wb_notes=${SBA_UAS_WB_NOTES:-Benchmark SBA-UAS Roach-compatible policy checkpoint.}" \
  "agent.ppo.ckpt=${POLICY_CKPT}" \
  "agent.ppo.wb_run_path=null" \
  "test_suites=${SUITE}" \
  "seed=${SEED}" \
  "carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh" \
  "$@"
