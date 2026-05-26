#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/carla-roach:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${ROOT_DIR}"

MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "driveadapter" ]]; then
  echo "warning: expected conda env 'driveadapter', current='${CONDA_DEFAULT_ENV:-none}'" >&2
fi

if [[ -z "${CARLA_ROOT:-}" ]]; then
  echo "CARLA_ROOT must point to a Linux CARLA install containing CarlaUE4.sh." >&2
  echo "Example: CARLA_ROOT=/opt/carla-0.9.11 scripts/run_train_sba_uas.sh smoke" >&2
  exit 2
fi

SBA_UAS_PAPER_CONFIG="{enabled:true,updates_per_ppo_train:${SBA_UAS_UPDATES_PER_PPO_TRAIN:-1},use_sba_critic_for_ppo_advantage:true,save_extra_state:true,standard_buffer_capacity:4096,familiar_buffer_capacity:4096,batch_size:32,environment_batch_size:32,reference_batch_size:32,warmup_transitions:256,shifted_vas_samples:64,environment_prediction_samples:1,hidden_units:2048,n_layers:6,san_feature_dim:256,san_shared_hidden_units:1024,environment_base_channels:32,environment_transition_hidden_dims:[512,512],latent_dim:256}"
SBA_UAS_SMOKE_CONFIG="{enabled:true,updates_per_ppo_train:1,use_sba_critic_for_ppo_advantage:true,save_extra_state:true,standard_buffer_capacity:256,familiar_buffer_capacity:256,batch_size:8,environment_batch_size:8,reference_batch_size:4,warmup_transitions:16,shifted_vas_samples:2,environment_prediction_samples:1,hidden_units:64,n_layers:2,san_feature_dim:32,san_shared_hidden_units:64,environment_base_channels:4,environment_transition_hidden_dims:[64],latent_dim:32}"

COMMON_OVERRIDES=(
  "agent.ppo.training.entry_point=sba_uas.training.roach_ppo_sidecar:SBAUASPPO"
  "+agent.ppo.training.kwargs.sba_uas=${SBA_UAS_PAPER_CONFIG}"
  "agent/ppo/policy=xtma_beta"
  "agent/ppo/training=ppo"
  "carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh"
  "seed=${SBA_UAS_SEED:-2021}"
)

case "${MODE}" in
  smoke)
    COMMON_OVERRIDES[1]="+agent.ppo.training.kwargs.sba_uas=${SBA_UAS_SMOKE_CONFIG}"
    RUN_OVERRIDES=(
      "total_timesteps=${SBA_UAS_TOTAL_TIMESTEPS:-4096}"
      "dummy=true"
      "wb_project=${SBA_UAS_WB_PROJECT:-sba_uas_smoke}"
      "wb_name=${SBA_UAS_WB_NAME:-sba_uas_town01_smoke}"
      "train_envs=[{env_id:Endless-v0,env_configs:{carla_map:Town01,num_zombie_vehicles:[0,30],num_zombie_walkers:[0,60],weather_group:dynamic_1.0},gpu:[0]}]"
    )
    ;;
  standard)
    RUN_OVERRIDES=(
      "total_timesteps=${SBA_UAS_TOTAL_TIMESTEPS:-100000000}"
      "train_envs=endless_all"
      "dummy=${SBA_UAS_DUMMY:-false}"
      "wb_project=${SBA_UAS_WB_PROJECT:-sba_uas}"
      "wb_name=${SBA_UAS_WB_NAME:-sba_uas_standard}"
    )
    ;;
  sequential)
    MAPS=(Town01 Town02 Town03 Town04 Town05 Town06)
    STEPS_PER_MAP="${SBA_UAS_STEPS_PER_MAP:-2000000}"
    RUN_ROOT="${SBA_UAS_RUN_ROOT:-outputs/sba_uas_sequential_${SBA_UAS_SEED:-2021}}"
    mkdir -p "${RUN_ROOT}"
    PREV_CKPT="${SBA_UAS_INITIAL_CKPT:-}"
    CUMULATIVE_STEPS=0
    for INDEX in "${!MAPS[@]}"; do
      MAP="${MAPS[${INDEX}]}"
      CUMULATIVE_STEPS=$((CUMULATIVE_STEPS + STEPS_PER_MAP))
      STAGE_DIR="${RUN_ROOT}/$(printf "%02d" "$((INDEX + 1))")_${MAP}"
      STAGE_OVERRIDES=(
        "hydra.run.dir=${STAGE_DIR}"
        "total_timesteps=${CUMULATIVE_STEPS}"
        "dummy=${SBA_UAS_DUMMY:-false}"
        "wb_project=${SBA_UAS_WB_PROJECT:-sba_uas_continual}"
        "wb_name=${SBA_UAS_WB_NAME:-sba_uas}_${MAP}"
        "train_envs=[{env_id:Endless-v0,env_configs:{carla_map:${MAP},num_zombie_vehicles:[0,160],num_zombie_walkers:[0,160],weather_group:dynamic_1.0},gpu:[0]}]"
      )
      if [[ -n "${PREV_CKPT}" ]]; then
        STAGE_OVERRIDES+=("agent.ppo.ckpt=${PREV_CKPT}")
        EXTRA_CKPT="${PREV_CKPT%.*}_sba_uas_extra_state.${PREV_CKPT##*.}"
        if [[ ! -f "${EXTRA_CKPT}" ]]; then
          echo "could not find SBA-UAS extra state for resume: ${EXTRA_CKPT}" >&2
          exit 1
        fi
        export SBA_UAS_RESUME_EXTRA_STATE="${EXTRA_CKPT}"
      else
        unset SBA_UAS_RESUME_EXTRA_STATE || true
      fi
      python -u carla-roach/train_rl.py \
        "${COMMON_OVERRIDES[@]}" \
        "${STAGE_OVERRIDES[@]}" \
        "$@"
      EXPECTED_CKPT="${STAGE_DIR}/ckpt/ckpt_${CUMULATIVE_STEPS}.pth"
      if [[ -f "${EXPECTED_CKPT}" ]]; then
        PREV_CKPT="${EXPECTED_CKPT}"
      else
        PREV_CKPT="$(find "${STAGE_DIR}/ckpt" -maxdepth 1 -name 'ckpt_*.pth' 2>/dev/null | sort -V | tail -n 1 || true)"
      fi
      if [[ -z "${PREV_CKPT}" || ! -f "${PREV_CKPT}" ]]; then
        echo "could not find checkpoint after ${MAP} in ${STAGE_DIR}/ckpt" >&2
        exit 1
      fi
      echo "stage ${MAP} complete: ${PREV_CKPT}"
    done
    exit 0
    ;;
  *)
    echo "unknown mode '${MODE}'. Use: smoke, standard, or sequential." >&2
    exit 2
    ;;
esac

python -u carla-roach/train_rl.py \
  "${COMMON_OVERRIDES[@]}" \
  "${RUN_OVERRIDES[@]}" \
  "$@"
