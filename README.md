# SBA-UAS Reproduction for CARLA-Roach

This repository contains a reproduction-oriented implementation of **Similarity-Based Activation and Uncertainty-Aware Stabilization (SBA-UAS)** for continual reinforcement learning in autonomous driving. The project uses `carla-roach/` as a read-only upstream baseline and adds Critic-side continual-learning components without changing the Roach Actor/policy architecture or checkpoint contract.

## Repository Layout

```text
.
├── carla-roach/                  # Read-only Roach upstream baseline
├── configs/sba_uas/              # SBA-UAS training, evaluation, and policy contract configs
├── docs/                         # Reproduction plan and implementation status
├── environment/                  # Environment setup notes and extra dependencies
├── scripts/                      # Training, evaluation, and checkpoint compatibility scripts
├── src/sba_uas/
│   ├── compat/                   # Read-only Roach policy import and compatibility helpers
│   ├── critic/                   # SAN, Gated Critic, and Critic losses
│   ├── stabilization/            # BNN, shifted VAS, buffers, reference networks, importance
│   └── training/                 # Roach PPO sidecar, adapters, checkpointing, metrics
└── tests/                        # Lightweight tests that do not require a CARLA server
```

`carla-roach/` is intentionally kept unchanged. SBA-UAS additions live in this repository's own package, configuration, script, and documentation paths.

## Environment

Training and evaluation should run on Linux, WSL2 Ubuntu, or a Linux GPU server. Native Windows is suitable only for editing, documentation, and lightweight checks.

Lightweight tests have been verified with:

- conda environment: `driveadapter`
- Python: 3.7
- PyTorch: 1.13.1

CARLA/Roach runtime dependencies:

- Linux CARLA package with `CarlaUE4.sh`
- CARLA Python egg matching the CARLA version
- Roach BEV map `.h5` files
- Bash, `killall`, Linux paths, and a GPU runtime environment

## Installation

If the `driveadapter` environment already exists:

```bash
conda activate driveadapter
cd /home/wsl/pythonWork/sba-uas
pip install -e .
```

To recreate the environment from Roach:

```bash
conda env create -f carla-roach/environment.yml --name driveadapter
conda activate driveadapter
conda env update -n driveadapter -f environment/sba_uas_extra_linux.yml
pip install -e .
```

Install the CARLA Python egg for the CARLA version in use. Example for CARLA 0.9.11:

```bash
export CARLA_ROOT=/path/to/carla-0.9.11
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
```

Roach RL training is often more stable with CARLA 0.9.10.x, while the paper-level target is CARLA 0.9.11. Record the exact CARLA version in every experiment.

## Pre-Training Checks

Run lightweight tests:

```bash
conda run -n driveadapter pytest -q
```

Check that the SBA-UAS policy configuration still matches Roach `xtma_beta.yaml`:

```bash
conda run -n driveadapter python scripts/check_policy_checkpoint_compat.py
```

Check an existing policy checkpoint:

```bash
conda run -n driveadapter python scripts/check_policy_checkpoint_compat.py \
  carla-roach/checkpoints/roach_rl/ckpt/ckpt_11833344.pth
```

## Training

Training scripts default to the `driveadapter` environment and require `CARLA_ROOT` to point to a Linux CARLA installation.

```bash
conda activate driveadapter
export CARLA_ROOT=/path/to/carla-0.9.11
export WANDB_MODE=offline
```

### 1. Single-Town Smoke Run

Use the smallest configuration first to confirm that the CARLA server, Roach wrapper, SBA-UAS sidecar, and checkpoint saving path work end to end.

```bash
scripts/run_train_sba_uas.sh smoke
```

Optional smoke-run overrides:

```bash
SBA_UAS_TOTAL_TIMESTEPS=8192 \
SBA_UAS_SEED=2021 \
scripts/run_train_sba_uas.sh smoke
```

### 2. Six-Town Joint Training

This corresponds to the paper's standard RL agent-sampling protocol.

```bash
SBA_UAS_TOTAL_TIMESTEPS=100000000 \
SBA_UAS_SEED=2021 \
scripts/run_train_sba_uas.sh standard
```

This mode uses Roach `endless_all` and covers Town01 through Town06.

### 3. Sequential Continual-Learning Training

This corresponds to the Town01 to Town06 continual-learning protocol. After each town stage, the next stage resumes from the previous Roach policy checkpoint and restores the matching SBA-UAS extra state through `SBA_UAS_RESUME_EXTRA_STATE`.

```bash
SBA_UAS_STEPS_PER_MAP=2000000 \
SBA_UAS_SEED=2021 \
scripts/run_train_sba_uas.sh sequential
```

For debugging, use a reduced protocol:

```bash
SBA_UAS_STEPS_PER_MAP=20000 \
SBA_UAS_RUN_ROOT=outputs/sba_uas_seq_debug \
scripts/run_train_sba_uas.sh sequential
```

### 4. Roach/Hydra Overrides

Additional arguments are passed through to Roach Hydra:

```bash
scripts/run_train_sba_uas.sh smoke kill_running=false dummy=true
```

## Checkpoints

Roach policy checkpoints:

```text
ckpt_*.pth
```

These contain only the Roach-loadable policy contract:

- `policy_state_dict`
- `policy_init_kwargs`
- `train_init_kwargs`

SBA-UAS extra checkpoints:

```text
ckpt_*_sba_uas_extra_state.pth
```

These contain training-side state:

- SAN, Gated Critic, and Bayesian Environment Model
- Standard Buffer and Familiar Experience Buffer
- Reference Critic and Reference SAN
- Reward-Parameter Correlation and SAN SI-style importance
- optimizer state and training metadata

## Evaluation

Evaluation uses the Roach benchmark and loads only the Roach-compatible policy checkpoint. SBA-UAS extra state is training-side state and is not required for inference.

```bash
conda activate driveadapter
export CARLA_ROOT=/path/to/carla-0.9.11
export WANDB_MODE=offline

scripts/run_benchmark_sba_uas.sh path/to/ckpt_*.pth
```

Select a benchmark suite:

```bash
SBA_UAS_TEST_SUITE=nocrash_dense \
scripts/run_benchmark_sba_uas.sh path/to/ckpt_*.pth
```

The evaluation script runs the checkpoint contract check before calling `carla-roach/benchmark.py`.

## Paper-Level Experiment Path

Recommended progression:

1. `smoke`: run a small Town01 training job and confirm real CARLA sampling, updates, saving, and resume.
2. benchmark smoke: evaluate a smoke-run output with `nocrash_dense` or another small suite.
3. reduced sequential: run Town01 to Town02 with a small step count and inspect `U_tilde_VAS`, buffer migration, and checkpoint restore.
4. standard joint: run Town01 through Town06 joint sampling, starting with at least 3 seeds for trend checks.
5. full continual: run Town01 through Town06 with 2,000,000 steps per town and 10 seeds.
6. ablations and diagnostics: `-SAN`, `-PS`, Actor-side, Actor-Critic, t-SNE, activation ratio, `D_pi`, and `C_Q`.

The current code implements the default Critic-side SBA-UAS training loop. Actor-side and Actor-Critic variants remain separate ablation work.

## Troubleshooting

### Missing `CarlaUE4.sh`

Confirm that `CARLA_ROOT` points to the CARLA root directory:

```bash
export CARLA_ROOT=/opt/carla-0.9.11
ls ${CARLA_ROOT}/CarlaUE4.sh
```

### Missing `carla` Python Package

Install the matching Python egg:

```bash
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
```

### Missing BEV Map Files

Generate birdview map files using the Roach utility. The command usually has this shape:

```bash
python -m carla_gym.utils.birdview_map \
  --save_dir carla-roach/carla_gym/core/obs_manager/birdview/maps \
  --pixels_per_meter 5.00 \
  --carla_sh_path ${CARLA_ROOT}/CarlaUE4.sh
```

### Check Whether a Checkpoint Can Be Evaluated

```bash
python scripts/check_policy_checkpoint_compat.py path/to/ckpt.pth
```

If the current environment has the full CARLA/Roach dependency stack, also run a Roach load smoke check:

```bash
python scripts/check_policy_checkpoint_compat.py path/to/ckpt.pth --load-with-roach
```

## Current Status

Lightweight tests that do not require a CARLA server have passed in the `driveadapter` environment:

```bash
conda run -n driveadapter pytest -q
# 41 passed
```

Paper-level numbers still require real CARLA runs: six-town joint sampling, Town01 to Town06 sequential training, 10 seeds, ablations, and diagnostic visualizations.
