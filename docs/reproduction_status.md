# SBA-UAS Reproduction Status

Check date: 2026-05-25.

This document compares the SBA-UAS paper with the current repository implementation. The `carla-roach/` directory is treated as a read-only upstream baseline.

## Summary

The project has implemented the core SBA-UAS modules, the Roach PPO integration path, and a Roach-compatible sidecar training loop that can be tested without a CARLA server. Real CARLA smoke runs and paper-level long runs are still pending.

| Milestone | Status | Notes |
| --- | --- | --- |
| Milestone 1: read-only Roach compatibility and static checks | Mostly complete | `compat/roach_policy.py` uses lazy import; `check_policy_checkpoint_compat.py` compares the SBA-UAS policy YAML with Roach `xtma_beta.yaml` and supports optional `--load-with-roach`. |
| Milestone 2: Critic/SAN prototype loop | Mostly complete | Vector SAN, Gated Double Critic, SAN loss, and Bellman loss are implemented and covered by tests. |
| Milestone 3: BNN, shifted VAS, and dual buffers | Mostly complete | Roach BEV autoencoder, Bayesian latent transition, shifted VAS, Standard Buffer, and Familiar Buffer are implemented; real CARLA rollout distribution checks are still needed. |
| Milestone 4: parameter stabilization | Mostly complete | Reference snapshots, Reward-Parameter Correlation, SAN SI-style importance, parameter stabilization loss, and D-only reference Critic/SAN updates are implemented. |
| Milestone 5: training entry point and CARLA smoke run | Partially complete | Roach rollout adapter, sidecar trainer, Roach PPO `SBAUASPPO` integration, split checkpoint save/restore, and training/evaluation scripts are implemented; a real CARLA smoke run is still missing. |
| Milestone 6: paper-level experiments and ablations | Not complete | Six-town joint sampling, Town01 to Town06 sequential training, 10 seeds, ablations, and visualizations have not been run. |

## Implemented Components

| Paper component | Current location | Status |
| --- | --- | --- |
| Read-only Roach policy reuse | `src/sba_uas/compat/roach_policy.py` | Provides an import layer while preserving the Actor/policy contract. |
| Policy checkpoint contract check | `scripts/check_policy_checkpoint_compat.py` | Checks `policy_init_kwargs` and core `policy_state_dict` fields. |
| SAN gating | `src/sba_uas/critic/san.py` | Supports sigmoid gates and defaults that can be configured as `n=6,c=2048,rho=0.2`. |
| Gated Double Critic | `src/sba_uas/critic/gated_critic.py` | Implements twin Q networks with SAN gates applied layer by layer. |
| SAN loss and Bellman loss | `src/sba_uas/critic/losses.py` | Includes activation similarity, budget loss, and clipped double-Q targets. |
| Bayesian BEV environment model | `src/sba_uas/stabilization/environment_model.py`, `bnn.py` | Supports Roach `birdview` masks plus `state`, internal normalization, CNN encoder/decoder, latent BNN transition, composite loss, and MC prediction. |
| shifted VAS | `src/sba_uas/stabilization/vas.py` | Scores transitions using MC next-state prediction error. |
| Dual buffers | `src/sba_uas/stabilization/replay_buffer.py` | Moves the lowest-`u_vas` samples from B to D when B is full, and evicts the highest-`u_vas` samples when D is full. |
| Reference network | `src/sba_uas/stabilization/reference_network.py` | Provides frozen deep copies and sync support. |
| Reward-Parameter Correlation | `src/sba_uas/stabilization/reward_parameter_correlation.py` | Implements reward-correlated Critic-side importance. |
| SAN SI-style importance | `src/sba_uas/stabilization/synaptic_importance.py` | Tracks `delta_theta * -grad` path importance and clips negative values when converting to `Omega`. |
| Parameter stabilization loss | `src/sba_uas/stabilization/regularization.py` | Implements `eta * sum Omega * (theta - theta_ref)^2`. |
| AP, AF, and FWT metrics | `src/sba_uas/training/metrics.py` | Implements paper experiment metrics with tests. |
| Split checkpointing | `src/sba_uas/training/checkpointing.py` | Separates Roach policy checkpoints from SBA-UAS extra state and is covered by tests. |
| Roach rollout adapter | `src/sba_uas/training/roach_env_adapter.py` | Converts Roach `birdview/state` rollout steps into replay transitions with shifted VAS. |
| Roach-compatible sidecar trainer | `src/sba_uas/training/trainer.py` | Keeps the Actor as the Roach policy and trains buffers, environment model, SAN/Critic, D-only references, importance, and checkpoints as extra state. |
| Roach PPO integration | `src/sba_uas/training/roach_ppo_sidecar.py` | Extends upstream PPO; the Actor is still updated by Roach PPO, while rollout data also trains the SBA-UAS sidecar. Warm-up can enable gated Double-Q advantage replacement without changing the Actor network. |
| Training script | `scripts/run_train_sba_uas.sh` | Supports `smoke`, `standard`, and `sequential` modes and checks `driveadapter` plus `CARLA_ROOT` by default. |
| Evaluation script | `scripts/run_benchmark_sba_uas.sh` | Runs a policy contract check before loading a Roach-compatible policy checkpoint into Roach benchmark evaluation. |

## Remaining Gaps

1. A real single-town CARLA/Roach smoke run has not been executed; the current training loop is validated with fake Roach-like vector environment tests.
2. Paper-level results are still missing: six-town joint sampling, Town01 to Town06 sequential training, 10 seeds, ablations, and visualizations.
3. Actor-side and Actor-Critic variants are only tracked in the experiment matrix. The default implementation remains Critic-side SBA-UAS. Reproducing Table IV requires separate ablation implementations.
4. Diagnostic scripts for t-SNE, activated neuron ratio, Actor policy drift `D_pi`, and Critic gradient alignment `C_Q` are not implemented yet.

## Verification

Base Python 3.9 test collection fails without `torch`. The existing conda environment has been verified as follows:

```bash
conda run -n driveadapter pytest -q
# 41 passed
```

The `driveadapter` environment uses PyTorch 1.13.1. The current training-side code was verified there. A previous compatibility issue around `torch.minimum` on PyTorch 1.4 has been fixed.

## Recommended Next Steps

1. Set `CARLA_ROOT` in `driveadapter` and run `scripts/run_train_sba_uas.sh smoke` to verify real sampling, updates, saving, and restore.
2. Run `scripts/run_benchmark_sba_uas.sh <policy_ckpt>` for a single-suite benchmark smoke test.
3. Inspect BEV environment-model loss, `U_tilde_VAS` distribution, and Familiar Buffer map coverage on real Roach rollouts.
4. Run a reduced Town01 to Town02 sequential job before scaling to Town01 through Town06 and 10 seeds.
