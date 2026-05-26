# SBA-UAS Reproduction Plan

This plan tracks the remaining work needed to turn the current implementation into a paper-level SBA-UAS reproduction while keeping `carla-roach/` read-only and preserving the Roach policy checkpoint contract.

## Scope

The implementation must keep the Actor/policy structure identical to CARLA-Roach `PpoPolicy`. SBA-UAS mechanisms are implemented as Critic-side or training-side additions:

- Similarity-based Activation Network (SAN)
- Gated Double Critic
- Bayesian BEV environment model
- shifted VAS uncertainty scoring
- Standard Buffer and Familiar Experience Buffer
- reference Critic/SAN updates
- Reward-Parameter Correlation
- parameter stabilization regularization

Roach-compatible policy checkpoints must continue to contain only the Roach-loadable policy contract. SBA-UAS extra state is stored separately.

## Engineering Milestones

### 1. Roach Compatibility

Deliverables:

- Lazy Roach import through `src/sba_uas/compat/roach_policy.py`
- policy YAML comparison against Roach `xtma_beta.yaml`
- checkpoint contract checks for `policy_init_kwargs`, `policy_state_dict`, and optional Roach load smoke

Acceptance criteria:

- `scripts/check_policy_checkpoint_compat.py` runs without a CARLA server.
- The check fails on incompatible policy initialization values.
- No Roach source file is modified.

### 2. Critic and SAN Loop

Deliverables:

- Vector SAN with configurable number of layers, hidden size, and activation budget
- Gated Double Critic
- SAN activation-similarity and budget losses
- Bellman target loss with clipped double-Q targets

Acceptance criteria:

- Synthetic batches can run forward, backward, and optimizer steps.
- The activation ratio tracks the configured budget.
- Actor/policy architecture remains unchanged.

### 3. BNN, shifted VAS, and Buffers

Deliverables:

- Roach BEV and measurement adapter for the environment model
- Bayesian latent transition model with MC next-state prediction
- shifted VAS scorer
- Standard Buffer B and Familiar Buffer D migration/eviction logic

Acceptance criteria:

- Familiar samples trend toward lower `u_vas` in synthetic tests.
- Buffer D preserves low-uncertainty samples under capacity pressure.
- Real Roach rollout distributions can be logged once CARLA is available.

### 4. Parameter Stabilization

Deliverables:

- frozen reference Critic and SAN snapshots
- D-only reference updates
- Reward-Parameter Correlation for the Critic
- SI-style importance for SAN parameters
- regularization loss for Critic and SAN parameters
- checkpoint save/restore for references and importance trackers

Acceptance criteria:

- Regularization loss is non-negative.
- Loss increases when protected parameters drift from references.
- Save/restore preserves importance values and reference parameters.

### 5. Training Integration

Deliverables:

- Roach rollout adapter
- sidecar trainer for SBA-UAS extra state
- Roach PPO subclass or entry point that mirrors rollouts into the sidecar
- split checkpoint writing for Roach policy and SBA-UAS extra state
- smoke, standard, and sequential training scripts

Acceptance criteria:

- A single-town smoke run completes sampling, updates, saving, and resume.
- Roach policy checkpoint compatibility checks continue to pass.
- SBA-UAS extra state can be restored independently of the policy checkpoint.

### 6. Paper-Level Experiments

Deliverables:

- six-town joint sampling results
- Town01 to Town06 sequential continual-learning results
- 10-seed runs or clearly documented reduced protocols
- ablations for `-SAN`, `-PS`, Actor-side, and Actor-Critic variants
- diagnostic plots for t-SNE, activation ratio, `D_pi`, and `C_Q`

Acceptance criteria:

- Every result is tied to a config snapshot, checkpoint path, seed, CARLA version, and Roach baseline snapshot.
- Reduced protocols are explicitly labeled and not mixed with full paper-level results.

## Experiment Protocols

### Standard Joint Sampling

Goal: reproduce the paper's standard RL agent-sampling protocol.

Settings:

- Town01 through Town06 sampled jointly
- traffic density aligned with the paper's busy setting
- training and evaluation cover all six towns

Process:

1. Run one seed at a reduced step count.
2. Run at least three seeds to check the trend.
3. Scale to 10 seeds for table-quality results.

### Sequential Continual Learning

Goal: reproduce the Town01 to Town06 continual-learning protocol.

Settings:

- map order: Town01 -> Town02 -> Town03 -> Town04 -> Town05 -> Town06
- after a map switch, do not collect new trajectories from previous maps
- old-domain retention must come only from replay, buffers, or checkpoints
- full protocol uses about 2,000,000 environment steps per map stage

Per-stage outputs:

- current-map learning curve
- evaluation matrix over all seen maps
- checkpoint and extra-state paths
- invalid-policy events, if any

### Ablations and Diagnostics

Required variants:

- Full SBA-UAS with Critic-side SAN and parameter stabilization
- SBA-UAS Actor-side variant
- SBA-UAS Actor-Critic variant
- `-SAN` without SAN gating
- `-PS` without Familiar Buffer and uncertainty-based parameter stabilization

Diagnostics:

- t-SNE of Context/SAN features
- Q-network activated neuron ratio by driving scenario
- `U_tilde_VAS` distribution across map switches
- Familiar Buffer D map and scenario coverage
- regularization loss versus final performance

## Test Strategy

Lightweight tests that do not require CARLA:

- policy config and checkpoint contract tests
- SAN shape, activation, and loss tests
- Gated Critic forward/backward tests
- BNN MC sampling and shifted VAS tests
- buffer migration and eviction tests
- regularization loss and state-restore tests

CARLA-dependent integration tests:

- single-town single-seed smoke run
- checkpoint resume smoke run
- small benchmark smoke run

Paper-level statistical checks:

- AP, AF, and FWT metrics
- invalid-policy event tracking
- unified result tables generated from config snapshots

## Risks

### Roach PPO differs from the paper formulas

The paper equations are closer to off-policy Actor-Critic or SAC-style training, while this repository must keep the Roach `PpoPolicy` Actor and checkpoint contract. The default path therefore applies SBA-UAS to Critic/value estimation and training-side state. Any additional off-policy Critic heads must be stored only in SBA-UAS extra state.

### BEV channels may not exactly match the paper

Roach observations are treated as the source of truth for Actor and environment inputs. Any compression or reshaping for the BNN or SAN belongs inside `src/sba_uas/` and should be documented in configuration.

### CARLA version differences

The paper targets CARLA 0.9.11, while the Roach training stack may be easier to run with CARLA 0.9.10.x. Smoke runs should prioritize a runnable setup. Paper-level results should use a fixed CARLA version and report it.

### Training cost is high

The default SAN/Gated Critic can use six 2048-unit layers, and the BNN can use many MC samples. Debug configurations should use smaller models and fewer samples. Full experiments should keep all non-ablated factors fixed.
