# Repository Rules

These rules apply to the entire `sba-uas` repository.

1. `carla-roach/` is a read-only upstream baseline. Do not modify, delete, move, rename, reformat, or generate experiment/configuration/documentation changes inside `carla-roach/`.
2. New SBA-UAS code belongs in `src/sba_uas/`; new configuration belongs in `configs/sba_uas/`; new scripts belong in `scripts/`; new documentation belongs in `docs/` or in root-level project documents.
3. The SBA-UAS policy must use the same model structure and checkpoint contract as CARLA-Roach. The default policy source is `PpoPolicy` from `carla-roach/agents/rl_birdview/models/ppo_policy.py`, initialized with parameters matching `carla-roach/config/agent/ppo/policy/xtma_beta.yaml`.
4. Do not change the Actor/policy network structure for SBA-UAS. Paper components such as SAN, Gated Critic, BNN environment modeling, Familiar Experience Buffer, Reward-Parameter Correlation, and parameter stabilization must be implemented only as Critic-side or training-side extensions.
5. When reusing the Roach policy or checking checkpoint compatibility, import Roach through `src/sba_uas/compat/roach_policy.py` or through standalone scripts that add `carla-roach/` to `PYTHONPATH`. Do not copy and rewrite Roach source code into a new policy implementation.
6. CARLA-Roach training and evaluation depend on a Linux CARLA package, `CarlaUE4.sh`, `killall`, and the Linux Python egg. Native Windows should be used only for code editing, documentation, and lightweight static checks. Real CARLA training and evaluation should run on Linux, WSL2/Ubuntu, or a Linux server.
