# 仓库规则

本规则作用于整个 `sba-uas` 仓库。

1. `carla-roach/` 是只读上游基线目录。任何复现、适配、实验、脚本、配置或文档改动都不得修改、删除、移动、重命名或格式化 `carla-roach/` 下的任何文件。
2. SBA-UAS 的新增代码放在 `src/sba_uas/`；新增配置放在 `configs/sba_uas/`；新增脚本放在 `scripts/`；新增说明放在 `docs/` 或根目录文档中。
3. SBA-UAS 的 policy 必须与 CARLA-Roach 的 policy 使用完全相同的模型结构和 checkpoint 合同。默认 policy 来源固定为 `carla-roach/agents/rl_birdview/models/ppo_policy.py` 中的 `PpoPolicy`，并使用与 `carla-roach/config/agent/ppo/policy/xtma_beta.yaml` 一致的初始化参数。
4. 不得为了实现 SBA-UAS 而修改 Actor/policy 网络结构。论文中的 SAN、Gated Critic、BNN 环境模型、Familiar Experience Buffer、Reward-Parameter Correlation 和参数稳定化机制只能作为 Critic 侧或训练侧扩展实现。
5. 如果需要复用 Roach policy 或检查 checkpoint 兼容性，应通过 `src/sba_uas/compat/roach_policy.py` 或独立脚本把 `carla-roach/` 加入 `PYTHONPATH`，不得把 Roach 源码复制后改写成新的 policy。
6. CARLA-Roach 的训练和评测入口依赖 Linux CARLA 包、`CarlaUE4.sh`、`killall` 和 Linux Python egg。Windows 原生环境只用于代码编辑、文档整理和轻量静态检查；实际 CARLA 训练/评测应在 Linux/WSL2/Ubuntu 或 Linux 服务器上执行。
