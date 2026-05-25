# SBA-UAS 当前复现状态

检查日期：2026-05-25。

本文档对照 `paper/main.pdf` 和当前仓库实现，记录 SBA-UAS 复现已经完成、部分完成和未完成的范围。`carla-roach/` 按仓库规则视为只读上游。

## 总体结论

当前项目处于“核心模块和 Roach-compatible sidecar 训练闭环已实现并可轻量测试，尚未完成真实 CARLA smoke run”的阶段。

按里程碑估计：

| 里程碑 | 状态 | 说明 |
| --- | --- | --- |
| Milestone 1：只读 Roach 兼容与静态检查 | 部分完成 | 有 `compat/roach_policy.py` 和 checkpoint 合同脚本，但缺少自动对比 Roach yaml、真实 checkpoint load smoke test。 |
| Milestone 2：Critic/SAN 原型闭环 | 基本完成 | 已实现向量版 SAN、Gated Double Critic、SAN loss、Bellman loss，并有单测覆盖。 |
| Milestone 3：BNN + shifted VAS + 双 buffer | 基本完成 | 已实现 Roach BEV autoencoder + Bayesian latent transition、shifted VAS、Standard/Familiar buffer；仍需接入真实训练循环验证。 |
| Milestone 4：Parameter Stabilization | 基本完成 | 已实现 reference snapshot、Reward-Parameter Correlation、SAN SI-style importance、参数稳定化 loss，以及 D-only reference Critic/SAN 更新。 |
| Milestone 5：训练入口与 CARLA smoke run | 部分完成 | 已实现 Roach rollout adapter、sidecar trainer、checkpoint 拆分保存/恢复，并用 fake Roach-like vector env 验证；仍缺真实 CARLA smoke run 和运行脚本接入。 |
| Milestone 6：论文级实验与消融 | 未完成 | 尚未跑标准六地图联合采样、Town01->Town06 顺序流、10 seeds、消融和可视化。 |

## 已落地代码

| 论文组件 | 当前落点 | 状态 |
| --- | --- | --- |
| Roach policy 只读复用 | `src/sba_uas/compat/roach_policy.py` | 有导入层，保持 Actor/policy 合同。 |
| policy checkpoint 合同检查 | `scripts/check_policy_checkpoint_compat.py` | 可检查 `policy_init_kwargs` 和 `policy_state_dict` 基本字段。 |
| SAN gating | `src/sba_uas/critic/san.py` | 支持 sigmoid gate，默认可配置为 `n=6,c=2048,rho=0.2`。 |
| Gated Double Critic | `src/sba_uas/critic/gated_critic.py` | 双 Q 网络，隐藏层逐层乘 SAN gate。 |
| SAN loss / Bellman loss | `src/sba_uas/critic/losses.py` | 有 activation similarity、budget loss、clipped double-Q target。 |
| Bayesian BEV environment model | `src/sba_uas/stabilization/environment_model.py`、`bnn.py` | 支持 Roach `birdview` masks + `state` 输入，内部归一化，包含 CNN encoder/decoder、latent BNN transition、复合 loss 和 MC prediction。 |
| shifted VAS | `src/sba_uas/stabilization/vas.py` | 按 MC next-state prediction error 打分。 |
| 双 buffer | `src/sba_uas/stabilization/replay_buffer.py` | 实现 B 满后迁移最低 `u_vas` 到 D，D 满后淘汰最高 `u_vas`。 |
| Reference network | `src/sba_uas/stabilization/reference_network.py` | 冻结 deep copy，支持 sync。 |
| Reward-Parameter Correlation | `src/sba_uas/stabilization/reward_parameter_correlation.py` | 实现 Critic 侧 reward-correlated importance。 |
| SAN SI-style importance | `src/sba_uas/stabilization/synaptic_importance.py` | 实现 SAN 侧 `delta_theta * -grad` 路径重要性，并在转为 `Omega` 时裁剪负值。 |
| Parameter stabilization loss | `src/sba_uas/stabilization/regularization.py` | 实现 `eta * sum Omega * (theta - theta_ref)^2`。 |
| AP/AF/FWT 指标 | `src/sba_uas/training/metrics.py` | 已实现论文实验统计指标，覆盖单测。 |
| checkpoint 拆分 | `src/sba_uas/training/checkpointing.py` | 已实现 Roach policy checkpoint 与 SBA-UAS extra state 分离保存，覆盖单测。 |
| Roach rollout adapter | `src/sba_uas/training/roach_env_adapter.py` | 将 Roach `birdview/state` rollout step 转为带 shifted VAS 的 replay transition。 |
| Roach-compatible sidecar trainer | `src/sba_uas/training/trainer.py` | Actor 保持 Roach policy；extra state 侧串起 B/D buffer、环境模型、SAN/Critic、D-only reference 更新、importance 和 checkpoint。 |

## 与论文仍有差距的关键点

1. 尚未在真实 CARLA/Roach 环境中执行单 Town smoke run；当前训练闭环通过 fake Roach-like vector env 单测验证。
2. `scripts/run_train_sba_uas.sh` 与 `scripts/run_benchmark_sba_uas.sh` 还不是完整可运行入口，只是提示 scaffold。
3. 缺少 CARLA smoke config：单 Town、小步数、可恢复 checkpoint、无 W&B 或 offline W&B 模式。
4. 缺少论文级实验配置矩阵：六地图联合采样、顺序流、消融 variants、10 seeds、结果聚合和可视化。
5. 工作区内 `carla-roach/` 已存在大量未提交修改；后续复现不能继续改动该目录，若需要可单独核对这些修改是否来自行尾转换或外部编辑。

## 当前验证结果

在 base Python 3.9 环境中，测试收集会因缺少 `torch` 失败。使用现有 conda 环境验证结果如下：

```bash
conda run -n driveadapter pytest -q
# 38 passed
```

`driveadapter` 环境中 PyTorch 为 1.13.1。本轮按用户要求使用 `driveadapter` 验证训练侧新增代码。此前已修复 `torch.minimum` 在 PyTorch 1.4 不存在的兼容问题。

## 推荐下一步

1. 把 `scripts/run_train_sba_uas.sh` 接到 `RoachCompatibleSBAUASTrainer` 或对应 Hydra entry point。
2. 配置单 Town 小步数 CARLA smoke run，确认真实采样、更新、保存、恢复。
3. 在真实 Roach rollout 上检查 BEV 环境模型的 loss 曲线和 `U_tilde_VAS` 分布。
4. 启动 Town01 -> Town02 sequential smoke run，然后扩展到 Town01 -> Town06 与 10 seeds。
