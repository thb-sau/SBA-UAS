# SBA-UAS 论文复现计划

本文档面向 `paper/main.pdf` 中的 SBA-UAS 方法复现，目标是在不修改 `carla-roach/` 的前提下，基于 CARLA-Roach 的环境、BEV observation、Actor/policy checkpoint 合同和 benchmark 入口，实现论文中的 Critic 侧持续强化学习机制，并复现标准多地图采样与顺序持续学习实验。

## 0. 复现边界与硬约束

### 0.1 不修改 Roach 上游

- `carla-roach/` 只作为只读基线使用。
- 不复制、改写或格式化 Roach 的 policy、模型、环境和训练源码。
- 新增代码只放在 `src/sba_uas/`，新增配置只放在 `configs/sba_uas/`，新增脚本只放在 `scripts/`，新增说明只放在 `docs/` 或根目录文档。

### 0.2 Actor/policy 保持 Roach 兼容

论文中 SBA-UAS 的持续学习机制主要作用于 Critic，Actor 不做结构性修改。项目实现必须保持：

- Actor/policy 默认使用 `carla-roach/agents/rl_birdview/models/ppo_policy.py` 中的 `PpoPolicy`。
- policy 初始化参数与 `carla-roach/config/agent/ppo/policy/xtma_beta.yaml` 一致。
- 面向 Roach policy 的 checkpoint 必须继续包含 Roach 所需的 `policy_init_kwargs` 与 `policy_state_dict`。
- SAN、Gated Critic、BNN、Familiar Experience Buffer、Reward-Parameter Correlation 等额外状态单独保存，不能污染 Roach policy checkpoint 合同。

### 0.3 复现优先级

第一优先级是可验证的模块闭环，而不是一次性追求论文表格数值：

1. 先确保 Roach policy 能被只读导入、初始化和加载。
2. 再实现可单测的 SBA-UAS 核心模块。
3. 然后接入训练循环，跑通小规模 smoke run。
4. 最后扩大到 CARLA 0.9.11 六地图实验、10 seeds 统计和消融。

## 1. 论文方法拆解

SBA-UAS 解决的是 task-agnostic continual reinforcement learning for autonomous driving 中的三类问题：

- Loss of plasticity：长期训练后网络吸收新场景知识的能力下降。
- Catastrophic forgetting：地图/交通流切换后旧场景价值结构被覆盖。
- Representation interference：异构驾驶场景共享同一 Critic 表征空间导致梯度互相干扰。

论文方法由两个子系统组成：

### 1.1 Module A：Actor-Critic 主干与 Critic 侧 Similarity-Based Activation

- Actor：保持标准 policy，不增加 SAN 或 gating 结构。
- SAN：从 BEV 和 measurement 编码状态特征，输出 gating matrix `R`。
- Gated Critic：用 `R` 调制 Q 网络隐藏层，使相似场景激活相似路径，异构场景激活部分隔离路径。
- SAN loss：约束 gating 相似度匹配状态特征相似度，并通过激活预算控制稀疏程度。
- Critic loss：基于 Double Q / off-policy Bellman loss 更新。
- Actor loss：由 Critic 提供价值梯度，Actor 结构不变。

### 1.2 Module B：Uncertainty-Aware Stabilization

- BNN Environment Model：在 latent space 中预测下一状态，用 Bayesian posterior sampling 估计 transition familiarity。
- shifted VAS：用多次 BNN stochastic forward 的预测误差估计 `U_tilde_VAS`，低值表示熟悉 transition。
- Standard Buffer `B`：保存常规训练经验。
- Familiar Experience Buffer `D`：保存从 `B` 中迁移出来的低不确定性熟悉经验，作为历史知识锚点。
- Reference Networks：用 `D` 训练 Critic/SAN 的 reference copy。
- Reward-Parameter Correlation：用参数更新与 batch reward 变化的关联估计 Critic 参数重要性。
- Parameter Stabilization：对 Critic 与 SAN 的关键参数施加正则，降低旧知识被覆盖的概率。

## 2. 模块化实现路线

### 2.1 兼容层与基线验证

目标：先证明项目能只读复用 Roach policy 和配置。

落点：

- `src/sba_uas/compat/roach_policy.py`
- `configs/sba_uas/policy_compatibility.yaml`
- `scripts/check_policy_checkpoint_compat.py`
- `tests/compat/`

步骤：

1. 固化 `PYTHONPATH=src:carla-roach` 的导入方式。
2. 校验 `PpoPolicy`、`XtMaCNN`、`BetaDistribution` 可从 Roach 导入。
3. 读取 `configs/sba_uas/policy_compatibility.yaml`，与 Roach `xtma_beta.yaml` 做字段级比较。
4. 增加 checkpoint 合同检查：
   - policy checkpoint 存在 `policy_init_kwargs`。
   - policy checkpoint 存在 `policy_state_dict`。
   - 初始化参数与 Roach baseline 完全一致。
   - checkpoint 可被 Roach `PpoPolicy.load()` 或等价路径加载。
5. 输出清晰错误信息，避免训练结束后才发现 checkpoint 不兼容。

验收：

- 在没有 CARLA server 的机器上也能跑通静态兼容性检查。
- 不触碰 `carla-roach/` 文件。

### 2.2 Critic 表征与 SAN

目标：实现论文中的 Similarity-Based Activation Network，并只用于 Critic 侧。

落点：

- `src/sba_uas/critic/san.py`
- `src/sba_uas/critic/gated_critic.py`
- `src/sba_uas/critic/losses.py`
- `tests/critic/`

步骤：

1. 定义 SAN 输入合同：
   - BEV observation `s`。
   - measurement vector `m`。
   - 输出 gating matrix `R`，形状为 `[batch, n_layers, units_per_layer]`。
2. 按论文配置实现默认结构：
   - `n_layers = 6`。
   - `units_per_layer = 2048`。
   - shared mapping hidden units 为 `1024`。
   - target active ratio `rho = 0.2`。
3. 实现 sigmoid gating，而不是 row-wise softmax。
4. 实现 SAN loss：
   - 用 CNN/encoder 特征计算状态相似度。
   - 用 cosine similarity 计算 gating 相似度。
   - 加入激活预算项 `abs(||R_i||_1 - rho * c)`。
5. 实现 Gated Critic：
   - Double Q critic。
   - 每层 hidden activation 乘对应 `R_i`。
   - Actor 不接收或复用 gating 结构。
6. 增加形状、梯度、稀疏度单测：
   - `R` 值域在 `(0, 1)`。
   - 激活比例接近配置目标。
   - 相同状态生成更接近的 gate。
   - Critic forward/backward 不断梯度。

验收：

- SAN 与 Gated Critic 可独立 forward/backward。
- 默认配置与 `configs/sba_uas/train_sba_uas.yaml` 对齐。
- Actor/policy 结构没有被改动。

### 2.3 BNN 环境模型与 shifted VAS

目标：实现论文中的 uncertainty/familiarity 估计，作为经验筛选信号。

落点：

- `src/sba_uas/stabilization/environment_model.py`
- `src/sba_uas/stabilization/bnn.py`
- `src/sba_uas/stabilization/vas.py`
- `tests/stabilization/`

步骤：

1. 定义环境模型输入输出：
   - 输入：`s, m, a`。
   - 编码：`z = ENC(s)`。
   - BNN 预测：`z_hat_next = BNN(z, m, a)`。
   - 解码：`s_hat_next = DEC(z_hat_next)`。
2. 默认 latent dim 为 `256`。
3. 实现 Bayesian layer 或可替换的 variational MLP：
   - 参数 posterior 为 factorized Gaussian。
   - 支持 Monte Carlo sampling。
   - 默认 `N = 64` 次采样。
4. 实现环境模型 loss：
   - 当前状态 reconstruction。
   - 下一状态 reconstruction。
   - latent transition prediction。
   - observation-space next-state prediction。
   - KL regularization。
5. 实现 shifted VAS：
   - 对同一 transition 多次采样。
   - 用平均 prediction error 作为 `U_tilde_VAS`。
   - 明确该分数只作为 familiarity/replay filtering，不作为安全关键事件优先级。
6. 增加轻量单测：
   - MC 采样输出形状稳定。
   - `U_tilde_VAS` 为非负标量。
   - 熟悉样本经过训练后分数下降。

验收：

- 环境模型可在离线 batch 上训练。
- `U_tilde_VAS` 可附加到 replay transition。

### 2.4 双 buffer 与熟悉经验维护

目标：实现 Standard Buffer `B` 与 Familiar Experience Buffer `D` 的数据流。

落点：

- `src/sba_uas/stabilization/replay_buffer.py`
- `src/sba_uas/stabilization/familiar_buffer.py`
- `tests/stabilization/test_buffers.py`

步骤：

1. 定义 transition 数据结构：
   - `s, m, a, r, s_next, m_next, done, u_vas`。
2. 新 transition 先由 BNN 标注 `u_vas`，再进入 `B`。
3. 当 `B` 满：
   - 找出 `B` 中 `u_vas` 最低的样本。
   - 将该样本迁移到 `D`。
   - 新样本进入 `B`。
4. 当 `D` 满：
   - 找出 `D` 中 `u_vas` 最高的样本。
   - 用更熟悉的迁移样本替换它。
5. 对相同极值的样本使用随机 tie-break，避免固定顺序偏差。
6. 支持保存与恢复：
   - `B` 用于常规训练恢复。
   - `D` 是 SBA-UAS 额外状态，保存到独立 extra checkpoint。

验收：

- buffer 容量、迁移、淘汰逻辑有单测覆盖。
- `D` 中样本整体 `u_vas` 分布低于 `B`。

### 2.5 Reference Networks 与参数稳定化

目标：复现 Uncertainty-Aware Stabilization 的核心正则。

落点：

- `src/sba_uas/stabilization/reference_network.py`
- `src/sba_uas/stabilization/reward_parameter_correlation.py`
- `src/sba_uas/stabilization/regularization.py`
- `tests/stabilization/test_regularization.py`

步骤：

1. 为 Critic 和 SAN 分别维护 reference network：
   - `Q_ref` 只用 `D` 训练。
   - `SAN_ref` 只用 `D` 训练。
2. 实现 Critic 的 Reward-Parameter Correlation：
   - 记录相邻训练 step 的参数差 `phi_i(t) - phi_i(t-1)`。
   - 记录 batch mean reward 差 `r_bar(t) - r_bar(t-1)`。
   - 累积 `omega_i` 与归一化分母 `omega_bar_i`。
   - 得到重要性 `Omega_i = omega_i / (omega_bar_i + zeta)`。
3. 实现 SAN 的 SI-style importance：
   - 用 SAN loss gradient 与参数变化累积。
   - 对负值做 `max(0, value)` 裁剪。
4. 实现参数保护项：
   - `L_ps = eta * sum_i Omega_i * (theta_i - theta_ref_i)^2`。
   - 分别加到 Critic loss 与 SAN loss。
5. 支持 importance 状态保存：
   - `Omega`。
   - reference network state dict。
   - 上一 step 参数 snapshot。
   - reward running state。

验收：

- 正则项为非负。
- 关键参数偏离 reference 时 loss 增大。
- 保存/恢复后 importance 和 reference 状态一致。

### 2.6 训练循环接入

目标：把上面模块接入 Roach/CARLA 采样与训练流程，形成可运行 SBA-UAS 训练入口。

落点：

- `src/sba_uas/training/trainer.py`
- `src/sba_uas/training/roach_env_adapter.py`
- `src/sba_uas/training/checkpointing.py`
- `scripts/run_train_sba_uas.sh`
- `configs/sba_uas/train_sba_uas.yaml`

步骤：

1. 复用 Roach 环境、wrapper、reward、terminal 和 BEV observation。
2. 统一 observation 适配：
   - Roach BEV channel 数与论文输入可能存在差异，先以 Roach 实际 observation 为准。
   - 如需给 BNN 使用论文中的压缩输入，转换逻辑放在 `sba_uas` 侧。
3. 每个环境 step：
   - Actor 用 Roach policy 产出 action。
   - SAN 为 Critic 产出 `R`。
   - BNN 计算 `u_vas`。
   - transition 进入双 buffer。
4. 每个训练 step：
   - 用 `B` 更新 Critic/SAN/Actor。
   - 用 `B ∪ D` 更新环境模型。
   - 用 `D` 更新 reference Critic/SAN。
   - 更新 importance。
   - 将 parameter stabilization 加入 Critic/SAN loss。
5. checkpoint 拆分：
   - `policy.pth`：保持 Roach policy 可加载。
   - `sba_uas_extra_state.pth`：保存 Critic、SAN、BNN、buffers、reference networks、importance。
6. 训练日志：
   - reward。
   - route completion / infraction / driving score。
   - SAN activation ratio。
   - `u_vas` 分布。
   - `D` buffer 地图/场景覆盖。
   - parameter regularization loss。

验收：

- 单环境小步数 smoke run 可完成。
- checkpoint 可恢复继续训练。
- Roach policy checkpoint 兼容性检查仍通过。

## 3. 实验复现路线

### 3.1 环境与版本

论文实验使用 CARLA 0.9.11 和 LeaderBoard benchmark。当前 Roach scaffold 中也存在 CARLA 0.9.10.1 相关配置，复现实验需要明确两个层级：

- 工程 smoke run：优先使用当前 Roach 环境最容易跑通的版本。
- 论文指标复现：以 CARLA 0.9.11 + LeaderBoard 六地图设置为准。

需要记录：

- CARLA 版本。
- Roach commit 或目录快照。
- Python、PyTorch、CUDA 版本。
- 每次实验的 config snapshot。
- random seed。

### 3.2 标准多地图联合采样实验

目标：复现论文 Table I 的标准 RL agent-sampling protocol。

设置：

- Town01 到 Town06 联合采样。
- traffic density 参考论文 busy setting。
- 训练和评测都覆盖六个地图。
- 10 seeds。

指标：

- AP(DS)。
- AP(RC)。
- AP(IS)。

对比：

- SBA-UAS。
- PPO/Roach。
- SAC large buffer。
- CH-HNN adapted baseline。
- XdG 不纳入该设置，原因是依赖显式 task ID。

阶段验收：

1. 先完成 1 seed 小步数 sanity check。
2. 再跑 3 seeds 验证趋势。
3. 最后跑 10 seeds 形成表格。

### 3.3 顺序持续学习压力测试

目标：复现论文 Table II、Table III、Fig. 4 的 continual-learning protocol。

设置：

- 地图顺序：Town01 -> Town02 -> Town03 -> Town04 -> Town05 -> Town06。
- 每次切换后不再从旧地图采集新轨迹。
- old-domain 保留只能来自 replay/buffer/checkpoint。
- 论文中每个 map stage 约 2,000,000 environment steps，正式复现实验按该规模执行。

每个阶段输出：

- 当前 map 学习曲线。
- 当前模型在已见地图上的评测矩阵。
- AP/AF/FWT。
- 是否出现 invalid policy。

全局指标：

- Average Performance：`AP(Y)`。
- Average Forgetting：`AF(Y)`。
- Forward Transfer：`FWT(Y)`。
- `Y ∈ {DS, RC, IS}`。

### 3.4 可视化与机制验证

目标：复现论文中支撑方法解释的诊断图。

内容：

- Context/SAN feature 的 t-SNE 可视化。
- 不同驾驶场景下 Q-network activated neuron ratio 分布。
- `U_tilde_VAS` 在新旧地图切换时的分布变化。
- Familiar Buffer `D` 中样本来源地图与场景类型分布。
- Critic/SAN regularization loss 与最终表现的关系。

### 3.5 消融实验

目标：复现论文 Table IV、Fig. 7、Fig. 8 的关键消融。

必要 variants：

- Full SBA-UAS：只在 Critic 侧使用 SAN 与 parameter stabilization。
- SBA-UAS (Actor)：将机制放到 Actor 侧，Critic 回到普通结构。
- SBA-UAS (Actor-Critic)：Actor 与 Critic 同时 gated。
- `-SAN`：移除 SAN，使用普通 Critic。
- `-PS`：移除 Familiar Buffer / uncertainty-based parameter stabilization。

诊断指标：

- AP(DS)、AF(DS)、FWT(DS)。
- Actor-only old-domain rollout retention。
- Actor policy drift `D_pi`。
- Critic gradient alignment `C_Q`。

注意：

- Actor-side 和 Actor-Critic variants 只作为消融，不应成为默认实现路径。
- 默认实现仍必须保证 Roach Actor checkpoint 合同。

## 4. 推荐开发里程碑

### Milestone 1：只读 Roach 兼容与静态检查

交付：

- policy 兼容性检查脚本完善。
- policy config diff 检查。
- 基础 README/运行说明。

完成标准：

- `scripts/check_policy_checkpoint_compat.py` 能在无 CARLA server 情况下运行。
- CI 或本地静态检查不依赖修改 `carla-roach/`。

### Milestone 2：Critic/SAN 原型闭环

交付：

- SAN。
- Gated Double Critic。
- SAN loss。
- 单测。

完成标准：

- synthetic batch 可训练。
- activation ratio 接近 20%。
- Actor 结构未改动。

### Milestone 3：BNN + shifted VAS + 双 buffer

交付：

- BNN environment model。
- shifted VAS scorer。
- Standard/Familiar buffer。
- buffer save/load。

完成标准：

- 熟悉样本 `u_vas` 下降。
- `D` buffer 能稳定保留低不确定性样本。

### Milestone 4：Parameter Stabilization

交付：

- Reference Critic/SAN。
- Reward-Parameter Correlation。
- SAN SI-style importance。
- regularization loss。

完成标准：

- 正则项可训练、可保存恢复。
- 关闭/打开 stabilization 的 loss 和梯度表现可区分。

### Milestone 5：训练入口与小规模 CARLA smoke run

交付：

- `src/sba_uas/training/` 训练入口。
- checkpoint 拆分保存。
- smoke config。

完成标准：

- 单 Town 小步数训练能完整采样、更新、保存、恢复。
- 兼容性检查确认 policy checkpoint 仍可由 Roach 合同读取。

### Milestone 6：论文级实验与消融

交付：

- 标准多地图联合采样结果。
- 六地图顺序持续学习结果。
- 消融实验。
- 可视化与统计表。

完成标准：

- 每个实验至少 10 seeds 或明确记录资源不足时的 reduced protocol。
- 所有结果配置可追溯。

## 5. 测试策略

### 5.1 不依赖 CARLA 的轻量测试

- policy config/checkpoint 合同测试。
- SAN shape/activation/loss 测试。
- Gated Critic forward/backward 测试。
- BNN MC sampling 与 VAS 测试。
- buffer 迁移与淘汰测试。
- regularization loss 与 state restore 测试。

### 5.2 依赖 CARLA 的集成测试

- 单 Town 单 seed smoke run。
- checkpoint resume。
- benchmark eval smoke run。
- sequential map switch smoke run。

### 5.3 论文级统计测试

- 10 seeds。
- mean ± std。
- 记录 invalid policy。
- 统一生成 AP/AF/FWT 表格。

## 6. 风险与处理策略

### 6.1 Roach PPO 与论文公式存在算法差异

论文公式更接近 off-policy Actor-Critic / SAC 风格，而项目硬约束要求 Actor/policy 与 Roach `PpoPolicy` 保持兼容。处理方式：

- 默认 Actor 使用 Roach `PpoPolicy`。
- SBA-UAS 增量优先作用于 Critic/value estimation 和训练侧状态。
- 若实现 off-policy Critic 需要额外 action-value head，必须作为 SBA-UAS extra state 保存，不改变 Roach policy checkpoint。
- 文档中明确“论文公式复现”和“Roach checkpoint 兼容复现”的差异。

### 6.2 BEV channel 与论文设置不完全一致

论文描述中存在 BEV channel/shape 细节，Roach 实际 observation 可能不同。处理方式：

- Actor 和环境输入以 Roach 实际接口为准。
- BNN/SAN 的输入适配在 `src/sba_uas/` 内完成。
- 配置中记录使用的 observation schema。

### 6.3 CARLA 版本差异

论文使用 CARLA 0.9.11，Roach 训练链路可能更贴近 0.9.10.x。处理方式：

- smoke run 使用可运行版本优先。
- 论文级复现单独建立 0.9.11 环境。
- 结果表中标注版本，不混用。

### 6.4 BNN 和大 Critic 训练成本高

默认 SAN/Gated Critic 使用 6 层、每层 2048 units，BNN 默认 64 MC samples，计算成本高。处理方式：

- 提供 `debug` 配置：更小 hidden size、更少 MC samples。
- 正式实验使用论文默认配置。
- 所有 ablation 保持除被测因素外配置一致。

## 7. 最小可执行顺序

建议按以下顺序开工：

1. 完善 `compat` 与 checkpoint 检查。
2. 实现 SAN 与 Gated Critic 的 synthetic batch 单测。
3. 实现 BNN shifted VAS scorer。
4. 实现双 buffer。
5. 实现 Reference Networks 与 Reward-Parameter Correlation。
6. 接入训练循环和 checkpoint 拆分。
7. 跑单 Town smoke run。
8. 跑 Town01 -> Town02 sequential smoke run。
9. 跑六地图 reduced protocol。
10. 跑论文级 10 seeds 与消融。

## 8. 预期最终产物

- `src/sba_uas/`：SBA-UAS 方法实现。
- `configs/sba_uas/`：训练、评测、debug、消融配置。
- `scripts/`：兼容性检查、训练、评测、结果汇总脚本。
- `docs/`：方法说明、实验记录、结果表、复现偏差说明。
- `tests/`：轻量单测与集成测试。
- `experiments/`：每次实验的 config snapshot、日志索引和分析结果。
- `checkpoints/`：Roach-compatible policy checkpoint 与 SBA-UAS extra state checkpoint。
