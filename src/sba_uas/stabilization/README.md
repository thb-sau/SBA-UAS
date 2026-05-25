# 稳定化模块

这里用于实现论文中的 Uncertainty-Aware Stabilization：

- BNN Environment Model
- shifted VAS / transition familiarity 估计
- Familiar Experience Buffer
- Reward-Parameter Correlation
- Critic/SAN 参数稳定化正则

环境模型的论文级实现位于 `environment_model.py`：

- `BayesianBEVEnvironmentModel` 直接使用 Roach wrapper 输出的 `birdview` masks 和 `state` measurement。
- 默认 BEV 合同为 `15 x 192 x 192`，对应 Roach `chauffeurnet` 的 3 个静态通道和 4 帧历史中的车辆、行人、交通灯/stop 通道。
- 输入可为 Roach 原始 `uint8 [0, 255]` masks；模型内部归一化到 `[0, 1]`。
- 结构为 CNN encoder、CNN decoder 和 latent-space Bayesian transition MLP，默认 latent dim 为 256。
- `predict_next_state_samples()` 输出归一化 BEV prediction，`ShiftedVASScorer` 会通过模型的 `normalize_prediction_target()` 将 raw Roach next BEV 归一化后再计算 shifted VAS。

这些机制应作为训练侧或 Critic 侧状态保存，不应改变 Roach policy checkpoint 的结构。
