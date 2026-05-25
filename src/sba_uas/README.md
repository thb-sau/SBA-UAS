# SBA-UAS Python 包

该目录只放 SBA-UAS 复现新增代码，不修改 `carla-roach/`。

推荐模块边界：

- `compat/`：只读复用 CARLA-Roach 的 policy、配置和 checkpoint 合同。
- `critic/`：实现 SAN、Gated Critic、Critic 表征隔离。
- `stabilization/`：实现 BNN 环境模型、熟悉经验缓存、奖励-参数相关性和参数稳定化。
- `training/`：实现训练循环、双 buffer 更新、评测调度和 Roach 环境适配。
