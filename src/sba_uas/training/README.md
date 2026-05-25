# 训练入口

这里用于实现 SBA-UAS 的训练和评测调度。

设计原则：

- 通过 `PYTHONPATH=src:carla-roach` 复用 Roach 的环境、wrapper、reward、terminal 和 policy。
- SBA-UAS 的额外状态单独保存，例如 SAN、Gated Critic、BNN 环境模型和熟悉经验缓存。
- 面向 Roach policy 的 checkpoint 文件必须继续可由 `PpoPolicy.load()` 加载。

当前训练侧实现：

- `roach_env_adapter.py`：接收 Roach `RlBirdviewWrapper` 输出的 `{"birdview": masks, "state": measurement}`，把 vectorized rollout step 转成带 `u_vas` 的 `Transition`。
- `trainer.py`：实现 Roach-compatible sidecar trainer。Actor 使用原 Roach policy 采样和 bootstrap action；SAN、Gated Critic、BEV BNN 环境模型、双 buffer、reference networks 和 importance tracker 都作为 SBA-UAS extra state 训练。
- `checkpointing.py`：拆分保存 `policy.pth` 和 `sba_uas_extra_state.pth`，不把 SBA-UAS extra state 写进 Roach policy checkpoint。

注意：该 trainer 已用 fake Roach-like vector env 单测验证训练闭环；真实 CARLA smoke run 仍需要 Linux CARLA server、Roach env config 和运行脚本接入。
