# 训练入口

这里用于实现 SBA-UAS 的训练和评测调度。

设计原则：

- 通过 `PYTHONPATH=src:carla-roach` 复用 Roach 的环境、wrapper、reward、terminal 和 policy。
- SBA-UAS 的额外状态单独保存，例如 SAN、Gated Critic、BNN 环境模型和熟悉经验缓存。
- 面向 Roach policy 的 checkpoint 文件必须继续可由 `PpoPolicy.load()` 加载。

当前训练侧实现：

- `roach_env_adapter.py`：接收 Roach `RlBirdviewWrapper` 输出的 `{"birdview": masks, "state": measurement}`，把 vectorized rollout step 转成带 `u_vas` 的 `Transition`。
- `trainer.py`：实现 Roach-compatible sidecar trainer。Actor 使用原 Roach policy 采样和 bootstrap action；SAN、Gated Critic、BEV BNN 环境模型、双 buffer、reference networks 和 importance tracker 都作为 SBA-UAS extra state 训练。
- `roach_ppo_sidecar.py`：提供 Roach Hydra 可加载的 `SBAUASPPO`。它继承上游 PPO，让 Roach Actor/policy 继续按 PPO 更新，同时把 rollout 镜像到 SBA-UAS sidecar；warm-up 后可用 gated Double-Q 估计重写 PPO advantage，使稳定化 Critic 参与 Actor 更新而不改变 Actor 网络结构。
- `checkpointing.py`：拆分保存 `policy.pth` 和 `sba_uas_extra_state.pth`，不把 SBA-UAS extra state 写进 Roach policy checkpoint。

运行入口：

```bash
conda activate driveadapter
export CARLA_ROOT=/path/to/carla-0.9.11
scripts/run_train_sba_uas.sh smoke
scripts/run_train_sba_uas.sh standard
scripts/run_train_sba_uas.sh sequential
```

`smoke` 使用小模型和 Town01 小步数；`standard` 对应六地图联合采样；`sequential` 依次运行 Town01 到 Town06，每个阶段从上一阶段 checkpoint 继续训练。

注意：轻量单测已验证 fake Roach-like vector env 下的 PPO+SBA-UAS 闭环；真实 CARLA smoke run 仍需要 Linux CARLA server、地图 h5、CARLA Python egg 和可用 GPU/显示环境。
