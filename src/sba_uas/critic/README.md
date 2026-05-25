# Critic 侧模块

这里用于实现论文中的 Critic 侧持续学习机制：

- Similarity-Based Activation Network (SAN)
- Gated Q-Network / Gated Critic
- Critic 表征隔离与共享路径复用

注意：Actor/policy 结构不得放在这里改写。policy 必须继续复用 CARLA-Roach 的 `PpoPolicy` checkpoint 合同。
