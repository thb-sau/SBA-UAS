# SBA-UAS 复现项目目录结构

本文档说明当前仓库的复现骨架。根目录下的 `carla-roach/` 是只读基线，所有 SBA-UAS 复现代码都应放在新建目录中，避免污染上游实现。

## 顶层目录

| 路径 | 作用 |
| --- | --- |
| `AGENTS.md` | 仓库规则：明确 `carla-roach/` 不可修改，并约束 SBA-UAS policy 与 CARLA-Roach policy 的 checkpoint 兼容性。 |
| `paper/main.pdf` | 待复现论文：Similarity-Based Activation and Uncertainty-Aware Stabilization (SBA-UAS)。 |
| `carla-roach/` | CARLA-Roach 只读基线目录，保留原 PPO/Roach 训练、评测、环境和 policy 实现。 |
| `src/sba_uas/` | SBA-UAS 新增 Python 包。这里只放复现方法本身，不修改 Roach 源码。 |
| `configs/sba_uas/` | SBA-UAS 新增 Hydra/实验配置，包含 policy 兼容性声明、训练配置和评测配置。 |
| `scripts/` | 新增运行脚本和检查脚本，用于启动 SBA-UAS 训练/评测或检查 checkpoint 合同。 |
| `environment/` | 环境说明与 Linux 追加依赖文件。由于 Roach 依赖 Linux CARLA，Windows 原生暂不创建 conda 环境。 |
| `docs/` | 中文复现说明、方法拆解、实验计划和后续开发记录。 |
| `checkpoints/` | 统一存放大模型 checkpoint。默认只保留目录占位文件，真实权重不提交。 |
| `experiments/` | 实验配置快照、运行记录和分析产物的占位目录。 |
| `outputs/` | 本地训练、评测输出目录。运行产物默认不提交。 |
| `tests/` | 后续放置单元测试与兼容性测试。 |

## SBA-UAS 包内部结构

| 路径 | 作用 |
| --- | --- |
| `src/sba_uas/compat/` | 与 CARLA-Roach 的兼容层，负责只读导入 Roach policy，并维护共享 checkpoint 合同。 |
| `src/sba_uas/critic/` | Critic 侧结构预留：SAN、Gated Critic、Critic 表征隔离等应放在这里。 |
| `src/sba_uas/stabilization/` | Uncertainty-Aware Stabilization 相关模块预留：BNN 环境模型、熟悉经验缓存、奖励-参数相关性和参数正则。 |
| `src/sba_uas/training/` | SBA-UAS 训练循环、Roach 环境适配、双 buffer 更新和评测调度入口。 |

## Policy 与 checkpoint 兼容规则

SBA-UAS 的 policy 不新建结构，也不复制后改写 Roach 的 policy。默认使用：

```text
agents.rl_birdview.models.ppo_policy:PpoPolicy
```

并保持以下结构与 Roach 一致：

- `policy_head_arch: [256, 256]`
- `value_head_arch: [256, 256]`
- `features_extractor_entry_point: agents.rl_birdview.models.torch_layers:XtMaCNN`
- `features_extractor_kwargs.states_neurons: [256, 256]`
- `distribution_entry_point: agents.rl_birdview.models.distributions:BetaDistribution`
- `distribution_kwargs.dist_init: null`

因此，SBA-UAS 的新增机制应以 Critic/训练侧扩展的形式保存到独立 checkpoint 或独立字段中；面向 Roach policy 的 checkpoint 文件必须继续满足 Roach `PpoPolicy.load()` 所需的 `policy_init_kwargs` 与 `policy_state_dict` 格式。

## 环境结论

当前 Windows 机器上没有可用的 `conda` 命令，而且 CARLA-Roach 代码依赖 `CarlaUE4.sh`、`killall -r CarlaUE4-Linux` 和 Linux 版 CARLA Python egg。因此本次没有在 Windows 原生环境创建 conda 环境。

在 Linux/WSL2/Ubuntu 或服务器上，建议先用 `carla-roach/environment.yml` 创建基础环境，再安装本项目包和追加依赖：

```bash
conda env create -f carla-roach/environment.yml --name sba-uas-roach
conda activate sba-uas-roach
conda env update -n sba-uas-roach -f environment/sba_uas_extra_linux.yml
pip install -e .
```
