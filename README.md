# SBA-UAS Reproduction for CARLA-Roach

SBA-UAS 是面向自动驾驶持续强化学习的论文复现项目，对应 `paper/main.pdf` 中的 **Similarity-Based Activation and Uncertainty-Aware Stabilization** 方法。项目以 `carla-roach/` 为只读上游基线，在不修改 Roach Actor/policy 网络结构和 checkpoint 合同的前提下，实现 Critic 侧的持续学习扩展，并提供训练、评估、兼容性检查和轻量单测入口。

## 核心特性

- **Roach policy 完全兼容**：默认 Actor 使用 `carla-roach/agents/rl_birdview/models/ppo_policy.py` 中的 `PpoPolicy`，初始化参数与 Roach `xtma_beta.yaml` 保持一致。
- **Critic-side SBA-UAS**：新增 SAN、Gated Double Critic、Bayesian BEV Environment Model、shifted VAS、Familiar Experience Buffer、Reference Network 和 Reward-Parameter Correlation。
- **Roach PPO 可训练接入**：`SBAUASPPO` 继承 Roach PPO，Actor 仍按 PPO 更新；SBA-UAS sidecar 同步训练 Critic 侧模块，warm-up 后可用 gated Double-Q 估计参与 PPO advantage。
- **checkpoint 分离保存**：Roach policy checkpoint 只保存 `policy_state_dict`、`policy_init_kwargs` 和 `train_init_kwargs`；SBA-UAS extra state 单独保存为 `*_sba_uas_extra_state.pth`。
- **可运行脚本**：提供 smoke、标准联合采样、顺序持续学习训练入口，以及 Roach benchmark 评估入口。

## 仓库结构

```text
.
├── carla-roach/                  # 只读 Roach 上游基线，不要修改
├── configs/sba_uas/              # SBA-UAS 训练、评估和 policy 合同配置
├── docs/                         # 复现计划与当前状态
├── environment/                  # 环境安装补充说明
├── paper/main.pdf                # 目标论文
├── scripts/                      # 训练、评估、checkpoint 兼容性检查脚本
├── src/sba_uas/
│   ├── compat/                   # Roach policy 只读导入与兼容层
│   ├── critic/                   # SAN、Gated Critic 和 Critic losses
│   ├── stabilization/            # BNN、VAS、buffer、reference、importance
│   └── training/                 # Roach PPO sidecar、adapter、checkpoint、metrics
└── tests/                        # 不依赖 CARLA server 的轻量单测
```

## 复现边界

`carla-roach/` 是只读目录。任何 SBA-UAS 代码、配置、脚本和文档改动都应放在本项目自己的目录中：

- 新增代码：`src/sba_uas/`
- 新增配置：`configs/sba_uas/`
- 新增脚本：`scripts/`
- 新增说明：`docs/` 或根目录文档

默认实现只在 Critic/训练侧加入持续学习机制，不修改 Roach Actor/policy 网络结构。

## 环境要求

推荐在 Linux、WSL2 Ubuntu 或 Linux GPU 服务器中运行训练和评估。Windows 原生环境只建议用于代码编辑和轻量静态检查。

已验证轻量测试环境：

- conda 环境：`driveadapter`
- Python：3.7
- PyTorch：1.13.1

CARLA/Roach 运行依赖：

- Linux CARLA 包，包含 `CarlaUE4.sh`
- 与 CARLA 版本匹配的 Python egg
- Roach 需要的 BEV map h5 文件
- Bash、`killall`、Linux 路径和 GPU 运行环境

## 安装

如果你已经有 `driveadapter` 环境，直接激活并安装本项目：

```bash
conda activate driveadapter
cd /home/wsl/pythonWork/sba-uas
pip install -e .
```

如果需要从 Roach 环境重新创建：

```bash
conda env create -f carla-roach/environment.yml --name driveadapter
conda activate driveadapter
conda env update -n driveadapter -f environment/sba_uas_extra_linux.yml
pip install -e .
```

安装 CARLA Python egg。以 CARLA 0.9.11 为例：

```bash
export CARLA_ROOT=/path/to/carla-0.9.11
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
```

Roach RL 训练链路有时使用 CARLA 0.9.10.x 更稳定；论文指标复现以 CARLA 0.9.11 为目标。请在实验记录中固定并注明 CARLA 版本。

## 训练前检查

运行轻量单测：

```bash
conda run -n driveadapter pytest -q
```

检查 policy 配置是否仍与 Roach `xtma_beta.yaml` 一致：

```bash
conda run -n driveadapter python scripts/check_policy_checkpoint_compat.py
```

检查已有 policy checkpoint：

```bash
conda run -n driveadapter python scripts/check_policy_checkpoint_compat.py \
  carla-roach/checkpoints/roach_rl/ckpt/ckpt_11833344.pth
```

## 开始训练

所有训练脚本默认使用 `driveadapter`，并要求 `CARLA_ROOT` 指向 Linux CARLA 安装目录。

```bash
conda activate driveadapter
export CARLA_ROOT=/path/to/carla-0.9.11
export WANDB_MODE=offline
```

### 1. 单 Town smoke run

先运行最小配置，确认 CARLA server、Roach wrapper、SBA-UAS sidecar、checkpoint 保存都能跑通：

```bash
scripts/run_train_sba_uas.sh smoke
```

可用环境变量缩短或调整 smoke：

```bash
SBA_UAS_TOTAL_TIMESTEPS=8192 \
SBA_UAS_SEED=2021 \
scripts/run_train_sba_uas.sh smoke
```

### 2. 六地图标准联合采样训练

对应论文 Table I 的 standard RL agent-sampling protocol：

```bash
SBA_UAS_TOTAL_TIMESTEPS=100000000 \
SBA_UAS_SEED=2021 \
scripts/run_train_sba_uas.sh standard
```

该模式使用 Roach `endless_all`，覆盖 Town01 到 Town06。

### 3. 顺序持续学习训练

对应论文 Town01 -> Town06 continual-learning protocol。每个地图阶段结束后，脚本会从上一阶段的 Roach policy checkpoint 继续，并通过 `SBA_UAS_RESUME_EXTRA_STATE` 自动恢复配套的 SBA-UAS extra state。

```bash
SBA_UAS_STEPS_PER_MAP=2000000 \
SBA_UAS_SEED=2021 \
scripts/run_train_sba_uas.sh sequential
```

调试时可以使用 reduced protocol：

```bash
SBA_UAS_STEPS_PER_MAP=20000 \
SBA_UAS_RUN_ROOT=outputs/sba_uas_seq_debug \
scripts/run_train_sba_uas.sh sequential
```

### 4. 传递 Roach/Hydra override

训练脚本会把额外参数原样传给 Roach Hydra。例如：

```bash
scripts/run_train_sba_uas.sh smoke kill_running=false dummy=true
```

## checkpoint 说明

Roach policy checkpoint：

```text
ckpt_*.pth
```

只包含 Roach 可加载的 policy 合同：

- `policy_state_dict`
- `policy_init_kwargs`
- `train_init_kwargs`

SBA-UAS extra checkpoint：

```text
ckpt_*_sba_uas_extra_state.pth
```

包含：

- SAN / Gated Critic / Bayesian Environment Model
- Standard Buffer / Familiar Experience Buffer
- Reference Critic / Reference SAN
- Reward-Parameter Correlation / SAN SI-style importance
- optimizer state 和训练元数据

## 评估

评估使用 Roach benchmark，只加载 Roach-compatible policy checkpoint；SBA-UAS extra state 是训练侧状态，不参与推理。

```bash
conda activate driveadapter
export CARLA_ROOT=/path/to/carla-0.9.11
export WANDB_MODE=offline

scripts/run_benchmark_sba_uas.sh path/to/ckpt_*.pth
```

选择测试 suite：

```bash
SBA_UAS_TEST_SUITE=nocrash_dense \
scripts/run_benchmark_sba_uas.sh path/to/ckpt_*.pth
```

评估脚本会先运行 checkpoint 合同检查，再调用 `carla-roach/benchmark.py`。

## 论文级实验建议

建议按以下顺序推进：

1. `smoke`：Town01 小步数训练，确认真实 CARLA 采样、更新、保存、恢复。
2. benchmark smoke：用 smoke 产物跑一次 `nocrash_dense` 或小 suite。
3. reduced sequential：Town01 -> Town02，小步数，观察 `U_tilde_VAS`、buffer 迁移和 checkpoint 恢复。
4. standard joint：Town01-Town06 联合采样，至少 3 seeds 验证趋势。
5. full continual：Town01 -> Town06，每地图 2,000,000 steps，10 seeds。
6. 消融与诊断：`-SAN`、`-PS`、Actor-side、Actor-Critic、t-SNE、activation ratio、`D_pi`、`C_Q`。

当前代码已经实现默认 Critic-side SBA-UAS 训练闭环；Actor-side 和 Actor-Critic variants 仍需作为 Table IV 消融单独实现。

## 常见问题

### 找不到 `CarlaUE4.sh`

确认 `CARLA_ROOT` 指向 CARLA 根目录：

```bash
export CARLA_ROOT=/opt/carla-0.9.11
ls ${CARLA_ROOT}/CarlaUE4.sh
```

### 找不到 `carla` Python 包

安装对应版本 Python egg：

```bash
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
```

### BEV map h5 缺失

按 Roach 提示生成 birdview map 文件。通常命令形式为：

```bash
python -m carla_gym.utils.birdview_map \
  --save_dir carla-roach/carla_gym/core/obs_manager/birdview/maps \
  --pixels_per_meter 5.00 \
  --carla_sh_path ${CARLA_ROOT}/CarlaUE4.sh
```

### 想确认 checkpoint 能否评估

```bash
python scripts/check_policy_checkpoint_compat.py path/to/ckpt.pth
```

如果当前环境已安装完整 CARLA/Roach 依赖，也可以加上 Roach load smoke：

```bash
python scripts/check_policy_checkpoint_compat.py path/to/ckpt.pth --load-with-roach
```

## 当前状态

不依赖 CARLA server 的轻量测试已通过：

```bash
conda run -n driveadapter pytest -q
# 41 passed
```

真实论文数值仍需要在 CARLA 环境中运行：标准六地图联合采样、Town01 -> Town06 顺序流、10 seeds、消融实验和可视化统计。

## 参考文档

- 复现计划：`docs/reproduction_plan.md`
- 当前状态：`docs/reproduction_status.md`
- 训练模块说明：`src/sba_uas/training/README.md`
- 稳定化模块说明：`src/sba_uas/stabilization/README.md`
- policy 合同：`configs/sba_uas/policy_compatibility.yaml`
