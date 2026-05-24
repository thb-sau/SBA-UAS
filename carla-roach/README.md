# CARLA-Roach RL Only

This checkout is a trimmed version of [zhejz/carla-roach](https://github.com/zhejz/carla-roach) that keeps only the reinforcement-learning expert workflow:

- RL training with PPO.
- RL evaluation/benchmarking in CARLA.
- The CARLA gym environments and benchmark suites needed by the RL workflow.
- The best RL expert checkpoint listed by the original repository, `Roach`.

The imitation-learning, CILRS, BC/DAGGER, and dataset-collection code paths have been removed.

## Kept Entry Points

- `train_rl.py`: train an RL expert.
- `benchmark.py`: evaluate an RL expert.
- `run/train_rl.sh`: shell wrapper for RL training.
- `run/benchmark.sh`: shell wrapper for evaluating the downloaded Roach checkpoint.

## Downloaded RL Checkpoint

The original project lists `iccv21-roach/trained-models/1929isj0` as the Roach RL expert. Its checkpoint and agent config have been downloaded locally:

- `checkpoints/roach_rl/ckpt/ckpt_11833344.pth`
- `checkpoints/roach_rl/config_agent.yaml`

`config/benchmark.yaml` and `run/benchmark.sh` default to this local checkpoint through:

```yaml
agent:
  ppo:
    ckpt: checkpoints/roach_rl/ckpt/ckpt_11833344.pth
```

The code still supports W&B checkpoints if you set `agent.ppo.wb_run_path`, but local `agent.ppo.ckpt` is preferred when provided.

## Installation

Refer to [doc/INSTALL.md](doc/INSTALL.md) for the original CARLA and conda environment setup. For RL training, the authors recommend CARLA `0.9.10.1`.

After creating and activating the environment, install the CARLA Python egg that matches your simulator version.

## Train RL

Edit `run/train_rl.sh` if you need to change CARLA paths, W&B project names, or PPO policy options, then run:

```bash
bash run/train_rl.sh
```

The training wrapper starts from scratch by default:

```bash
agent.ppo.ckpt=null
agent.ppo.wb_run_path=null
```

## Evaluate RL

To evaluate the downloaded Roach checkpoint:

```bash
bash run/benchmark.sh
```

The default benchmark suite is `nocrash_dense`. Other available suites remain under `config/test_suites`.

## Citation

Please cite the original work if you use this code:

```bibtex
@inproceedings{zhang2021roach,
  title = {End-to-End Urban Driving by Imitating a Reinforcement Learning Coach},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  author = {Zhang, Zhejun and Liniger, Alexander and Dai, Dengxin and Yu, Fisher and Van Gool, Luc},
  year = {2021},
}
```

## License

This software keeps the original CC-BY-NC 4.0 license from CARLA-Roach. It is intended for personal and research use only.
