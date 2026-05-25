# 环境说明

CARLA-Roach 的原始环境是 Linux 优先环境，不建议在 Windows 原生环境中直接训练或评测。

原因如下：

- `carla-roach/doc/INSTALL.md` 使用 Ubuntu 18.04、Linux CARLA 包和 Linux Python egg。
- `carla-roach/utils/server_utils.py` 会调用 `killall -9 -r CarlaUE4-Linux`。
- `carla-roach/run/*.sh` 依赖 Bash、`CarlaUE4.sh` 和 Linux 路径。
- `carla-roach/environment.yml` 锁定了大量 `linux-64` conda 包，例如 `libgcc-ng`、`ld_impl_linux-64`、`ncurses` 等。

因此当前 Windows 机器只适合做代码编辑、文档维护和轻量静态检查。实际运行建议使用 Linux/WSL2/Ubuntu 或 Linux GPU 服务器。

## Linux 创建方式

```bash
conda env create -f carla-roach/environment.yml --name sba-uas-roach
conda activate sba-uas-roach
conda env update -n sba-uas-roach -f environment/sba_uas_extra_linux.yml
pip install -e .
```

随后安装与 CARLA 版本匹配的 Python egg。例如 CARLA 0.9.11：

```bash
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
```

若用于 Roach RL 训练，原安装说明建议 CARLA 0.9.10.1 更稳定，此时 egg 文件名通常为：

```bash
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```
