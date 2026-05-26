# Environment Notes

The original CARLA-Roach environment is Linux-first. Native Windows is not recommended for training or evaluation.

Reasons:

- `carla-roach/doc/INSTALL.md` uses Ubuntu 18.04, a Linux CARLA package, and a Linux Python egg.
- `carla-roach/utils/server_utils.py` calls `killall -9 -r CarlaUE4-Linux`.
- `carla-roach/run/*.sh` depends on Bash, `CarlaUE4.sh`, and Linux paths.
- `carla-roach/environment.yml` pins many `linux-64` conda packages, including `libgcc-ng`, `ld_impl_linux-64`, and `ncurses`.

Native Windows should be used only for editing, documentation, and lightweight static checks. Real CARLA runs should use Linux, WSL2 Ubuntu, or a Linux GPU server.

## Linux Setup

```bash
conda env create -f carla-roach/environment.yml --name sba-uas-roach
conda activate sba-uas-roach
conda env update -n sba-uas-roach -f environment/sba_uas_extra_linux.yml
pip install -e .
```

Install the CARLA Python egg that matches the CARLA version. Example for CARLA 0.9.11:

```bash
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
```

Roach RL training may be more stable with CARLA 0.9.10.1. In that case, the egg file name is usually:

```bash
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```
