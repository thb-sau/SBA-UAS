#!/bin/bash

benchmark () {
  python -u benchmark.py resume=true log_video=true \
  wb_project=carla-roach-rl-benchmark \
  agent.ppo.ckpt=checkpoints/roach_rl/ckpt/ckpt_11833344.pth \
  agent.ppo.wb_run_path=null \
  'wb_group="Roach-RL"' \
  'wb_notes="Benchmark the Roach RL expert on NoCrash-dense."' \
  test_suites=nocrash_dense \
  seed=2021 \
  +wb_sub_group=nocrash_dense-2021 \
  no_rendering=true \
  carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
}


# NO NEED TO MODIFY THE FOLLOWING
# activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carla

# remove checkpoint files
rm -f outputs/checkpoint.txt
rm -f outputs/wb_run_id.txt
rm -f outputs/ep_stat_buffer_*.json

# resume benchmark in case carla is crashed.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  benchmark
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

killall -9 -r CarlaUE4-Linux
echo "Bash script done."

# To shut down the aws instance after the script is finished
# sleep 10
# sudo shutdown -h now
