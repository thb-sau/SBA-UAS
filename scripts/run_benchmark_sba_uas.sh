#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/carla-roach:${PYTHONPATH:-}"

cd "${ROOT_DIR}"
echo "SBA-UAS benchmark scaffold is ready. Implement src/sba_uas/training before launching CARLA runs."
echo "Benchmark scaffold config: configs/sba_uas/benchmark_sba_uas.yaml"
