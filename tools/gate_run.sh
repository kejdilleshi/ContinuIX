#!/usr/bin/env bash
# Gate/dev run of one igm_run with a timestamped log (epoch seconds prefixed to every line).
#   tools/gate_run.sh GPU OUT_DIR EXPERIMENT [hydra overrides...]
# OUT_DIR is relative to ContinuIX/. Intended to be launched with: setsid nohup ... &
set -uo pipefail
GPU=$1; OUT=$2; EXP=$3; shift 3
cd /home/klleshi/Documents/ContinuIX
unset DISPLAY; export MPLBACKEND=Agg   # never X (hangs in XOpenDisplay when the ssh forward stalls)
export PATH=/home/klleshi/miniconda3/envs/igm-pretrain/bin:$PATH
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
echo "$(date +%s) START gpu$GPU $EXP $*" > "$OUT.log"
igm_run +experiment="$EXP" "$@" hydra.run.dir="$OUT" 2>&1 | awk '{ print systime(), $0; fflush() }' >> "$OUT.log"
echo "$(date +%s) END rc=${PIPESTATUS[0]}" >> "$OUT.log"
