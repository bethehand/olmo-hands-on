#!/bin/bash
# Usage: run_debug.sh <run name> [extra flags...]
# Runs open-instruct's own one-GPU debug script, scripts/train/debug/grpo_fast.sh (GPU 0 unless CUDA_VISIBLE_DEVICES is set).
D=$(cd "$(dirname "$0")" && pwd)    # this folder, as an absolute path (env.sh changes directory)
source "$D/env.sh" && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NAME=$1; shift; L=$OI_WORK/logs; mkdir -p $L
"$D/gpumon.sh" $L/$NAME.gpumem.csv & MON=$!
date +%s > $L/$NAME.start
bash scripts/train/debug/grpo_fast.sh --output_dir $OI_WORK/runs/$NAME "$@" > $L/$NAME.log 2>&1
echo "EXIT=$?" >> $L/$NAME.log
date +%s > $L/$NAME.end; kill $MON
