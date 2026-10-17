#!/bin/bash
# Usage: run_olmo2_1b.sh <run name> [extra flags...]
# Ai2's OLMo 2 1B RLVR recipe (docs/olmo2.md, "modern reproduction"), scaled down to 4× RTX 4090 and 10 steps, just to check it runs.
# Changes from the recipe: 2 learners + 2 vLLM engines; 8 prompts x 8 samples per step; 1024-token prompts and responses; 10 steps; no W&B; no upload.
D=$(cd "$(dirname "$0")" && pwd)    # this folder, as an absolute path (env.sh changes directory)
source "$D/env.sh" && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NAME=$1; shift; L=$OI_WORK/logs; mkdir -p $L
"$D/gpumon.sh" $L/$NAME.gpumem.csv & MON=$!
date +%s > $L/$NAME.start
python open_instruct/grpo_fast.py \
    --model_name_or_path allenai/OLMo-2-0425-1B-DPO \
    --tokenizer_name_or_path allenai/OLMo-2-0425-1B-DPO \
    --add_bos \
    --chat_template_name tulu \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --apply_verifiable_reward true \
    --remap_verifier ifeval=ifeval_old \
    --stop_strings "</answer>" \
    --non_stop_penalty \
    --non_stop_penalty_value 0.0 \
    --beta 0.01 \
    --kl_estimator 3 \
    --learning_rate 5e-7 \
    --lr_scheduler_type constant \
    --temperature 1.0 \
    --max_prompt_token_length 1024 \
    --response_length 1024 \
    --pack_length 2048 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 2 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 8 \
    --total_episodes 640 \
    --num_epochs 1 \
    --deepspeed_stage 2 \
    --gradient_checkpointing \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --local_eval_every 5 \
    --seed 1 \
    --push_to_hub false \
    --output_dir $OI_WORK/runs/$NAME \
    "$@" > $L/$NAME.log 2>&1
echo "EXIT=$?" >> $L/$NAME.log
date +%s > $L/$NAME.end; kill $MON
