# Running open-instruct GRPO on 4× RTX 4090, outside Ai2

Goal: run Ai2's own RL code ([open-instruct](https://github.com/allenai/open-instruct) `grpo_fast.py`) on consumer GPUs. First the built-in debug script, then the OLMo 2 1B RLVR recipe for 10 steps. Not to train a good model, just to make sure it runs and to write down what breaks.

## Machine
- Ubuntu 22.04, 4× RTX 4090 (24 GB each), 128 CPU threads, 1 TB RAM.
- NVIDIA driver 550.144.03 (CUDA 12.4). No peer-to-peer between the cards (`nvidia-smi topo -p2p r` shows `CNS`), so NCCL goes through host memory.
- open-instruct at commit `ff32794` (2026-10-17).

## Steps

| Step | What | Status |
|---|---|---|
| 1 | `git clone https://github.com/allenai/open-instruct.git` | done |
| 2 | `uv sync` (Python 3.12, torch 2.10, vLLM 0.19) | done: 27 min, 14 GB |
| 2b | [`scripts/check_env.py`](scripts/check_env.py): torch sees the GPU, bf16 matmul runs | done |
| 3 | `scripts/train/debug/grpo_fast.sh` as-is, one GPU | running |
| 4 | OLMo 2 1B RLVR recipe from `docs/olmo2.md`, scaled down to 4 GPUs and 10 steps | to do |

## Snags found so far

1. **The desktop RTX 4090 isn't in `GPU_SPECS`.** `grpo_fast.py` builds `ModelDims` at startup (line 1292 at `ff32794`). That looks up the GPU name in `open_instruct/utils.py:GPU_SPECS`, which only has `"4090 laptop"`. A desktop card reports `NVIDIA GeForce RTX 4090`, nothing matches, and the run stops with `ValueError: Unknown device name`. The numbers are only used for the MFU log line. Fix: add a `"4090"` entry *after* `"4090 laptop"` (the lookup takes the first key that matches, so the order matters). Found in the code; confirming on the machine next.
2. **`push_to_hub` defaults to `True`.** Pass `--push_to_hub false` for local runs, or the trainer tries to upload the model at the end. This matters if you're logged in to Hugging Face.
3. **`scripts/train/debug/single_gpu_grpo.sh` needs Ai2's Beaker cluster.** Use `scripts/train/debug/grpo_fast.sh`, which calls the trainer directly.
4. **`--use_cpu_adam` isn't a quick fix for memory.** It also needs `--deepspeed_offload_optimizer true`, and DeepSpeed compiles an op the first time, which needs a CUDA 12 `nvcc`. This machine has 11.5, so it would fail here.
5. **On a shared machine, give Ray its own temp dir** (`RAY_TMPDIR=/tmp/ray_$USER`), so you don't create a `/tmp/ray` that other users can't write to.

Good news: torch 2.10's CUDA 12.8 wheels work on driver 550 (CUDA 12.4). No driver upgrade needed so far.

## Scripts
- [`scripts/env.sh`](scripts/env.sh): paths and caches. Set `OI_WORK` to a big disk first.
- [`scripts/check_env.py`](scripts/check_env.py): torch, CUDA and a bf16 matmul on GPU 0; versions of vLLM, DeepSpeed and FlashAttention.
- [`scripts/run_debug.sh`](scripts/run_debug.sh): the debug script on one GPU, with a log and a GPU memory trace.
- [`scripts/run_olmo2_1b.sh`](scripts/run_olmo2_1b.sh): OLMo 2 1B RLVR, 4 GPUs (2 learners + 2 vLLM engines), 10 steps.
- [`scripts/gpumon.sh`](scripts/gpumon.sh): logs memory and utilization of every GPU every 5 seconds.
