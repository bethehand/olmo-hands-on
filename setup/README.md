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
| 3 | `scripts/train/debug/grpo_fast.sh` as-is, one GPU | done: stops at startup with `Unknown device name` ([traceback](notes/traceback_desktop_4090.txt)), as expected from the code |
| 3b | Same script with the desktop 4090 added to `GPU_SPECS` | running |
| 4 | OLMo 2 1B RLVR recipe from `docs/olmo2.md`, scaled down to 4 GPUs and 10 steps | to do |

## Snags found so far

1. **The desktop RTX 4090 isn't in `GPU_SPECS`.** `grpo_fast.py` builds `ModelDims` at startup (line 1292 at `ff32794`). That looks up the GPU name in `open_instruct/utils.py:GPU_SPECS`, which only has `"4090 laptop"`. A desktop card reports `NVIDIA GeForce RTX 4090`, nothing matches, and the run stops with `ValueError: Unknown device name`. The numbers are only used for the MFU log line. Fix: add a `"4090"` entry *after* `"4090 laptop"` (the lookup takes the first key that matches, so the order matters). Confirmed on the machine: the [traceback](notes/traceback_desktop_4090.txt) goes `grpo_fast.py:1292` → `ModelDims.from_hf_config` → `get_device_name`. The numbers (165.2 TFLOPS dense BF16 with FP32 accumulate, 24 GB, 1008 GB/s) are from Table 2 of NVIDIA's Ada Lovelace architecture whitepaper.
2. **`push_to_hub` defaults to `True`.** Pass `--push_to_hub false` for local runs, or the trainer tries to upload the model at the end. This matters if you're logged in to Hugging Face.
3. **`scripts/train/debug/single_gpu_grpo.sh` needs Ai2's Beaker cluster.** Use `scripts/train/debug/grpo_fast.sh`, which calls the trainer directly.
4. **`--use_cpu_adam` isn't a quick fix for memory.** It also needs `--deepspeed_offload_optimizer true`, and DeepSpeed compiles an op the first time, which needs a CUDA 12 `nvcc`. This machine has 11.5, so it would fail here.
5. **On a shared machine, give Ray its own temp dir** (`RAY_TMPDIR=/tmp/ray_$USER`), so you don't create a `/tmp/ray` that other users can't write to.
6. **Ray workers can fail to start when the driver runs under `uv run`.** The debug script launches with `uv run --active`. Ray 2.53 then starts its workers with `uv run` too (`RAY_ENABLE_UV_RUN_RUNTIME_ENV`, on by default). With uv 0.12, this project's environment lives inside uv's cache (`.venv` is a link into `<cache>/environments-v2/`), so uv refuses: `The project directory ... is inside the cache directory`. The workers never come up, and the run just hangs with no GPU use. Fix: `export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, so the workers use the same Python directly.
7. **Slow downloads from Hugging Face here (0.3–1 MB/s).** The first run sat at 88% of a model download for 15+ minutes. Fetch models with `hf download <repo>` before a run (it resumes), and set `HF_HUB_DOWNLOAD_TIMEOUT=60`.

Good news: torch 2.10's CUDA 12.8 wheels work on driver 550 (CUDA 12.4). No driver upgrade needed so far.

## Scripts
- [`scripts/env.sh`](scripts/env.sh): paths and caches. Set `OI_WORK` to a big disk first.
- [`scripts/check_env.py`](scripts/check_env.py): torch, CUDA and a bf16 matmul on GPU 0; versions of vLLM, DeepSpeed and FlashAttention.
- [`scripts/run_debug.sh`](scripts/run_debug.sh): the debug script on one GPU, with a log and a GPU memory trace.
- [`scripts/run_olmo2_1b.sh`](scripts/run_olmo2_1b.sh): OLMo 2 1B RLVR, 4 GPUs (2 learners + 2 vLLM engines), 10 steps.
- [`scripts/gpumon.sh`](scripts/gpumon.sh): logs memory and utilization of every GPU every 5 seconds.
