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
| 3b | Same script with the desktop 4090 added to `GPU_SPECS` | got past startup (peak 21 GB on GPU 0), then failed in the first training step: snag 8 |
| 3c | Same, with `CUDA_HOME` pointing at a CUDA 12 toolkit | compiled fine, then out of memory on one 24 GB card, even with `--vllm_gpu_memory_utilization 0.25 --gradient_checkpointing` (trainer 15.6 GB + vLLM 6.8 GB, wanted 2.8 GB more): snag 9 |
| 3d | Same on two GPUs: `--single_gpu_mode false`, trainer on one card, vLLM on the other | **done: 4 steps in 204 s, `Saving final model at step 4`, exit 0.** Peak 20.8 GB (trainer) and 11.7 GB (vLLM). Mean reward per step 4.4, 5.0, 2.5, 5.6 (10 per correct answer), as noisy as you'd expect from 4 steps |
| 4 | OLMo 2 1B RLVR recipe ([`run_olmo2_1b.sh`](scripts/run_olmo2_1b.sh)), 4 GPUs (2 learners + 2 vLLM engines), 10 steps | **done on the first try: 447 s in all, `Saving final model at step 10`, exit 0.** vLLM start-up 1 min 43 s; training part 13–38 s per step (mean 21 s) at 64 answers per step. Learner GPUs peaked at 24.2 and 23.6 GB, so there's no headroom for bigger batches without a third learner. Answers 85–347 tokens; correct rate around 0.5 and noisy, as expected |

## Snags found so far

1. **The desktop RTX 4090 isn't in `GPU_SPECS`.** `grpo_fast.py` builds `ModelDims` at startup (line 1292 at `ff32794`). That looks up the GPU name in `open_instruct/utils.py:GPU_SPECS`, which only has `"4090 laptop"`. A desktop card reports `NVIDIA GeForce RTX 4090`, nothing matches, and the run stops with `ValueError: Unknown device name`. The numbers are only used for the MFU log line. Fix: add a `"4090"` entry *after* `"4090 laptop"` (the lookup takes the first key that matches, so the order matters). Sent upstream as [open-instruct #2018](https://github.com/allenai/open-instruct/pull/2018). Confirmed on the machine: the [traceback](notes/traceback_desktop_4090.txt) goes `grpo_fast.py:1292` → `ModelDims.from_hf_config` → `get_device_name`. The numbers (165.2 TFLOPS dense BF16 with FP32 accumulate, 24 GB, 1008 GB/s) are from Table 2 of NVIDIA's Ada Lovelace architecture whitepaper.
2. **`push_to_hub` defaults to `True`.** Pass `--push_to_hub false` for local runs, or the trainer tries to upload the model at the end. This matters if you're logged in to Hugging Face.
3. **`scripts/train/debug/single_gpu_grpo.sh` needs Ai2's Beaker cluster.** Use `scripts/train/debug/grpo_fast.sh`, which calls the trainer directly.
4. **`--use_cpu_adam` isn't a quick fix for memory.** It also needs `--deepspeed_offload_optimizer true`, and DeepSpeed compiles an op the first time, which needs a CUDA 12 `nvcc`. The `nvcc` on `PATH` here is 11.5; point `CUDA_HOME` at a CUDA 12 toolkit first (snag 8).
5. **On a shared machine, give Ray its own temp dir** (`RAY_TMPDIR=/tmp/ray_$USER`), so you don't create a `/tmp/ray` that other users can't write to.
6. **Ray workers can fail to start when the driver runs under `uv run`.** The debug script launches with `uv run --active`. Ray 2.53 then starts its workers with `uv run` too (`RAY_ENABLE_UV_RUN_RUNTIME_ENV`, on by default). With uv 0.12, this project's environment lives inside uv's cache (`.venv` is a link into `<cache>/environments-v2/`), so uv refuses: `The project directory ... is inside the cache directory`. The workers never come up, and the run just hangs with no GPU use. Fix: `export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, so the workers use the same Python directly.
7. **Slow downloads from Hugging Face here (0.3–1.5 MB/s, while the same machine pulls PyPI's CDN at 47 MB/s).** It's the route to Hugging Face's CDN, not the bandwidth. The first run sat at 88% of a model download for 15+ minutes. Fetch models with `hf download <repo>` before a run (it resumes), and set `HF_HUB_DOWNLOAD_TIMEOUT=60`. I relayed the big checkpoints through another machine. Also, newer repos (Olmo 3, Qwen3.5) are stored on Hugging Face's Xet backend, and from this network the `hf_xet` transfer path stalled at 0 bytes, while plain HTTP still gave about 1 MB/s. `HF_HUB_DISABLE_XET=1` makes `huggingface_hub` use plain HTTP. Older LFS-only repos such as OLMo 2 weren't affected, which is why only some downloads hung.
8. **Qwen3.5's linear-attention backward compiles a kernel at run time, and an old `nvcc` can't target the 4090.** flash-linear-attention dispatches `chunk_bwd_dqkwg` to TileLang, which calls whatever `nvcc` it finds. Here that was Ubuntu's `/usr/bin/nvcc` (CUDA 11.5): `nvcc fatal : Value 'sm_89' is not defined for option 'gpu-architecture'` (sm_89 needs CUDA 11.8+). The machine also had CUDA 12.4 in `/usr/local/cuda-12.4`, just not on `PATH`. Fix: `export CUDA_HOME=/usr/local/cuda-12.4` and put `$CUDA_HOME/bin` first on `PATH`. Models with plain attention (OLMo 2, Olmo 3) don't go through this.
9. **The one-GPU debug script doesn't fit on a 24 GB card with Qwen3.5-0.8B.** Lowering vLLM's share to 25% and turning on gradient checkpointing still left it 2.8 GB short. Running it on two cards (`--single_gpu_mode false`) works.

Good news: torch 2.10's CUDA 12.8 wheels work on driver 550 (CUDA 12.4). No driver upgrade needed so far.

## Snags when running Olmo 3 checkpoints and Ai2's eval tools (2026-10-18)

10. **Olmo 3 configs vs transformers 5.x.** Olmo 3 checkpoints ship a transformers-4.57-style YaRN config: `rope_scaling` with integer `beta_fast`/`beta_slow`, and `rope_theta` outside it. transformers 5.x rejects it (`KeyError: 'rope_theta'`, then "beta_fast must be a float"). For your own scripts, load a folder that links the weights and holds a converted config ([`make_model_views.py`](https://github.com/bethehand/olmo-rlvr-side-effects/blob/main/experiments/make_model_views.py)). For Ai2's eval tools, pin transformers 4.57.1 (snag 16).
11. **Olmo-3-7B-RL-Zero-General's `model_type` is `olmo2-retrofit`**, the pre-release name of the Olmo 3 architecture. Every other field matches; map it to `olmo3` / `Olmo3ForCausalLM`.
12. **vLLM's offline `LLM` (V1 engine) needs an `if __name__ == "__main__":` guard.** It spawns worker processes that re-import your script.
13. **`hf download --exclude` takes one pattern per flag** (hub 1.x CLI). `--exclude "*.bin" "*.pt"` treats the extra patterns as file names: it downloads 0 files and exits 0, so a retry loop reports success. Repeat the flag: `--exclude "*.bin" --exclude "*.pt"`. Some repos (Olmo-3-7B-Think-DPO, allenai/wildguard) carry duplicate `.bin` weights next to the safetensors.
14. **One connection to Hugging Face got about 3 MB/s here; many at once reached 60 MB/s.** `hf download` fetches up to 8 files of a repo in parallel, so each repo's last big file crawls. Start the next repos instead of queueing them.
15. **vLLM's default `gpu_memory_utilization=0.9` runs out of memory on 24 GB cards with Olmo 3 7B**, during start-up ("CUDA out of memory occurred when warming up sampler with 256 dummy requests"). 0.85 works. With tensor parallel 2, that leaves a 49,920-token KV cache. A Think model that loops can fill its whole 65,536-token window at about 44 tokens/s for a single sequence.
16. **allenai/safety-eval outside OLMES** (OLMES installs it with `uv pip install -e` and runs it with API keys set):
    - put the repo on `PYTHONPATH`, or `python evaluation/run_all_generation_benchmarks.py` can't import `evaluation`;
    - set a placeholder `OPENAI_API_KEY`, because `src/dispatch_openai_requests.py` builds an OpenAI client at import time, even for tasks that only use local judges;
    - its requirements are open-ended (`transformers>=4.57.0`, `datasets>=2.19.0`). Today that resolves to transformers 5.x (snag 10) and to a `datasets` that needs huggingface_hub 2.x. Pin transformers 4.57.1, huggingface_hub 0.36.2 and datasets 4.4.1 (the releases current when Ai2 evaluated Olmo 3);
    - its vLLM calls take no memory settings, so on 24 GB cards you need a small patch for snag 15 ([ours](https://github.com/bethehand/olmo-rlvr-side-effects/blob/main/experiments/2026-10-18_think7b-safety/safety-eval-memory.patch) reads it from an environment variable; unset means upstream behaviour).
    - StrongReject's judge merges a LoRA onto google/gemma-2b, which is gated (manual approval on Hugging Face).

## Scripts
- [`scripts/env.sh`](scripts/env.sh): paths and caches. Set `OI_WORK` to a big disk first.
- [`scripts/check_env.py`](scripts/check_env.py): torch, CUDA and a bf16 matmul on GPU 0; versions of vLLM, DeepSpeed and FlashAttention.
- [`scripts/run_debug.sh`](scripts/run_debug.sh): the debug script on one GPU, with a log and a GPU memory trace.
- [`scripts/run_olmo2_1b.sh`](scripts/run_olmo2_1b.sh): OLMo 2 1B RLVR, 4 GPUs (2 learners + 2 vLLM engines), 10 steps.
- [`scripts/gpumon.sh`](scripts/gpumon.sh): logs memory and utilization of every GPU every 5 seconds.
