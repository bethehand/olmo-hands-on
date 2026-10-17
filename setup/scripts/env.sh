# Usage: export OI_WORK=/path/on/a/big/disk; source env.sh
# Expects open-instruct cloned at $OI_WORK/open-instruct.
export OI_WORK=${OI_WORK:-$HOME/oi_work}
export PATH="$HOME/.local/bin:$PATH"           # uv
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}  # a CUDA 12 toolkit: kernels compiled at run time need an nvcc that knows sm_89
export PATH="$CUDA_HOME/bin:$PATH"
export UV_CACHE_DIR=$OI_WORK/uv-cache           # keep uv's download cache off the system disk
export HF_HUB_CACHE=$OI_WORK/hf-hub             # model cache (the login token stays in ~/.cache/huggingface)
export HF_DATASETS_CACHE=$OI_WORK/hf-datasets
export HF_HUB_DISABLE_XET=1                       # on a network where Xet transfers stall, fall back to plain HTTP
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export RAY_TMPDIR=/tmp/ray_$USER                 # don't take over /tmp/ray on a shared machine
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0            # start Ray workers with this Python, not `uv run` (see setup/README.md, snag 6)
export HF_HUB_DOWNLOAD_TIMEOUT=60                 # slow links: wait longer before a download read times out
cd $OI_WORK/open-instruct
