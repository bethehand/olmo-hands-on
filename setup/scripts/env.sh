# Usage: export OI_WORK=/path/on/a/big/disk; source env.sh
# Expects open-instruct cloned at $OI_WORK/open-instruct.
export OI_WORK=${OI_WORK:-$HOME/oi_work}
export PATH="$HOME/.local/bin:$PATH"           # uv
export UV_CACHE_DIR=$OI_WORK/uv-cache           # keep uv's download cache off the system disk
export HF_HUB_CACHE=$OI_WORK/hf-hub             # model cache (the login token stays in ~/.cache/huggingface)
export HF_DATASETS_CACHE=$OI_WORK/hf-datasets
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export RAY_TMPDIR=/tmp/ray_$USER                 # don't take over /tmp/ray on a shared machine
cd $OI_WORK/open-instruct
