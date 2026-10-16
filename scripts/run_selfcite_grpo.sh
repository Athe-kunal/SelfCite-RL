#!/bin/bash
# SelfCite GRPO with miles: the one script that configures and runs everything.
#
# Trains Qwen3-4B (thinking on) with GRPO on LongCite-45k. The reward is
# SelfCite's context ablation (hold - drop), scored by the live policy, plus a
# format gate. See src/selfcite_rl/ for the reward code.
#
# Usage (from the host; it starts the pinned miles Docker image itself):
#   scripts/run_selfcite_grpo.sh [all|prepare|train]      # default: all
#
#   prepare  download the model and build data/selfcite_train.jsonl
#   train    launch GRPO (needs the outputs of prepare)
#   all      prepare (skipping what already exists), then train
#
# Every setting below can be overridden from the environment, e.g.
#   TRAIN_MODE=full LR=1e-6 scripts/run_selfcite_grpo.sh train
#
# Outputs: runs/<RUN_NAME>/{checkpoints,tensorboard,train.log}
#   tensorboard --logdir runs/

set -euo pipefail

STAGE="${1:-all}"

# =============================================================================
# 1. Configuration
# =============================================================================

# --- What to train -----------------------------------------------------------
TRAIN_MODE="${TRAIN_MODE:-lora}"            # full | lora
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"         # also the miles model-args name
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/${MODEL_NAME}}"
RUN_NAME="${RUN_NAME:-selfcite-${MODEL_NAME}-${TRAIN_MODE}-$(date +%Y%m%d-%H%M%S)}"

# --- Hardware ----------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-4}"
TP_SIZE="${TP_SIZE:-2}"                      # tensor parallel
CP_SIZE="${CP_SIZE:-2}"                      # context parallel (long prompts)
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"  # x CP_SIZE = per-sample cap
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.7}"

# --- Data and lengths (32,768 total = Qwen3-4B native context) ----------------
NUM_PREP_EXAMPLES="${NUM_PREP_EXAMPLES:-}"   # empty = all of LongCite-45k
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-24576}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-8192}"

# --- GRPO ----------------------------------------------------------------------
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"            # rollout + train iterations
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"  # prompts per rollout
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"  # GRPO group size
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"  # samples per optimizer step
TEMPERATURE="${TEMPERATURE:-1.0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

# --- Optimizer: LoRA usually wants a ~10x higher LR than full fine-tuning ---
if [[ "${TRAIN_MODE}" == "lora" ]]; then
  LR="${LR:-1e-5}"
else
  LR="${LR:-1e-6}"
fi
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# --- SelfCite reward (read by src/selfcite_rl/miles_reward.py) ----------------
SELFCITE_UNCITED_PENALTY="${SELFCITE_UNCITED_PENALTY:--0.5}"
SELFCITE_MAX_SPANS="${SELFCITE_MAX_SPANS:-3}"
SELFCITE_MAX_SPAN_CHUNKS="${SELFCITE_MAX_SPAN_CHUNKS:-10}"
SELFCITE_MAX_CONCURRENCY="${SELFCITE_MAX_CONCURRENCY:-64}"

# --- Environment ---------------------------------------------------------------
# Pinned by digest: patched SGLang + Megatron. miles itself comes from the
# third_party/miles submodule, installed over the image's copy at start-up.
MILES_IMAGE="${MILES_IMAGE:-radixark/miles@sha256:b836fec3693e7cbf5104d6d7b8530cd9c82e2a187eee29449cc95f41f6f69159}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

# Everything above, forwarded into the container.
CONFIG_VARS=(
  TRAIN_MODE MODEL_NAME HF_MODEL_ID RUN_NAME
  NUM_GPUS TP_SIZE CP_SIZE MAX_TOKENS_PER_GPU SGLANG_MEM_FRACTION
  NUM_PREP_EXAMPLES MAX_PROMPT_TOKENS MAX_RESPONSE_TOKENS
  NUM_ROLLOUT ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE
  TEMPERATURE SAVE_INTERVAL LR LORA_RANK LORA_ALPHA
  SELFCITE_UNCITED_PENALTY SELFCITE_MAX_SPANS SELFCITE_MAX_SPAN_CHUNKS
  SELFCITE_MAX_CONCURRENCY
)

# =============================================================================
# 2. Host side: re-run this script inside the miles container
# =============================================================================

if [[ -z "${IN_MILES_CONTAINER:-}" ]]; then
  REPO_HOST="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  if [[ ! -f "${REPO_HOST}/third_party/miles/train.py" ]]; then
    echo "third_party/miles is empty; run: git submodule update --init" >&2
    exit 1
  fi
  env_flags=(-e IN_MILES_CONTAINER=1 -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)")
  for var in "${CONFIG_VARS[@]}"; do
    env_flags+=(-e "${var}=${!var}")
  done
  # Interactive only with a terminal, so nohup / CI runs also work.
  [[ -t 0 ]] && env_flags+=(-it)
  mkdir -p "${HF_CACHE}"
  exec docker run --rm \
    --gpus all --ipc=host --shm-size=32g --network=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${REPO_HOST}:/workspace/SelfCite-RL" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -w /workspace/SelfCite-RL \
    "${env_flags[@]}" \
    "${MILES_IMAGE}" \
    bash scripts/run_selfcite_grpo.sh "${STAGE}"
fi

# =============================================================================
# 3. Container side
# =============================================================================

REPO=/workspace/SelfCite-RL
MILES_DIR="${REPO}/third_party/miles"
MODEL_PATH="${REPO}/models/${MODEL_NAME}"
DATA_PATH="${REPO}/data/selfcite_train.jsonl"
RUN_DIR="${REPO}/runs/${RUN_NAME}"
export PYTHONPATH="${REPO}/src:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

# The repo belongs to the host user; let git read it and give outputs back.
git config --global --add safe.directory '*'
return_ownership() {
  for dir in models data runs; do
    [[ -d "${REPO}/${dir}" ]] && chown -R "${HOST_UID}:${HOST_GID}" "${REPO}/${dir}"
  done
  return 0
}
trap return_ownership EXIT

print_summary() {
  cat <<EOF
==================== SelfCite GRPO ====================
 stage        ${STAGE}
 run          ${RUN_NAME}  ->  runs/${RUN_NAME}
 model        ${HF_MODEL_ID}  (${TRAIN_MODE}$(
                [[ "${TRAIN_MODE}" == lora ]] && echo ", rank ${LORA_RANK}, alpha ${LORA_ALPHA}"))
 data         data/selfcite_train.jsonl
 lengths      prompt ${MAX_PROMPT_TOKENS} + response ${MAX_RESPONSE_TOKENS}
 gpus         ${NUM_GPUS} (colocated)  TP ${TP_SIZE}  CP ${CP_SIZE}
 grpo         ${NUM_ROLLOUT} rollouts x ${ROLLOUT_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} samples, global batch ${GLOBAL_BATCH_SIZE}, lr ${LR}
 reward       uncited ${SELFCITE_UNCITED_PENALTY}, max spans ${SELFCITE_MAX_SPANS}, max span chunks ${SELFCITE_MAX_SPAN_CHUNKS}
 miles        $(git -C "${MILES_DIR}" rev-parse --short HEAD 2>/dev/null || echo "?") (submodule)
 tensorboard  tensorboard --logdir runs/
=======================================================
EOF
}

prepare() {
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo ">>> Downloading ${HF_MODEL_ID}"
    hf download "${HF_MODEL_ID}" --local-dir "${MODEL_PATH}"
  fi
  if [[ ! -f "${DATA_PATH}" ]]; then
    echo ">>> Building ${DATA_PATH}"
    mkdir -p "$(dirname "${DATA_PATH}")"
    python3 -m selfcite_rl.prepare_data \
      --output "${DATA_PATH}" \
      --tokenizer "${MODEL_PATH}" \
      --max_prompt_tokens "${MAX_PROMPT_TOKENS}" \
      ${NUM_PREP_EXAMPLES:+--num_examples "${NUM_PREP_EXAMPLES}"}
  fi
}

train() {
  mkdir -p "${RUN_DIR}"
  echo ">>> Installing miles from the submodule"
  pip install -q -e "${MILES_DIR}" --no-deps

  local model_args
  model_args="$(python3 "${MILES_DIR}/miles/utils/external_utils/model_args_utils.py" "${MODEL_NAME}")"
  read -ra MODEL_ARGS <<< "${model_args}"

  CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --megatron-to-hf-mode bridge     # load HF weights directly, no conversion
    --save "${RUN_DIR}/checkpoints"
    --save-interval "${SAVE_INTERVAL}"
  )

  MODE_ARGS=()
  if [[ "${TRAIN_MODE}" == "lora" ]]; then
    MODE_ARGS=(
      --lora-rank "${LORA_RANK}"
      --lora-alpha "${LORA_ALPHA}"
      --lora-dropout 0.0
      --target-modules all-linear
    )
  elif [[ "${TRAIN_MODE}" != "full" ]]; then
    echo "TRAIN_MODE must be 'full' or 'lora', got '${TRAIN_MODE}'" >&2
    exit 1
  fi

  ROLLOUT_ARGS=(
    --prompt-data "${DATA_PATH}"
    --input-key prompt
    --metadata-key metadata
    --apply-chat-template            # Qwen3 default: thinking on
    --rollout-shuffle
    --custom-rm-path selfcite_rl.miles_reward.reward
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-prompt-len "${MAX_PROMPT_TOKENS}"
    --rollout-max-response-len "${MAX_RESPONSE_TOKENS}"
    --rollout-temperature "${TEMPERATURE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
  )

  GRPO_ARGS=(
    --advantage-estimator grpo
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    # No KL term: bridge mode has no Megatron-format reference checkpoint.
  )

  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
  )

  PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size "${CP_SIZE}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  )

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
    --sglang-attention-backend flashinfer   # A100: no FlashAttention-3
  )

  LOG_ARGS=(
    --use-tensorboard
    --tb-project-name selfcite
    --tb-experiment-name "${RUN_NAME}"
    --custom-rollout-log-function-path selfcite_rl.miles_logging.log_rollout_data
  )

  MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
  )

  local runtime_env
  runtime_env="$(python3 - <<EOF
import json, os
env = {
    "PYTHONPATH": os.environ["PYTHONPATH"],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TENSORBOARD_DIR": "${RUN_DIR}/tensorboard",
}
env.update({k: v for k, v in os.environ.items() if k.startswith("SELFCITE_")})
print(json.dumps({"env_vars": env}))
EOF
)"

  ray stop --force >/dev/null 2>&1 || true
  ray start --head --num-gpus "${NUM_GPUS}" --disable-usage-stats >/dev/null

  echo ">>> Launching GRPO; log: runs/${RUN_NAME}/train.log"
  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${runtime_env}" \
    -- python3 "${MILES_DIR}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --num-gpus-per-node "${NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/train.log"
}

print_summary
case "${STAGE}" in
  prepare) prepare ;;
  train) train ;;
  all) prepare && train ;;
  *) echo "Unknown stage '${STAGE}'; use all, prepare or train." >&2; exit 1 ;;
esac
