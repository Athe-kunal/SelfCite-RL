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
#   LoRA checkpoints hold only the adapter (HF PEFT format under
#   checkpoints/iter_*/adapter/); the base model is never re-saved.
#   tensorboard --logdir runs/

set -euo pipefail

STAGE="${1:-all}"

# =============================================================================
# 1. Configuration
# =============================================================================

# --- What to train -----------------------------------------------------------
TRAIN_MODE="${TRAIN_MODE:-lora}"            # full | lora
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
MILES_MODEL_TYPE="${MILES_MODEL_TYPE:-qwen3-4B}"  # third_party/miles/scripts/models/<this>.py
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/${MODEL_NAME}}"
RUN_NAME="${RUN_NAME:-selfcite-${MODEL_NAME}-${TRAIN_MODE}-$(date +%Y%m%d-%H%M%S)}"

# --- Hardware ----------------------------------------------------------------
# Default: colocated. Every GPU runs SGLang during rollout (one data-parallel
# engine per GPU, TP 1) and Megatron during training (data parallel =
# TRAIN_GPUS / (TP x CP)); miles offloads whichever side is idle. Set
# ROLLOUT_GPUS>0 to give SGLang its own GPUs instead.
GPU_IDS="${GPU_IDS:-2,3}"                    # host GPUs given to the container
TRAIN_GPUS="${TRAIN_GPUS:-2}"                # Megatron trainer
ROLLOUT_GPUS="${ROLLOUT_GPUS:-0}"            # SGLang; 0 = colocate with trainer
TP_SIZE="${TP_SIZE:-1}"                      # tensor parallel
CP_SIZE="${CP_SIZE:-1}"                      # context parallel (long prompts)
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"  # x CP_SIZE = per-sample cap
if (( ROLLOUT_GPUS == 0 )); then
  SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.7}"
else
  SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
fi

# --- Data and lengths (32,768 total = Qwen3-4B native context) ----------------
DATA_DIR="${DATA_DIR:-data}"
NUM_PREP_EXAMPLES="${NUM_PREP_EXAMPLES:-2300}"  # first N rows; empty = all 45k
NUM_EVAL="${NUM_EVAL:-100}"                  # of those, held out for eval
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-24576}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-8192}"

# --- GRPO ----------------------------------------------------------------------
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"            # rollout + train iterations
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"  # prompts per rollout
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"  # GRPO group size
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"  # samples per optimizer step
TEMPERATURE="${TEMPERATURE:-1.0}"
# Off-policy correction. The trainer always recomputes the old-policy
# log-probs (an extra forward pass), so the PPO ratio is against its own
# policy; TIS then reweights each token by min(pi_train / pi_sglang, TIS_CLIP)
# to correct the SGLang-vs-Megatron numeric mismatch. Bitwise
# --true-on-policy-mode needs FlashAttention 3 (Hopper), so not on A100.
USE_TIS="${USE_TIS:-1}"
TIS_CLIP="${TIS_CLIP:-2.0}"                  # upper bound on the IS weight
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"         # rollouts between adapter saves
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"         # rollouts between held-out evals
DUMP_ROLLOUTS="${DUMP_ROLLOUTS:-1}"          # 1 = keep every sample in runs/<RUN>/rollouts/

# --- Optimizer: LoRA usually wants a ~10x higher LR than full fine-tuning ---
if [[ "${TRAIN_MODE}" == "lora" ]]; then
  LR="${LR:-1e-5}"
else
  LR="${LR:-1e-6}"
fi
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"     # alpha / rank = 1

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
  TRAIN_MODE MODEL_NAME MILES_MODEL_TYPE HF_MODEL_ID RUN_NAME
  GPU_IDS TRAIN_GPUS ROLLOUT_GPUS TP_SIZE CP_SIZE MAX_TOKENS_PER_GPU
  SGLANG_MEM_FRACTION DATA_DIR NUM_PREP_EXAMPLES NUM_EVAL
  MAX_PROMPT_TOKENS MAX_RESPONSE_TOKENS
  NUM_ROLLOUT ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE
  TEMPERATURE USE_TIS TIS_CLIP SAVE_INTERVAL EVAL_INTERVAL DUMP_ROLLOUTS LR LORA_RANK LORA_ALPHA
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
    --gpus "\"device=${GPU_IDS}\"" --ipc=host --shm-size=32g --network=host \
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
DATA_PATH="${REPO}/${DATA_DIR}/selfcite_train.jsonl"
EVAL_DATA_PATH="${REPO}/${DATA_DIR}/selfcite_eval.jsonl"
NUM_GPUS=$((TRAIN_GPUS + ROLLOUT_GPUS))
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
 data         ${DATA_DIR}/selfcite_{train,eval}.jsonl  (${NUM_PREP_EXAMPLES:-all} rows, ${NUM_EVAL} eval)
 lengths      prompt ${MAX_PROMPT_TOKENS} + response ${MAX_RESPONSE_TOKENS}
 gpus         host ${GPU_IDS}: train ${TRAIN_GPUS}, rollout ${ROLLOUT_GPUS} (0 = colocated)  TP ${TP_SIZE}  CP ${CP_SIZE}
 save / eval  every ${SAVE_INTERVAL} / ${EVAL_INTERVAL} rollouts
 old logprobs trainer recompute$([[ "${USE_TIS}" == 1 ]] && echo " + TIS (clip ${TIS_CLIP})")
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
  if [[ ! -f "${DATA_PATH}" || ! -f "${EVAL_DATA_PATH}" ]]; then
    echo ">>> Building ${DATA_PATH} and ${EVAL_DATA_PATH}"
    mkdir -p "$(dirname "${DATA_PATH}")"
    python3 -m selfcite_rl.prepare_data \
      --output "${DATA_PATH}" \
      --eval_output "${EVAL_DATA_PATH}" \
      --num_eval "${NUM_EVAL}" \
      --tokenizer "${MODEL_PATH}" \
      --max_prompt_tokens "${MAX_PROMPT_TOKENS}" \
      ${NUM_PREP_EXAMPLES:+--num_examples "${NUM_PREP_EXAMPLES}"}
  fi
}

train() {
  mkdir -p "${RUN_DIR}"
  echo ">>> Installing miles from the submodule"
  pip install -q -e "${MILES_DIR}" --no-deps
  # Transformer Engine prefers FlashAttention 4 whenever it is installed, but
  # FA4's GQA kernels fail to compile on A100 and cuDNN fused attention rejects
  # these inputs; without FA4, TE falls back to FA2, which works. (Container
  # only; SGLang uses flashinfer and its own vendored kernels.)
  pip uninstall -y -q flash-attn-4 >/dev/null 2>&1 || true

  local model_args
  model_args="$(python3 "${MILES_DIR}/miles/utils/external_utils/model_args_utils.py" "${MILES_MODEL_TYPE}")"
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
      --no-save-optim                # checkpoints hold the adapter only
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

  # Held-out LongCite rows, scored with the same SelfCite reward, every
  # EVAL_INTERVAL rollouts (and once before training). Qwen3's recommended
  # thinking sampling.
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data selfcite "${EVAL_DATA_PATH}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${MAX_RESPONSE_TOKENS}"
    --eval-temperature 0.6
    --eval-top-p 0.95
  )

  GRPO_ARGS=(
    --advantage-estimator grpo
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    # No KL term: bridge mode has no Megatron-format reference checkpoint.
  )
  if [[ "${USE_TIS}" == 1 ]]; then
    GRPO_ARGS+=(--use-tis --tis-clip "${TIS_CLIP}")
  fi

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
    --log-probs-chunk-size 4096
  )

  if (( ROLLOUT_GPUS == 0 )); then
    PLACEMENT_ARGS=(--colocate)
    # Keep SGLang's base weights in CPU memory so only the LoRA adapter is
    # re-synced after each onload.
    if [[ "${TRAIN_MODE}" == lora ]]; then
      PLACEMENT_ARGS+=(--lora-base-cpu-backup)
    fi
  else
    PLACEMENT_ARGS=(--rollout-num-gpus "${ROLLOUT_GPUS}")
  fi

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1          # one engine per GPU = data parallel
    # Each engine has its own radix (prefix) cache; cache-aware routing sends
    # requests sharing a prompt (a GRPO group, its reward-scoring prefills) to
    # the engine that already holds it.
    --sglang-router-policy cache_aware
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
    --sglang-attention-backend flashinfer   # A100: no FlashAttention-3
  )

  LOG_ARGS=(
    --use-tensorboard
    --tb-project-name selfcite
    --tb-experiment-name "${RUN_NAME}"
    --custom-rollout-log-function-path selfcite_rl.miles_logging.log_rollout_data
    --custom-eval-rollout-log-function-path selfcite_rl.miles_logging.log_eval_rollout_data
  )
  if [[ "${DUMP_ROLLOUTS}" == 1 ]]; then
    LOG_ARGS+=(--save-debug-rollout-data "${RUN_DIR}/rollouts/{rollout_id}.pt")
  fi

  MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash        # FA2 (see the FA4 note in train())
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
    --actor-num-gpus-per-node "${TRAIN_GPUS}" \
    --num-gpus-per-node "${NUM_GPUS}" \
    "${PLACEMENT_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
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
