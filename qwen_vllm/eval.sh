#!/usr/bin/env bash
set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn

TASK=${1:-minerva_math500}
DATASET=${2:-math500}
MODEL_PATH=${3:-Qwen/Qwen3-8B}
OUT=${4:-1.log}

M_CANDIDATES=${M_CANDIDATES:-15}
K_MONTE_CARLO=${K_MONTE_CARLO:-3}
BLOCK_SIZE=${BLOCK_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-512}
TEMPERATURE=${TEMPERATURE:-0.7}
SEED=${SEED:-0}

GEN_DP=${GEN_DP:-8}
UTILIZATION=${UTILIZATION:-0.8}

python eval_qwen.py \
  --tasks ${TASK} \
  --model qwen-ets \
  --output_path ${OUT} \
  --log_samples \
  --model_args dataset=${DATASET},model_path=${MODEL_PATH},m_candidates=${M_CANDIDATES},k_monte_carlo=${K_MONTE_CARLO},block_size=${BLOCK_SIZE},max_length=${MAX_LENGTH},temperature=${TEMPERATURE},data_parallel_size=${GEN_DP},tensor_parallel_size=1,gpu_memory_utilization=${UTILIZATION},vllm_seed=${SEED}
