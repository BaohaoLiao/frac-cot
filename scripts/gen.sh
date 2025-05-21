#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

MODEL_NAME_OR_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
OUTPUT_DIR=./outputs
NUM_TEST_SAMPLE=-1
DATA_NAME="aime24"

python3 -u main.py \
    --data_dir "./benchmarks" \
    --data_name ${DATA_NAME} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --max_tokens_per_call 32768 \
    --max_model_len 40000 \
    --max_num_seqs 64 \
    --top_p 0.95 \
    --min_p 0.0 \
    --temperature 0.6 \
    --n_sampling 16 \
    --num_think_chunks 16 \
    --num_solutions_per_chunk 4 \
    --max_tokens_per_solution 2048 \
    --save_think_solutions \
    --output_dir ${OUTPUT_DIR} \
    --seed 0