#!/bin/bash
set -x

INPUT_FILE=./outputs/aime24/DeepSeek-R1-Distill-Qwen-1.5B_seed0_t0.6topp0.95minp0.0_len32768_num-1_n16H16m4.json

echo "pass@k in single dimension"

# n is also the vanilla sampling
for DIM in n H; do

python3 -u evaluations/passk_single_dim.py \
    --input_file  ${INPUT_FILE} \
    --dimension ${DIM} \
    --k_values "1,2,4,8,16"

done

for DIM in m; do

python3 -u evaluations/passk_single_dim.py \
    --input_file ${INPUT_FILE} \
    --dimension ${DIM} \
    --k_values "1,2,4"

done

echo "pass@k in multiple dimensions"

python3 -u evaluations/passk_multi_dim.py \
    --input_file ${INPUT_FILE} \
    --k_values "1,2,4,8,16" \
    --h_chunks 16 \
    --m_solutions 4 \