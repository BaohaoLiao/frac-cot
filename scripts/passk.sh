#!/bin/bash
set -x

echo "pass@k in single dimension"

# n is also the vanilla sampling
for DIM in n H; do

python3 -u passk_single_dim.py \
    --input_file  \
    --dimension ${DIM} \
    --k_values "1,2,4,8,16"

done

for DIM in m; do

python3 -u passk_single_dim.py \
    --input_file  \
    --dimension ${DIM} \
    --k_values "1,2,4"

done

echo "pass@k in multiple dimensions"

python3 -u passk_multi_dim.py \
    --input_file  \
    --k_values "1,2,4,8,16" \
    --h_chunks 16 \
    --m_solutions 4 \