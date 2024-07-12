#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
tracklets_dir='/playpen-storage/levlevi/player-re-id/src/data/_50_game_reid_benchmark_/labeled-tracks'
results_out_fp='/playpen-storage/levlevi/player-re-id/src/player_reid/hplr/results.json'
num_gpus=1
compile_model='True'
precision='fp32'
profile='True'
dummy_benchmarking='False'

python extract_jersey_numbers.py \
    --tracklets_dir "$tracklets_dir" \
    --results_out_fp "$results_out_fp" \
    --num_gpus "$num_gpus" \
    --compile_model "$compile_model" \
    --precision "$precision" \
    --profile "$profile" \
    &
wait