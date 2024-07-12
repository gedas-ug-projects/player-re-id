#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tracklets_dir="/mnt/sun/levlevi/nba-plus-statvu-dataset/player-tracklets"
results_dir="/playpen-storage/levlevi/player-re-id/src/player_reid/hplr/data/results"
temp_dir="/playpen-storage/levlevi/player-re-id/src/player_reid/hplr/data/temp"
half="True"
compile_model="True"
model_variant="base"

for rank in {0..0}; do
    python generate_gallery.py \
        --tracklets_dir "$tracklets_dir" \
        --results_dir "$results_dir" \
        --temp_dir "$temp_dir" \
        --half "$half" \
        --compile_model "$compile_model" \
        --model_variant "$model_variant" \
        --device "$rank" \
        &
done
wait