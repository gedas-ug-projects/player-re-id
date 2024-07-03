#!/bin/bash

tracklets_dir='/mnt/opr/levlevi/player-re-id/src/data/_50_game_reid_benchmark_/labeled-tracks'
results_out_fp='/mnt/opr/levlevi/player-re-id/src/player_reid/hplr/results.json'
num_gpus=1
compile_model=True
nohup python3 extract_jersey_numbers.py --tracklets_dir "$tracklets_dir" --results_out_fp "$results_out_fp" --num_gpus "$num_gpus" --compile_model "$compile_model" &
wait