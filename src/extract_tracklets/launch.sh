#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1
cd MixSort/

## NBA 15'-16' ##
tracklets_out_dir="/mnt/sun/levlevi/player-tracklets-backup/player-tracklets"
videos_src_dir="/mnt/sun/levlevi/nba-plus-statvu-dataset/game-replays"

## TEST ##
videos_src_dir="/playpen-storage/levlevi/player-re-id/__old__/clips"
tracklets_out_dir="/playpen-storage/levlevi/player-re-id/src/extract_tracklets/testing_tracks_out"
tracklets_temp_data_dir="/mnt/meg/levlevi/tmp"
dataloader_batch_size=8
dataloader_workers=2
torch_compile="False"

for rank in {0..0}; do
    python pipeline.py \
        --tracklets_out_dir "$tracklets_out_dir" \
        --videos_src_dir "$videos_src_dir" \
        --tracklets_temp_data_dir "$tracklets_temp_data_dir" \
        --torch_compile "$torch_compile" \
        --device "$rank" \
        --dataloader_batch_size "$dataloader_batch_size" \
        --dataloader_workers "$dataloader_workers" \
        -expn "levi-test-exp" \
        -f "exps/example/mot/yolox_x_sportsmot.py" \
        -n "yolox_x_sportsmot_mix" \
        -c "pretrained/yolox_x_sportsmot_mix.pth.tar" \
        --batch-size "1" \
        --num_machines "1" \
        --devices "1" \
        --test \
        --conf "0.01" \
        --nms "0.7" \
        --tsize "640" \
        --track_thresh "0.6" \
        --config "track" \
        &
done
wait