#!/bin/bash

# for rank in {0..3}
# do
#     nohup python3 pipeline.py --rank $rank &
#     python3 pipeline.py --rank $rank &
# done

# Wait for all background processes to complete
# wait

export CUDA_VISIBLE_DEVICES=0,1,2
cd MixSort/

## NBA 15'-16' ##
# tracklets_out_dir="/mnt/sun/levlevi/nba-plus-statvu-dataset/player-tracklets"
# videos_src_dir="/mnt/sun/levlevi/nba-plus-statvu-dataset/game-replays"

## TEST ##
videos_src_dir="/playpen-storage/levlevi/player-re-id/__old__/clips"
tracklets_out_dir="/playpen-storage/levlevi/player-re-id/src/extract_tracklets/testing_tracks_out"
tracklets_temp_data_dir="/mnt/meg/levlevi/tmp"
device=0

python pipeline.py \
    --tracklets_out_dir "$tracklets_out_dir" \
    --videos_src_dir "$videos_src_dir" \
    --tracklets_temp_data_dir "$tracklets_temp_data_dir" \
    --device "$device" \
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
wait