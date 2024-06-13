export CUDA_VISIBLE_DEVICES=1

python3 tools/demo_track.py \
video \
-expn yolox_x_sports_mix \
-f exps/example/mot/yolox_x_sportsmot.py \
-c pretrained/yolox_x_sports_mix.pth.tar \
-n yolox_x_sports_mix \
--path "../sample-videos/clips/18139_10-31-2015_80_Houston Rockets_2_Golden State Warriors_period2_clip.mp4" \
--save_result \