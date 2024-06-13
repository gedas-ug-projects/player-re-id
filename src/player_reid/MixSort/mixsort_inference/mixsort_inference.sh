export CUDA_VISIBLE_DEVICES=1

python3 tools/track_mixsort_simple.py \
-expn yolox_x_sample_test \
-f exps/example/mot/yolox_x_sample_test.py \
-c pretrained/yolox_x_sports_mix.pth.tar \
-b 1 \
-d 1 \
--config track

# -expn yolox_x_inference -f exps/example/mot/yolox_x_inference.py -c pretrained/yolox_x_sports_mix.pth.tar -b 1 -d 1 --config track