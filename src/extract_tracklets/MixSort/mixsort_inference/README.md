# Steps to run inference on MixSort with a single video (6/7/2024)
## Environment set up
1. Follow directions on MixSort repo for environment set up. Ensure you're using numpy>=1.2.* for MixSort inference to work. The vast majority of troubleshooting was straightforward versioning that didn't require too much detective work. It's still tricky, so hit me up if there's an issue you've been dealing with for a while.
## Data preprocessing

0. Clip video optionally using ```clip.py``` util in ```mixsort_inference/utils/```
1. Create a copy of ```datasets/sample_mot``` such as ```datasets/{your_video_name}_mot```. Also, create a COCO format dataset by making a copy of ```datasets/sample_coco``` such as ```datasets/{your_video_name}_coco```. 
2. Split input video into frames using ```mixsort_inference/utils/video_to_frames.py``` and ensure the destination is ```datasets/{your_video_name}_mot/dataset/train/v_100/img1/```.
3. Insert the frames into the COCO dataset by replacing ```datasets/{your_video_name}_coco/val/v_100/img1``` with the frames folder created in step 2 located at ```datasets/{your_video_name}_mot/dataset/train/v_100/img1/```.
4. Create the appropriate annotations file in COCO format from your MOT dataset. Start by running the ```mixsort_inference/utils/convert_sportsmot_to_coco.py``` script setting the datapath to ```datasets/{your_video_name}_mot/dataset/```. The output should appear as ```datasets/{your_video_name}_mot/dataset/annotations/annotations.json```. Replace the copied sample annotations ```datasets/{your_video_name}_coco/annotations/annotations.json``` with the newly created ```datasets/{your_video_name}_mot/dataset/annotations/annotations.json```.
## Inference
1. Create a copy of ```exps/example/mot/yolox_x_inference.py``` renaming to ```exps/example/mot/yolox_x_{your_vid_name}.py```. This exp file determines how to configure a subclassed DataLoader to load COCO datasets. On line 98, change ```"sample_coco"``` with ```"{your_video_name}_coco``` ensuring it matches the COCO-version of your dataset.
2. Modify ```mixsort_inference/mixsort_inference.sh``` to change the ```-f``` flag to the created exp file: ```exps/example/mot/yolox_x_{your_vid_name}.py``` as well as the ```-expn ``` flag to ```yolox_x_{your_video_name}```.
3. Run ```bash mixsort_inference.sh``` and resulting bounding box results appear in MOT format under ```YOLOX_outputs/yolox_x_{your_vid_name}/track_results/v_100.txt``` or something similar at least.
### Create video demo using outputs
Optionally, the bounding boxes produced in MOT format ```v_100.txt``` can be used along with one of the frames directory ```img1/``` by using ```mixsort_inference/utils/bb_to_video.py``` to produce a video. Note that the frame rate is a constant at the top defaulted to 30fps, but double check your source video. The output should be generated in the same directory as ```v_100.txt```.