# Multi-video inference
## Data preprocessing
folder of videos -> coco_dataset
1. first built MOT dataset \
1a. input is folder of videos \
1b. create MOT folder with same name as input folder_MOT \
1c. create subfolders /dataset and /splits_txt\
1c. for each video create /dataset/{video_name}/. each video folder contains gt/gt.txt which is empty, img1/ which contains the video frames in .jpg format, and extract video/frame information to make seqifo.ini \
1d. create splits_txt/train.txt which has the names of the video per line \
2. convert mot dataset to coco \
2a. create COCO folder with same name as input folder_COCO \
2b. create subfolders /annotations/annotations.json (which should be the save location from the other script)
2c. create subfolders /annotations/val/{vid_name} which is just a copy of each video directory from the MOT dataset creation
3. create exp file, 