import os
import numpy as np
import json
import cv2
from tqdm import tqdm
import argparse


# still unclear, what is MOT?
# why not take vid in and create dataset

def convert_mot_to_coco(
        data_path, 
        out_path, 
        splits=["train"], 
        half_video=False, 
        create_splitted_ann=True, 
        use_det=False, 
        create_splitted_det=False
        ):
    os.makedirs(out_path, exist_ok=True)

    for split in splits:
        split_data_path = os.path.join(data_path, split)
        split_out_path = os.path.join(out_path, "annotations.json")

        # example annotations file format
        out = {
            "images": [],
            "annotations": [],
            "videos": [],
            "categories": [{
                "id": 1,
                "name": "pedestrian"
            }]
        }

        video_list = os.listdir(split_data_path)
        print(video_list)

        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0

        for seq in tqdm(sorted(video_list)):
            if not ".mp4" in seq:
                continue
            video_cnt += 1  # video sequence number.
            out["videos"].append({"id": video_cnt, "file_name": seq})
            seq_path = os.path.join(split_data_path, seq)
            img_path = os.path.join(seq_path, "img1")
            ann_path = os.path.join(seq_path, "gt/gt.txt")
            images = os.listdir(img_path)

            # do we need to split 50-50?
            num_images = len([image for image in images if "jpg" in image])  # half and half
            
            info_path = os.path.join(seq_path, "seqinfo.ini")
            with open(info_path, "r") as f:
                for line in f:
                    if 'imWidth' in line:
                        width = int(line.split('=')[1])
                    elif 'imHeight' in line:
                        height = int(line.split('=')[1])

            if half_video and ("half" in split):
                image_range = [0, num_images // 2] if "train" in split else [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                image_info = {
                    "file_name": "{}/img1/{:06d}.jpg".format(seq, i + 1),  # image name.
                    "id": image_cnt + i + 1,  # image number in the entire training set.
                    "frame_id": i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                    "prev_image_id": image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                    "next_image_id": image_cnt + i + 2 if i < num_images - 1 else -1,
                    "video_id": video_cnt,
                    "height": height,
                    "width": width
                }
                out["images"].append(image_info)
            print("{}: {} images".format(seq, num_images))
            if split != "test":
                det_path = os.path.join(seq_path, "det/det.txt")
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=",")
                if use_det:
                    dets = np.loadtxt(det_path, dtype=np.float32, delimiter=",")
                if create_splitted_ann and ("half" in split):
                    anns_out = np.array([
                        anns[i] for i in range(anns.shape[0])
                        if int(anns[i][0]) - 1 >= image_range[0]
                        and int(anns[i][0]) - 1 <= image_range[1]
                    ], np.float32)
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(seq_path, "gt/gt_{}.txt".format(split))
                    with open(gt_out, "w") as fout:
                        for o in anns_out:
                            fout.write(
                                "{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n".format(
                                    int(o[0]), int(o[1]), int(o[2]), int(o[3]),
                                    int(o[4]), int(o[5]), int(o[6]), int(o[7]),
                                    o[8])
                            )
                if create_splitted_det and ("half" in split) and use_det:
                    dets_out = np.array([
                        dets[i] for i in range(dets.shape[0])
                        if int(dets[i][0]) - 1 >= image_range[0]
                        and int(dets[i][0]) - 1 <= image_range[1]
                    ], np.float32)
                    dets_out[:, 0] -= image_range[0]
                    det_out = os.path.join(seq_path, "det/det_{}.txt".format(split))
                    with open(det_out, "w") as dout:
                        for o in dets_out:
                            dout.write(
                                "{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n".format(
                                    int(o[0]), int(o[1]), float(o[2]), float(o[3]),
                                    float(o[4]), float(o[5]), float(o[6]))
                            )

                print("{} ann images".format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    if not ("15" in data_path):
                        if not (float(anns[i][8]) >= 0.25):  # visibility.
                            continue
                        if not (int(anns[i][6]) == 1):  # whether ignore.
                            continue
                        if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                            continue
                        if int(anns[i][7]) in [2, 7, 8, 12]:  # Ignored person
                            category_id = -1
                        else:
                            category_id = 1  # pedestrian(non-static)
                    else:
                        category_id = 1
                    ann = {
                        "id": ann_cnt,
                        "category_id": category_id,
                        "image_id": image_cnt + frame_id,
                        "track_id": track_id,
                        "bbox": anns[i][2:6].tolist(),
                        "conf": float(anns[i][6]),
                        "iscrowd": 0,
                        "area": float(anns[i][4] * anns[i][5])
                    }
                    out["annotations"].append(ann)
            image_cnt += num_images
        print("loaded {} for {} images and {} samples".format(
            split, len(out["images"]), len(out["annotations"])))
        with open(split_out_path, "w") as f:
            json.dump(out, f, indent=2)