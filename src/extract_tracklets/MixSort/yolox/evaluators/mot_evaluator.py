import os
import concurrent
import time
import torch
import logging
import numpy as np
import concurrent.futures as cc

from yolox.utils import (
    postprocess,
    xyxy2xywh,
)
from yolox.mixsort_tracker.mixsort_iou_tracker import MIXTracker
from yolox.data.dataloading import DataLoader
from loguru import logger
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# TODO: optimize
# maybe use a different library?
# find some way to do this in paralell?
def write_results(filename, results):
    save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    s=round(score, 2),
                )
                f.write(line)


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader: DataLoader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate_mixsort(self, model, tracklets_out_path=None, args=None):
        
        def track(result):
            
            min_box_area = self.args.min_box_area
            device = args.device
            img_size = self.img_size
    
            outputs = result[0][0]
            info_imgs = result[1]
            frame_id = result[3]
            
            # set defaults
            online_tlwhs, online_ids, online_scores = [], [], []
            origin_imgs = (
                result[4].squeeze(0).to(device)
            )  # Move origin_imgs to device once
            if outputs is not None and len(outputs) > 0:
                online_targets = tracker.update(
                    outputs, info_imgs, img_size, origin_imgs
                )
                valid_targets = [
                    (t.tlwh, t.track_id, t.score)
                    for t in online_targets
                    if t.tlwh[2] * t.tlwh[3] > min_box_area
                    and t.tlwh[2] / t.tlwh[3] <= 1.6
                ]
                if valid_targets:
                    online_tlwhs, online_ids, online_scores = zip(*valid_targets)
            return (frame_id, online_tlwhs, online_ids, online_scores)
            
        # MARK: main inference loop
        def predict(iterable_dataloader):
            results = []
            for _, (origin_imgs, imgs, _, info_imgs, _) in tqdm(
                enumerate(iterable_dataloader),
                total=len(iterable_dataloader),
                desc="Creating Tracklets",
            ):
                with torch.no_grad():
                    frame_id = info_imgs[2]
                    # necessary move from CPU -> GPU
                    imgs = imgs.to(args.device)
                    outputs = model(imgs)
                    batch_outputs = [
                        [
                            postprocess(
                                torch.unsqueeze(outputs[i], dim=0),
                                self.num_classes,
                                self.confthre,
                                self.nmsthre,
                            ),
                            [item[i] for item in info_imgs],
                            1,
                            frame_id[i],
                            origin_imgs[i],
                        ]
                        for i in range(outputs.shape[0])
                    ]
                    # track results
                    for result in batch_outputs:
                        # will this work?
                        results.append(track(result))
                        
            torch.cuda.empty_cache()
            return results

        # assert tracklet path is valid
        assert tracklets_out_path.endswith(
            ".txt"
        ), f"Result path must end with .txt, but got {tracklets_out_path}"

        model.eval()
        tracker: MIXTracker = MIXTracker(self.args)

        # [(output_prediction_tensor, img_info_array, id (always 1))]
        iterable_dataloader = self.dataloader
        results = predict(iterable_dataloader)
        
        # remove existing tracklet is we are overwriting a file
        if os.path.isfile(tracklets_out_path):
            os.remove(tracklets_out_path)
        write_results(tracklets_out_path, results)
        return None

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for output, img_h, img_w, img_id in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue

            # expensive GPU -> CPU copy opp
            # is this necessisary
            # TODO: can we process a big tensor of results in paralell rather than iterating over results?
            # output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )

            # TODO: add explict types for bbxs
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            # TODO: maybe a faster way to do this?
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        return data_list
