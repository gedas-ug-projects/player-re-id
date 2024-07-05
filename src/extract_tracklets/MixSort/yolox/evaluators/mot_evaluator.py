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
    # logger.info('save results to {}'.format(filename))


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

        # we probably need to use this somewhere
        rank = args.local_rank
        half = args.fp16

        # assert tracklet path is valid
        assert tracklets_out_path.endswith(
            ".txt"
        ), f"Result path must end with .txt, but got {tracklets_out_path}"

        # TODO: half precision, dosen't work rn ):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model.eval()
        if half:
            model = model.half()

        # TODO: is there slow code in this MixTracker obj?
        tracker: MIXTracker = MIXTracker(self.args)
        data_list = []
        results = []

        # TODO: add explicit typing for dataloader obj
        iterable_dataloader = iter(self.dataloader)

        # Q: how long does it take just to iterate through the dataloader
        # start_time = time.time()
        # load_data_batches(iterable_dataloader, args.batch_size)
        # end_time = time.time()
        # logger.debug(f"Parsing data took {end_time - start_time} seconds")

        # batch size for forward pass
        batch_size = args.dataloader_batch_size
        outputs_post_proccessed = []

        # MARK: main inference loop
        for cur_iter, (origin_imgs, imgs, _, info_imgs, ids) in tqdm(
            enumerate(iterable_dataloader),
            total=len(iterable_dataloader),
            desc="Creating Tracklets",
        ):
            # (5, 32, 3, 720, 1280)
            
            # MARK: forward pass
            with torch.no_grad():

                # raw items in data loader
                # TODO: is there a better way to stack these items
                # then appending items to a list w/ a for loop?
                frame_id = info_imgs[2]
                video_id = info_imgs[3]
                img_file_name = info_imgs[4]

                logger.debug(f"info_imgs: {info_imgs}")
                logger.debug(f"frame_id: {frame_id}")
                logger.debug(f"video_id: {video_id}")
                logger.debug(f"img_file_name: {img_file_name}")

                # imgs = imgs.type(tensor_type)
                # imgs.type(tensor_type)
                logger.debug(f"imgs.shape: {imgs.shape}")

                # forward pass
                # do we need to copy to GPU?
                start_time = time.time()
                imgs = imgs.cuda()
                end_time = time.time()
                print(f"copying to GPU took {end_time - start_time} seconds")
                
                # TODO: what is the optimal batch size?
                    # 1:   1.6780129030399678
                    # 2:   2.5556285711239166
                    # 4:   3.9093613984134867
                    # 8:   3.3346755078720745
                    # 16:  9.771818710255356
                    # 32:  17.471120875596277
                
                start_time = time.time()
                outputs = model(imgs)
                end_time = time.time()
                items_per_sec = batch_size / (end_time - start_time)
                print(f"forward pass took {end_time - start_time} seconds")
                print(f"items per second: {items_per_sec}")
                print(f"outputs.shape: {outputs.shape}")

                # MARK: no significant speed-up from paralell processing
                start_time = time.time()
                outputs_post_proccessed = [
                        postprocess(
                            torch.unsqueeze(outputs[i, :, :], dim=0),
                            self.num_classes,
                            self.confthre,
                            self.nmsthre,
                        )
                            for i in range(outputs.shape[0])
                        ]
                end_time = time.time()
                print(f"Post processing took {end_time - start_time} seconds")
                
                start_time = time.time()
                torch.cuda.empty_cache()
                end_time = time.time()
                print(f"empty cache took {end_time - start_time} seconds")

                # logger.debug(f"outputs post-proccessed shape: {outputs}")

                # seems to be we are just calculating
                # TODO: this code needs to modularized
                # 1. generate all batches from generator as a single iterable tensor
                # we should perform all expensive copy ops from CPU -> GPU in paralell up front
                # 2. perform inference on batches and append results (already on GPU) to a list
                # 3. post process all results
                # 4. calculate all results

                # output, image info, video id
        #         # TODO: calculate results
        #         for o, ii, id in zip(
        #             outputs_post_proccessed, info_imgs, imgs
        #         ):

        #             logger.debug(f"o: {o}")
        #             logger.debug(f"ii: {ii}")
        #             logger.debug(f"id: {id}")

        #             # TODO: can most likely optimize this function
        #             # TODO: do this in paralell
        #             output_results = self.convert_to_coco_format(o, ii, id)

        #             logger.debug(f"output_results: {output_results}")
        #             data_list.extend(output_results)
        #             logger.debug(f"data_list: {data_list}")

        #             # what is at o[0]?
        #             if o[0] is not None:
        #                 logger.debug(f"o[0]: {o[0]}")

        #                 # what are these online targets?
        #                 # TODO: copy all origin images to cuda as a single tensor
        #                 # TODO: optimize `update` function
        #                 online_targets = tracker.update(
        #                     o[0], ii, self.img_size, origin_imgs.squeeze(0).cuda()
        #                 )
        #                 logger.debug(f"online_targets: {online_targets}")

        #                 # what is `online_tlwhs` var?
        #                 # what is `t` temp var?
        #                 online_tlwhs, online_ids, online_scores = [], [], []
        #                 for t in online_targets:

        #                     # what is tid, do i care?
        #                     tlwh, tid, score = t.tlwh, t.track_id, t.score

        #                     # silly hard coded threshold
        #                     vertical = tlwh[2] / tlwh[3] > 1.6
        #                     if (
        #                         tlwh[2] * tlwh[3] > self.args.min_box_area
        #                         and not vertical
        #                     ):
        #                         online_tlwhs.append(tlwh)
        #                         online_ids.append(tid)
        #                         online_scores.append(score)

        #                 logger.debug(f"online_tlwhs: {online_tlwhs}")
        #                 logger.debug(f"online_ids: {online_ids}")
        #                 logger.debug(f"online_scores: {online_scores}")

        #                 # results.append((frame_id, online_tlwhs, online_ids, online_scores))
        #                 results.append(
        #                     (
        #                         id,
        #                         online_tlwhs,
        #                         online_ids,
        #                         online_scores,
        #                     )
        #                 )
        #                     # why write results every forward pass?

        #             # reset
        #             # TODO: come up w/ a better memory management system
        #             # TODO: benchmark: are larger forward passes actually faster than processing videos one at a time?

        # # TODO: write results to file in paralell
        # # this function just iterates over results and writes each line to a file
        # write_results(tracklets_out_path, results)

        # TODO: maybe give this function a meaningful return value?
        return None

    # TODO: some explicit typing / documentation of these variable names
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
            output = output.cpu()

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
