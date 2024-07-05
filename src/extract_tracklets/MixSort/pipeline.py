import os
import shutil
import warnings
import argparse
import cProfile
import pstats
import logging
import torch
import random
import sys
import torch.backends.cudnn as cudnn

sys.path.append(
    "/playpen-storage/levlevi/player-re-id/src/extract_tracklets/MixSort/MixViT"
)

from yolox.exp import get_exp
from yolox.utils import fuse_model
from yolox.evaluators import MOTEvaluator
from yolox.data.datasets.datasets_wrapper import Dataset 
from exps.example.mot.yolox_x_sportsmot import Exp
from utils.convert_vid_coco import format_video_to_coco_dataset
from glob import glob

warnings.simplefilter("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

### MIXSORT CODE ###


def run(exp: Exp, args, coco_dataset_dir: str='', tracklet_out_path: str =''):

    def set_seeds(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting."
        )
    rank = args.device
    if args.seed is not None:
        set_seeds(args.seed)
    # hard-coded for now
    cudnn.benchmark = True
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    # is this bad?, we seem to get a new model w/ every call to main
    model = exp.get_model()
    val_loader: Dataset = exp.get_eval_loader(
        args,
        return_origin_img=True, # must be true
        data_dir=coco_dataset_dir,
    )
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
    )
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()
    if not args.speed and not args.trt:
        ckpt_file = args.ckpt or os.path.join(coco_dataset_dir, "best_ckpt.pth.tar")
        ckpt = torch.load(ckpt_file, map_location=f"cuda:{rank}")
        model.load_state_dict(ckpt["model"])
    if args.fuse:
        model = fuse_model(model)
    if args.torch_compile == "True":
        logger.info("Compiling model...")
        model = torch.compile(model)
    evaluator.evaluate_mixsort(model, tracklet_out_path, args)


### MY CODE ###


def generate_player_tracks(coco_dataset_dir: str, tracklet_out_path: str, args):
    exp = get_exp(args.f, args.name)
    exp.merge(args.opts)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    run(exp, args, coco_dataset_dir, tracklet_out_path)


def process_video(fp, args):
    skip_redundant = args.skip_redundant_videos
    tracklets_out_dir = args.tracklets_out_dir
    temp_coco_dir = args.tracklets_temp_data_dir
    vid_name = os.path.basename(fp).replace(".mp4", "").lower()
    coco_dst_path = os.path.join(temp_coco_dir, vid_name)
    track_dst_path = os.path.join(tracklets_out_dir, f"{vid_name}.txt")
    # check if the track output file already exists
    if skip_redundant == "True" and os.path.exists(track_dst_path):
        logger.info(f"Skipping {vid_name}: {track_dst_path} already exists.")
        return False
    format_video_to_coco_dataset(fp, coco_dst_path)
    generate_player_tracks(coco_dst_path, track_dst_path, args)
    # remove tmp dir
    shutil.rmtree(coco_dst_path)
    return True


def process_dir(args):
    videos_dir = args.videos_src_dir
    device = args.device
    video_files = glob(os.path.join(videos_dir, "*.mp4"))
    num_videos = int((1 / 8) * len(video_files))
    start_idx = device * num_videos
    end_idx = (
        start_idx + num_videos
        if start_idx + num_videos < len(video_files)
        else len(video_files)
    )
    ## MARK: DO NOT USE SUBSET ##
    # video_files = video_files[start_idx:end_idx]
    logger.info(f"Rank {device} processing videos {start_idx} to {end_idx}")
    for fp in video_files:
        res = process_video(fp, args)
        ### BREAK ###
        break
        ### BREAK ###


def main(args):
    process_dir(args)


if __name__ == "__main__":

    ### MY ARGS ###
    parser = argparse.ArgumentParser(description="Process some videos.")
    parser.add_argument(
        "--tracklets_out_dir", type=str, required=True, help="Tracklets directory"
    )
    parser.add_argument(
        "--videos_src_dir", type=str, required=True, help="Videos to process directory"
    )
    parser.add_argument(
        "--tracklets_temp_data_dir",
        type=str,
        required=True,
        help="Temporary data directory",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="GPU device to process videos on",
    )
    parser.add_argument(
        "--profile", type=str, required=False, default="False", help="Enable profiling"
    )
    parser.add_argument(
        "--skip_redundant_videos",
        type=str,
        required=False,
        default="False",
        help="Skip redundant videos",
    )
    parser.add_argument("--torch_compile", type=str, required=False, default="False")
    parser.add_argument("--dataloader_batch_size", type=int, required=True, default=1)
    parser.add_argument("--dataloader_workers", type=int, required=False, default=1)

    ### MIXSORT ARGS ###
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f", type=str, help="pls input your expriment description file"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default="pretrained/yolox_x_sportsmix_ch.pth.tar",
        type=str,
        help="ckpt for eval",
    )
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--track_thresh", type=float, default=0.6, help="tracking confidence threshold"
    )
    parser.add_argument(
        "--track_buffer", type=int, default=30, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.9,
        help="matching threshold for tracking",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument(
        "--min-box-area", type=float, default=100, help="filter out tiny boxes"
    )
    parser.add_argument(
        "--mot20", dest="mot20", default=False, action="store_true", help="test mot20."
    )
    parser.add_argument("--script", type=str, default="mixformer_deit")
    parser.add_argument("--config", type=str, default="track")
    parser.add_argument("--alpha", type=float, default=0.6, help="fuse parameter")
    parser.add_argument(
        "--radius", type=int, default=0, help="radius for computing similarity"
    )
    parser.add_argument(
        "--iou_only",
        dest="iou_only",
        default=False,
        action="store_true",
        help="only use iou for similarity",
    )

    args = parser.parse_args()
    profile = args.profile
    if profile == "True":
        profile_filename = "profiling_results.prof"
        # profile the main function
        cProfile.run("main(args.rank)", profile_filename)
        # read the profiling results and print them
        with open("profiling_stats.txt", "w") as stream:
            p = pstats.Stats(profile_filename, stream=stream)
            p.sort_stats(pstats.SortKey.CUMULATIVE)
            p.print_stats()
    else:
        main(args)
