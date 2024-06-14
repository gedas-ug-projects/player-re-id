import os
import warnings

from loguru import logger
from MixSort.tools.track_mixsort_simple import *

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument")

MIXSORT_FP = "/playpen-storage/levlevi/player-re-id/src/player_reid/MixSort"
DATA_DIR = "/playpen-storage/levlevi/player-re-id/src/player_reid/testing/datasets/nba"

os.chdir(MIXSORT_FP)


def generate_player_tracks(coco_dataset_path: str, track_out_path: str, rank: int=0):
    
    parser = make_parser()
    args = parser.parse_args([
        "-expn", "levi-test-exp",
        "-f", "exps/example/mot/yolox_x_sportsmot.py",
        "-n", "yolox_x_sportsmot_mix",
        "-c", "pretrained/yolox_x_sportsmot_mix.pth.tar",
        "--batch-size", "1",
        "--num_machines", "1",
        "--devices", "1",
        "--test",
        "--conf", "0.01",
        "--nms", "0.7",
        "--tsize", "640",
        "--track_thresh", "0.6",
        "--config track"
    ])
    
    exp = get_exp(args.f, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = 1
    
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu, coco_dataset_path, track_out_path, rank),
    )


if __name__ == '__main__':
    dataset_fp = '/playpen-storage/levlevi/player-re-id/src/player_reid/testing/datasets/nba'
    track_out_fp = '/playpen-storage/levlevi/player-re-id/src/player_reid/testing/datasets/nba/track_results/test.txt'
    generate_player_tracks(dataset_fp, track_out_fp)