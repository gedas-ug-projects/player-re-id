import os
import warnings

from loguru import logger
from MixSort.tools.track_mixsort_simple import *
from .paths import MIXSORT_DIR

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument")
os.chdir(MIXSORT_DIR)

def generate_player_tracks(coco_dataset_path: str, track_out_path: str, args):
    exp = get_exp(args.f, args.name)
    exp.merge(args.opts)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    num_gpu = 1
    main(args)
    
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu, coco_dataset_path, track_out_path, rank),
    )