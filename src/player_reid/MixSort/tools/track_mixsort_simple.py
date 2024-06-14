import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import random
import warnings
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../MixViT'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MixSort'))

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import fuse_model
from yolox.evaluators import MOTEvaluator

import motmetrics as mm

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("-f", type=str, help="pls input your expriment description file")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.")
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("-c", "--ckpt", default='pretrained/yolox_x_sportsmix_ch.pth.tar', type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--iou_thresh",type=float,default=0.3)
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--script",type=str,default='mixformer_deit')
    parser.add_argument("--config",type=str,default='track')
    parser.add_argument("--alpha",type=float,default=0.6,help='fuse parameter')
    parser.add_argument("--radius",type=int,default=0,help='radius for computing similarity')
    parser.add_argument("--iou_only",dest="iou_only",default=False, action="store_true",help='only use iou for similarity')
    return parser

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    warnings.warn("You have chosen to seed testing. This will turn on the CUDNN deterministic setting.")

def main(exp, args, num_gpu, data_dir=None, results_path=None, rank: int=0):
    if args.seed is not None:
        set_seeds(args.seed)

    is_distributed = num_gpu > 1
    cudnn.benchmark = True

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test, return_origin_img=True, data_dir=data_dir)

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
        ckpt_file = args.ckpt or os.path.join(data_dir, "best_ckpt.pth.tar")
        ckpt = torch.load(ckpt_file, map_location=f"cuda:{rank}")
        model.load_state_dict(ckpt["model"])

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        model = fuse_model(model)

    trt_file, decoder = None, None
    evaluator.evaluate_mixsort(model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_path, rank=rank)

def compare_dataframes(gts, ts):
    accs, names = [], []
    for k, tsacc in ts.items():
        if k in gts:
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            warnings.warn(f'No ground truth for {k}, skipping.')
    return accs, names

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    exp = get_exp(args.f)
    num_gpu = torch.cuda.device_count()
    main(exp, args, num_gpu)