import os

MIXSORT_FP = "/playpen-storage/levlevi/player-re-id/__vish__/player-reidentification/MixSort"
os.chdir(MIXSORT_FP)

from MixSort.tools.track_mixsort_simple import *

if __name__ == '__main__':
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

    # num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    num_gpu = 1
    print(torch.cuda.is_available())
    print(num_gpu, torch.cuda.device_count())
    assert num_gpu <= torch.cuda.device_count()

    data_dir = "/playpen-storage/levlevi/player-re-id/__vish__/player-reidentification/MixSort/datasets/levi-test-ds-2"
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu, data_dir),
    )