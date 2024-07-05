# encoding: utf-8
import os
import torch
import torch.distributed as dist
import logging

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train.json"
        self.val_ann = "val.json"  # change to train.json when running on training set
        self.input_size = (800, 1440)

        # can we reduce this input size?
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 1
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "SportsMOT"),
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=600,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1200,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader

    def get_eval_loader(
        self,
        args,
        return_origin_img=False,
        data_dir=os.path.join(get_yolox_datadir(), "SportsMOT"),
    ):
        from yolox.data import MOTDataset, ValTransform
        valdataset = MOTDataset(
            data_dir=data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name="val",  # change to train when running on training set
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            return_origin_img=return_origin_img,
        )
        logger.info(f"Batch Size: {args.dataloader_batch_size}")
        batch_size = args.dataloader_batch_size
        
        # TODO: use batch sampler?
            # greater num workers yields faster dataset parsing speeds
            # dataloader_workers=8 seems to be optimal
            
        seq_sampler = torch.utils.data.SequentialSampler(valdataset)
        
        # TODO: batch size MUST BE HARD CODED TO 1
        batch_sampler = torch.utils.data.BatchSampler(seq_sampler, batch_size=1, drop_last=False)
        dataloader_kwargs = {
            "num_workers": args.dataloader_workers,
            "pin_memory": True, # need to copy tensors to GPU
            "sampler": batch_sampler,
            "batch_size": batch_size,
        }
        logger.info("Dataloader args are: {}".format(dataloader_kwargs))
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
