import torch

from ultralytics import YOLO
from paddleocr import PaddleOCR
# from transformers import (
#     TrOCRProcessor,
#     VisionEncoderDecoderModel,
#     AutoModel,
#     AutoTokenizer,
# )
from PIL import Image

YOLO_MODEL_PATH = r"/mnt/opr/levlevi/nba-positions-videos-dataset/models/yolo/weights/tr_roi_finetune_60_large.pt"
TR_OCR_MODEL_PATH = "microsoft/trocr-base-stage1"


class YOLOModel:

    def __init__(self, device: int = 0) -> None:

        self.model = YOLO(YOLO_MODEL_PATH).to(device)


class PaddleModel:

    def __init__(self, device: int) -> None:
        self.model = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            det_db_score_mode="slow",
            ocr_version="PP-OCRv4",
            rec_algorithm="SVTR_LCNet",
            det_db_thresh=0.50,
            drop_score=0.80,
            use_gpu=True,
            gpu_id=device,
            gpu_mem=1000,
        )


# class TrOCRModel:

#     def __init__(self, device: int) -> None:

#         self.processor = TrOCRProcessor.from_pretrained(TR_OCR_MODEL_PATH)
#         self.model = VisionEncoderDecoderModel.from_pretrained(TR_OCR_MODEL_PATH).to(
#             device
#         )


# class MiniCPMModel:

#     def __init__(self, device: int) -> None:
#         self.model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.bfloat16)
#         self.model = self.model.to(device=device, dtype=torch.bfloat16)
#         self.model.eval()
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "openbmb/MiniCPM-V-2", trust_remote_code=True, torch_dtype=torch.bfloat16
#         )
