from paddleocr import PaddleOCR

class PaddleModel:

    def __init__(self, device: int = 0) -> None:
        self.model = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            det_db_score_mode="slow",
            ocr_version="PP-OCRv4",
            rec_algorithm="SVTR_LCNet",
            det_db_thresh=0.50,
            drop_score=0.0,
            use_gpu=True,
            gpu_id=device,
            gpu_mem=1000,
        )