import numpy as np

from PIL import Image
from typing import List

from models import PaddleModel


def extract_text_with_paddle(image: Image.Image, model: PaddleModel) -> List[str]:
    """
    Returns a [str] containing all words found in a
    provided PIL image.
    """

    if image is None:
        return []
    ideal_height = 100
    scale_factor = ideal_height / image.height
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size)
    img_arr = np.array(image)
    # cv2.imwrite("preprocessed_img.png", img_arr)
    results = []

    # pred w/ paddleocr
    raw_result = model.model(img_arr)
    return results