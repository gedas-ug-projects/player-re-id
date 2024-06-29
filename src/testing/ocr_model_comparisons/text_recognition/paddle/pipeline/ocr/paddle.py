import numpy as np

from PIL import Image
from typing import List

from ocr.models import PaddleModel
from ocr.helpers import find_time_remaining_from_results


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

    text_arr = raw_result[1]
    for pred in text_arr:
        word = pred[0]
        results.append(word)
    return results


def extract_time_remaining_from_image_paddle(image: Image.Image, model: PaddleModel):
    """
    Given a PIL Image object,
    returns either a valid formatted time-remaining str (e.g., '11:30')
    or None.
    """
    rgb_img = image.convert("RGB")
    results = extract_text_with_paddle(rgb_img, model=model)
    time_remaining = find_time_remaining_from_results(results)
    return time_remaining
