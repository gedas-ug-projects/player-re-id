import torch

from typing import List
from PIL import Image
from ocr.models import TrOCRModel
from ocr.helpers import find_time_remaining_from_results


def extract_text_with_tr_ocr(
    image: Image.Image, model: TrOCRModel, device: str = "cuda:0"
) -> List[str]:
    """
    Returns a List[str] containing all words found in a
    provided list of PIL images.
    """

    pixel_values = model.processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.model.generate(pixel_values)
    decoded_text = model.processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return [decoded_text]


def extract_text_with_tr_ocr_batch(
    images: List[Image.Image], model, device: str = "cuda:0"
) -> List[str]:
    """
    Returns a List[str] containing all words found in a
    provided list of PIL images.
    """
    pixel_values_list = [
        model.processor(image, return_tensors="pt").pixel_values.to(device)
        for image in images
    ]
    pixel_values = torch.cat(pixel_values_list, dim=0)
    with torch.no_grad():
        generated_ids = model.model.generate(pixel_values)
    decoded_texts = model.processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )
    return decoded_texts


def extract_time_remaining_from_image_tr(
    image: Image.Image, model: TrOCRModel, device: int = 0
):
    """
    Given a PIL Image object,
    returns either a valid formatted time-remaining str (e.g., '11:30')
    or None.
    """
    rgb_img = image.convert("RGB")
    results = extract_text_with_tr_ocr(rgb_img, model=model, device=device)
    time_remaining = find_time_remaining_from_results(results)
    return time_remaining


def extract_time_remaining_from_images_tr(
    images: List[Image.Image], model: TrOCRModel, device: str = "cuda:0"
) -> List[str]:
    """
    Given a list of PIL Image objects,
    returns a list of valid formatted time-remaining strings (e.g., '11:30')
    or None for each image.
    """
    rgb_images = [image.convert("RGB") for image in images]
    results = extract_text_with_tr_ocr_batch(rgb_images, model=model, device=device)
    time_remaining_list = [
        find_time_remaining_from_results([result]) for result in results
    ]
    return time_remaining_list
