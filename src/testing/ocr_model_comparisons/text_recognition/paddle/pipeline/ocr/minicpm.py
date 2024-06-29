import torch
from torchvision import transforms

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from PIL import Image
from typing import List

from ocr.models import MiniCPMModel
from ocr.helpers import find_time_remaining_from_results


def process_image(image: Image.Image, model: MiniCPMModel, device: str) -> List[str]:
    question = "The following is an image of a game clock from a FIBA basketball broadcast. Perform OCR."
    msgs = [{"role": "user", "content": question}]

    with torch.no_grad():
        res, _, _ = model.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=model.tokenizer,
            sampling=True,
            temperature=0.7,
        )
    return res.split()


def extract_text_with_minicpm(
    images: List[Image.Image],
    model: MiniCPMModel,
    device: str = "cuda:0",
    max_workers = 8
) -> List[str]:
    """
    Returns a List[str] containing all words found in a
    provided list of PIL images, processed concurrently.
    """
    
    def process_image_concurrently(image):
        return process_image(image, model, device)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image_concurrently, images))
        
    return results


def extract_time_remaining_from_images_minicpm(
    images: List[Image.Image], model: MiniCPMModel, device: str = "cuda:0"
) -> List[str]:
    """
    Given a list of PIL Image objects,
    returns a list of valid formatted time-remaining strings (e.g., '11:30')
    or None for each image.
    """

    time_remaining = []
    results = extract_text_with_minicpm(images, model=model, device=device)
    for result in results:
        time_remaining.append(find_time_remaining_from_results(result))
    return time_remaining