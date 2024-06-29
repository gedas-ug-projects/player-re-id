import json
import os
import logging
import warnings
import random
import easyocr
import numpy as np

from glob import glob
from PIL import Image, ImageOps
from tqdm import tqdm

MODEL = easyocr.Reader(['en'])
logger = logging.getLogger(__name__)

def ocr(image_file_path: str) -> str:
    def augment_image(image):
        angle = random.uniform(-1, 1)
        image = image.rotate(angle)
        max_translate = 5
        translate_x = random.uniform(-max_translate, max_translate)
        translate_y = random.uniform(-max_translate, max_translate)
        image = image.transform(
            image.size, Image.AFFINE,
            (1, 0, translate_x, 0, 1, translate_y)
        )
        scale = random.uniform(0.99, 1.01)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.BICUBIC)
        if scale < 1:
            pad_x = (image.size[0] - new_size[0]) // 2
            pad_y = (image.size[1] - new_size[1]) // 2
            image = ImageOps.expand(image, border=(pad_x, pad_y, pad_x, pad_y))
        else:
            crop_x = (new_size[0] - image.size[0]) // 2
            crop_y = (new_size[1] - image.size[1]) // 2
            image = image.crop((crop_x, crop_y, crop_x + image.size[0], crop_y + image.size[1]))
        return image
    def is_valid_jersey_number(text):
        if text.isdigit():
            number = int(text)
            return 0 <= number <= 99
        return False
    image = Image.open(image_file_path)
    temp_results = []
    for bootstrap in range(10):
        augmented_image = augment_image(image)
        augmented_image_np = np.array(augmented_image)
        results_texts = MODEL.readtext(augmented_image_np, detail=1)
        valid_results = [(text, confidence) for (bbox, text, confidence) in results_texts if is_valid_jersey_number(text)]
        temp_results.extend(valid_results)
    best_result = None
    if temp_results:
        best_result = max(temp_results, key=lambda x: x[1])
    return best_result

def ocr_dir(dir_fp):
    img_paths = glob(os.path.join(dir_fp, '*.jpg'))
    results = {}
    for idx, img_path in enumerate(tqdm(img_paths, total=len(img_paths))):
        result = ocr(img_path)
        if result:
            results[idx] = result
    logger.info(f"Results for directory {dir_fp}: {results}")
    return dir_fp, results

def main():
    
    tracks_dir = '/mnt/opr/levlevi/player-re-id/src/data/_50_game_reid_benchmark_/labeled-tracks'
    out_fp = '/mnt/opr/levlevi/player-re-id/src/data/florence_100_track_bm_results.json'
    dirs = glob(os.path.join(tracks_dir, '*', '*'))
    
    all_results = {}
    for dir in dirs:
        all_results[dir] = {}
        dir_fp, results = ocr_dir(dir)
        all_results[dir_fp] = results
        
    logger.info(f"All results: {all_results}")
    with open(out_fp, 'w') as f:
        for dir_fp, results in all_results.items():
            f.write(f"{dir_fp}: {json.dumps(results)}\n")
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    main()