import cv2
import shutil
import json
import os
import logging
import sys
import torch.multiprocessing as mp
import warnings
import random

from glob import glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import List

MODEL_NAME = 'openbmb/MiniCPM-Llama3-V-2_5'
MINI_CPM_DIR = "/playpen-storage/levlevi/player-re-id/src/testing/ocr_analysis/mini_cpm/MiniCPM-V"
PROMPT = """Analyze the basketball player shown in the provided still tracklet frame and describe the following details:

1. Jersey Number: Identify the number on the player's jersey. If not visible, respond with None.
2. Jersey Colors: List the colors visible on the player's jersey. Format this as a list of color names in lowercase (e.g., ["<color_one>", "<color_two>"].
3. Race: Determine the race or ethnicity of the player. Choose one from "white", "black", or "mixed"
4. Position: Identify the player's position. Use one of the following abbreviations: "G" (Guard), "C" (Center), "F" (Forward), "SG" (Shooting Guard), "PF" (Power Forward), or "SF" (Small Forward).

Based on the frame description, produce an output prediction in the following JSON format:
{
  "jersey_number": "<predicted_jersey_number>",
  "jersey_colors": ["<predicted_color_1>", "<predicted_color_2>"],
  "race": "<predicted_race>",
  "position": "<predicted_position>"
}
[EOS]"""

if os.path.exists(MINI_CPM_DIR):
    sys.path.append(MINI_CPM_DIR)
    os.chdir(MINI_CPM_DIR)
else:
    raise FileNotFoundError(f"Directory {MINI_CPM_DIR} does not exist")

from chat import MiniCPMVChat, img2base64

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(device: int = 0):
    try:
        logger.info("Loading model and tokenizer...")
        model = MiniCPMVChat(MODEL_NAME, device)
        logger.info("Model and tokenizer loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise

def ocr(image_base64, model):
    try:
        question = PROMPT
        msgs = [{'role': 'user', 'content': question}]
        inputs = {"image": image_base64, "question": json.dumps(msgs)}
        answer = model.chat(inputs)
        result = answer
        return result
    except Exception as e:
        logger.error(f"Failed to perform OCR: {e}")
        return ""

def load_and_convert_image(fp: str):
    try:
        return img2base64(fp)
    except Exception as e:
        logger.error(f"Failed to load or convert image {fp}: {e}")
        return None

def convert_images_to_pil(images: List[str]):
    grouped_images = [images[i:i + 4] for i in range(0, len(images), 4) if len(images[i:i + 4]) == 4]
    return grouped_images

def concatenate_images(image_group):
    try:
        pil_images = [ImageOps.exif_transpose(Image.open(fp)) for fp in image_group]
        widths, heights = zip(*(i.size for i in pil_images))
        max_width, max_height = max(widths), max(heights)
        new_image = Image.new('RGB', (2 * max_width, 2 * max_height))

        new_image.paste(pil_images[0], (0, 0))
        new_image.paste(pil_images[1], (max_width, 0))
        new_image.paste(pil_images[2], (0, max_height))
        new_image.paste(pil_images[3], (max_width, max_height))

        return new_image
    except Exception as e:
        logger.error(f"Failed to concatenate images: {e}")
        return None

def process_images(image_groups: List[List[str]], model):
    results = {}
    for idx, image_group in enumerate(tqdm(image_groups, total=len(image_groups))):
        concatenated_image = concatenate_images(image_group)
        if concatenated_image:
            rand_idx = random.randint(0, 1000000)
            temp_fp = f'/playpen-storage/levlevi/player-re-id/src/testing/ocr_analysis/temp_{rand_idx}.png'
            concatenated_image.save(temp_fp)
            image_base64 = img2base64(temp_fp)
            os.remove(temp_fp)
            result = ocr(image_base64, model)
            if result:
                results[idx] = result
    return results

def ocr_dir(dir_fp: str, model):
    img_paths = glob(os.path.join(dir_fp, '*.jpg'))
    image_groups = convert_images_to_pil(img_paths)
    results = process_images(image_groups, model)
    logger.info(f"Results for directory {dir_fp}: {results}")
    return dir_fp, results

def process_dir(dirs, device: int = 0, all_results=None):
    model = load_model_and_tokenizer(device)
    for dir_fp in dirs:
        logger.info(f"Processing directory: {dir_fp}")
        dir_fp, dir_result = ocr_dir(dir_fp, model)
        all_results[dir_fp] = dir_result

def main():
    mp.set_start_method('spawn')
    
    tracks_dir = '/playpen-storage/levlevi/player-re-id/src/testing/ocr_analysis/_50_game_reid_benchmark_/labeled-tracks'
    out_fp = '/playpen-storage/levlevi/player-re-id/src/testing/ocr_analysis/results.json'
    dirs = glob(os.path.join(tracks_dir, '*', '*'))[0:1]
    
    manager = mp.Manager()
    all_results = manager.dict()
    
    processes = []
    num_devices = 1

    for i in range(num_devices):
        sub_dirs_arr = [dirs[j] for j in range(len(dirs)) if j % num_devices == i]
        p = mp.Process(target=process_dir, args=(sub_dirs_arr, i, all_results))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
        
    logger.info(f"All results: {all_results}")
    with open(out_fp, 'w') as f:
        for dir_fp, results in all_results.items():
            f.write(f"{dir_fp}: {json.dumps(results)}\n")
    logger.info("Processing completed successfully.")


if __name__ == "__main__":
    main()
