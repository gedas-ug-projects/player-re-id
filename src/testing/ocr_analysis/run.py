import json
import os
import logging
import sys
import torch.multiprocessing as mp
import warnings

from glob import glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
from typing import List

MODEL_NAME = 'openbmb/MiniCPM-Llama3-V-2_5'
MINI_CPM_DIR = "/playpen-storage/levlevi/player-re-id/src/testing/ocr_analysis/mini_cpm/MiniCPM-V"
PROMPT = """Analyze the basketball player shown in the provided still tracklet frame and describe the following details:

1. Jersey Number: Identify the number on the player's jersey. If not visible, respond with None.
2. Jersey Colors: List the colors visible on the player's jersey. Format this as a list of color names in lowercase (e.g., ["red", "white"]). If colors are not visible, respond with None.
3. Race: Determine the race or ethnicity of the player. Choose one from "white", "black", or "mixed". If race is not visible, respond with None.
4. Position: Identify the player's position. Use one of the following abbreviations: "G" (Guard), "C" (Center), "F" (Forward), "SG" (Shooting Guard), "PF" (Power Forward), or "SF" (Small Forward). If position is not visible, respond with None.

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

def ocr(image, model):
    try:
        question = PROMPT
        msgs = [{'role': 'user', 'content': question}]
        inputs = {"image": image, "question": json.dumps(msgs)}
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

def convert_images_to_pil(fps: List[str]):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_and_convert_image, fps))
    images_trim = []
    step = 1
    index = 0
    for image in images:
        if image and index % step == 0:
            images_trim.append(image)
        index += 1
    return images_trim

def process_images(images: List[Image.Image], model, output_dir: str):
    results = {}
    for idx, image in enumerate(tqdm(images, total=len(images))):
        result = ocr(image, model)
        if result:
            results[idx] = result
    try:
        with open(os.path.join(output_dir, f"output_{idx}.json"), 'w') as f:
            json.dump(results, f)
    except:
        logger.error(f"Failed to write results to file: {output_dir}")
    return results

def ocr_dir(dir_fp: str, model, output_dir: str):
    img_paths = glob(os.path.join(dir_fp, '*.jpg'))
    images = convert_images_to_pil(img_paths)
    results = process_images(images, model, output_dir)
    logger.info(f"Results for directory {dir_fp}: {results}")
    return dir_fp, results

def process_dir(dirs, device: int = 0, all_results = None):
    model = load_model_and_tokenizer(device)
    for dir_fp in dirs:
        logger.info(f"Processing directory: {dir_fp}")
        output_dir = os.path.join("results", os.path.basename(dir_fp))
        os.makedirs(output_dir, exist_ok=True)
        dir_fp, dir_result = ocr_dir(dir_fp, model, output_dir)
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