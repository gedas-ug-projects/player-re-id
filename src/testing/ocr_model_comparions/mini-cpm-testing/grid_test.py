import cv2
import shutil
import json
import os
import logging
import sys
import torch.multiprocessing as mp
import warnings

from glob import glob
from PIL import Image, ImageOps
from tqdm import tqdm

MODEL_NAME = 'openbmb/MiniCPM-Llama3-V-2_5'
MINI_CPM_DIR = "/mnt/opr/levlevi/player-re-id/src/testing/mini_cpm_testing/mini_cpm/MiniCPM-V"
PROMPT = """Analyze the basketball player shown in the provided still tracklet frame and describe the following details:
1. Jersey Number: Identify the number on the player's jersey. If the player has no jersey, provide None.
Based on the frame description, produce an output prediction in the following JSON format:
{
  "jersey_number": "<predicted_jersey_number>",
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

def process_image(image_fp: str, model):
    image_base64 = load_and_convert_image(image_fp)
    if image_base64:
        result = ocr(image_base64, model)
        return result
    return None

def ocr_dir(dir_fp: str, model):
    img_paths = glob(os.path.join(dir_fp, '*.jpg'))
    results = {}
    for idx, img_path in enumerate(tqdm(img_paths, total=len(img_paths))):
        result = process_image(img_path, model)
        if result:
            results[idx] = result
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
    
    tracks_dir = '/mnt/opr/levlevi/player-re-id/src/testing/ocr_analysis/_50_game_reid_benchmark_/labeled-tracks'
    out_fp = '/mnt/opr/levlevi/player-re-id/src/testing/ocr_analysis/results.json'
    dirs = glob(os.path.join(tracks_dir, '*', '*'))
    
    manager = mp.Manager()
    all_results = manager.dict()
    
    processes = []
    num_devices = 8

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