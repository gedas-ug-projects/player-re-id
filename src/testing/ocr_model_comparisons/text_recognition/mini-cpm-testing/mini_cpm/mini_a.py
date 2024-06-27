import json
import ast
import os
import logging
import sys
import torch.multiprocessing as mp

from glob import glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
from typing import List

MODEL_NAME = 'openbmb/MiniCPM-Llama3-V-2_5'
MINI_CPM_DIR = "/mnt/opr/levlevi/player-re-id/src/testing/ocr_analysis/mini_cpm/MiniCPM-V"

if os.path.exists(MINI_CPM_DIR):
    sys.path.append(MINI_CPM_DIR)
    os.chdir(MINI_CPM_DIR)
else:
    raise FileNotFoundError(f"Directory {MINI_CPM_DIR} does not exist")

from chat import MiniCPMVChat, img2base64

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
        question = """This is an image of a basketball player. Identify the colors of the player's jersey and perform OCR to find the jersey number. 
        If the jersey number or colors are not visible, respond with None for each missing variable. 
        Format your result as '{"jersey_colors": ["color1", "color2", ...], "jersey_number": number}[EOS]'. 
        If multiple colors are present on the jersey, list all the colors. The jersey number should be an integer.""".replace('\n', '')
        msgs = [{'role': 'user', 'content': question}]
        inputs = {"image": image, "question": json.dumps(msgs)}
        answer = model.chat(inputs)
        
        try:
            result = '{' + answer.split('[EOS]')[0].split('{')[-1]
        except:
            result = {
                "jersey_colors": None,
                "jersey_number": None
            }
            return result
        return ast.literal_eval(result)
    except Exception as e:
        logger.info(f"Failed to perform OCR on image: {e}")
        result = {
                "jersey_colors": None,
                "jersey_number": None
            }
        return result

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
        os.makedirs(output_dir, exis
                    t_ok=True)
        dir_fp, dir_result = ocr_dir(dir_fp, model, output_dir)
        all_results[dir_fp] = dir_result

def main():
    mp.set_start_method('spawn')
    dirs = glob(os.path.join('/mnt/opr/levlevi/player-re-id/src/testing/ocr_analysis/sample_tracks_nba_100/__tracks_nba_50_grouped__/_hand-labeled', '*'))
    
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
        
    print(f"All results: {all_results}")

    result_out_fp = '/mnt/opr/levlevi/player-re-id/src/testing/ocr_analysis/nba_100_tracks_ocr_results.json'
    with open(result_out_fp, 'w') as f:
        for dir_fp, results in all_results.items():
            f.write(f"{dir_fp}: {json.dumps(results)}\n")
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    main()