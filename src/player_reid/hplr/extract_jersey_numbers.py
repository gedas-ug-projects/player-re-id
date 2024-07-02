import json
import os
import logging
import random
import warnings
import torch.multiprocessing as mp

from glob import glob
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOOTSTRAPS = 9
PROMPT = """Analyze the basketball player shown in the provided still tracklet frame and describe the following details:
1. Jersey Number: Identify the number on the player's jersey. If the player has no jersey, provide None.
Based on the frame description, produce an output prediction in the following JSON format:
{
  "jersey_number": "<predicted_jersey_number>",
}
[EOS]"""

def load_model_and_tokenizer(device: int = 0):
    try:
        logger.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True, device_map="cuda").to(device).eval()
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
        return model, processor
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise

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

def ocr(image_file_path: str, model, processor, device) -> str:
    bootstraped_results = []
    for _ in range(BOOTSTRAPS):
        image = Image.open(image_file_path)
        image = augment_image(image)
        if not image:
            return None
        inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(f'cuda:{device}')
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=True,
            num_beams=10
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task="<OCR>", image_size=(image.width, image.height))
        bootstraped_results.append(parsed_answer)
    return bootstraped_results

def ocr_dir(dir_fp: str, model, processor, deivce):
    img_paths = glob(os.path.join(dir_fp, '*.jpg'))
    results = {}
    for idx, img_path in enumerate(tqdm(img_paths, total=len(img_paths))):
        result = ocr(img_path, model, processor, deivce)
        if result:
            results[idx] = result
    logger.info(f"Results for directory {dir_fp}: {results}")
    return dir_fp, results

def process_dir(dirs, device: int = 0, all_results=None):
    model, processor = load_model_and_tokenizer(device)
    for dir_fp in dirs:
        logger.info(f"Processing directory: {dir_fp}")
        dir_fp, dir_result = ocr_dir(dir_fp, model, processor, device)
        all_results[dir_fp] = dir_result

def main():
    mp.set_start_method('spawn')
    tracks_dir = '/mnt/opr/levlevi/player-re-id/src/data/_50_game_reid_benchmark_/labeled-tracks'
    out_fp = '/mnt/opr/levlevi/player-re-id/src/data/florence_100_track_bm_results.json'
    dirs = glob(os.path.join(tracks_dir, '*', '*'))
    manager = mp.Manager()
    all_results = manager.dict()
    processes = []
    start_device = 4
    num_devices = 8
    for i in range(start_device, num_devices):
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