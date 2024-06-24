import json
import os
import logging
import torch.multiprocessing as mp
import warnings

from glob import glob
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from tqdm import tqdm

PROMPT = """Analyze the basketball player shown in the provided still tracklet frame and describe the following details:
1. Jersey Number: Identify the number on the player's jersey. If the player has no jersey, provide None.
Based on the frame description, produce an output prediction in the following JSON format:
{
  "jersey_number": "<predicted_jersey_number>",
}
[EOS]"""

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(device: int = 0):
    try:
        logger.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True, device_map="cuda").to(device).eval()
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
        return model, processor
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise

def ocr(image_file_path: str, model, processor, device) -> str:
    image = Image.open(image_file_path)
    inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="<OCR>", image_size=(image.width, image.height))
    return parsed_answer

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