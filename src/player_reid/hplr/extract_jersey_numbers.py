_ = """
TODO
1. process images/prompts in batches

# 4.02s/it: fp32, no compile
# 4.08s/it: fp32, compile
# 2.78s/it: fp32, no compile, sample=true, num_beams=20
# 2.07s/it: fp32, no compile, sample=false, num_beams=1
"""

import json
import os
import torch
import logging
import random
import warnings
import argparse
import cProfile
import pstats
import torch.multiprocessing as mp

from argparse import Namespace
from glob import glob
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOTAL_GPUS = 8
BOOTSTRAPS = 9
PROMPT = """Analyze the basketball player shown in the provided still tracklet frame and describe the following details:
1. Jersey Number: Identify the number on the player's jersey. If the player has no jersey, provide None.
Based on the frame description, produce an output prediction in the following JSON format:
{
  "jersey_number": "<predicted_jersey_number>",
}
[EOS]"""

def load_model_and_tokenizer(device: int = 0, args=None):
    compile_model = args.compile_model
    precision = args.precision
    try:
        logger.info("Loading model and tokenizer...")
        if precision == "int4":
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large-ft", 
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True,),
                device_map="cuda"
            ).eval()
        elif precision == "int8":
               model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large-ft", 
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True,),
                device_map="cuda"
            ).eval()
        else:     
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large-ft", 
                trust_remote_code=True,
                device_map="cuda"
            ).eval().to(device)
        # attempt to speed up inference by compiling model JIT
        if compile_model == 'True':
            logger.info("Compiling model...")
            model = torch.compile(model, mode='max-autotune')
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large-ft", 
            trust_remote_code=True
            )
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

def ocr(image_file_path: str, model, processor, device: int=0, args=None,) -> str:
    precision = args.precision
    bootstraped_results = []
    for _ in range(BOOTSTRAPS):
        image = Image.open(image_file_path)
        image = augment_image(image)
        if not image:
            return None
        
        # optionally use quantization
        if precision == "int4":
            inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(f'cuda:{device}')
            inputs['pixel_values'] = inputs['pixel_values'].half()
        elif precision == "int8":
            inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(f'cuda:{device}')
            inputs['pixel_values'] = inputs['pixel_values'].half()
        else:
            inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(f'cuda:{device}')
        # forward pass
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            early_stopping=False,
            num_beams=1,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task="<OCR>", image_size=(image.width, image.height))
        bootstraped_results.append(parsed_answer)
    return bootstraped_results

def ocr_dir(dir_fp: str, model, processor, device:int=0, args=None):
    img_paths = glob(os.path.join(dir_fp, '*.jpg'))
    results = {}
    for idx, img_path in enumerate(tqdm(img_paths, total=len(img_paths))):
        result = ocr(img_path, model, processor, device, args)
        if result:
            results[idx] = result
    logger.debug(f"Results for directory {dir_fp}: {results}")
    return dir_fp, results

def process_dir(
    dirs,
    device: int = 0, 
    all_results=None,
    args=None,
    ):
    model, processor = load_model_and_tokenizer(device, args)
    for dir_fp in dirs:
        logger.debug(f"Processing directory: {dir_fp}")
        dir_fp, dir_result = ocr_dir(dir_fp, model, processor, device, args)
        all_results[dir_fp] = dir_result
        ### MARK: BREAK ###
        break

def main(args: Namespace):
    
    tracklets_dir = args.tracklets_dir
    results_out_fp = args.results_out_fp
    num_gpus = args.num_gpus
    
    mp.set_start_method('spawn')
    dirs = glob(os.path.join(tracklets_dir, '*', '*'))
    manager = mp.Manager()
    all_results = manager.dict()
    processes = []
    
    # start_device = TOTAL_GPUS - num_gpus # ex: 8 - 1 = 7
    for rank in range(num_gpus):
        sub_dirs_arr = [dirs[j] for j in range(len(dirs)) if j % TOTAL_GPUS == rank]
        logger.debug(f"Rank: {rank}, sub_dirs_arr: {sub_dirs_arr}")
        p = mp.Process(
            target=process_dir,
            args=(
                sub_dirs_arr, 
                rank, 
                all_results,
                args
                ))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
        
    logger.debug(f"All results: {all_results}")
    # write all results
    with open(results_out_fp, 'w') as f:
        for dir_fp, results in all_results.items():
            f.write(f"{dir_fp}: {json.dumps(results)}\n")
    logger.info("All jersey numbers extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract jersey numbers from raw tracklets.')
    parser.add_argument('--tracklets_dir', type=str, required=True)
    parser.add_argument('--results_out_fp', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, required=False, default=1)
    parser.add_argument('--compile_model', type=str, required=False, default='False')
    parser.add_argument('--precision', type=str, required=False, default="fp32")
    parser.add_argument('--profile', type=str, required=False, default='False')
    args = parser.parse_args()
    # run main w/ profiling
    profile = args.profile
    if profile == 'True':
        profile_filename = 'profiling_results.prof'
        cProfile.run('main(args)', profile_filename)
        with open('profiling_stats.txt', 'w') as stream:
            p = pstats.Stats(profile_filename, stream=stream)
            p.sort_stats(pstats.SortKey.CUMULATIVE)
            p.print_stats()
    else:
        main(args)