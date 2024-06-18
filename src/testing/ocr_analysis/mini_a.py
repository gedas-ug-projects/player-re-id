import torch
import json
import os
import logging

from chat import MiniCPMVChat, img2base64
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = 'openbmb/MiniCPM-Llama3-V-2_5'
# MODEL_NAME = 'openbmb/MiniCPM-Llama3-V-2_5-gguf'

# Load model and tokenizer
def load_model_and_tokenizer():
    try:
        logger.info("Loading model and tokenizer...")
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16)
        model = model.to(device='cuda')
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model.eval()
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise

# OCR function
def ocr(image: Image.Image, model, tokenizer):
    try:
        question = """This is an image of basketball player. Please identify the colors of this player's jersey and perform OCR on this image. Format the result as '{"jersey_colors": [colors], "jersey_number"': {number}}[EOS]"""
        msgs = [{'role': 'user', 'content': question}]
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            system_prompt='You are a helpful assistant.'
        )
        return '{' + res.split('[EOS]')[0].split('{')[-1]
    except Exception as e:
        logger.error(f"Failed to perform OCR on image: {e}")
        return None

# Load and convert image
def load_and_convert_image(fp: str) -> Image.Image:
    try:
        return Image.open(fp).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to load or convert image {fp}: {e}")
        return None

# Convert images to PIL with concurrent processing
def convert_images_to_pil(fps: List[str]) -> List[Image.Image]:
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_and_convert_image, fps))
    return [img for img in images if img is not None]

# Process images
def process_images(images: List[Image.Image], model, tokenizer):
    results = []
    for image in tqdm(images, total=len(images)):
        result = ocr(image, model, tokenizer)
        if result:
            results.append(result)
    return results

# Process directory
def ocr_dir(dir_fp: str, model, tokenizer):
    img_paths = glob(os.path.join(dir_fp, '*.jpg'))
    images = convert_images_to_pil(img_paths)
    results = process_images(images, model, tokenizer)
    return dir_fp, results

# Main function
def main():
    try:
        model, tokenizer = load_model_and_tokenizer()
        dirs = glob(os.path.join('/mnt/opr/levlevi/player-re-id/src/testing/ocr_analysis/sample_tracks_nba_100/__tracks_nba_50_grouped__/_hand-labeled', '*'))
        all_results = []
        with open('out.txt', 'w') as f:
            for dir_fp in dirs:
                logger.info(f"Processing directory: {dir_fp}")
                dir_result = ocr_dir(dir_fp, model, tokenizer)
                f.write(f"{dir_result[0]}: {json.dumps(dir_result[1])}\n")
                all_results.append(dir_result)
        logger.info("Processing completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()