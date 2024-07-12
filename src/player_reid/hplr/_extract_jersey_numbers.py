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
import time
from typing import Optional, List

from argparse import Namespace
from glob import glob
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from optimum.bettertransformer import BetterTransformer
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTOptimizer

from transformers import BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.cuda.amp import autocast

PROMPT = """Identify the jersey number of the basketball player in the frame. If none, return None. Output only the digits:
<jersey_number>
[EOS]"""
BOOTSTRAPS = 1

warnings.simplefilter("ignore", FutureWarning)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FlorenceModel:

    # ensure we only load our model into memory once
    global _model, _processor, _compile_model, _model_varient, _half
    _model = None
    _processor = None
    _compile_model = None
    _model_varient = None
    _half = None

    @staticmethod
    def load_model_and_tokenizer(args=None):
        if (
            _model is not None
            and _processor is not None
            and _compile_model == args.compile_model
            and _model_varient == args.model_varient
            and _half == args.half
        ):
            return _model, _processor
        compile_model = args.compile_model
        model_varient = args.model_varient
        half = args.half
        try:
            logger.info("Loading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(
                f"microsoft/Florence-2-{model_varient}-ft",  # either 'base' or 'large'
                trust_remote_code=True,
                device_map="cuda",
            ).eval()
            processor = AutoProcessor.from_pretrained(
                f"microsoft/Florence-2-{model_varient}-ft",
                trust_remote_code=True,
            )
            if compile_model == "True":
                model = torch.compile(model)
            if args.half:
                model = model.half()
            _model = model
            _processor = processor
            _compile_model = compile_model
            _model_varient = model_varient
            return model, processor
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
        raise


def ocr(
    image_file_paths: List[str],
    model,
    processor,
    device: int = 0,
    args: Optional[dict] = None,
) -> Optional[List[str]]:
    def load_image(fp):
        try:
            image = Image.open(fp)
            image.load()
            return image
        except Exception as e:
            logger.error(f"Failed to load image {fp}: {e}")
            return None
        
    # all ocr results
    bootstraped_results = []
    # load images (pretty quick)
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_file_paths))
    images = [img for img in images if img is not None]
    if not images:
        logger.error("No valid images loaded.")
        return None
    prompts = [PROMPT] * len(images)
    inputs = processor(text=prompts, images=images, return_tensors="pt")
    # copy inputs to device
    input_ids = inputs["input_ids"].to(device, non_blocking=True)
    pixel_values = inputs["pixel_values"].to(device, non_blocking=True).half()
    del inputs
    # forward pass
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=5,
            do_sample=False,
            early_stopping=False,
            num_beams=BOOTSTRAPS,
            num_return_sequences=BOOTSTRAPS,
        )
    # decode the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # post-process the output
    for gt, image in zip(generated_text, images):
        parsed_answer = processor.post_process_generation(
            gt, task="<OCR>", image_size=(image.width, image.height)
        )
        bootstraped_results.append(parsed_answer)
    return bootstraped_results if bootstraped_results else None
