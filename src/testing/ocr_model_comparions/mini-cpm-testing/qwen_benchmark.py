import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from glob import glob
from PIL import Image

os.environ['HF_HOME'] = '/mnt/opr/levlevi/tmp'
torch.manual_seed(1234)

MODEL = 'deepseek-ai/deepseek-vl-7b-chat'
PROMPT = """Analyze the basketball player shown in the provided still tracklet frame and describe the following details:
1. Jersey Number: Identify the number on the player's jersey. If the player has no jersey, provide None.
Based on the frame description, produce an output prediction in the following JSON format:
{
  "jersey_number": "<predicted_jersey_number>",
}
[EOS]"""

test_set_dir = '/mnt/opr/levlevi/player-re-id/src/testing/ocr_model_comparions/nba_100_test_set'
image_file_paths = glob(os.path.join(test_set_dir, '*.jpg'))

# load qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
# load qwen model
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda", trust_remote_code=True).eval()

def perform_ocr(image_file_path: str) -> str:
    query = tokenizer.from_list_format([
    {'image': image_file_path},
    {'text': PROMPT},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response

ground_truth_labels = []
results = []
for image_file_path in tqdm(image_file_paths):
    # get human annotation
    ground_truth_label = image_file_path.split('/')[-1].split('_')[1].split('.')[0]
    ground_truth_labels.append(ground_truth_label)
    # perform ocr
    result = perform_ocr(image_file_path)
    results.append(result)
    
all_results_data = [gt + "__" + r + '\n' for gt, r in zip(ground_truth_labels, results)]
out_fp = '/mnt/opr/levlevi/player-re-id/src/testing/ocr_model_comparions/qwen_v_results.txt'
with open(out_fp, 'w') as f:
    f.writelines(all_results_data)