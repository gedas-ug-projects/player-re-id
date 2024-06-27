from ultralytics import YOLO
from PIL import Image
from glob import glob
import ast
import sys

THRESH = 0.50

model = YOLO("/playpen-storage/levlevi/player-re-id/src/testing/ocr_model_comparisons/text_detection/jersey_num_det.pt")
img_dir = '/playpen-storage/levlevi/player-re-id/src/testing/ocr_model_comparisons/text_recognition/nba_100_test_set'
cropped_imgs_dir = '/playpen-storage/levlevi/player-re-id/src/testing/ocr_model_comparisons/text_detection/cropped_imgs_dir'
img_fps = glob(f'{img_dir}/*.jpg')
cropped_image_bounding_boxes = []

results = model(img_fps)  # return a list of Results objects
for idx, result in enumerate(results):
    x1_min = sys.maxsize
    x2_max = 0
    y1_min = sys.maxsize
    y2_max = 0
    json_obj = ast.literal_eval(result.tojson())
    if len(json_obj) == 0:
        cropped_image_bounding_boxes.append(None)
    for pred in json_obj:
        if pred.get('confidence') < THRESH:
            continue
        x1, y1, x2, y2 = [pred.get('box').get(k) for k in pred.get('box')]
        if x1 < x1_min:
            x1_min = x1
        if x2 > x2_max:
            x2_max = x2
        if y1 < y1_min:
            y1_min = y1
        if y2 > y2_max:
            y2_max = y2
    if x2_max == 0:
        cropped_image_bounding_boxes.append(None)
    else:
        cropped_image_bounding_boxes.append((int(x1_min), int(y1_min), int(x2_max), int(y2_max)))
print(cropped_image_bounding_boxes)