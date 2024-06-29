import os
import json
import cv2
import numpy as np

from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pipeline import extract_roi_from_video


# TODO: also viz bounding box representing roi

def visualize_timestamps(video_path, timestamps_path, viz_path, tr_roi=None):

    print(f"Generating visualization for video at: {video_path}")
    with open(timestamps_path, 'r') as f:
        timestamps = json.load(f)
    reader = cv2.VideoCapture(video_path)
    frame_cnt = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width, fps = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(reader.get(
        cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(
            cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(
        viz_path, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))
    font = ImageFont.truetype('utilities/os-eb.ttf', 30)
    for frame_index in tqdm(range(frame_cnt)):
        ret, frame = reader.read()
        if not ret:
            break

        quarter, time_remaining = None, None
        minutes, seconds, decimal_seconds = -1, -1, -1
        index = str(frame_index)
        if index in timestamps:
            quarter = timestamps[str(index)]["quarter"]
            time_remaining = timestamps[str(index)]["time_remaining"]
            if time_remaining is not None:
                minutes = int(time_remaining) // 60
                seconds = int((time_remaining - (minutes * 60)))
                decimal_seconds = int(
                    time_remaining - (minutes * 60) - seconds) * 10
        img = Image.fromarray(frame)
        draw: ImageDraw = ImageDraw.Draw(img)
        draw.text(
            (10, 10), text=f"Q: {quarter} T: {minutes:02d}:{seconds:02d}.{decimal_seconds}", font=font, fill=(255, 255, 255))
        writer.write(np.array(img))

    writer.release()


def visualize_roi(video_path, viz_path, roi):

    reader = cv2.VideoCapture(video_path)
    frame_cnt = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width, fps = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(reader.get(
        cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(
            cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(
        viz_path, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

    x1, y1, x2, y2 = None, None, None, None
    if roi is not None:
        x1, y1, x2, y2 = roi.tolist()

    color = (0, 0, 255)
    thickness = 2
    for _ in tqdm(range(frame_cnt)):
        ret, frame = reader.read()
        if not ret:
            break

        if x1 is not None:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        writer.write(np.array(frame))

    writer.release()
    reader.release()


def viz_dir_roi(in_dir: str, out_dir: str):
    in_dir_file_paths = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    for in_file_path in tqdm(in_dir_file_paths):
        out_file_path = os.path.join(out_dir, os.path.basename(in_file_path))
        roi = extract_roi_from_video(in_file_path)
        visualize_roi(in_file_path, out_file_path, roi)