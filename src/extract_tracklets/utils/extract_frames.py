import cv2
import os
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def save_frame(frame, save_path, overwrite):
    if not os.path.exists(save_path) or overwrite:
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    video_dir, video_filename = os.path.split(video_path)
    assert os.path.exists(video_path)
    
    vr = VideoReader(video_path, ctx=cpu(0))
                     
    if start < 0:
        start = 0
    if end < 0:
        end = len(vr)
    
    frames_list = list(range(start, end, every))
    saved_count = 0
    
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)
    
    if every > 25 and len(frames_list) < 1000:
        frames = vr.get_batch(frames_list).asnumpy()
        
        with ThreadPoolExecutor() as executor:
            for index, frame in zip(frames_list, frames):
                save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(index))
                executor.submit(save_frame, frame, save_path, overwrite)
                saved_count += 1
    else:
        with ThreadPoolExecutor() as executor:
            for index in tqdm(range(start, end), desc="Extracting Frames"):
                frame = vr[index]
                if index % every == 0:
                    save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(index))
                    executor.submit(save_frame, frame.asnumpy(), save_path, overwrite)
                    saved_count += 1
    
    return saved_count

def video_to_frames(video_path, frames_dir, overwrite=False, every=1):
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    video_dir, video_filename = os.path.split(video_path)
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)
    print("Extracting frames from {}".format(video_filename))
    extract_frames(video_path, frames_dir, overwrite=overwrite, every=every)
    return os.path.join(frames_dir, video_filename)

if __name__ == '__main__':
    video_to_frames(video_path='test.mp4', frames_dir='test_frames', overwrite=False, every=5)
