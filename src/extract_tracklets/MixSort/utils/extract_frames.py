import gc
import cv2
import os
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def save_frame(frame, save_path, overwrite):
    if not os.path.exists(save_path) or overwrite:
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    del frame

def extract_frames(vr, video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    
    # basic idea
        # create new vr for every 100 frames
    
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    assert os.path.exists(video_path)
                    
    saved_count = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        for index in range(start, end):
            try:
                frame = vr[index]
                save_path = os.path.join(frames_dir, "{:010d}.jpg".format(index))
                executor.submit(save_frame, frame.asnumpy(), save_path, overwrite)
                saved_count += 1
                del frame
            except:
                print(f"Error: could not open frame at index {index}")
    return saved_count

def video_to_frames(video_path, frames_dir, overwrite=False, every=1):
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    video_dir, video_filename = os.path.split(video_path)
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)
    print("Extracting Frames From {}".format(video_filename))
    
    vr = VideoReader(video_path, ctx=cpu(0))
    end = len(vr)
        
    step_size = 1000
    for index in tqdm(range(0, end-step_size, step_size), desc="Extracting Frames"):
        s = index
        e = index + step_size
        extract_frames(vr, video_path, frames_dir, overwrite=overwrite, every=every, start=s, end=e)
        
    return os.path.join(frames_dir, video_filename)

if __name__ == '__main__':
    video_to_frames(video_path='test.mp4', frames_dir='test_frames', overwrite=False, every=5)