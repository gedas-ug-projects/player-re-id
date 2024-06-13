import os
import random
import subprocess
from glob import glob

def get_video_duration(video_path):
    """Get the duration of a video file in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)

def sample_clips(src_dir: str, dst_dir: str):
    """Sample 100 random 10s clips from videos in src_dir and save them to dst_dir."""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    video_files = glob(os.path.join(src_dir, "*"))[0:100]
    clip_count = 0

    for video_file in video_files:
        duration = get_video_duration(video_file)
        if duration > 10:
            for _ in range(100 // len(video_files)):
                start_time = random.uniform(0, duration - 10)
                output_file = os.path.join(dst_dir, f"clip_{clip_count}.mp4")

                ffmpeg_command = [
                    "ffmpeg", "-ss", str(start_time), "-i", video_file,
                    "-t", "10", "-c:v", "copy", "-c:a", "copy", output_file
                ]

                subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                clip_count += 1
                if clip_count >= 100:
                    return

    # If we haven't reached 100 clips and there are more videos, continue sampling
    remaining_clips = 100 - clip_count
    while remaining_clips > 0:
        for video_file in video_files:
            if remaining_clips <= 0:
                break

            duration = get_video_duration(video_file)
            
            if duration > 10:
                start_time = random.uniform(0, duration - 10)
                output_file = os.path.join(dst_dir, f"clip_{clip_count}.mp4")

                ffmpeg_command = [
                    "ffmpeg", "-ss", str(start_time), "-i", video_file,
                    "-t", "10", "-c:v", "copy", "-c:a", "copy", output_file
                ]

                subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                clip_count += 1
                remaining_clips -= 1

src_dir = '/mnt/sun/levlevi/nba-plus-statvu-dataset/game-replays'
dst_dir = '/playpen-storage/levlevi/player-re-id/src/player_reid/sample-videos'
sample_clips(src_dir, dst_dir)