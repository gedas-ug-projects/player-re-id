import os
import shutil
import warnings
import argparse

MIXSORT_DIR = '/mnt/opr/levlevi/player-re-id/src/player_reid/MixSort'
os.chdir(MIXSORT_DIR)

from glob import glob
from run_inference import generate_player_tracks
from utils.convert_vid_coco import format_video_to_coco_dataset

TEMP_COCO_DIR = '/playpen-storage/levlevi/temp'
os.makedirs(TEMP_COCO_DIR, exist_ok=True)
warnings.simplefilter("ignore", category=UserWarning)

def process_video(fp, track_output_dir, rank):
    vid_name = os.path.basename(fp).replace('.mp4', '').lower()
    coco_dst_path = os.path.join(TEMP_COCO_DIR, vid_name)
    track_dst_path = os.path.join(track_output_dir, f"{vid_name}.txt")

    # Check if the track output file already exists
    if os.path.exists(track_dst_path):
        print(f"Skipping {vid_name}: {track_dst_path} already exists.")
        return
    
    format_video_to_coco_dataset(fp, coco_dst_path)
    generate_player_tracks(coco_dst_path, track_dst_path, rank)
    shutil.rmtree(coco_dst_path)
def process_dir(videos_dir, track_output_dir, num_videos=1, rank=0):
    video_files = glob(os.path.join(videos_dir, "*.mp4"))
    if num_videos > 0:
        num_videos = int((1/8) * len(video_files))
        start_idx = rank * num_videos
        end_idx = start_idx + num_videos if start_idx + num_videos < len(video_files) else len(video_files)
        video_files = video_files[start_idx:end_idx]
        print(f"Rank {rank} processing videos {start_idx} to {end_idx}")
    for fp in video_files:
        process_video(fp, track_output_dir, rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some videos.')
    parser.add_argument('--rank', type=int, required=True, help='Rank value to process a subset of videos')
    args = parser.parse_args()

    vids_dir = '/mnt/sun/levlevi/nba-plus-statvu-dataset/game-replays'
    tracks_dir = '/mnt/sun/levlevi/nba-plus-statvu-dataset/player-tracklets'
    process_dir(vids_dir, tracks_dir, rank=args.rank)