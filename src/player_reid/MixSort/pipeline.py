import os
import shutil
import concurrent.futures as cf
import warnings

from glob import glob
from run_inference import generate_player_tracks
from utils.convert_vid_coco import format_video_to_coco_dataset

TEMP_COCO_DIR = '/mnt/opr/levlevi/player-re-id/src/player_reid/ocr_analysis/temp_dir_2'
warnings.simplefilter("ignore", category=UserWarning)

def process_video(fp, track_output_dir, rank):
    vid_name = os.path.basename(fp).replace('.mp4', '').lower()
    coco_dst_path = os.path.join(TEMP_COCO_DIR, vid_name)
    track_dst_path = os.path.join(track_output_dir, f"{vid_name}.txt")
    format_video_to_coco_dataset(fp, coco_dst_path)
    generate_player_tracks(coco_dst_path, track_dst_path, rank)
    shutil.rmtree(coco_dst_path)

def process_dir(videos_dir, track_output_dir, num_videos=1, rank:int = 0):
    video_files = glob(os.path.join(videos_dir, "*.mp4"))
    if num_videos > 0:
        num_videos = int((1/8) * len(video_files))
        start_idx = rank * num_videos
        end_idx = start_idx + num_videos if start_idx + num_videos < len(video_files) else len(video_files)
        video_files = video_files[start_idx:end_idx]
        print(f"Rank {rank} processing videos {start_idx} to {end_idx}")
    for fp in video_files:
        process_video(fp, track_output_dir, rank)
        
    # with cf.ProcessPoolExecutor(max_workers=1) as executor:
    #     futures = []
    #     for idx, fp in enumerate(video_files):
    #         rank = idx % 1  # Distribute work among 8 GPUs
    #         futures.append(executor.submit(process_video, fp, track_output_dir, rank))
    #     # Ensure all futures are completed
    #     for future in cf.as_completed(futures):
    #         future.result()

if __name__ == "__main__":
    
    # test dir
    # vids_dir = '/playpen-storage/levlevi/player-re-id/__old__/sample_vids'
    # nba 15'-16' replays
    vids_dir = '/mnt/sun/levlevi/nba-plus-statvu-dataset/game-replays'
    tracks_dir = '/mnt/sun/levlevi/nba-plus-statvu-dataset/player-tracklets'
    process_dir(vids_dir, tracks_dir, rank=3)
