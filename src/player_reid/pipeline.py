import os
import shutil
import concurrent.futures as cf

from glob import glob
from run_inference import generate_player_tracks
from utils.convert_vid_coco import format_video_to_coco_dataset

TEMP_COCO_DIR = '/playpen-storage/levlevi/player-re-id/src/player_reid/ocr_analysis/temp_dir'


def process_video(fp, track_output_dir, rank):
    vid_name = os.path.basename(fp).replace('.mp4', '').lower()
    coco_dst_path = os.path.join(TEMP_COCO_DIR, vid_name)
    track_dst_path = os.path.join(track_output_dir, f"{vid_name}.txt")
    
    format_video_to_coco_dataset(fp, coco_dst_path)
    generate_player_tracks(coco_dst_path, track_dst_path, rank)
    shutil.rmtree(coco_dst_path)


def process_dir(videos_dir, track_output_dir, num_videos=-1):
    video_files = glob(os.path.join(videos_dir, "*.mp4"))
    if num_videos > 0:
        video_files = video_files[:num_videos]

    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for idx, fp in enumerate(video_files):
            rank = idx % 8  # Distribute work among 8 GPUs
            futures.append(executor.submit(process_video, fp, track_output_dir, rank))
        
        # Ensure all futures are completed
        for future in cf.as_completed(futures):
            future.result()
            

if __name__ == "__main__":
    
    # vids_dir = '/playpen-storage/levlevi/player-re-id/__old__/6_13_24_player-reidentification/sample-videos/clips'
    vids_dir = '/mnt/sun/levlevi/nba-plus-statvu-dataset/game-replays'
    tracks_dir = '/playpen-storage/levlevi/player-re-id/src/data/full_game_tracks'
    process_dir(vids_dir, tracks_dir)