import os
from glob import glob
from run_inference import generate_player_tracks
from utils.convert_vid_coco import format_video_to_coco_dataset

TEMP_COCO_DIR = '/playpen-storage/levlevi/player-re-id/src/player_reid/ocr_analysis/temp_dir'


def process_dir(videos_dir, track_output_dir, num_videos=-1):
    video_files = glob(os.path.join(videos_dir, "*"))
    if num_videos > 0:
        video_files = video_files[:num_videos]
    for fp in video_files:
        vid_name = os.path.basename(fp).replace('.mp4', '').lower()
        coco_dst_path = os.path.join(TEMP_COCO_DIR, vid_name)
        track_dst_path = os.path.join(track_output_dir, f"{vid_name}.txt")
        format_video_to_coco_dataset(fp, coco_dst_path)
        generate_player_tracks(coco_dst_path, track_dst_path)
        ## TODO: remove temp files
        

if __name__ == "__main__":
    
    # vids_dir = '/playpen-storage/levlevi/player-re-id/src/player_reid/ocr_analysis/sample_vids'
    vids_dir = '/mnt/sun/levlevi/nba-plus-statvu-dataset/game-replays'
    tracks_dir = '/playpen-storage/levlevi/player-re-id/src/player_reid/ocr_analysis/full_video_tracks'
    num_videos = 5
    process_dir(vids_dir, tracks_dir)