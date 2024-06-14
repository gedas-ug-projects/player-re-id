import os
from glob import glob

VIDEOS_DIR = '/mnt/sun/levlevi/nba-plus-statvu-dataset/game-replays'
TRACKS_DIR = '/playpen-storage/levlevi/player-re-id/src/player_reid/ocr_analysis/full_video_tracks'

"""
ffmpeg_command = [
                    "ffmpeg", "-ss", str(start_time), "-i", video_file,
                    "-t", "10", "-c:v", "copy", "-c:a", "copy", output_file
                ]
"""


def sample_player_track_vids(videos_dir: str, tracks_dir: str, out_dir: str, num_samples: int =-1):
    tracks_fps = glob(os.path.join(tracks_dir, "*.txt"))
    print(tracks_fps)
    return


if __name__ == "__main__":
    out_dir = '/playpen-storage/levlevi/player-re-id/src/player_reid/ocr_analysis/sample_vids_from_full_length'
    sample_player_track_vids(VIDEOS_DIR, TRACKS_DIR, out_dir, 100)