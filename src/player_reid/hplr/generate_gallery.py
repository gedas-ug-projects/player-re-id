import argparse

from _extract_frames_from_tracklets import format_tracklets_for_reid, extract_frames_from_tracklet_df
from _extract_jersey_numbers import FlorenceModel, ocr

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracklets_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)