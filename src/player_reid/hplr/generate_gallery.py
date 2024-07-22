import argparse
import os
import random

from glob import glob
from _extract_frames_from_tracklets import (
    format_tracklets_for_reid,
    extract_frames_from_tracklet_df,
)
from _extract_jersey_numbers import FlorenceModel, ocr


def process_tracklet(args, fp: str, model, processor):
    results = []
    # 1. extract frames from tracklet
    try:
        video_file_paths = glob(args.replays_dir + "/*.mp4")
        video_file_paths_map = {
            os.path.basename(fp).lower(): fp for fp in video_file_paths
        }
        tracklet_df = format_tracklets_for_reid(fp, video_file_paths_map)
    except Exception as e:
        print(f"Failed to format tracklet {fp}: {e}")
        return
    try:
        tracklet_tmp_dir = os.path.join(args.temp_dir, os.path.basename(fp))
        extract_frames_from_tracklet_df(tracklet_df, tracklet_tmp_dir)
    except Exception as e:
        print(f"Failed to extract frames from tracklet {fp}: {e}")
    # 2. extract jersey numbers from each tracklet subdir
    try:
        subdir_paths = glob(os.path.join(tracklet_tmp_dir, "*"))
        for subdir_path in subdir_paths:
            frames_file_paths = glob(os.path.join(subdir_path, "*.jpg"))
            tracklet_results = ocr(args, frames_file_paths, model, processor)
            results.extend(tracklet_results)
    except Exception as e:
        print(f"Failed to extract jersey numbers: {e}")
        return
    # 3. save results as txt file
    out_fp = os.path.join(
        args.results_dir, os.path.basename(fp).replace(".txt", "_jersey_numbers.txt")
    )
    try:
        with open(os.path.join(args.results_dir, out_fp, "w")) as f:
            f.write("\n".join(results))
    except Exception as e:
        print(f"Failed to save results to {out_fp}: {e}")


def get_remaining_tracklet_file_paths(args):
    tracklet_file_paths = glob(args.tracklets_dir + "/*.txt")
    tmp_folder_names = [
        os.path.basename(x).replace(".txt", "") for x in glob(args.temp_dir + "/*")
    ]
    result_file_names = [
        os.path.basename(x).replace(".txt", "")
        for x in glob(args.results_dir + "/*.txt")
    ]
    # return a list of all full tracklet file paths that do not have a matching basename subdir
    # in the `tmp_dir` (without '.txt') folder or matching basename .txt file in the `results_dir`
    return [
        x
        for x in tracklet_file_paths
        if os.path.basename(x).replace(".txt", "") not in tmp_folder_names
        and os.path.basename(x).replace(".txt", "") not in result_file_names
    ]


def main(args):
    model, processor = FlorenceModel.load_model_and_tokenizer(args)
    while True:
        tracklet_file_paths = get_remaining_tracklet_file_paths(args)
        if len(tracklet_file_paths) == 0:
            break
        random.shuffle(tracklet_file_paths)
        process_tracklet(args, tracklet_file_paths[0], model, processor)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracklets_dir", type=str, required=True)
    parser.add_argument("--replays_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--temp_dir", type=str, required=True)
    parser.add_argument("--device", type=int, required=False, default=0)
    parser.add_argument("--half", type=str, required=False, default="False")
    parser.add_argument("--compile_model", type=str, required=False, default="False")
    parser.add_argument("--model_variant", type=str, required=False, default="base")
    args = parser.parse_args()
    main(args)
