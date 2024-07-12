import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

GAME_REPLAYS = "/mnt/sun/levlevi/nba-plus-statvu-dataset/game-replays"
PLAYER_TRACKLETS = "/mnt/sun/levlevi/nba-plus-statvu-dataset/player-tracklets"
MIN_ARR_SIZE = 30
NAMES = ["frame", "entity_id", "x1", "y1", "width", "height", "conf"]


def format_tracklets_for_reid(tracklet_fp: str, video_file_paths_map) -> pd.DataFrame:
    """
    Given a path to a raw tracklet `.txt` file, return a formated DataFrame object.
    """

    # read csv -> df
    df = pd.read_csv(tracklet_fp, sep=",", names=NAMES, usecols=NAMES[:7])
    # extract video file path
    video_path_name = os.path.basename(tracklet_fp).replace(".txt", ".mp4")
    video_file_path = video_file_paths_map.get(video_path_name)
    rows = []
    temp_entity_id = 0
    # all unique tracklet entity ids
    unique_entities_df = df["entity_id"].unique()
    for entity_id in unique_entities_df:
        df_entity = df[df["entity_id"] == entity_id]
        last_frame_idx = -1
        temp_entity_rows = []
        # filter out entities w/ less than 30 consecutive frames
        for row in df_entity.itertuples(index=False):
            temp_frame_idx = row.frame
            if last_frame_idx != -1 and temp_frame_idx != last_frame_idx + 1:
                if len(temp_entity_rows) >= MIN_ARR_SIZE:
                    rows.append(
                        [
                            video_file_path,
                            temp_entity_id,
                            pd.DataFrame(
                                temp_entity_rows,
                                columns=[
                                    "video_file_path",
                                    "frame",
                                    "x1",
                                    "y1",
                                    "width",
                                    "height",
                                    "conf",
                                ],
                            ),
                        ]
                    )
                    temp_entity_id += 1
                temp_entity_rows = []
            temp_entity_rows.append(
                [
                    video_file_path,
                    row.frame,
                    row.x1,
                    row.y1,
                    row.width,
                    row.height,
                    row.conf,
                ]
            )
            last_frame_idx = temp_frame_idx
        if len(temp_entity_rows) >= MIN_ARR_SIZE:
            rows.append(
                [
                    video_file_path,
                    temp_entity_id,
                    pd.DataFrame(
                        temp_entity_rows,
                        columns=[
                            "video_file_path",
                            "frame",
                            "x1",
                            "y1",
                            "width",
                            "height",
                            "conf",
                        ],
                    ),
                ]
            )
            temp_entity_id += 1
    tracklets_df = pd.DataFrame(
        rows, columns=["video_path_name", "entity_id", "tracklet_dataframe"]
    )
    return tracklets_df


def save_cropped_image(cropped_img_path: str, cropped_img: np.ndarray) -> None:
    try:
        cv2.imwrite(cropped_img_path, cropped_img)
    except Exception as e:
        pass


def process_frame(frame: np.ndarray, row: pd.Series, output_dir: str) -> None:
    x1, y1, width, height = (
        int(row["x1"]),
        int(row["y1"]),
        int(row["width"]),
        int(row["height"]),
    )
    cropped_img = frame[y1 : y1 + height, x1 : x1 + width]
    cropped_img_path = os.path.join(output_dir, f"cropped_frame_{row['frame']}.jpg")
    save_cropped_image(cropped_img_path, cropped_img)


def save_cropped_images_from_video(
    output_dir: str, rows: pd.DataFrame, cap: cv2.VideoCapture
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    start_frame = rows.iloc[0]["frame"]
    end_frame = rows.iloc[-1]["frame"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    frame_idx = 0
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(rows) and current_frame == rows.iloc[frame_idx]["frame"]:
            process_frame(frame, rows.iloc[frame_idx], output_dir)
            frame_idx += 1
        current_frame += 1


def process_tracklet(
    output_dir: str, tracklet_df: pd.DataFrame, video_path: str
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    save_cropped_images_from_video(output_dir, tracklet_df, cap)
    cap.release()


def extract_frames_from_tracklet_df(tracklet_df: pd.DataFrame, dst_path: str) -> None:
    os.makedirs(dst_path, exist_ok=True)
    tasks = []
    with ThreadPoolExecutor() as executor:
        for entity_id, entity_df in enumerate(tracklet_df["tracklet_dataframe"]):
            entity_subdir = os.path.join(dst_path, f"{entity_id:06d}")
            os.makedirs(entity_subdir, exist_ok=True)
            video_path = entity_df.iloc[0]["video_file_path"]
            tasks.append(
                executor.submit(process_tracklet, entity_subdir, entity_df, video_path)
            )
        for future in tqdm(
            as_completed(tasks), total=len(tasks), desc="Extracting Tracklets"
        ):
            future.result()
