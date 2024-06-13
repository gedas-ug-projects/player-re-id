import os
import subprocess
import json

from PIL import Image
from tqdm import tqdm

ANNOTATIONS_DIR = 'annotations'
VAL_DIR = 'val'
GT_ANNOTATIONS_DIR = 'gt'
FRAMES_DIR = 'frames'

ANNOTATIONS_FILE_NAME = 'val.json'
SEQINFO_FILE_NAME = 'seqinfo.ini'
GT_ANNOTATIONS_FP = 'gt.txt'

IMG_EXT = '.jpeg'

MIXSORT_FP = "/playpen-storage/levlevi/player-re-id/__vish__/player-reidentification/MixSort"
os.chdir(MIXSORT_FP)


def get_video_metadata(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,r_frame_rate', '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    metadata = json.loads(result.stdout)
    width = metadata['streams'][0]['width']
    height = metadata['streams'][0]['height']
    fps = eval(metadata['streams'][0]['r_frame_rate'])
    return fps, width, height


def extract_frames(video_path: str, to_path: str):
    assert os.path.exists(to_path)
    command = [
        'ffmpeg',
        '-i', video_path,
        os.path.join(to_path, '%06d.jpeg'),
        '-loglevel', 'panic'
    ]
    subprocess.run(command, check=True)


def create_annotations(video_path: str, image_dir: str, annotations_dir: str):

    vid_name = os.path.basename(video_path).replace('.mp4', '').lower()
    anotations = {
        'images': [],
        'annotations': [],
        'videos': {
            "id": 1,
            "file_name": vid_name
            },
        'categories':[{
            "id": 1,
            "name": "pedestrian"
        }]
    }

    frame_files = sorted([f for f in os.listdir(image_dir) if f.endswith(IMG_EXT)])
    for index, frame_file in enumerate(frame_files):
        frame_id = index + 1
        prev_image_id = frame_id - 1 if frame_id > 1 else -1
        next_image_id = frame_id + 1 if frame_id < len(frame_files) else -1
        frame_path = os.path.join(image_dir, frame_file)
        with Image.open(frame_path) as img:
            width, height = img.size
        image_data = {
            "file_name": frame_path,
            "id": frame_id,
            "frame_id": frame_id,
            "prev_image_id": prev_image_id,
            "next_image_id": next_image_id,
            "video_id": 1,
            "height": height,
            "width": width
        }
        anotations["images"].append(image_data)

    annotation_out_path = os.path.join(annotations_dir, ANNOTATIONS_FILE_NAME)
    with open(annotation_out_path, 'w') as f:
        json.dump(anotations, f, indent=2)
        

def create_misc_metadata(vid_out_dir: str, video_path: str, img_dir: str):

    vid_name = os.path.basename(video_path).replace('.mp4', '').lower()
    gt_dir = os.path.join(vid_out_dir, GT_ANNOTATIONS_DIR)
    os.makedirs(gt_dir, exist_ok=True)

    gt_txt_fp = os.path.join(gt_dir, GT_ANNOTATIONS_FP)
    with open(gt_txt_fp, 'w') as f:
        f.write('')

    frame_rate, im_width, im_height = get_video_metadata(video_path)
    num_images = len([f for f in os.listdir(img_dir) if f.endswith(IMG_EXT)])

    sequence_path = os.path.join(vid_out_dir, SEQINFO_FILE_NAME)
    sequence_str = f"[Sequence]\nname={vid_name}]\nimDir={img_dir}\nframeRate={int(frame_rate)}\nseqLength={num_images}\nimWidth={im_width}\nimHeight={im_height}\nimExt={IMG_EXT}"

    with open(sequence_path, 'w') as f:
        f.write(sequence_str)


def format_video_to_coco_dataset(video_path: str, to_path: str):

    # make dirs
    os.makedirs(to_path, exist_ok=True)
    os.makedirs(os.path.join(to_path, ANNOTATIONS_DIR), exist_ok=True)
    os.makedirs(os.path.join(to_path, VAL_DIR), exist_ok=True)

    vid_name = os.path.basename(video_path).replace('.mp4', '').lower()
    vid_dir = os.path.join(to_path, VAL_DIR, vid_name)
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(os.path.join(to_path, VAL_DIR, vid_name, GT_ANNOTATIONS_DIR), exist_ok=True)

    frames_dir = os.path.join(to_path, VAL_DIR, vid_name, FRAMES_DIR)
    os.makedirs(frames_dir, exist_ok=True)

    annotations_dir = os.path.join(to_path, ANNOTATIONS_DIR)
    extract_frames(video_path, frames_dir)
    create_annotations(video_path, frames_dir, annotations_dir)
    create_misc_metadata(vid_dir, video_path, frames_dir)   

if __name__ == "__main__":

    vid_path = "/playpen-storage/levlevi/player-re-id/__vish__/player-reidentification/sample-videos/clips/Rockets_2_Warriors_10_31_2015_clip_1.mp4"
    dst_path = "/playpen-storage/levlevi/player-re-id/__vish__/player-reidentification/MixSort/datasets/levi-test-ds-2"
    format_video_to_coco_dataset(vid_path, dst_path)