import os
import cv2
from glob import glob
import shutil
from convert_sportsmot_to_coco import convert_mot_to_coco

def create_mot_dataset(video_folder, output_folder, frame_rate=None):
    # Create the necessary folders
    if os.path.exists(output_folder):
        # If the output directory exists, remove it and create a new one
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    dataset_folder = os.path.join(output_folder, 'dataset/train')
    splits_folder = os.path.join(output_folder, 'splits_txt')
    
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(splits_folder, exist_ok=True)

    train_txt_path = os.path.join(splits_folder, 'train.txt')

    video_paths = glob(os.path.join(video_folder, '*.mp4'))  # Adjust the video extension if necessary

    with open(train_txt_path, 'w') as train_txt_file:
        for video_path in video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_folder = os.path.join(dataset_folder, video_name)
            img1_folder = os.path.join(video_output_folder, 'img1')
            gt_folder = os.path.join(video_output_folder, 'gt')
            
            os.makedirs(img1_folder, exist_ok=True)
            os.makedirs(gt_folder, exist_ok=True)
            
            gt_txt_path = os.path.join(gt_folder, 'gt.txt')
            with open(gt_txt_path, 'w') as f:
                f.write('1, 0, 1, 1, 1, 1, 1, 1, 1\n')
                f.write('2, 0, 1, 1, 1, 1, 1, 1, 1\n')
            
            cap = cv2.VideoCapture(video_path)
            if frame_rate is None:
                frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            seqinfo_path = os.path.join(video_output_folder, 'seqinfo.ini')
            
            with open(seqinfo_path, 'w') as seqinfo_file:
                seqinfo_file.write('[Sequence]\n')
                seqinfo_file.write(f'name={video_name}\n')
                seqinfo_file.write(f'imDir=img1\n')
                seqinfo_file.write(f'frameRate={frame_rate}\n')
                seqinfo_file.write(f'seqLength={total_frames}\n')
                seqinfo_file.write(f'imWidth={width}\n')
                seqinfo_file.write(f'imHeight={height}\n')
                seqinfo_file.write(f'imExt=.jpg\n')
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = os.path.join(img1_folder, f'{frame_idx:06d}.jpg')
                cv2.imwrite(frame_filename, frame)
                frame_idx += 1
            
            cap.release()
            train_txt_file.write(f'{video_name}\n')

def create_coco_from_mot(mot_output_folder, coco_output_folder):
    if os.path.exists(coco_output_folder):
        # If the output directory exists, remove it and create a new one
        shutil.rmtree(coco_output_folder)
    os.makedirs(coco_output_folder, exist_ok=True)
    
    annotations_folder = os.path.join(coco_output_folder, 'annotations')
    os.makedirs(annotations_folder, exist_ok=True)
    
    mot_dataset_folder = os.path.join(mot_output_folder, 'dataset/')

    convert_mot_to_coco(mot_dataset_folder, annotations_folder)
    
    # Copy the dataset from MOT to COCO structure
    mot_dataset_folder = os.path.join(mot_output_folder, 'dataset/train')
    coco_dataset_folder = os.path.join(coco_output_folder, 'val')
    shutil.copytree(mot_dataset_folder, coco_dataset_folder, dirs_exist_ok=True)

if __name__ == "__main__":
    video_folder = '/mnt/opr/vishravi/player-reidentification/sample-videos/clips/'  # Replace with the path to your input folder of videos
    mot_output_folder = '/mnt/opr/vishravi/player-reidentification/MixSort/datasets/sample_multi_mot_auto/'  # Replace with the path to your desired MOT output folder
    coco_output_folder = '/mnt/opr/vishravi/player-reidentification/MixSort/datasets/sample_multi_coco_auto/'  # Replace with the path to your desired COCO output folder
    
    create_mot_dataset(video_folder, mot_output_folder)
    create_coco_from_mot(mot_output_folder, coco_output_folder)
