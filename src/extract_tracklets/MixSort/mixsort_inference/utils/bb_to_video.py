import cv2
import os
import numpy as np
import argparse

BOUNDING_BOX_FILE_PATH = "YOLOX_outputs/yolox_x_sample_test/track_results/v_100.txt"
FRAME_DIR = "datasets/sample_coco_test/val/v_100/img1"
FRAME_RATE = 30  # if using the same nba data, it's probably this, but double check if the video input length != output length

# Define a function to generate a unique color for each ID
# rand color
def get_color(id):
    np.random.seed(id)
    return tuple(np.random.randint(0, 255, 3).tolist())

# Load bounding box data
bounding_boxes = np.loadtxt(BOUNDING_BOX_FILE_PATH, delimiter=',')

# Parse bounding box data
parsed_bboxes = []

## TODO: legend for bbx files
for parts in bounding_boxes:
    frame = int(parts[0])
    id = int(parts[1])
    bb_left = float(parts[2])
    bb_top = float(parts[3])
    bb_width = float(parts[4])
    bb_height = float(parts[5])
    parsed_bboxes.append((frame, id, bb_left, bb_top, bb_width, bb_height))

# Directory containing frames
frame_dir = FRAME_DIR

# Output video file path (same directory as BOUNDING_BOX_FILE_PATH)
output_video = os.path.join(os.path.dirname(BOUNDING_BOX_FILE_PATH), "output_video.mp4")

# Get frame size
frame_sample = cv2.imread(os.path.join(frame_dir, "000001.jpg"))
height, width, layers = frame_sample.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, FRAME_RATE, (width, height))

# Process frames
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
frame_files = [os.path.join(frame_dir, f) for f in frame_files]

# Dictionary to store bounding boxes by frame
bboxes_by_frame = {}
for bbox in parsed_bboxes:
    frame, id, bb_left, bb_top, bb_width, bb_height = bbox
    if frame not in bboxes_by_frame:
        bboxes_by_frame[frame] = []
    bboxes_by_frame[frame].append((id, bb_left, bb_top, bb_width, bb_height))

# Read each frame and draw bounding boxes
for i, frame_file in enumerate(frame_files):
    frame = cv2.imread(frame_file)
    frame_index = i + 1  # since count starts at 1

    if frame_index in bboxes_by_frame:
        for bbox in bboxes_by_frame[frame_index]:
            id, bb_left, bb_top, bb_width, bb_height = bbox
            color = get_color(id)
            top_left = (int(bb_left), int(bb_top))
            bottom_right = (int(bb_left + bb_width), int(bb_top + bb_height))
            cv2.rectangle(frame, top_left, bottom_right, color, 2)
            cv2.putText(frame, str(id), (int(bb_left), int(bb_top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    out.write(frame)

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print(f"Video has been rendered successfully and saved to {output_video}.")
