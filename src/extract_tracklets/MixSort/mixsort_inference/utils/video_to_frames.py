import cv2
import os

# Constants
VIDEO_PATH = "/mnt/opr/vishravi/player-reidentification/sample-videos/clips/Rockets_2_Warriors_10_31_2015_clipped.mp4"
SAVE_FOLDER = "/mnt/opr/vishravi/player-reidentification/MixSort/datasets/sample_mot_test/dataset/train/v_100/img1/"

def deconstruct_video_to_frames(video_path, save_folder):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        return
    
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open the video file '{video_path}'.")
        return
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the current frame as an image with zero-padded frame number
        frame_filename = os.path.join(save_folder, f"{frame_count + 1:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    print(f"Extracted {frame_count} frames from '{video_path}' and saved to '{save_folder}'.")

if __name__ == "__main__":
    deconstruct_video_to_frames(VIDEO_PATH, SAVE_FOLDER)
