import cv2
import os

# Constants
VIDEO_PATH = '/mnt/opr/vishravi/player-reidentification/sample-videos/Rockets_2_Warriors_10_31_2015.mp4'#"path/to/your/video.mp4"
SAVE_PATH =  '/mnt/opr/vishravi/player-reidentification/sample-videos/clips/Rockets_2_Warriors_10_31_2015_clipped.mp4' #path/to/save/clipped/video.mp4"
CLIP_START_TIME = 10  # Start time in seconds
CLIP_DURATION = 20    # Duration of the clip in seconds

def clip_video(video_path, save_path, start_time, duration):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        return
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open the video file '{video_path}'.")
        return
    
    # Get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the starting frame and the ending frame
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    
    # Set the video to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    current_frame = start_frame
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video clipped and saved to '{save_path}'.")

if __name__ == "__main__":
    clip_video(VIDEO_PATH, SAVE_PATH, CLIP_START_TIME, CLIP_DURATION)
