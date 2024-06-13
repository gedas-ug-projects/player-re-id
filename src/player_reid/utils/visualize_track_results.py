import cv2
import csv
import random

def generate_random_color():
    """Generate a random color in BGR format."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def viz_track_results(video_fp: str, results_fp: str, output_fp: str):
    # Read tracking results from file
    tracking_results = {}
    with open(results_fp, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            frame = int(row[0])
            entity_id = int(row[1])
            x1 = float(row[2])
            y1 = float(row[3])
            width = float(row[4])
            height = float(row[5])
            # The rest of the row elements are ignored as per given format
            if frame not in tracking_results:
                tracking_results[frame] = []
            tracking_results[frame].append((entity_id, x1, y1, width, height))

    # Assign random colors to each unique entity ID
    entity_colors = {}
    for frame in tracking_results:
        for entity_id, _, _, _, _ in tracking_results[frame]:
            if entity_id not in entity_colors:
                entity_colors[entity_id] = generate_random_color()

    # Open the video file
    cap = cv2.VideoCapture(video_fp)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_fp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process each frame
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in tracking_results:
            for (entity_id, x1, y1, width, height) in tracking_results[frame_num]:
                # Draw the bounding box
                top_left = (int(x1), int(y1))
                bottom_right = (int(x1 + width), int(y1 + height))
                color = entity_colors[entity_id]
                cv2.rectangle(frame, top_left, bottom_right, color, 3)  # Thicker lines
                # Put the ID text above the bounding box
                cv2.putText(frame, str(entity_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)
        frame_num += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == '__main__':
    input_video_fp = '/playpen-storage/levlevi/player-re-id/src/player_reid/sample-videos/clips/Rockets_2_Warriors_10_31_2015_clip_1.mp4'
    track_results_fp = '/playpen-storage/levlevi/player-re-id/src/player_reid/testing/datasets/nba/track_results/.txt'
    out_fp = '/playpen-storage/levlevi/player-re-id/src/player_reid/testing/datasets/nba/viz/viz.mp4'
    viz_track_results(input_video_fp, track_results_fp, out_fp)
