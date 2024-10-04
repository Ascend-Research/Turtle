import cv2
import os
import shutil

"""
This script converts a video into individual frames and saves them to a specified directory.
Only a subset of frames is saved based on a defined interval to match the desired frame rate for further processing.
"""


# Path to the video file
video_path = 'video.mp4'

# Directory to save the frames
output_folder = 'output_dir'

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)



# Load the video
cap = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_second = 10
interval = int(fps / frames_per_second)

frame_count = 0
saved_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame if it's in the interval
    if frame_count % interval == 0:
        frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

cap.release()
print(f'Extracted {saved_frame_count} frames to {output_folder}')
