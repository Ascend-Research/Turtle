import cv2
import os


"""
This script creates a side-by-side comparison video from pairs of input and predicted frames stored in a directory. 
A sliding line moves across the frames to visually compare the differences, and the resulting video is saved to an output file.
"""


# Directory path for Input and Restored frames
frames_dir = 'path to the low quality ad=nd high quality frames'

# Output video parameters
output_video_path = 'path_to_save_video/x.mp4'
fps = 20  # Set the frames per second for the output video



# Initialize video writer
frame_example = cv2.imread(os.path.join(frames_dir, os.listdir(frames_dir)[1]))
height, width, layers = frame_example.shape
print(height, width)
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Get sorted list of frame filenames
all_files = os.listdir(frames_dir)
input_frames = sorted([f for f in all_files if 'Input' in f], key=lambda x: int(x.split('_')[1]))
pred_frames = sorted([f for f in all_files if 'Pred' in f], key=lambda x: int(x.split('_')[1]))

# Total number of frames
total_frames = min(len(input_frames), len(pred_frames))

for i in range(total_frames):
    # Construct the filenames based on the sorted lists
    low_quality_filename = input_frames[i]
    high_quality_filename = pred_frames[i]

    # Read frames
    frame1 = cv2.imread(os.path.join(frames_dir, low_quality_filename))
    frame2 = cv2.imread(os.path.join(frames_dir, high_quality_filename))
    # print(frame1.shape,frame2.shape)

    # Compute the position of the sliding line
    slider_position = int((i / total_frames) * width)

    # Create a combined frame
    combined_frame = frame1.copy()
    combined_frame[:, :slider_position] = frame2[:, :slider_position]

    # Draw the sliding line
    cv2.line(combined_frame, (slider_position, 0), (slider_position, height), (0, 255, 0), 2)

    # Write the combined frame to the output video
    out.write(combined_frame)

# Release video writer
out.release()

print("Video has been created and saved as", output_video_path)
