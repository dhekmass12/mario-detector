import cv2
import os
from ultralytics import YOLO

video_path = "test video (open)/OTV5.mp4"  # Path to the input video
output_frames_dir = "./video_frames"
os.makedirs(output_frames_dir, exist_ok=True)

# Load the video
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Save each frame as an image
    frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    frame_count += 1
    print(f"Loaded frame count: {frame_count}")

cap.release()
print(f"Extracted {frame_count} frames from the video.")

# Load trained YOLO model
model = YOLO("runs/obb/train28/weights/best.pt")

# Paths
input_frames_dir = "./video_frames"
output_frames_dir = "./annotated_frames"
os.makedirs(output_frames_dir, exist_ok=True)

# Process each frame
frame_files = sorted(os.listdir(input_frames_dir))
for frame_file in frame_files:
    frame_path = os.path.join(input_frames_dir, frame_file)
    frame = cv2.imread(frame_path)

    # Run YOLO inference
    results = model(frame)

    # Annotate frame with predictions
    annotated_frame = results[0].plot()

    # Save annotated frame
    output_path = os.path.join(output_frames_dir, frame_file)
    cv2.imwrite(output_path, annotated_frame)

print(f"Processed {len(frame_files)} frames.")

input_frames_dir = "./annotated_frames"
output_video_path = "output_video.mp4"

# Get list of annotated frames
frame_files = sorted(os.listdir(input_frames_dir))
frame_path = os.path.join(input_frames_dir, frame_files[0])
frame = cv2.imread(frame_path)
height, width, _ = frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30  # Adjust FPS as needed
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write each frame to the video
for frame_file in frame_files:
    frame_path = os.path.join(input_frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()
print(f"Output video saved as {output_video_path}")
