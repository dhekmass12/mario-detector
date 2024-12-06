import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs/obb/train4/weights/best.pt")

# Load video
video_path = "Test video (open)/OTV1.mp4"
cap = cv2.VideoCapture(video_path)
output_video_path = "output_video.mp4"

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Write frame to output video
    out.write(annotated_frame)

    # Optional: Display the frame
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved as {output_video_path}")
