from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO('runs/detect/train16/weights/best.pt')  # Path to your trained weights

# Load image or video
input_path = 'input/videos/OTV1/mario.mp4'  # Change to your test video path if needed
image = cv2.imread(input_path)

# Perform inference
results = model.predict(source=image, conf=0.5)

# Visualize detections
for result in results:
    boxes = result.boxes  # Get detected bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy)
        label = f"{model.names[int(box.cls)]} {box.conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
