import cv2
import numpy as np
import os
import random

output_images_dir = "./synthetic_backgrounds"
output_labels_dir = "./synthetic_labels"
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Generate 10 synthetic backgrounds
for i in range(10):
    # Create random background
    background = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    image_path = os.path.join(output_images_dir, f"background_{i}.jpg")
    cv2.imwrite(image_path, background)

    # Create random bounding box annotations
    with open(os.path.join(output_labels_dir, f"background_{i}.txt"), "w") as label_file:
        for _ in range(random.randint(1, 5)):  # Random number of bounding boxes
            x_center = random.uniform(0.1, 0.9)
            y_center = random.uniform(0.1, 0.9)
            width = random.uniform(0.1, 0.3)
            height = random.uniform(0.1, 0.3)
            label_file.write(f"1 {x_center} {y_center} {width} {height}\n")
