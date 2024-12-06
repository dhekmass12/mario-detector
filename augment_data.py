import albumentations as A
from albumentations import (
    Compose, HorizontalFlip, Rotate, RandomBrightnessContrast, CoarseDropout, Perspective, ElasticTransform
)
import cv2
import os

# Paths
input_image_path = 'coco_converted/images/mario.png'  # Path to the original image
input_label_path = 'coco_converted/labels/mario.txt'  # Path to the YOLOv8 OBB annotation file
output_train_images = './dataset/images/train'
output_train_labels = './dataset/labels/train'
output_val_images = './dataset/images/val'
output_val_labels = './dataset/labels/val'
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(output_train_labels, exist_ok=True)
os.makedirs(output_val_images, exist_ok=True)
os.makedirs(output_val_labels, exist_ok=True)

image = cv2.imread(input_image_path)
height, width, _ = image.shape

# Load and parse the YOLOv8 OBB annotation
with open(input_label_path, "r") as f:
    lines = f.readlines()

# Parse polygon annotations
annotations = []
for line in lines:
    parts = list(map(float, line.strip().split()))
    class_id = int(parts[0])
    polygon = [(parts[i] * width, parts[i + 1] * height) for i in range(1, len(parts), 2)]  # Denormalize
    annotations.append({"class_id": class_id, "polygon": polygon})

# Augmentation for training set
train_augmentations = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=75, p=0.5),
    RandomBrightnessContrast(p=0.2),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

# Augmentation for validation set
val_augmentations = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=210, p=0.5),
    RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.6, p=0.2),
    Perspective(scale=(0.02, 0.05), p=0.5),
    CoarseDropout(max_holes=25, max_height=10, max_width=10, p=0.5),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

# Function to save augmented data
def save_augmented_data(augmented, filename, output_image_dir, output_label_dir, class_id):
    augmented_image = augmented["image"]
    augmented_polygon = augmented["keypoints"]

    # Filter keypoints within bounds
    augmented_polygon = [
        (x, y) for x, y in augmented_polygon if 0 <= x <= width and 0 <= y <= height
    ]

    # Skip if no valid keypoints remain
    if not augmented_polygon:
        print(f"Skipping {filename}: no valid keypoints after augmentation.")
        return

    # Save augmented image
    image_output_path = os.path.join(output_image_dir, filename)
    cv2.imwrite(image_output_path, augmented_image)

    # Normalize polygon and clip coordinates
    normalized_polygon = [
        f"{max(0, min(1, x / width))} {max(0, min(1, y / height))}"
        for x, y in augmented_polygon
    ]
    label_output_path = os.path.join(output_label_dir, f"{os.path.splitext(filename)[0]}.txt")
    with open(label_output_path, "w") as label_file:
        label_file.write(f"{class_id} {' '.join(normalized_polygon)}\n")

# Generate augmented training data
for i in range(100):
    for annotation in annotations:
        augmented = train_augmentations(image=image, keypoints=annotation["polygon"])
        save_augmented_data(augmented, f"mario_train_aug_{i}.jpg", output_train_images, output_train_labels, annotation["class_id"])

# Generate augmented validation data
for i in range(100):
    for annotation in annotations:
        augmented = val_augmentations(image=image, keypoints=annotation["polygon"])
        save_augmented_data(augmented, f"mario_val_aug_{i}.jpg", output_val_images, output_val_labels, annotation["class_id"])