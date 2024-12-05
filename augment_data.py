from albumentations import (
    Compose, HorizontalFlip, Rotate, RandomBrightnessContrast, CoarseDropout, Perspective, ElasticTransform
)
import cv2
import os

# Paths
image_path = './mario.png'
train_images_path = './dataset/images/train'
val_images_path = './dataset/images/val'
train_labels_path = './dataset/labels/train'
val_labels_path = './dataset/labels/val'

# Create directories
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# Load the original image
image = cv2.imread(image_path)

# Original label for Mario
label_content = "0 0.505455 0.502762 0.989091 0.994475"

# Augmentation for training set
train_augmentations = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=20, p=0.5),
    RandomBrightnessContrast(p=0.2),
])

# Augmentation for validation set
val_augmentations = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=20, p=0.5),
    RandomBrightnessContrast(p=0.2),
    Perspective(scale=(0.05, 0.1), p=0.5),
    ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
    CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.5),
])

# Generate training set
for i in range(100):
    augmented_train = train_augmentations(image=image)
    train_image = augmented_train["image"]
    train_image_path = os.path.join(train_images_path, f"mario_train_aug_{i}.jpg")
    train_label_path = os.path.join(train_labels_path, f"mario_train_aug_{i}.txt")
    cv2.imwrite(train_image_path, train_image)
    with open(train_label_path, 'w') as label_file:
        label_file.write(label_content)

# Generate validation set
for i in range(100):
    augmented_val = val_augmentations(image=image)
    val_image = augmented_val["image"]
    val_image_path = os.path.join(val_images_path, f"mario_val_aug_{i}.jpg")
    val_label_path = os.path.join(val_labels_path, f"mario_val_aug_{i}.txt")
    cv2.imwrite(val_image_path, val_image)
    with open(val_label_path, 'w') as label_file:
        label_file.write(label_content)
