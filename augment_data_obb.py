import albumentations as A
from albumentations import Compose, HorizontalFlip, Rotate, RandomBrightnessContrast, CoarseDropout, Perspective, ElasticTransform
import cv2
import numpy as np
import os

# Paths
image_path = './mario_limited/images/mario.png'  # Path to the original image
train_images_path = './mario_limited_obb_white/images/train'
train_labels_path = './mario_limited_obb_white/labels/train'
val_images_path = './mario_limited_obb_white/images/val'
val_labels_path = './mario_limited_obb_white/labels/val'

# Create directories
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

def replace_transparent_with_white(image):
    """
    Replaces the transparent background in an RGBA image with a white background.
    :param image: Input image with an alpha channel (RGBA).
    :return: Image with transparency replaced by white (RGB).
    """
    if image.shape[-1] == 4:  # Check if the image has an alpha channel
        # Split the image into RGB and Alpha
        rgb_image = image[:, :, :3]
        alpha_channel = image[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

        # Create a white background
        white_background = np.ones_like(rgb_image, dtype=np.uint8) * 255

        # Blend the image with the white background using the alpha channel
        blended_image = (rgb_image * alpha_channel[:, :, None] + 
                         white_background * (1 - alpha_channel[:, :, None]))
        return blended_image.astype(np.uint8)  # Ensure image is uint8
    return image  # If no alpha channel, return the original image

# Load the original image
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Includes alpha channel if present
image = replace_transparent_with_white(image)  # Replace transparency with white

# Read the OBB annotation for Mario (YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4)
obb_label_content = "0 0.012108469716879703 0.01182654402102497 0.9600286704097477 0.01182654402102497 0.9600286704097477 0.9724047306176086 0.012108469716879703 0.9724047306176086"

# Parse OBB label content
def parse_obb_label(label_content):
    parts = list(map(float, label_content.strip().split()))
    class_id = int(parts[0])
    box_points = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
    return {"class_id": class_id, "box_points": box_points}

# Get the box points from the OBB label
annotation = parse_obb_label(obb_label_content)

# Augmentation for training set
train_augmentations = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=270, p=0.5),
    RandomBrightnessContrast(p=0.2),
    Perspective(scale=(0.1, 0.2), p=0.5),
])

# Augmentation for validation set
val_augmentations = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=270, p=0.5),
    RandomBrightnessContrast(p=0.2),
    Perspective(scale=(0.05, 0.5), p=0.5),
    ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
    CoarseDropout(max_holes=10, max_height=25, max_width=45, p=0.5),
])

# Function to adjust OBB labels after augmentation
def adjust_obb_label(obb, augmentation, image_width, image_height):
    box_points = np.array(obb["box_points"], dtype=np.float32)

    # Apply flip augmentation (flip x coordinates)
    if isinstance(augmentation, A.HorizontalFlip):
        box_points[:, 0] = image_width - box_points[:, 0]

    # Apply rotation augmentation (apply matrix rotation)
    if isinstance(augmentation, A.Rotate):
        angle = augmentation.get_params()['limit'][0]
        center = (image_width // 2, image_height // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        box_points = cv2.transform(np.array([box_points]), rot_matrix)[0]

    # Ensure the bounding box is within the image
    box_points = np.clip(box_points, 0, [image_width, image_height])

    # Normalize the coordinates to the [0, 1] range
    normalized_points = [(x / image_width, y / image_height) for x, y in box_points]

    return {"class_id": obb["class_id"], "box_points": normalized_points}

# Function to save the augmented data
def save_augmented_data(augmented, filename, output_image_dir, output_label_dir, annotations):
    augmented_image = augmented["image"]
    augmented_labels = []

    # Adjust OBB labels for augmentation
    for annotation in annotations:
        adjusted_label = adjust_obb_label(annotation, augmented, image.shape[1], image.shape[0])
        augmented_labels.append(adjusted_label)

    # Save the augmented image
    image_output_path = os.path.join(output_image_dir, filename)
    cv2.imwrite(image_output_path, augmented_image)

    # Save augmented labels in YOLO OBB format (8 points)
    label_output_path = os.path.join(output_label_dir, f"{os.path.splitext(filename)[0]}.txt")
    with open(label_output_path, "w") as label_file:
        for label in augmented_labels:
            box_points = label['box_points']
            label_file.write(f"{label['class_id']} " + ' '.join([f"{x} {y}" for x, y in box_points]) + "\n")

# Function to apply grayscale augmentation with probability
def mario_to_grayscale(image, probability=0.5):
    """
    Converts the image to grayscale with a given probability.
    :param image: Input image.
    :param probability: Probability of applying grayscale.
    :return: Grayscale or original image.
    """
    if np.random.rand() < probability:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def place_multiple_marios(image, box_points, num_instances_range=(3, 6), probability=0.5):
    """
    Places multiple Mario instances into the image on a white background.
    :param image: Input RGB image with a white background.
    :param box_points: Bounding box points (denormalized).
    :param num_instances_range: Range of the number of Mario instances to add.
    :param probability: Probability of adding multiple Marios.
    :return: Image with multiple Mario instances.
    """
    if np.random.rand() < probability:
        num_instances = np.random.randint(*num_instances_range)  # Randomize number of instances
        image_height, image_width = image.shape[:2]

        # Denormalize box_points to pixel values
        box_points = np.array(box_points, dtype=np.float32)
        box_points[:, 0] *= image_width
        box_points[:, 1] *= image_height
        box_points = np.round(box_points).astype(int)

        # Crop the original Mario cutout
        mario_cutout = image[box_points[0][1]:box_points[2][1], box_points[0][0]:box_points[2][0]]

        for _ in range(num_instances):
            # Randomly position Mario in the image
            x_offset = np.random.randint(0, image_width - mario_cutout.shape[1])
            y_offset = np.random.randint(0, image_height - mario_cutout.shape[0])

            # Overlay Mario at the random position
            image[y_offset:y_offset + mario_cutout.shape[0], x_offset:x_offset + mario_cutout.shape[1]] = mario_cutout

    return image


# def clear_original_area(image, box_points):
#     """
#     Clears the original area where Mario was, making it fully transparent.
#     :param image: Input RGBA image.
#     :param box_points: Bounding box points (denormalized).
#     :return: Image with cleared original area.
#     """
#     box_points = np.array(box_points, dtype=int)  # Ensure integer coordinates
#     x_min, y_min = np.min(box_points[:, 0]), np.min(box_points[:, 1])
#     x_max, y_max = np.max(box_points[:, 0]), np.max(box_points[:, 1])

#     # Set the alpha channel of the specified area to 0 (fully transparent)
#     image[y_min:y_max, x_min:x_max, 3] = 0
#     return image


# def overlay_with_transparency(background, overlay, x, y):
#     """
#     Overlays an RGBA image onto a background image at position (x, y),
#     preserving transparency.
#     """
#     bh, bw = background.shape[:2]
#     oh, ow = overlay.shape[:2]

#     # Ensure the overlay fits within the background dimensions
#     x_end = min(x + ow, bw)
#     y_end = min(y + oh, bh)
#     overlay = overlay[:y_end - y, :x_end - x]

#     # Extract the alpha mask
#     alpha = overlay[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
#     alpha_inv = 1.0 - alpha

#     for c in range(3):  # Blend each RGB channel
#         background[y:y_end, x:x_end, c] = (
#             alpha * overlay[:, :, c] + alpha_inv * background[y:y_end, x:x_end, c]
#         )

#     # Update the alpha channel in the background
#     background[y:y_end, x:x_end, 3] = (
#         alpha * 255 + alpha_inv * background[y:y_end, x:x_end, 3]
#     )
#     return background

# Function to apply pixelation randomly with probability
def pixelate(image, pixel_size_range=(5, 15), probability=0.5):
    """
    Pixelates the image by downscaling and upscaling with random pixel size and probability.
    :param image: Input image.
    :param pixel_size_range: Range of the pixel size.
    :param probability: Probability of applying pixelation.
    :return: Pixelated image.
    """
    if np.random.rand() < probability:
        pixel_size = np.random.randint(*pixel_size_range)  # Randomize pixel size within range
        small = cv2.resize(image, (image.shape[1] // pixel_size, image.shape[0] // pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated_image = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return pixelated_image
    return image

def ensure_alpha_channel(image):
    """
    Ensures the image has an alpha channel. Adds a fully opaque alpha channel if absent.
    :param image: Input image (RGB or RGBA).
    :return: Image with alpha channel.
    """
    if image.shape[-1] == 3:  # If no alpha channel
        alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype) * 255  # Fully opaque
        image = cv2.merge((image, alpha_channel))
    return image

# Generate augmented training data
for i in range(20):

    augmented_image = place_multiple_marios(image.copy(), annotation["box_points"], num_instances_range=(3, 6), probability=0.3)

    # Apply Albumentations on RGB
    augmented_train = train_augmentations(image=augmented_image)
    augmented_image = augmented_train["image"]

    # Apply grayscale transformation
    augmented_image = mario_to_grayscale(augmented_image, probability=0.2)  # 10% chance of grayscale
    augmented_image = pixelate(augmented_image, pixel_size_range=(5, 15), probability=0.1)

    save_augmented_data({"image": augmented_image}, f"mario_train_aug_{i}.png", train_images_path, train_labels_path, [annotation])

image = ensure_alpha_channel(image)  # Ensure transparency

# Generate augmented validation data
for i in range(20):

    augmented_image = place_multiple_marios(image.copy(), annotation["box_points"], num_instances_range=(3, 6), probability=0.3)

    # Apply Albumentations on RGB
    augmented_val = val_augmentations(image=augmented_image)
    augmented_image = augmented_val["image"]

    # Apply grayscale transformation
    augmented_image = mario_to_grayscale(augmented_image, probability=0.2)
    augmented_image = pixelate(augmented_image, pixel_size_range=(5, 15), probability=0.1)

    save_augmented_data({"image": augmented_image}, f"mario_val_aug_{i}.png", val_images_path, val_labels_path, [annotation])
