import os
import cv2

# Paths
input_images_dir = "./dataset/images/train"
input_labels_dir = "./dataset/labels/train"
output_images_dir = "./dataset/images/backgrounds"
output_labels_dir = "./dataset/labels/backgrounds"
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Generate hard negatives from existing images
for image_file in os.listdir(input_images_dir):
    image_path = os.path.join(input_images_dir, image_file)
    label_path = os.path.join(input_labels_dir, os.path.splitext(image_file)[0] + ".txt")
    
    # Load image and annotations
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    mario_boxes = []

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                polygon = [(parts[i] * width, parts[i + 1] * height) for i in range(1, len(parts), 2)]
                x_min = min([point[0] for point in polygon])
                y_min = min([point[1] for point in polygon])
                x_max = max([point[0] for point in polygon])
                y_max = max([point[1] for point in polygon])
                mario_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

    # Define non-Mario regions
    step = 50  # Size of cropped sections
    for y in range(0, height, step):
        for x in range(0, width, step):
            crop_box = (x, y, min(x + step, width), min(y + step, height))
            
            # Check overlap with Mario bounding boxes
            overlap = False
            for mx1, my1, mx2, my2 in mario_boxes:
                if not (crop_box[2] <= mx1 or crop_box[0] >= mx2 or crop_box[3] <= my1 or crop_box[1] >= my2):
                    overlap = True
                    break
            
            if not overlap:
                # Save non-Mario region
                cropped = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                output_image_path = os.path.join(output_images_dir, f"bg_{image_file}_{x}_{y}.jpg")
                cv2.imwrite(output_image_path, cropped)

                # Create label for background
                norm_x_center = ((crop_box[0] + crop_box[2]) / 2) / width
                norm_y_center = ((crop_box[1] + crop_box[3]) / 2) / height
                norm_width = (crop_box[2] - crop_box[0]) / width
                norm_height = (crop_box[3] - crop_box[1]) / height
                label_path = os.path.join(output_labels_dir, f"bg_{image_file}_{x}_{y}.txt")
                with open(label_path, "w") as label_file:
                    label_file.write(f"1 {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")