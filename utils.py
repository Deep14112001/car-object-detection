import os
import shutil
import cv2
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

def plot_image_with_bbox(image_dir, image_name, bbox):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Image {image_name} not found.")
        return
    
    x_min, y_min, x_max, y_max = bbox
    
    # Draw the bounding box on the image
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    
    # Convert BGR to RGB for displaying with matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image: {image_name}")
    plt.axis('off')
    plt.show()


def copy_images(csv_path, original_image_dir, img_dir):
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        image_name = row['image']
        src_path = os.path.join(original_image_dir, image_name)
        dst_path = img_dir
        shutil.copy(src_path, dst_path)


def convert_to_yolo_format(csv_file, output_dir, image_dir):
    df = pd.read_csv(csv_file)
    for idx, row in df.iterrows():
        image_file = row['image']
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        
        # YOLO format requires [class, x_center, y_center, width, height] normalized by image size
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h
        
        # Create a text file for each image with YOLO formatted bounding boxes
        label_file = os.path.join(output_dir, Path(image_file).stem + ".txt")
        with open(label_file, 'w') as f:
            # Class '0' for cars (you can change this if there are multiple classes)
            f.write(f"0 {x_center} {y_center} {width} {height}\n")
