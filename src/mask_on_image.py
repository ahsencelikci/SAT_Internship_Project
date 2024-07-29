import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *

# Create a list which contains every file name in masks folder
mask_list = os.listdir(MASK_DIR)
# Remove hidden files if any
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)

# For every mask image
for mask_name in tqdm.tqdm(mask_list):
    # Name without extension
    mask_name_without_ex = mask_name.split('.')[0]

    # Access required folders
    mask_path = os.path.join(MASK_DIR, mask_name)
    image_path = os.path.join(IMAGE_DIR, mask_name_without_ex + '.jpeg')
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)

    # Debugging: Print file paths
    print(f"Mask path: {mask_path}")
    print(f"Image path: {image_path}")

    # Read mask and corresponding original image
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    
    
    if image is None:
        print(f"Image not found: {image_path}")
        continue

    # Ensure the image dimensions match the mask dimensions
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Change the color of the pixels on the original image that corresponds
    # to the mask part and create new image
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colored_mask[:, :, 1:] = 0  # Keep only the red channel

    new_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    # Write output image into IMAGE_OUT_DIR folder
    cv2.imwrite(image_out_path, new_image)

    # Visualize created image if VISUALIZE option is chosen
    if VISUALIZE:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Masked Image')
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.show()
