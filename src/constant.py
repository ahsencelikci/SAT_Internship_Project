import os

# Path to jsons
JSON_DIR = 'C:/Users/Ahsen/SAT_Internship_Project/data/jsons'

# Path to mask
MASK_DIR = 'C:/Users/Ahsen/SAT_Internship_Project/data/masks'
if not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = 'C:/Users/Ahsen/SAT_Internship_Project/data/masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.makedirs(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = 'C:/Users/Ahsen/SAT_Internship_Project/data/images'

# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = True

# Bacth size
BATCH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS = 2
