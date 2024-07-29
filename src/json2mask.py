import json
import os
import numpy as np
import cv2
import tqdm
from constant import JSON_DIR, MASK_DIR

# Create a list which contains every file name in "jsons" folder
json_list = os.listdir(JSON_DIR)

# Example iterator to demonstrate tqdm usage
iterator_example = range(1000000)

# Progress bar for example iterator
for i in tqdm.tqdm(iterator_example):
    pass

# For every json file
for json_name in tqdm.tqdm(json_list):
    # Access and open json file as dictionary
    json_path = os.path.join(JSON_DIR, json_name)
    with open(json_path, 'r') as json_file:
        # Load json data
        json_dict = json.load(json_file)

    # Create an empty mask whose size is the same as the original image's size
    
    ###########################################################################################
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    ###########################################################################################
    
    # For every object
    for obj in json_dict["objects"]:
        # Check if the object's 'classTitle' is 'Freespace' or not
        if obj['classTitle'] == 'Freespace':
            
            #################################################################################
            exterior_points = np.array([obj['points']['exterior']], dtype=np.int32)
            mask = cv2.fillPoly(mask, exterior_points, color=255)  # Use 255 for white (freespace)
            ##################################################################################

    # Write mask image into MASK_DIR folder
    
    ##########################################################################################
        mask_path = os.path.join(MASK_DIR, json_name[:-5])
        cv2.imwrite(mask_path + '.png', mask.astype(np.uint8))

print("Mask generation complete.")
     ##########################################################################################