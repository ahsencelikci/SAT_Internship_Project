from model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 4
epochs = 20
cuda = torch.cuda.is_available()
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = 'C:/Users/Ahsen/SAT_Internship_Project'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
###############################

# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# DATA CHECK
if not image_mask_check(image_path_list, mask_path_list):
    raise ValueError("Images and masks do not match.")

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = [image_path_list[i] for i in indices[:test_ind]]
test_label_path_list = [mask_path_list[i] for i in indices[:test_ind]]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = [image_path_list[i] for i in indices[test_ind:valid_ind]]
valid_label_path_list = [mask_path_list[i] for i in indices[test_ind:valid_ind]]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = [image_path_list[i] for i in indices[valid_ind:]]
train_label_path_list = [mask_path_list[i] for i in indices[valid_ind:]]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list) // batch_size

# CALL MODEL
model = FoInternNet(input_size=input_shape, n_classes=n_classes)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

# TRAINING THE NEURAL NETWORK
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for batch_idx in range(steps_per_epoch):
        batch_indices = np.random.choice(len(train_input_path_list), batch_size, replace=False)
        batch_input_paths = [train_input_path_list[i] for i in batch_indices]
        batch_label_paths = [train_label_path_list[i] for i in batch_indices]
        
        try:
            # Load the images and masks
            inputs = tensorize_image(batch_input_paths, output_shape=input_shape, cuda=cuda)
            masks = tensorize_mask(batch_label_paths, output_shape=input_shape, n_class=n_classes, cuda=cuda)
        except ValueError as e:
            print(e)
            continue  # or other error handling code

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{steps_per_epoch}], Loss: {running_loss / (batch_idx + 1):.4f}')

    # Validation phase
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        for i in range(len(valid_input_path_list)):
            inputs = tensorize_image([valid_input_path_list[i]], output_shape=input_shape, cuda=cuda).squeeze(0)
            masks = tensorize_mask([valid_label_path_list[i]], output_shape=input_shape, n_class=n_classes, cuda=cuda).squeeze(0)

            outputs = model(inputs.unsqueeze(0))
            loss = criterion(outputs, masks.unsqueeze(0))
            
            valid_loss += loss.item()
        
        print(f'Validation Loss after Epoch {epoch+1}: {valid_loss / len(valid_input_path_list):.4f}')

print("Training complete.")
