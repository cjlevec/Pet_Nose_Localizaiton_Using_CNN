# custom dataloader classes must implement three specific functions:
# init, len, and getitem

import os # module used for interacting with files, directory paths, and env vars
import pandas as pd # library used for data analysis, structuring, and filtering
import numpy as np
import torch
from mpl_toolkits.mplot3d.proj3d import transform
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset # for dataloader, and parent Dataset class
import torchvision.transforms.functional as F
from torchvision.transforms import v2

transform1 = v2.Compose([  # v2 is an instance of torchvisions transforms module
    v2.Resize((227, 227)),  # resize to desired shape
    # v2.ToTensor() # transform img data into tensor, depreciated
    v2.ToDtype(torch.float32, scale=True)
])

class SnoutNoseDataset (Dataset):
    def __init__ (self, annotations_file, img_dir, transform = transform1, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
    # can directly put file name here but not optimal since we have seperate train and test sets 
    # also we can use read_csv on the .txt files since the data is seperated by commas
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    """def __getitem__(self, idx):
        # Combines img directory and the specific image file name into one complete path (os independant)
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # Load image from file and convert to tensor all at once, func comes from torchvision
        image = read_image(img_path)

        label_str = self.img_labels.iloc[idx, 1] # label in string form
        xy_coords = tuple(map(int, label_str.strip("()").split(",")))  # Convert to tuple of integers
        label = torch.tensor(xy_coords, dtype=torch.float32)

        # Ensure the image is RGB (3 channels)
        if image.size(0) == 4:  # If it has an alpha channel
            image = image[:3, :, :]

        if image.size(0) == 1:  # If it's grayscale
            image = image.repeat(3, 1, 1)

        # Apply image transformation
        image = self.transform(image)
        """
    """for idx in range(0, 54):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        print(f"Image at index {idx}: shape = {image.shape}")""" """

    return image, label"""


    def __getitem__(self, idx):
        # Combines the img directory and the specific image file name into one complete path
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # start
        # Check the file extension
        for i in range (0, 150):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])
            valid_extensions = ('.jpg', '.jpeg', '.png', '.gif')
            if not img_path.lower().endswith(valid_extensions):
                print(f"Warning: Unsupported image format for {img_path}. Skipping.")
        return None, None  # Skip this entry or handle it as needed

        #end
        image = read_image(img_path)  # function comes from torchvision

        # Convert to RGB if it has an alpha channel or is grayscale
        if image.size(0) == 1:  # If it's grayscale
            image = image.repeat(3, 1, 1)  # Convert to RGB by repeating the single channel
        elif image.size(0) == 4:  # If it has an alpha channel
            image = image[:3, :, :]  # Keep only the RGB channels

        # Extract and parse the label (coordinates)
        # start
        # Check the file extension
        for i in range (0, 150):
            label_str = self.img_labels.iloc[i, 1]  # This will be something like "(311, 152)"
            valid_extensions = ('.jpg', '.jpeg', '.png', '.gif')
            if not img_path.lower().endswith(valid_extensions):
                print(f"Warning: Unsupported image format for {img_path}. Skipping.")
        return None, None  # Skip this entry or handle it as needed

        # end
        # label_str = self.img_labels.iloc[idx, 1]  # This will be something like "(311, 152)"



        xy_coords = tuple(map(int, label_str.strip("()").split(",")))  # Convert to tuple of integers
        label = torch.tensor(xy_coords, dtype=torch.float32)
        # Get original image dimensions
        original_height, original_width = label[0], label[1]

        if self.transform:
            image = self.transform(image)

        # Adjust coordinates according to new image dimensions
        scaled_x = int(label[0] * 227 / original_width)
        scaled_y = int(label[1] * 227 / original_height)

        # Ensure coordinates are within the bounds of the resized image
        scaled_x = min(scaled_x, 227 - 1)
        scaled_y = min(scaled_y, 227 - 1)

        scaled_coordinates = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)

        return image, scaled_coordinates


import random
import matplotlib.pyplot as plt

# Number of random images to display
num_images = 1

trainSet = SnoutNoseDataset("/Users/christianlevec/Documents/475 Lab 2/oxford-iiit-pet-noses/train_noses.txt",
                            "/Users/christianlevec/Documents/475 Lab 2/oxford-iiit-pet-noses/images-original/images"
                            , transform=transform1)
testSet = SnoutNoseDataset("/Users/christianlevec/Documents/475 Lab 2/oxford-iiit-pet-noses/test_noses.txt",
                           "/Users/christianlevec/Documents/475 Lab 2/oxford-iiit-pet-noses/images-original/images"
                           , transform=transform1)

train_dataloader = DataLoader(trainSet, batch_size=64, shuffle=True)  # experiment with batch_size
test_dataloader = DataLoader(testSet, batch_size=64, shuffle=True)

# Get random indices from the dataset
random_indices = random.sample(range(len(trainSet)), num_images)

# Loop over the random indices
for idx in random_indices:
    image_test, label = trainSet[idx]  # Get the image and label
    # Display the image using matplotlib
    plt.imshow(image_test.permute(1, 2, 0))  # Permute the dimensions to (height, width, channels)
    plt.title(f"Label: {label}")  # Set the title to the label
    plt.plot(label[0], label[1], marker='o', color='red', markersize=10)

    plt.show()  # Show the image

    """image1, label1 = trainSet[idx]
    img = np.transpose(image1, (1, 2, 0))
    plt.imshow(img)
    print(label1)
    plt.plot(label1[0], label1[1], marker='o', color='red', markersize=10)
    plt.show()"""




# Image is initially in the shape (C, H, W) for PyTorch (C is colour aka RGB)
# have to change to (H, W, C) for matplotlib
