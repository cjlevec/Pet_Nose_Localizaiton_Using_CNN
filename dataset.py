
# Added so that I can run the code on PyCharm and Colab
ColabPath = "/content/drive/My Drive/oxford-iiit-pet-noses/"
HomePath = "/Users/christianlevec/Documents/475 Lab 2/oxford-iiit-pet-noses/"

file_id = 'YOUR_FILE_ID'
data_url = f'https://drive.google.com/uc?id={file_id}'

import os # module used for interacting with files, directory paths, and env vars
import pandas as pd # library used for data analysis, structuring, and filtering
import torch
#from mpl_toolkits.mplot3d.proj3d import transform
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset # for dataloader, and parent Dataset class
from torchvision.transforms import v2

# Transform for orignial images -> 227x227 -> tensor
transform1 = v2.Compose([  # v2 is an instance of torchvisions transforms module
    v2.Resize((227, 227)),  # resize to desired shape
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

    def __getitem__(self, idx):

        # Combines the img directory and the specific image file name into one complete path
        img_filename = self.img_labels.iloc[idx, 0]

        # Combines the img directory and the specific image file name into one complete path
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)  # function comes from torchvision

        # Convert to RGB if it has an alpha channel or is grayscale
        if image.size(0) == 1:  # If it's grayscale
            image = image.repeat(3, 1, 1)  # Convert to RGB by repeating the single channel
        elif image.size(0) == 4:  # If it has an alpha channel
            image = image[:3, :, :]  # Keep only the RGB channels

        # Save original image dimensions for scaling calculation
        original_height, original_width = image.shape[1], image.shape[2]

        # Extract labels in string format: "(311, 152)"
        label_str = self.img_labels.iloc[idx, 1]
        # Remove parentheses and split string into two parts: ['311', '152']
        xy_coords = label_str.strip("()").split(",")
        # Convert parts into ints: x = 311, y = 152
        x, y = int(xy_coords[0]), int(xy_coords[1])
        # Make into tensor
        label_tensor = torch.tensor([x, y], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        new_height, new_width = image.shape[1], image.shape[2]

        # Correct scaling calculation using the original label coordinates
        scaled_x = label_tensor[0] * new_width / original_width
        scaled_y = label_tensor[1] * new_height / original_height

        # Convert to tensor with 2 elements (x, y)
        scaled_coordinates = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)

        return image, scaled_coordinates    # Both in tensor form


##### MAIN ####
"""
import random
import matplotlib.pyplot as plt

trainSet = SnoutNoseDataset(ColabPath+"train_noses.txt",
                            ColabPath+"images-original/images"
                            , transform=transform1)
testSet = SnoutNoseDataset(HomePath+"test_noses.txt",
                           HomePath+"images-original/images"
                           , transform=transform1)

train_dataloader = DataLoader(trainSet, batch_size=61, shuffle=True)  # experiment with batch_size
test_dataloader = DataLoader(testSet, batch_size=61, shuffle=True)

# Number of random images to display
num_images = 5
# Get random indices from the dataset
random_indices = random.sample(range(len(trainSet)), num_images)

# Loop over the random indices
for idx in range (0, num_images):
    image_test, label = trainSet[idx]  # Get the image and label
    plt.imshow(image_test.permute(1, 2, 0))  # Permute the dimensions to (height, width, channels) from (ch, height, width)
    plt.title(f"Label: {label}")  # Set the title to the label
    plt.plot(label[0], label[1], marker='o', color='red', markersize=10)"""
    #plt.show()  # Show the image
