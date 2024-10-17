import torch
import torch.nn.functional as F
import torch.nn as nn


class SnoutNet(nn.Module):

	def __init__(self):

		super(SnoutNet, self).__init__()

        # Convolutional Layers
        # Padding size of 1 ensures the output size is equal to the input
        # 2 by 2 max pooling after each conv layer to reduce computation intensity
		self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
		self.MP1 = nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 2)
		self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
		self.MP2 = nn.MaxPool2d(kernel_size = 4, stride = 4, padding =2)
		self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
		self.MP3 = nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 2) 
		# Kernel Size, stride, and padding for MP layers was found through experimentation
		# so that conv3 output dim == 4 x 4. 

        # Fully connected layers
		self.fc1 = nn.Linear(4096, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 2)


	def forward(self, X):
        # Calling pooling function MP1 directly as it was defined above using MaxPool2d
		# Using the functional interface (F) for the relu activation function
		X = self.MP1(F.relu(self.conv1(X)))	# Order is from inside to out
		# print(X.shape)
		X = self.MP2(F.relu(self.conv2(X)))
		# print(X.shape)
		X = self.MP3(F.relu(self.conv3(X)))
		# print(X.shape)

		# print(X.shape)
		X = X.view(X.size(0), -1)	# tensor.size(0) corresponds to batch size
									# 1 = number of channels (RGB)
									# 2 = img height
									# 3 = img width

		X = self.fc1(X)
		X = F.relu(X)
		X = self.fc2(X)
		X = F.relu(X)
		X = self.fc3(X)
		X = F.relu(X)

		return X


### MAIN ###

"""SnoutModel = SnoutNet() # Must create instance of model
A = torch.rand(1, 3, 227, 227) # Create dummy tensor to test
print(A.shape) # Print initial tensor shape
output = SnoutModel(A) # Pass tensor into the model
print(output.shape) # Check output shape"""

"""
Calculation Mistake: When you defined your first fully connected layer, 
you need to make sure it matches the size that results from the last convolutional layer 
after pooling and flattening.

"""
