#!/usr/bin/env python3

import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import cv2

from data import TrainingData


# Use CUDA if available -> replacing every .cuda() with .to(device)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# LOAD IMAGES
print("Loading data")
td = TrainingData()
image_train, y_train, image_val, y_val, output_size = td.getTrainingImages("beauty")
print("Output Size:", output_size)

current_image_example = -1
def getImageTrainingExample():
	global current_image_example
	current_image_example += 1
	if current_image_example > len(image_train) - 1:
		current_image_example = 0
	img = cv2.imread(image_train[current_image_example], 1)
	img_tensor = torch.div(torch.from_numpy(np.array(img)).float(), 255.0).transpose(0, 2).unsqueeze(0).to(device)
	return img_tensor, torch.tensor([y_train[current_image_example]]).to(device)

class CNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(CNN, self).__init__()

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0).to(device)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0).to(device)
		self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0).to(device)
		self.pool2 = nn.AdaptiveAvgPool2d((1, hidden_size)).to(device)

		self.fc1 = nn.Linear(hidden_size, output_size).to(device)

	def forward(self, input):
		output = F.relu(self.conv1(input)).to(device)
		output = self.pool1(output)
		output = F.relu(self.conv2(output)).to(device)
		output = self.pool2(output)
		output = F.log_softmax(self.fc1(output[0][-1]), dim=1).to(device)
		return output

model = CNN(300, output_size)

lr = 0.01
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr)


def train(x, y):
	model.zero_grad()

	output = model(x)
	loss = criterion(output, y)
	loss.backward()
	optimizer.step()

	return loss.item()

n_iters = 10000
print_every = 50
for iter in range(1, n_iters + 1):
	x, y = getImageTrainingExample()
	loss = train(x, y)

	if iter % print_every == 0:
		print(iter, loss)