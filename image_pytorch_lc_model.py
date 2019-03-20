#!/usr/bin/env python3
"""
Li Chao Image Recognition Model

This model runs on pretrained resnet50 model. Reason being that pretrained image models are better 
at detecting different types of surfaces from the pretrained filters and weights. 

The model merely adds a final Linear (aka Dense in keras) layer to the pretrained model. 

The resnet model however is preset to run on cpu. I have included a switch to convert the 
model to gpu if required. 

Always try with gpu, as script should load faster. However, some gpu may not have enough memory space. 
I included a switch to convert to cpu if it is found that cuda doesnt have enough memory. 

I have also decided to validate on the val samples only after the full training for one epoch is done. 
Reason being that validation takes as much time as training, so excessive validation will increase
computational time.

"""

import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import cv2
import time
import math

from data import TrainingData
from torchvision import models

# Use CUDA if available -> replacing every .cuda() with .to(device)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') # use this if gpu doesnt provide sufficient memory

# LOAD IMAGES
print("Loading data")
td = TrainingData()
image_train, y_train, image_val, y_val, output_size = td.getTrainingImages("beauty")
print("Output Size:", output_size)
print('Train on', len(image_train), 'Test on', len(image_val))

# Extracting images one by one
current_image_example = -1
def getImageTrainingExample():
	global current_image_example
	current_image_example += 1
	if current_image_example > len(image_train) - 1:
		current_image_example = 0
	img = cv2.imread(image_train[current_image_example], 1)
	img_tensor = torch.div(torch.from_numpy(np.array(img)).float(), 255.0).transpose(0, 2).unsqueeze(0).to(device)
	return img_tensor, torch.tensor([y_train[current_image_example]]).to(device)

current_val_example = -1
def getImageValidationExample():
	global current_val_example
	current_val_example += 1
	if current_val_example > len(image_val) - 1:
		current_val_example = 0
	img = cv2.imread(image_val[current_val_example], 1)
	img_tensor = torch.div(torch.from_numpy(np.array(img)).float(), 255.0).transpose(0, 2).unsqueeze(0).to(device)
	return img_tensor, torch.tensor([y_val[current_val_example]]).to(device)

def train(x, y):
	model.zero_grad()
	output = model(x)
	loss = criterion(output, y)
	loss.backward()
	optimizer.step()

	return loss.item()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

######################  MODEL DEFINITION  ###############################
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

# model = models.resnet50(pretrained=True)
# print(model)

# # preventing the model from retraining the weights that are already preset in this model. 
# for param in model.parameters():
#     param.requires_grad = False
    
# model.fc = nn.Sequential(nn.Linear(2048, 512),
#                                  nn.ReLU(),
#                                  nn.Dropout(0.2),
#                                  nn.Linear(512, 17),
#                                  nn.LogSoftmax(dim=1))

# # # Converting the model to run on cuda gpu
# model.cuda()

lr = 0.01
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr)

######################  TRAINING  ###############################
n_iters = 10000
print_every = 500
validate_every = 2000

start = time.time()

print("Training the model\n")
print("CATEGORY: Beauty")
print("NO OF CLASSES:", output_size,'\n')

nb_epochs = 10
val_acc = []
val_loss = []

for epoch in range(nb_epochs):
	print('Running Training Epoch', epoch + 1)
	for iter in range(1, n_iters + 1):
		x, y = getImageTrainingExample()
		loss = train(x, y)

		#  to monitor the loss every 500 images
		if iter % print_every == 0:
			print('Percentage Complete:', 100 * iter / (n_iters + 1), 'Training loss:', loss, 'Time passed:', timeSince(start))

		if iter % validate_every == 0:
			correct = 0
			losses = 0
			start2 = time.time()

			model.eval()
			with torch.no_grad():
				for i in range(1, 1001):
					x_temp, y_temp = getImageValidationExample()
					output = model(x_temp)
					losses += criterion(output, y_temp)

					_, predicted = torch.max(output[0], 0)

					if predicted == y_temp.item():
						correct += 1

			correct = 100 * correct / 1000
			loss = losses / 1000
			print("val_loss:", loss, " val_acc:", correct, "%", 'time passed:', timeSince(start2))


	print('Finished Training Epoch', epoch + 1)

	# FINAL VALIDATION
	print('Validating')

	correct = 0
	losses = 0
	start2 = time.time()

	model.eval()
	with torch.no_grad():
		for i in range(1, 1001):
			x_temp, y_temp = getImageValidationExample()
			output = model(x_temp)
			losses += criterion(output, y_temp)

			_, predicted = torch.max(output[0], 0)

			if predicted == y_temp.item():
				correct += 1

	correct = 100 * correct / 1000
	loss = losses / 1000
	print("val_loss:", loss, " val_acc:", correct, "%", 'time passed:', timeSince(start2))

	val_acc.append(correct)
	val_loss.append(loss)

	model.train()

# Plotting validation results 
plt.plot(val_acc, label = 'Val Acc')
plt.plot(val_loss, label = 'Val Loss')
plt.legend(frameon=False)
plt.show()