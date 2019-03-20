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

category = "beauty"

# LOAD IMAGES
print("Loading data")
td = TrainingData()
image_train, y_train, image_val, y_val, output_size = td.getTrainingImages(category)
print("Category", category)
print("Output Size:", output_size)

current_image_example = -1
def getImageTrainingExample():
    global current_image_example
    current_image_example += 1
    if current_image_example > len(image_train) - 1:
        current_image_example = 0
    img = cv2.imread(image_train[current_image_example], 1)
    cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    img_tensor = torch.div(torch.from_numpy(np.array(img)).float(), 255.0).transpose(0, 2).unsqueeze(0).to(device)
    return img_tensor, torch.tensor([y_train[current_image_example]]).to(device)

class CNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=10, stride=1, padding=0).to(device)
        self.lrelu = nn.LeakyReLU(0.1).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=0).to(device)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=10, stride=1, padding=0).to(device)
        self.pool2 = nn.AdaptiveAvgPool2d((1, hidden_size)).to(device)

        self.fc1 = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, input):
        output = self.lrelu(self.conv1(input))
        output = self.pool1(output)
        output = self.lrelu(self.conv2(output))
        output = self.pool2(output)
        output = F.log_softmax(self.fc1(output[0][-1]), dim=1).to(device)
        return output

model = CNN(300, output_size)

lr = 0.001
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
validate_every = 50
for iter in range(1, n_iters + 1):
    x, y = getImageTrainingExample()
    loss = train(x, y)

    # Print
    if iter % print_every == 0:
        print(iter, loss)

    # Validate
    # Take out 200 samples to validate
    if iter % validate_every == 0:
        correct = 0
        losses = 0
        for index in [random.randint(0, len(image_val) - 1) for x in range(200)]:
            img = cv2.imread(image_val[index], 1)
            cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
            img_tensor = torch.div(torch.from_numpy(np.array(img)).float(), 255.0).transpose(0, 2).unsqueeze(0).to(device)
            output = model(img_tensor)
            losses += criterion(output, torch.tensor([y_val[index]]).to(device)).item()
            if torch.max(output[0], 0)[1].item() == y_val[index]:
                correct += 1
        correct = correct / 200 * 100
        loss = losses / 200
        print("val_loss:", loss, " val_acc:", correct, "%")

