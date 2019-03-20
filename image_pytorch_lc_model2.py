"""

Experimenting with using transforms and dataloader 
to resize the images, and normalising the image data

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import math

from data import TrainingData
from imagedatapreprocessing import DatasetProcessing
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Use CUDA if available -> replacing every .cuda() with .to(device)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') # use this if gpu doesnt provide sufficient memory

# LOAD IMAGES
print("Loading data")
td = TrainingData()
image_train, y_train, image_val, y_val, output_size = td.getTrainingImages("beauty")
print("Output Size:", output_size)
print('Train on', len(image_train), 'Test on', len(image_val))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
								transforms.ToPILImage(),
								transforms.Resize((128, 128)),
								transforms.RandomHorizontalFlip(),
								transforms.RandomRotation(60),
								transforms.ToTensor(),
								normalize])

val_transform = transforms.Compose([
								transforms.ToPILImage(),
								transforms.Resize((128, 128)),
								transforms.ToTensor(),
								normalize])

dset_train = DatasetProcessing(image_train, y_train, train_transform)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=100,
                                          shuffle=True, num_workers=0)

dset_val = DatasetProcessing(image_val, y_val, val_transform)
val_loader = torch.utils.data.DataLoader(dset_train, batch_size=100,
                                          shuffle=True, num_workers=0)

def train(train_loader, model):
	model.train()
	start = time.time()
	for i, (x, y) in enumerate(train_loader):
		x, y = x.cuda(), y.cuda()

		model.zero_grad()
		output = model(x)
		loss = criterion(output, y)
		loss.backward()
		optimizer.step()

		# prints loss after every 5000 training data (prints every 50 batches)
		if (i+1) * 100 % loss_every == 0:
			img_total = len(image_train)
			nb_batches = img_total / 100
			progress = round(100 * (i+1) / nb_batches , 2)
			print('Progress:', progress, '%' ,' Time passed:', timeSince(start), ' Training Loss:', loss)

		# Validate after every 20000 data (prints every 200 batches)
		if (i+1) * 100 % validate_every == 0:
			correct = 0
			losses = 0
			start2 = time.time()

			model.eval()
			with torch.no_grad():
				for i, (x_temp, y_temp) in enumerate(val_loader):
					# break out of loop once 20000 val images have been selected
					if i == (20000 / 100):
						break

					x_temp, y_temp = x_temp.cuda(), y_temp.cuda()

					output = model(x_temp)
					losses += criterion(output, y_temp)
					predicted = output.data.max(1, keepdim=True)[1]

					correct += predicted.eq(y_temp.data.view_as(predicted)).cpu().sum()

			correct = 100 * correct / 20000
			loss = losses / 20000
			print("val_loss:", loss, " val_acc:", correct, "%", 'time passed:', timeSince(start2))

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

######################  MODEL DEFINITION  ###############################
model = models.resnet50(pretrained=True)
print(model)

# preventing the model from retraining the weights that are already preset in this model. 
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 17),
                                 nn.LogSoftmax(dim=1))

# # Converting the model to run on cuda gpu
model.cuda()

lr = 0.001
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr)

model.cuda()
criterion.cuda()

######################  TRAINING  ###############################
loss_every = 5000
validate_every = 20000

start = time.time()

print("Training the model\n")
print("CATEGORY: Beauty")
print("NO OF CLASSES:", output_size,'\n')

nb_epochs = 10
val_acc = []
val_loss = []

for epoch in range(nb_epochs):
	print('Running Training Epoch', epoch + 1)

	train(train_loader, model)

	print('Finished Training Epoch', epoch + 1)