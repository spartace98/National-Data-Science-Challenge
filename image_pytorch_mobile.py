"""
#  FOR MOBILE IMAGES
Experimenting with using transforms and dataloader 
to resize the images, and normalising the image data

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import time
import math

from data import TrainingData
from imagedatapreprocessing import DatasetProcessing
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Use CUDA if available -> replacing every .cuda() with .to(device)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# LOAD IMAGES
print("Loading data")
td = TrainingData()
image_train, y_train, image_val, y_val, output_size = td.getTrainingImages("mobile")

print("Output Size:", output_size)
print('Train on', len(image_train), 'Test on', len(image_val))
print(image_train[0])

######################### DATA PREPROCESSING ########################
batch_size = 300
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
								transforms.ToPILImage(),
								transforms.RandomAffine(degrees = 60, 
														translate = (0.2, 0.2), 
														scale = (0.7, 1.3)),
								transforms.Resize((128, 128)),
								transforms.RandomHorizontalFlip(),
								transforms.ToTensor(),
								normalize])

val_transform = transforms.Compose([
								transforms.ToPILImage(),
								transforms.Resize((128, 128)),
								transforms.ToTensor(),
								normalize])

dset_train = DatasetProcessing(image_train, y_train, train_transform)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

dset_val = DatasetProcessing(image_val, y_val, val_transform)
val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

###################  MODEL DEFINITION  #############################
model = models.resnet50(pretrained=True)
print(model)

# preventing the model from retraining the weights that are already preset in this model. 
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, output_size),
                                 nn.LogSoftmax(dim=1))

# # Converting the model to run on cuda gpu
lr = 0.001
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr)

model.to(device)

######################## METHODS ###################################
def train(model):
	global current_val_acc
	acc_l = []
	loss_l = []
	model.train()
	start = time.time()
	for i, (x, y) in enumerate(train_loader):
		x, y = x.to(device), y.to(device)

		model.zero_grad()
		output = model(x)
		loss = criterion(output, y)
		loss.backward()
		optimizer.step()

		# prints loss after every 3000 training data (prints every 10 batches)
		if (i+1) * batch_size % loss_every == 0:
			img_total = len(image_train)
			nb_batches = img_total / batch_size
			progress = round(100 * (i+1) / nb_batches , 2)
			print('Progress:', progress, '%' ,' Time passed:', timeSince(start), ' Training Loss:', loss.item())

		# Validate after every 21000 data (prints every 70 batches)
		if (i+1) * batch_size % validate_every == 0:
			start2 = time.time()
			val_acc, val_loss = validate(model, batch_size)
			acc_l.append(val_acc)
			loss_l.append(val_loss)

		# Validate when last batch is trained on, then break out of loop for next batch
		if x.shape[0] < batch_size:
			start3 = time.time()
			val_acc, val_loss = validate(model, batch_size)
			acc_l.append(val_acc)
			loss_l.append(val_loss)

			break

		# empty cache memory
		torch.cuda.empty_cache()

	# plot val acc and val loss graph after every epoch 
	plt.plot(acc_l, label = 'Validation Accuracy')
	plt.plot(loss_l, label = 'Validation Loss')
	plt.legend()
	plt.show()

# validate on unseen results, save weights if model has more accurate predictions
def validate(model, batch_size):
	global current_val_acc
	correct = 0
	losses = 0
	model.eval()
	with torch.no_grad():
		for i, (x_temp, y_temp) in enumerate(val_loader):
			# break out of loop once 21000 val images have been selected
			if i == (validate_every / batch_size):
				break

			x_temp, y_temp = x_temp.to(device), y_temp.to(device)

			output = model(x_temp)
			losses += criterion(output, y_temp).item()
			predicted = output.data.max(1, keepdim=True)[1]

			correct += predicted.eq(y_temp.data.view_as(predicted)).to(device).sum().item() 

	val_acc = 100 * correct / validate_every
	val_loss = losses / validate_every

	print("val_loss:", val_loss, " val_acc:", val_acc, "%", 'time passed:', timeSince(start2))
			
	if val_acc > current_val_acc:
		print("Saving the better model's weights")
		torch.save(model.state_dict(), "models/" + "mobile" + ".image.pth")
		current_val_acc = val_acc

	return val_acc, val_loss

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

######################  TRAINING  ###############################
loss_every = 3000
validate_every = 21000
current_val_acc = 0

start = time.time()

print("Training the model\n")
print("CATEGORY: Mobile")
print("NO OF CLASSES:", output_size,'\n')

nb_epochs = 2

for epoch in range(nb_epochs):
	print('Running Training Epoch', epoch + 1)

	train(model)

	print('Finished Training Epoch', epoch + 1)