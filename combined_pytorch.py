#!/usr/bin/env python3

import pickle
import random
import Levenshtein

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

import numpy as np

from data import TrainingData
from imagedatapreprocessing import DatasetProcessing
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Use CUDA if available -> replacing every .cuda() with .to(device)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

validation_percent = 0.1

# LOAD WORD2VEC DATA
print("Loading dictionary of embeddings")
f = open("data/dict.pkl", "rb")
dictionary = pickle.load(f)
f.close()

words = list(dictionary.keys())

# COPY WORD2VEC EMBEDDINGS INTO NN.EMBEDDING LAYER
print("Copying embeddings")
vocab_size = len(dictionary)
vector_size = 100
pretrained_weights = list(dictionary.values())
# vocab_size is the number of words in your train, val and test set
# vector_size is the dimension of the word vectors you are using
embed = nn.Embedding(vocab_size, vector_size).to(device)

# intialize the word vectors, pretrained_weights is a
# numpy array of size (vocab_size, vector_size) and
# pretrained_weights[i] retrieves the word vector of
# i-th word in the vocabulary
embed.weight.data.copy_(torch.tensor(pretrained_weights).to(device))

######################### IMAGE DATA PREPROCESSING ########################
batch_size = 200
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

image_transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                normalize])

############################################################################

def processSentence(sentence):
    newX = []
    for word in sentence.split(" "):
        if word in words:
            newX.append(words.index(word))
        else:
            jaros = [Levenshtein.jaro(word, w) for w in words]
            highest_index = jaros.index(max(jaros))
            newX.append(highest_index)
    newX = torch.tensor(newX).to(device).long()
    return newX

# DECLARE LSTM CLASS
class LSTM(nn.Module):
    def __init__(self, embeddings, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True).to(device)

        self.embeddings = embeddings
        self.fc = nn.Linear(hidden_size * 2, output_size).to(device)
        self.hidden = self.initHidden()

    def forward(self, input):
        embed = self.embeddings(input)
        embed = embed.unsqueeze(1)
        lstm_out, _ = self.lstm(embed, self.hidden)
        out = F.log_softmax(self.fc(lstm_out[-1]), dim=1).to(device)
        return out

    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size).to(device), torch.zeros(2, 1, self.hidden_size).to(device))

td = TrainingData(validation_percent)

for cat in ["fashion", "beauty", "mobile"]:

    x_train, y_train, x_val, y_val, output_size = td.getTrainingData(cat)

    print(cat)

    if cat == "fashion":
        model = LSTM(embed, 100, 150, output_size)
    else: 
        model = LSTM(embed, 100, 100, output_size)
    model.load_state_dict(torch.load("models/"+cat+".text.pth"))
    model.eval()
    print(model)

    x_train = [model(processSentence(sentence)) for sentence in x_train]
    y_train = [torch.tensor(y).to(device).long() for y in y_train]

    x_val = [model(processSentence(sentence)) for sentence in x_val]
    y_val = [torch.tensor(y).to(device).long() for y in y_val]

    # ADD UR IMAGE CODE HERE, PROCESS IMAGE TRAIN AND IMAGE VAL AND OUTPUT A [0, 0, 0.... 1, 0, 0] TENSOR
    # @LICHAO

    image_train_X, image_train_y, image_val_X, image_val_y, output_size = td.getTrainingImages(cat)

    dset_train = DatasetProcessing(image_train_X, image_train_y, image_transform)
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size,
                                          shuffle=True, num_workers=0)


    dset_val = DatasetProcessing(image_val_X, image_val_y, image_transform)
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

    image_model = models.resnet50(pretrained=True)

    for param in image_model.parameters():
        param.requires_grad = False
    
    image_model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, output_size),
                                 nn.LogSoftmax(dim=1))

    image_model.load_state_dict(torch.load("models/"+cat+".image.pth"))
    image_model.to(device)
    image_model.eval()
    print(image_model)

    nb_batches = round(len(image_train_X) / batch_size)
    image_X_train = []
    image_y_train = []
    image_X_val = []
    image_y_val = []

    for i, (x_temp, y_temp) in enumerate(train_loader):
        output = image_model(x_temp)
        image_X_train.append(output)
        image_y_train.append(y_temp)
        print('Progress:', 100 * i / nb_batches, '%')

        if len(x_temp) < batch_size:
            break

    image_X_train = torch.cat((image_X_train), 0)
    image_y_train = torch.cat((image_y_train), 0)
    print(image_X_train.size(), image_y_train.size())

    for i, (x_temp, y_temp) in enumerate(val_loader):
        output = image_model(x_temp)
        image_X_val.append(output)
        image_y_val.append(y_temp)
        print('Progress:', 100 * i / nb_batches, '%')

        if len(x_temp) < batch_size:
            break

    image_X_val = torch.cat((image_X_val), 0)
    image_y_val = torch.cat((image_y_val), 0)
    print(image_X_val.size(), image_y_val.size())

    pickle.dump([x_train, y_train, x_val, y_val], open("data/" + cat + ".pickle", "wb"))