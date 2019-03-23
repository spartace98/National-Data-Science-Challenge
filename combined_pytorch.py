#!/usr/bin/env python3

import pickle
import random
import Levenshtein

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from data import TrainingData

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


    image_train, _, image_val, _ = td.getTrainingImages(cat)

    # ADD UR IMAGE CODE HERE, PROCESS IMAGE TRAIN AND IMAGE VAL AND OUTPUT A [0, 0, 0.... 1, 0, 0] TENSOR
    # @LICHAO


    pickle.dump([x_train, y_train, x_val, y_val], open("data/" + cat + ".pickle", "wb"))