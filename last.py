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
import pandas as pd

from data import TrainingData


# Use CUDA if available -> replacing every .cuda() with .to(device)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def processSentence(sentence, i):
    newX = []
    for word in sentence.split(" "):
        if word in words:
            newX.append(words.index(word))
        else:
            jaros = [Levenshtein.jaro(word, w) for w in words]
            highest_index = jaros.index(max(jaros))
            newX.append(highest_index)
    newX = torch.tensor(newX).to(device).long()
    print(str(i)+"\r", end="")
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

cats = ["beauty", "mobile", "fashion"]
models = []
td = TrainingData(0.1)
for cat in cats:
    _, _, _, _, output_size = td.getTrainingData(cat)
    if cat == "fashion":
        model = LSTM(embed, 100, 150, output_size)
    else: 
        model = LSTM(embed, 100, 100, output_size)
    model.load_state_dict(torch.load("models/"+cat+".text.pth"))
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    models.append(model)

data = pd.read_csv("data/test.csv", delimiter=",")

ids = []
outputs = []
for i, sentence in enumerate(data["title"]):
    category = data["image_path"][i].split("/")[0].split("_")[0]
    index = cats.index(category)
    output = torch.max(models[index](processSentence(sentence, i))[0], 0)[1].item()
    if category == "fashion":
        print("yess")
        output += 17
    elif category == "mobile":
        output += 31
    outputs.append(output)
    ids.append(data["itemid"][i])

output = np.hstack((np.array(ids).reshape((-1, 1)), np.array(outputs).reshape((-1, 1))))
np.savetxt("data/submission.csv", output, fmt="%i", delimiter=",", header="itemid,Category", comments="")