#!/usr/bin/env python3

import pickle
import random
import Levenshtein

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# LOAD WORD2VEC DATA
print("Loading dictionary of embeddings")
f = open("data/dict.pkl", "rb")
dictionary = pickle.load(f)
f.close()

words, embeddings = list(dictionary.keys()), list(dictionary.values())

characters = list('abcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()<>-_+=`\'\"\\/[]:;~')
numChar = len(characters)

def replace(word):
    arr = []
    for c in word:
        carr = [0] * numChar 
        carr[characters.index(c)] = 1
        arr.append(carr)
    return arr

words = [replace(word) for word in words]


index = 0
def getNextTrainingExample():
    global index
    if index > len(words) - 1:
        index = 0
    return torch.tensor(words[index]).to(device).float(), torch.tensor(embeddings[index]).to(device).float()

# DECLARE LSTM CLASS
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True).to(device)

        self.fc = nn.Linear(hidden_size * 2, output_size).to(device)
        self.hidden = self.initHidden()

    def forward(self, input):
        input = input.unsqueeze(1)
        lstm_out, _ = self.lstm(input, self.hidden)
        out = torch.tanh(self.fc(torch.cat((lstm_out[-1, :, :self.hidden_size], lstm_out[0, :, self.hidden_size:]), 1))).to(device)
        return out

    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size).to(device), torch.zeros(2, 1, self.hidden_size).to(device))


lstm = LSTM(numChar, 500, 100)


criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(lstm.parameters(), 0.01)

def train(x, y):
    lstm.zero_grad()

    lstm.hidden = lstm.initHidden()

    output = lstm(x)

    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    return output, loss.item()


import time
import math

n_iters = len(words) * 10
print_every = 1000
validate_every = 1000
plot_every = 50

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()


print("Training the model\n")
print("Number of words:", len(words))

for iter in range(1, n_iters + 1):
    # Get a random example and train
    x, y = getNextTrainingExample()
    if len(x) == 0:
        continue
    output, loss = train(x, y)
    current_loss += loss

    # Print iter number, loss
    if iter % print_every == 0:
        print("Iter: %6d / %7d %5.2f%% %s" % (iter, n_iters, iter / n_iters * 100.0, timeSince(start)), "trg_loss:", loss)

    # Validate
    if iter % validate_every == 0:
        print(lstm(torch.tensor(replace("bb")).to(device).float()))
        print(lstm(torch.tensor(replace("cream")).to(device).float()))
        print(dictionary["cream"])


    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
