#!/usr/bin/env python3

import os
import pickle
import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

import unicodedata
import string


# LOAD DATA
print("Loading dictionary of embeddings")
f = open("data/dict.pkl", "rb")
dictionary = pickle.load(f)
f.close()

words = list(dictionary.keys())
embeddings = list(dictionary.values())

print("Number of unique words:", len(words))

all_letters = string.ascii_letters + ".,:;'\"/\\-!$()@#%^&`~?"
n_letters = len(all_letters)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self.initHidden()

    def forward(self, input):
        lstm_out, _ = self.lstm(input, self.hidden)
        out = self.tanh(self.fc(lstm_out[-1]))
        return out

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

# NEURAL NET

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def randomTrainingExample():
	i = random.randint(0, len(words) - 1)
	training_example = lineToTensor(words[i])
	output_tensor = torch.zeros(1, 100)
	for li, x in enumerate(embeddings[i]):
		output_tensor[0][li] = x
	return output_tensor, training_example

lstm = LSTM(n_letters, 200, 100)

criterion = nn.MSELoss()
crtierion2 = nn.L1Loss()
lr = 0.01
optimizer = optim.Adam(lstm.parameters(), lr)

def train(category_tensor, line_tensor):
    lstm.zero_grad()

    lstm.hidden = lstm.initHidden()

    output = lstm(line_tensor)

    loss = criterion(output, category_tensor)
    loss2 = crtierion2(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss2.item()

# TRAINING
import time
import math

n_iters = len(words) * 20
#n_iters = 10000
print_every = 500
decay_every = 15000
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

for iter in range(1, n_iters + 1):
    category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        print("Iter: %6d %3.2f%% %s" % (iter, iter / n_iters * 100, timeSince(start)), loss)

    # Decay learning rate
    if iter % decay_every == 0:
    	lr = lr * 0.5
    	for param_group in optimizer.param_groups:
        	param_group['lr'] = lr

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

print(embeddings[words.index("beauty")])

print(lstm(lineToTensor("beauty")))
print(lstm(lineToTensor("mobile")))
print(F.cosine_similarity(lstm(lineToTensor("beauty")), lstm(lineToTensor("mobile")), dim=1))
print(F.cosine_similarity(lstm(lineToTensor("cream")), lstm(lineToTensor("bb"))))
print(F.cosine_similarity(lstm(lineToTensor("ebook")), lstm(lineToTensor("ebooks"))))
print(F.cosine_similarity(lstm(lineToTensor("lipstick")), lstm(lineToTensor("glossy"))))
print(F.cosine_similarity(lstm(lineToTensor("skirt")), lstm(lineToTensor("men"))))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)
plt.show()