#!/usr/bin/env python3

import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

import numpy as np

from data import TrainingData


validation_percent = 0.1

# LOAD TRAINING DATA
td = TrainingData(validation_percent)
x_train = td.x_train
y_train = td.y_train
x_val = td.x_val
y_val = td.y_val


# LOAD WORD2VEC DATA
print("Loading dictionary of embeddings")
f = open("data/dict.pkl", "rb")
dictionary = pickle.load(f)
f.close()

words = list(dictionary.keys())


# COPY WORD2VEC EMBEDDINGS INTO NN.EMBEDDING LAYER
vocab_size = len(dictionary)
vector_size = 100
pretrained_weights = list(dictionary.values())
# vocab_size is the number of words in your train, val and test set
# vector_size is the dimension of the word vectors you are using
embed = nn.Embedding(vocab_size, vector_size)

# intialize the word vectors, pretrained_weights is a 
# numpy array of size (vocab_size, vector_size) and 
# pretrained_weights[i] retrieves the word vector of
# i-th word in the vocabulary
embed.weight.data.copy_(torch.tensor(pretrained_weights))


# DECLARE LSTM CLASS
class LSTM(nn.Module):
    def __init__(self, embeddings, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

        self.embeddings = embeddings
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self.initHidden()

    def forward(self, input):
        embed = self.embeddings(input)
        embed = embed.unsqueeze(1)
        lstm_out, _ = self.lstm(embed, self.hidden)
        out = F.log_softmax(self.fc(lstm_out[-1]), dim=1)
        return out

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))


lstm = LSTM(embed, 100, 200, 58)

criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.Adam(lstm.parameters(), lr)

def train(x, y):
    lstm.zero_grad()

    lstm.hidden = lstm.initHidden()

    output = lstm(x)

    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    return output, loss.item()


############
# TRAINING #
############
import time
import math

n_iters = len(x_train) * 10
print_every = 1000
validate_every = 5000
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

# THIS FUNCTION WILL RETURN ONE RANDOM SET OF (INPUT, EXPECTED OUTPUT)
# TAKEN FROM THE TRAINING DATA. CURRENTLY DOES ON-THE-FLY PROCESSING OF 
# THE INPUT, MIGHT WANT TO MOVE THE PROCESSING TO BEFORE TRAINING
def randomTrainingExample():
	i = random.randint(0, len(x_train) - 1)
	newX = []
	for word in x_train[i].split(" "):
		if word in words:
			newX.append(words.index(word))

	newY = torch.tensor([y_train[i]])
	return torch.tensor(newX).long(), newY.long()


print("Training the model")
print("Number of iterations:", n_iters)

for iter in range(1, n_iters + 1):
	# Get a random example and train
    x, y = randomTrainingExample()
    if len(x) == 0:
        continue
    output, loss = train(x, y)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        print("Iter: %6d %.1f%% %s" % (iter, iter / n_iters * 100, timeSince(start)), loss)

    # Validate
    if iter % validate_every == 0:
        correct = 0
        for i, x in enumerate(x_val):
            newX = []
            for word in x.split(" "):
                if word in words:
                    newX.append(words.index(word))
            newX = torch.tensor(newX).long()
            if len(newX) == 0:
            	continue
            output = lstm(newX)
            if torch.max(output[0], 0)[1].item() == y_val[i]:
                correct += 1
        correct = correct / len(x_val) * 100
        print("val_acc:", correct, "%")

    # Decay learning rate
    if iter % decay_every == 0:
    	lr = lr * 0.5
    	for param_group in optimizer.param_groups:
        	param_group['lr'] = lr

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0