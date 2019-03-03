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


# ADJUSTABLE PARAMETERS
validation_percent = 0.1
category = "fashion"
chunk_size = 20000


# LOAD TRAINING DATA
print("Loading training data")
td = TrainingData(validation_percent)
x_train, y_train, x_val, y_val, output_size = td.getTrainingData(category)

# LOAD WORD2VEC DATA
print("Loading dictionary of embeddings")
f = open("data/dict.pkl", "rb")
dictionary = pickle.load(f)
f.close()

words = list(dictionary.keys())


# PROCESS TRAINING DATA
print("Processing training data")
x_train_chunked = [x_train[i:i + chunk_size] for i in range(0, len(x_train), chunk_size)]

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

current_training_example = -1
current_chunk = 0
def processChunk(current_chunk):
    global x_train
    x_train = [processSentence(sentence) for sentence in x_train_chunked[current_chunk]]
processChunk(current_chunk)

def getNextTrainingExample():
    global current_training_example
    global current_chunk
    current_training_example += 1
    if current_training_example > len(x_train) - 1:
        current_chunk += 1
        processChunk(current_chunk % len(x_train_chunked))
        current_training_example = -1

    newY = torch.tensor([y_train[current_training_example + current_chunk * chunk_size]]).to(device)
    return x_train[current_training_example], newY.long()

# PROCESS VALIDATION DATA
print("Processing validation data")
processed_x_val = []
processed_y_val = []
for i, sentence in enumerate(x_val):
    processed_x_val.append(processSentence(sentence))
    processed_y_val.append(torch.tensor([y_val[i]]).to(device).long())


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



# MODEL PARAMETERS
hidden_size = 100
lr = 0.01

lstm = LSTM(embed, 100, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss().to(device)
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

n_iters = (len(x_train_chunked) * (chunk_size - 1) + len(x_train_chunked[-1])) * 10
print_every = 1000
validate_every = 5000
decay_every = len(x_train)
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
print("CATEGORY:", category)
print("NO OF CLASSES:", output_size)

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
        correct = 0
        losses = 0
        for i, x in enumerate(processed_x_val):
            output = lstm(x)
            losses += criterion(output, processed_y_val[i]).item()
            if torch.max(output[0], 0)[1].item() == processed_y_val[i].item():
                correct += 1
        correct = correct / len(x_val) * 100
        loss = losses / len(x_val)
        print("val_loss:", loss, " val_acc:", correct, "%")

    # Decay learning rate
    if iter % decay_every == 0:
    	lr = lr * 0.5
    	for param_group in optimizer.param_groups:
        	param_group['lr'] = lr

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
