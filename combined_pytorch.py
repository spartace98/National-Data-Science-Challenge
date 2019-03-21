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

try:
    processed = pickle.load(open("data/processed.pickle", "rb"))
except (OSError, IOError) as e:
    td = TrainingData(validation_percent)
    x_combined = []
    y_combined = []

    for cat in ["beauty", "fashion", "mobile"]:
        x_train, y_train, x_val, y_val, output_size = td.getTrainingData(cat)

        # LOAD WORD2VEC DATA
        print("Loading dictionary of embeddings")
        f = open("data/dict.pkl", "rb")
        dictionary = pickle.load(f)
        f.close()

        words = list(dictionary.keys())

        x_train = [processSentence(sentence) for sentence in x_train]
        y_train = [torch.tensor(y).to(device).long() for y in y_train]

        x_val = [processSentence(sentence) for sentence in x_val]
        y_val = [torch.tensor(y).to(device).long() for y in y_val]


        image_train, _, image_val, _ = td.getTrainingImages(cat)

    pickle.dump([x_combined, y_combined, x_val, y_val], open("data/processed.pickle", "wb"))