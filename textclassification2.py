import os, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import keras
from keras import preprocessing
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras import Sequential


training_data_filepath = "data/train.csv"
chunk_size = 1000

csv_headers = ["itemid", "title", "Category", "image_path"]

number_of_categories = 58


# Takes a string and converts it to an array of numbers
# Each word is a single number
# Each char is 2 digits
# Case insensitive (a = 10, b = 11, ... z = 25)
# Numbers are mapped like this (0 -> 50, 1 -> 51, ... 9 -> 59)
# Special symbols (./#-_!@...etc) ---> 99

def categoryToBinaryVector(category):
	binVec = [0] * number_of_categories
	binVec[category] = 1
	return binVec

def stringToNumberArray(title):
	numberArr = []
	title = title.lower()
	titleArr = title.split(" ")
	for word in titleArr:
		number = ""
		letterArr = list(word)
		for char in letterArr:
			if not char.isalpha():
				if char.isdigit():
					number += str(int(char) + 50)
				else:
					number += str(99)
				break

			number += str(ord(char) - 87)
		numberArr.append(int(number))
	return numberArr


model = Sequential()
model.add(layers.Conv1D(filters=30, kernel_size=2, input_shape=(None, None), activation= 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'tanh'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(58, activation = 'softmax'))
model.summary()

# compiling the model
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])



for chunk in pd.read_csv(training_data_filepath, chunksize=chunk_size):
	titles = chunk[csv_headers[1]].apply(stringToNumberArray)
	expected_output = chunk[csv_headers[2]].apply(categoryToBinaryVector)

	history = model.fit(titles[:800], expected_output[:800], epochs = 100, batch_size = 10, validation_data = [titles[800:], expected_output[800:]])

