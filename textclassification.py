# TEXT CLASSIFICATION

import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import csv

import pandas as pd
import keras
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras import Sequential 

# Reading csv file, delimiter = ','

# DATAFRAME OF TRAINING SAMPLES
data_df = pd.read_csv('data/train.csv', delimiter = ',')
print(data_df.head())
print('------------------------------------------------')

# extract the description of the image, this is the texts sequence that will be used to train the model
texts = data_df['title']

# determine the maximum length of the texts
max_sentence_length = 0
for sentence in texts:
	sentence_length = len(sentence)
	if sentence_length > max_sentence_length:
		max_sentence_length = sentence_length

print('The maximum length of the texts is ', max_sentence_length)

# extract the category of the image, this is the target values to be predicted from the training set
targets = data_df['Category']

nb_categories = targets.nunique()

maxlen = max_sentence_length
training_samples = len(texts)
#Number of training samples found to be 666615
print('Number of training samples', training_samples)
# split training set into train and validation
training_samples = 600000
validation_samples = 66615
max_words = 10000

# encode text into vector 
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# sequences = tokenizer.texts_to_matrix(texts, mode='binary')

word_index = tokenizer.word_index
print('Found %s unique tokens.' %len(word_index))

texts = pad_sequences(sequences, maxlen= maxlen)
print('Shape of data tensor', texts.shape)

# encode targets into vectors
targets = keras.utils.to_categorical(targets, nb_categories)
print('Shape of target tensor', targets.shape)

x_train = texts[:training_samples]
y_train = targets[:training_samples]
print('Shape of x_train is ', x_train.shape)
print('Shape of y_train is ', y_train.shape)

x_val = texts[training_samples:]
y_val = targets[training_samples:]

# MODEL LAYERS FOR DENSE MODEL
# defining the structure of the model
model = Sequential()
model.add(layers.Embedding(max_words, 100, input_length = maxlen))
model.add(layers.Flatten())
model.add(Dropout())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(nb_categories, activation = 'softmax'))
model.summary()

# compiling the model
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

# train the model
history = model.fit(x_train, y_train, epochs = 100, batch_size = 10, validation_data = [x_val, y_val])

train_acc = history.history['acc']
train_loss = history.history['loss']

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epoch_index = range(1, 10 + 1)

plt.plot(epoch_index, train_acc, 'bo', label = 'Training Accuracy')
plt.plot(epoch_index, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epoch_index, train_loss, 'bo', label = 'Training Loss')
plt.plot(epoch_index, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()