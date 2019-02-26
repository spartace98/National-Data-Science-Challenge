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

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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

nb_categories = 58

maxlen = max_sentence_length
training_samples = len(texts)
#Number of training samples found to be 666615
print('Number of training samples', training_samples)
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

texts, targets = unison_shuffled_copies(texts, targets)

# MODEL LAYERS FOR DENSE MODEL
# defining the structure of the model
model = Sequential()
model.add(layers.Embedding(max_words, 100, input_length = maxlen))
model.add(layers.Conv1D(32, 2, activation='linear'))
model.add(layers.PReLU())
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(32, 2, activation='linear'))
model.add(layers.PReLU())
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='tanh'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(nb_categories, activation = 'softmax'))
model.summary()

# compiling the model
model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

nb_epochs = 10

# train the model
history = model.fit(texts, targets, epochs = nb_epochs, batch_size = 600, validation_split = 0.15)

print("Done training")
# DATAFRAM OF TEST DATA
test_df = pd.read_csv('data/test.csv')

print("Reading test data")

texts = test_df['title']
ids = test_df['itemid']

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
texts = pad_sequences(sequences, maxlen= maxlen)

print("Predicting")
predictions = model.predict(texts)

print("Outputting")
i = 0
catNumber = np.zeros(predictions.shape[0])
for p in predictions:
	catNumber[i] = np.argmax(p)
	i += 1
output = np.hstack((ids.to_numpy().reshape((-1, 1)), catNumber.astype(int).reshape(-1, 1)))
np.savetxt("data/submission.csv", output, fmt="%i", delimiter=",", header="itemid,Category", comments="")

print("Done")

train_acc = history.history['acc']
train_loss = history.history['loss']

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epoch_index = range(1, nb_epochs + 1)

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