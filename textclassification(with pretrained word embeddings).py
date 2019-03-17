# text classification with pretrained word embeddings
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
from keras.utils import to_categorical

# DATAFRAME OF TRAINING SAMPLES
# Reading csv file, delimiter = ','
data_df = pd.read_csv('data/train.csv', delimiter = ',')
print(data_df.head())
print('------------------------------------------------')

# extract the description of the image, this is the texts sequence that will be used to train the model
texts = data_df['title']

# extract the category of the image, this is the target values to be predicted from the training set
targets = data_df['Category']

# Number of training samples found to be 666615, 58 categories

# encode text into vector 
embed_size = 50
# max_features = 30000
max_features = 50000
maxlen = 100
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# sequences = tokenizer.texts_to_sequences(texts)
texts = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor', texts.shape)

# encode targets into vectors
targets = keras.utils.to_categorical(targets, 58)
print('Shape of target tensor', targets.shape)
print(targets[98])

# This is the folder with the EMBEDDING MATRIX TXT FILE
glove_dir = '/Users/User/Desktop/NDSC'

embeddings_index = {} # We create a dictionary of word -> embedding

# with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), 'r', encoding='utf-8') as f:
#     for line in f:
#         values = line.split()
#         word = values[0] # The first value is the word, the rest are the values of the embedding
#         embedding = np.asarray(values[1:], dtype='float32') # Load embedding
#         embeddings_index[word] = embedding # Add embedding to our embedding dictionary

# print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))

# Since the choice of GloVe embedding matrix used here has 100 dimensions
embedding_dim = 300

word_index = tokenizer.word_index

# How many words are there actually
nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embedding_dim))

# The vectors need to be in the same position as their index. 
# Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on

# Loop over all words in the word index
for word, i in word_index.items():
    # If we are above the amount of words we want to use we do nothing
    if i >= max_features: 
        continue
    # Get the embedding vector for the word
    embedding_vector = embeddings_index.get(word)
    # If there is an embedding vector, put it in the embedding matrix
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector

# SHUFFLING THE DATA
indices = np.arange(texts.shape[0])
np.random.shuffle(indices)
texts = texts[indices]
targets = targets[indices]

x_train = texts
y_train = targets
print('Shape of x_train is ', x_train.shape)
print('Shape of y_train is ', y_train.shape)

# MODEL LAYERS FOR DENSE MODEL
# defining the structure of the model
model = Sequential()
model.add(layers.Embedding(max_features, 
                    embedding_dim, 
                    input_length=maxlen, 
                    weights = [embedding_matrix], 
                    trainable = False))
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.MaxPooling1D())
model.add(layers.LSTM(256, dropout=0.1, recurrent_dropout=0.1))
# model.add(layers.LSTM(300, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
# model.add(layers.Flatten())
model.add(layers.Dense(58, activation='softmax'))
model.summary()

# compiling the model
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

nb_epochs = 6

# train the model
history = model.fit(x_train, y_train, epochs = nb_epochs, batch_size = 600, validation_split=0.1)

print("Done training")
# DATAFRAM OF TEST DATA
test_df = pd.read_csv('data/test.csv')

print("Reading test data")

texts = test_df['title']
ids = test_df['itemid']

tokenizer = Tokenizer(num_words = max_features)
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
output = np.hstack((ids.values.reshape((-1, 1)), catNumber.astype(int).reshape(-1, 1)))
np.savetxt("data/submission-lc.csv", output, fmt="%i", delimiter=",", header="itemid,Category", comments="")

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