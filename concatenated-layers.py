# Combining both layers
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
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing import image

# DATAFRAME OF TRAINING SAMPLES
# Reading csv file, delimiter = ','
data_df = pd.read_csv('data/train.csv', delimiter = ',')
print(data_df.head())
print('------------------------------------------------')

# Extract the iamge text and image category
texts = data_df['title']
targets = data_df['Category']
nb_categories = targets.nunique()

# encode targets into vectors of dimension 58
targets = keras.utils.to_categorical(targets, nb_categories)

"""
-----------------------SPECIFICALLY FOR TEXT PREDICTIONS ONLY-------------------------
"""
# Determine the maximum length of the texts
max_sentence_length = 0
for sentence in texts:
	sentence_length = len(sentence)
	if sentence_length > max_sentence_length:
		max_sentence_length = sentence_length
print('The maximum length of the texts is ', max_sentence_length)

maxlen = max_sentence_length
training_samples = len(texts)
# Number of training samples found to be 666615
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
print('Shape of target tensor', targets.shape)

x_train = texts[:training_samples]
y_train = targets[:training_samples]
x_val = texts[training_samples:]
y_val = targets[training_samples:]

y_train = targets[:training_samples]
y_val = targets[training_samples:]

"""
-----------------------SPECIFICALLY FOR IMAGE PREDICTIONS ONLY-------------------------
"""
img_path = data_df['image_path']
train_dir = []
for path in img_path:
	train_dir.append(os.path.join(base_dir, path))

print('Total images found', len(train_dir))

# images must be resized, converted to matrix before passing into .flow since .flow cannot 
# read image directories
n = 1
image_matrix = []
print('Image to matrix conversion starting...')
for path in train_dir:
	img = load_img(path, target_size = (150, 150))
	x = image.img_to_array(img)
	image_matrix.append(x)
	# print('Converted', n, 'image', 'Length of image matrix', len(image_matrix))
	if n % 100000 == 0:
		percentage = n * 100 / len(train_dir)
		print('Converted ', '%.2f' %percentage, '%')

	n += 1

image_matrix = np.array(image_matrix)
print('Shape of image matrix is ', image_matrix.shape)
print('Shape of target vector', targets.shape)

# split to train and validation from training set
x_train = image_matrix[:training_samples]
y_train = targets[:training_samples]
x_val = image_matrix[training_samples:]
y_val = targets[training_samples:]
print('Total training set', len(x_train))
print('Total validation set', len(x_val))

# DATA PREPROCESSING
nb_train_samples = len(x_train)
nb_validation_samples = len(x_val)
batch_size = 10
train_sample_size =  nb_train_samples // batch_size
validation_sample_size = nb_validation_samples // batch_size
nb_epochs = 100

train_datagen = ImageDataGenerator(rescale = 1./255,
									shear_range = 0.2, 
									zoom_range = 0.2, 
									horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow(x_train, y_train, batch_size = batch_size)
validation_generator = test_datagen.flow(x_val, y_val, batch_size = batch_size)

"""
----------------------------INITIALISING MODEL LAYERS----------------------------------
"""
# MODEL LAYERS FOR TEXT CLASSIFICATION 
text_input = Input(shape = (None, ), dtype = 'int32', name = 'texts')
embedded_text = layers.Embedding(max_words, 100)(text_input)
text_model = layers.LSTM(64)(embedded_text)

# MODEL LAYERS FOR IMAGE CLASSIFICATION
image_input = Input(shape = (150, 150, 3), dtype = 'float32', name = 'images')
image_model = layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3))(image_input)
image_model = layers.MaxPooling2D((2, 2))(image_model)
image_model = layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (150, 150, 3))(image_model)
image_model = layers.MaxPooling2D((2, 2))(image_model)
image_model = layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (150, 150, 3))(image_model)
image_model = layers.MaxPooling2D((2, 2))(image_model)
image_model = layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (150, 150, 3))(image_model)
image_model = layers.MaxPooling2D((2, 2))(image_model)
image_model = layers.Flatten()(image_model)

# merging both models
merge = concatenate([text_model, image_model])

# output layers
output_layer = Dense(128, activation = 'relu')(merge)
output = Dense(58, activation = 'softmax')(output_layer)

model = Model(inputs = [image_input, text_input], outputs = output)

# summarize model
print(model.summary())

# Compiling the model
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

model.fit([], targets, epochs = nb_epochs, batch_size = batch_size)

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