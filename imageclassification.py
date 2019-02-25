# IMAGE CLASSIFICATION
import os, shutil
import numpy as np
import matplotlib.pyplot as plt

import keras
import pandas as pd
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing import image

# IMPORTING DATASET
# creating links to the relevant folders 
base_dir = '/Users/User/Desktop/NDSC'
# words in quotes is the name of the folder where images are stored
train_dir = os.path.join(base_dir, 'train')
# test_dir = os.path.join(base_dir, 'test')

data_df = pd.read_csv('data/train.csv', delimiter = ',')
# print(data_df.head())
# print('------------------------------------------------')

# extracting image path and store in array
img_path = data_df['image_path']
train_dir = []
for path in img_path:
	train_dir.append(os.path.join(base_dir, path))

print('Total images found', len(train_dir))

# extracting categories  
targets = data_df['Category']
nb_categories = targets.nunique()

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

# encode targets into vectors
targets = keras.utils.to_categorical(targets, nb_categories)
print('Shape of target tensor', targets.shape)

# split to train and validation from training set
x_train = image_matrix[:600000]
y_train = targets[:600000]
x_val = image_matrix[600000:]
y_val = targets[600000:]
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

# Initialising the model layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

# adding a classifer on top of the convnet
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(58, activation = 'sigmoid'))

model.summary()

# compiling the model
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

# train the model
history = model.fit(x_train, y_train, epochs = nb_epochs, batch_size = 10, validation_data = [x_val, y_val])

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