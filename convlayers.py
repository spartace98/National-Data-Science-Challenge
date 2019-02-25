# DNN FOR TEXT AND IMAGE CLASSIFICATION
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd
from keras import layers
from keras import Model
from keras.preprocessing.image import ImageDataGenerator, load_img

# IMPORTING DATASET
base_dir = ''
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

for files in os.listdir(train_dir):
    print(files)

# Data Preprocessing
# for image
# since the dataset isnt complicated items, a 150x150 pixel array should be sufficient
img_width, img_height = 224, 224
nb_train_samples = len(os.listdir(train_dir))
nb_validation_samples = len(os.listdir(validation_dir))
batch_size = # to be decided
train_sample_size = nb_train_samples // batch_size
validation_sample_size = nb_validation_samples // batch_size
epochs = 20

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size = (img_width, img_height), 
                                                    batch_size = batch_size, 
                                                    class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, 
                                                        target_size = (img_width, img_height), 
                                                        batch_size = batch_size, 
                                                        class_mode = 'binary')
test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size = (img_width, img_height), 
                                                  batch_size = batch_size, 
                                                  class_mode = 'binary')

# for text sequences
max_features = 10000 # dimensions of embedded layer
maxlen = 100
batch_size = 

# MODEL LAYERS FOR IMAGE CLASSIFICATION
image_input = Input(shape = (150, 150, 3), dtype = 'float32', name = 'imagetensor')
image_model = layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3))(image_input)
image_model = layers.MaxPooling2D((2, 2))(image_model)
image_model = layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (150, 150, 3))(image_model)
image_model = layers.MaxPooling2D((2, 2))(image_model)
image_model = layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (150, 150, 3))(image_model)
image_model = layers.MaxPooling2D((2, 2))(image_model)
image_model = layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (150, 150, 3))(image_model)
image_model = layers.MaxPooling2D((2, 2))(image_model)
image_model = layers.Flatten()(image_model)

# MODEL LAYERS FOR LTSM 
text_input = Input(shape = (None, ), dtype = 'int32', name = 'textmatrix')
embedded_text = layers.Embedding(64, max_features)(text_input)
text_model = layers.LSTM(64)(embedded_text)

# merging both models
merge = concatenate([image_model, text_model])

# output layers
output_layer = Dense(64, activation = 'relu')(merge)
output = Dense(1, activation = 'sigmoid')(output_layer)

model = Model(inputs = [image_input, text_input], outputs = output)

# summarize model
print(model.summary())