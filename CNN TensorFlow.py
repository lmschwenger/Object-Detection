# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:57:48 2022

@author: lasse
"""
# import the necessary packages
import os

current_folder = 'E:\GitHub\Object Detection'
os.chdir(current_folder)

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


#%% Loading images
batch_size = 390
img_height = 180
img_width = 180
data_dir = 'E:\GitHub\Image Recognition\Test Images'

train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#train_ds = builder.as_dataset(split='test+train[:75%]')
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train = train_ds.map(lambda x, y: (normalization_layer(x), y))

normalized_val = test_ds.map(lambda x, y: (normalization_layer(x), y)) 

#%% Defining model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'], learning_rate=0.001)

history = model.fit(normalized_train, epochs=10, 
                    validation_data=normalized_val)


#%%
model.save_weights('CNN_tf_weights')
