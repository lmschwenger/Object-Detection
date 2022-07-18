# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:57:48 2022

@author: lasse
"""
# import the necessary packages
import os

current_folder = 'D:\GitHub\Object Detection'
os.chdir(current_folder)

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


#%% Loading images
batch_size = 390
img_height = 180
img_width = 180
data_dir = 'D:\GitHub\Image Recognition\Test Images'

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

print(train_ds.class_names)
n_classes = len(train_ds.class_names)
#train_ds = builder.as_dataset(split='test+train[:75%]')
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train = train_ds.map(lambda x, y: (normalization_layer(x), y))

normalized_val = test_ds.map(lambda x, y: (normalization_layer(x), y)) 

#%% Defining model
def create_model(img_height, img_width, n_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

model = create_model(img_height, img_width, n_classes)
normalized_train =normalized_train.prefetch(tf.data.experimental.AUTOTUNE )
normalized_val =normalized_val.prefetch(tf.data.experimental.AUTOTUNE )
history = model.fit(normalized_train, epochs=50, 
                    validation_data=normalized_val)
# Save the weights using the `checkpoint_path` format
model.save('D:/savedModels/OD_model_TF')

