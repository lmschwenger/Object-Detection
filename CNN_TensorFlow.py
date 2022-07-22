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
from pyimagesearch.detection_helpers import process

#%% Loading images
batch_size = 300
img_height = 180
img_width = 180
data_dir = r'D:\GitHub\Object Detection\trainingImages'

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

data_augmentation = tf.keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

#%% Defining model
def create_model(img_height, img_width, n_classes):
    model = models.Sequential()
    model.add(data_augmentation),
    model.add(layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))),
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu')),
    model.add(layers.MaxPooling2D(2, 2)),
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu')),
    model.add(layers.MaxPooling2D(2, 2)),
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu')),
    model.add(layers.MaxPooling2D(2, 2)),
    model.add(layers.Flatten()),
    model.add(layers.Dense(64, activation='relu')),
    #model.add(layers.Dropout(0.2)),

    model.add(layers.Dense(n_classes))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

model = create_model(img_height, img_width, n_classes)
model.summary()

# Defining Hatchback Class
class haltCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('val_accuracy') >= 0.99):
			print("\n\n\nReached 0.025 loss value so cancelling training!\n\n\n")
			self.model.stop_training = True

LearningCallback = haltCallback()

EPOCHS = 100

history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)
import plots
if input("See Results of training? (y/n): ") == "y":
    plots.training_results(history, EPOCHS)

# Save the weights using the `checkpoint_path` format
model.save('D:/savedModels/OD_model_TF')
