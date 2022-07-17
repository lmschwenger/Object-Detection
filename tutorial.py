# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:43:01 2022

@author: lasse
"""
import os

current_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_folder)

# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from pyimagesearch.detection_helpers import sliding_window
from pyimagesearch.detection_helpers import image_pyramid
import keras
from matplotlib import pyplot as plt
import numpy as np
import argparse
import imutils
import time
import cv2
import torch
def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	fig, ax = plt.subplots(1, dpi=200, figsize=(12,7))
	ax.imshow(image)
	ax.set_title(title)
	ax.grid(False)
	plt.show()

args = {
	"image": "images/test_circuit2.png",
	"size": "(30, 30)",
	"min_conf": 0.9,
	"visualize": -1
}



# initialize variables used for the object detection procedure
img_height = 128
img_width = 128
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (img_height, img_width)
CLASSES = sorted(['Ammeter', 'Voltmeter', 'Motor', 'Resistor', 'Inductor', 'Lamp'])
# load our network weights from disk
print("[INFO] loading network...")
#model = ResNet50(weights="imagenet", include_top=True)
#%% Model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from my_model import make_model
from sklearn.model_selection import train_test_split
batch_size = 256

data_dir = 'E:\GitHub\Image Recognition\Test Images'

train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
  validation_split=0.2,
  subset="training",
  color_mode='grayscale',
  seed=30,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  color_mode='grayscale',
  seed=30,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#train_ds = builder.as_dataset(split='test+train[:75%]')
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train = train_ds.map(lambda x, y: (normalization_layer(x), y))

normalized_val = test_ds.map(lambda x, y: (normalization_layer(x), y)) 
num_classes = 6
model = models.Sequential([

    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(6, 1, activation='relu', trainable=True),
    layers.MaxPooling2D(2),
   # layers.Conv2D(16, 1, activation='relu', trainable=True),
    # layers.MaxPooling2D(2),    
    # layers.Conv2D(120, 1, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu', trainable=True),
    layers.Dropout(0.8),
    # layers.Dense(1024, activation='relu'),
    # layers.Dropout(0.6),
    layers.Dense(1024, activation='relu', trainable=True),
    layers.Dropout(0.8),
    layers.Dense(num_classes, activation='softmax')
])
lr_schedule = 0.00005

train_ds = train_ds.prefetch(buffer_size=64)
test_ds = test_ds.prefetch(buffer_size=64)
#model = make_model(input_shape= (img_height, img_width) + (3,), num_classes=6)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, decay=1e-6),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), # from_logits=True
              metrics=['accuracy'])
history = model.fit(train_ds, epochs=150, validation_data=test_ds)

model.evaluate(test_ds)

model.save_weights('./checkpoints/my_checkpoint')
#%%
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
plt.figure(figsize=(10, 10))

for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
        
#%%
from plots import performance_plot
performance_plot(history)

#%% Showing subset of training images
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
  for i in range(9):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(CLASSES[labels[i]])
    plt.axis("off")
    #%%
# load the input image from disk, resize it such that it has the
# supplied width, and then grab its dimensions
orig = cv2.imread(args["image"])
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
orig = imutils.resize(orig, width=256)
(H, W) = orig.shape[:2]

model.load_weights('./checkpoints/my_checkpoint')
# initialize the image pyramid
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)

# initialize two lists, one to hold the ROIs generated from the image
# pyramid and sliding window, and another list used to store the
# (x, y)-coordinates of where the ROI was in the original image
rois = []
locs = []

# time how long it takes to loop over the image pyramid layers and
# sliding window locations
start = time.time()

# loop over the image pyramid
for image in pyramid:
	# determine the scale factor between the *original* image
	# dimensions and the *current* layer of the pyramid
	scale = W / float(image.shape[1])

	# for each layer of the image pyramid, loop over the sliding
	# window locations
	for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
		# scale the (x, y)-coordinates of the ROI with respect to the
		# *original* image dimensions
		x = int(x * scale)
		y = int(y * scale)
		w = int(ROI_SIZE[0] * scale)
		h = int(ROI_SIZE[1] * scale)

		# take the ROI and preprocess it so we can later classify
		# the region using Keras/TensorFlow
		roi = cv2.resize(roiOrig, INPUT_SIZE)
		roi = img_to_array(roi)
		#roi = preprocess_input(roi)

		# update our list of ROIs and associated coordinates
		rois.append(roi)
		locs.append((x, y, x + w, y + h))

        # check to see if we are visualizing each of the sliding
		# windows in the image pyramid
		if args["visualize"] > 0:
			# clone the original image and then draw a bounding box
			# surrounding the current region
			clone = orig.copy()
			cv2.rectangle(clone, (x, y), (x + w, y + h),
				(0, 255, 0), 2)

			# show the visualization and current ROI
			plt_imshow("Visualization", clone)
			plt_imshow("ROI", roiOrig)

# show how long it took to loop over the image pyramid layers and
# sliding window locations
end = time.time()
print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
	end - start))

# convert the ROIs to a NumPy array
rois = np.array(rois, dtype="float32")

# classify each of the proposal ROIs using ResNet and then show how
# long the classifications took
print("[INFO] classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(
	end - start))
highest_prob_col = preds.argmax(axis=-1)
preds_dir = []
for i in range(0, len(preds)):
    preds_dir.append((i,CLASSES[highest_prob_col[i]], preds[i,highest_prob_col[i]]))
# decode the predictions and initialize a dictionary which maps class
# labels (keys) to any ROIs associated with that label (values)
#preds = imagenet_utils.decode_predictions(preds, top=1)
#preds = np.argmax(preds)
#%%
labels = {}

# loop over the predictions
for (i, p) in enumerate(preds_dir):
	# grab the prediction information for the current ROI
	(imagenetID, label, prob) = p

	# filter out weak detections by ensuring the predicted probability
	# is greater than the minimum probability
	if prob >= args["min_conf"]:
		# grab the bounding box associated with the prediction and
		# convert the coordinates
		box = locs[i]

		# grab the list of predictions for the label and add the
		# bounding box and probability to the list
		L = labels.get(label, [])
		L.append((box, prob))
		labels[label] = L
        
# loop over the labels for each of detected objects in the image
for label in labels.keys():
	# clone the original image so that we can draw on it
	print("[INFO] showing results for '{}'".format(label))
	clone = orig.copy()

	# loop over all bounding boxes for the current label
	for (box, prob) in labels[label]:
		# draw the bounding box on the image
		(startX, startY, endX, endY) = box
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)

	# show the results *before* applying non-maxima suppression, then
	# clone the image again so we can display the results *after*
	# applying non-maxima suppression
	plt_imshow("Before", clone)
	clone = orig.copy()

    # extract the bounding boxes and associated prediction
	# probabilities, then apply non-maxima suppression
	boxes = np.array([p[0] for p in labels[label]])
	proba = np.array([p[1] for p in labels[label]])
	boxes = non_max_suppression(boxes, proba)

	# loop over all bounding boxes that were kept after applying
	# non-maxima suppression
	i=0
	for (startX, startY, endX, endY) in boxes:
        
		# draw the bounding box and label on the image
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(clone, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
		cv2.putText(clone, str(round(proba[i], 2)), (endX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
		i+=1

	# show the output after apply non-maxima suppression
	plt_imshow("After", clone)