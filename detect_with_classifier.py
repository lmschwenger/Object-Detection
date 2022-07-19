# USAGE
# python detect_with_classifier.py --image images/stingray.jpg --size "(300, 150)"
# python detect_with_classifier.py --image images/hummingbird.jpg --size "(250, 250)"
# python detect_with_classifier.py --image images/lawn_mower.jpg --size "(200, 200)" --min-conf 0.95

# import the necessary packages
import os

current_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_folder)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from pyimagesearch.detection_helpers import sliding_window
from pyimagesearch.detection_helpers import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2


#%%
# initialize variables used for the object detection procedure
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (25, 25)
INPUT_SIZE = (250, 250)

# load our the network weights from disk
print("[INFO] loading network...")
model = tf.keras.models.load_model('D:\savedModels\OD_model_TF')


image_path = r'D:\GitHub\Object Detection\images\test_circuit5.jpg'
min_conf = .20
visualize=0


# load the input image from disk, resize it such that it has the
# has the supplied width, and then grab its dimensions
orig = cv2.imread(image_path)
(H_o,W_o) = orig.shape[:2]
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]
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
i = 1
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

		# take the ROI and pre-process it so we can later classify
		# the region using Keras/TensorFlow
		roi = cv2.resize(roiOrig, INPUT_SIZE)
		roi = img_to_array(roi)
		#roi = preprocess_input(roi)

		# update our list of ROIs and associated coordinates
		rois.append(roi)
		locs.append((x, y, x + w, y + h))

		# check to see if we are visualizing each of the sliding
		# windows in the image pyramid
		if visualize > 0:
			# clone the original image and then draw a bounding box
			# surrounding the current region
			clone = orig.copy()
			cv2.rectangle(clone, (x, y), (x + w, y + h),
				(0, 255, 0), 2)

			# show the visualization and current ROI
			cv2.imshow("Visualization", clone)
			cv2.imshow("ROI", roiOrig)
			cv2.waitKey(0)
			#if i > 307:
			#	cv2.imwrite(r'D:\GitHub\Image Recognition\Test Images\wire\wire_%s.png' % i, roiOrig)
			
			#i+=1
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
preds = model.predict(rois, verbose=1)

print(preds)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(
	end - start))

# decode the predictions and initialize a dictionary which maps class
# labels (keys) to any ROIs associated with that label (values)

data_dir = 'D:\GitHub\Image Recognition\Test Images'

train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(180, 180),
  batch_size=390)

#Load method for prediction decoding
from pyimagesearch.detection_helpers import decode_predictions


data_labels = train_ds.class_names

labels = {}
filler_features = ['Other', 'wire', 'paper']
# loop over the predictions
for (i, p) in enumerate(preds):
	# grab the prediction information for the current ROI
	p = tf.nn.softmax(p)
	p_max = np.max(p)
	label = data_labels[np.argmax(p)]
	print(label, p)
	# filter out weak detections by ensuring the predicted probability
	# is greater than the minimum probability
	if p_max >= min_conf and label not in filler_features:
		# grab the bounding box associated with the prediction and
		# convert the coordinates
		box = locs[i]
		#print(prob)

		# grab the list of predictions for the label and add the
		# bounding box and probability to the list
		L = labels.get(label, [])
		L.append((box, p_max))
		labels[label] = L
		print(labels)

# loop over the labels for each of detected objects in the image
for label in labels.keys():
	# clone the original image so that we can draw on it
	print("[INFO] showing results for '{}'".format(label))
	clone = orig.copy()

	# loop over all bounding boxes for the current label
	for (box, p_max) in labels[label]:
		# draw the bounding box on the image
		(startX, startY, endX, endY) = box
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)

	# show the results *before* applying non-maxima suppression, then
	# clone the image again so we can display the results *after*
	# applying non-maxima suppression
	#cv2.imshow("Before", clone)
	clone = orig.copy()

	# extract the bounding boxes and associated prediction
	# probabilities, then apply non-maxima suppression
	boxes = np.array([p[0] for p in labels[label]])
	proba = np.array([p[1] for p in labels[label]])
	boxes = non_max_suppression(boxes, proba)

	# loop over all bounding boxes that were kept after applying
	# non-maxima suppression
	for (startX, startY, endX, endY) in boxes:
		# draw the bounding box and label on the image
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(clone, label+" "+str(round(p_max, 2)), (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# show the output after apply non-maxima suppression
	cv2.imshow("After", clone)
	cv2.waitKey(0)