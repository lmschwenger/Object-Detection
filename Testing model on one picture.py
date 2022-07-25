from pickle import TRUE
import tensorflow as tf
import cv2
import imutils
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
#%% Loading model
img_height = 60
img_width = 60


data_dir = r'D:\trainingImagesContrast'
test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=100)


model = tf.keras.models.load_model('D:\savedModels\OD_model_TF')
model.evaluate(test_ds, verbose=2)


image_path = 'D:/testImagesContrast/roi.png'
#orig = cv2.imread(image_path)
orig = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
input_arr = tf.keras.preprocessing.image.img_to_array(orig)
input_arr = np.array([input_arr])  # Convert single image to a batch.

#orig = np.asarray(orig).astype('float32')
#orig = np.expand_dims(orig, axis=0)
#orig = cv2.resize(orig, dsize=(img_height, img_width))
#orig = np.reshape(orig, (1, img_height, img_width, 3))

preds = model.predict(input_arr)
probs = tf.nn.softmax(preds[0])

labels = test_ds.class_names
label = labels[np.argmax(probs)]
print(np.argmax(probs))
print(labels)
print(tf.nn.softmax(preds[0]))

fig, ax = plt.subplots(1)

ax.imshow(orig)
ax.set_title(f"{label} ({round(np.max(probs), 2) * 100}%)")
plt.show()