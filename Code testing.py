# Code testing
import cv2
import tensorflow as tf
import imutils
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
#%% Loading model
img_height = 28
img_width = 28


#model = tf.keras.models.load_model('D:\savedModels\OD_model_TF')
create_contrast_directory = 0
if create_contrast_directory:
    name = 'resistor'
    directory = f'D:/trainingImages/{name}/'
    output_dir = f'D:/trainingImagesContrast/{name}/'
    print(f"[INFO] Printing contrast versions of subset '{name}'")
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
       # cv2.imshow("contrast", equ)
        #cv2.waitKey(0)
        for i in range(0, len(equ)):
            for j in range(0, len(equ[i])):
                if equ[i][j] > 15:
                    equ[i][j] = 255
                else:
                    equ[i][j] = 0
        #print("[INFO] PRINTING TO: %s" %os.path.join(output_dir, filename))
        cv2.imwrite(os.path.join(output_dir, filename), equ)
    print("[INFO] Task Finished ...")

print("1")
contrast_one_image = 0

if contrast_one_image:

    from preprocessing.img_contrast import increase_contrast
    threshold = 2
    f = 'D:/GitHub/Object Detection/images/test_circuit5.jpg'
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("2")
    equ = increase_contrast(cv2.equalizeHist(gray), threshold)
    print("3")
    output_name = 'D:/testImagesContrast/'+f.rsplit('/', 1)[1]
    print(output_name)
    cv2.imwrite(output_name, equ)

image_path = 'D:/testImagesContrast/roi_2.png'
roiOrig = tf.keras.preprocessing.image.load_img(image_path)
print(roiOrig)
input_arr = tf.keras.preprocessing.image.img_to_array(roiOrig)
#input_arr = np.array([input_arr])  # Convert single image to a batch.
roi = cv2.resize(input_arr, (60, 60), interpolation=cv2.INTER_CUBIC)
print(roi.shape)
cv2.imwrite('D:/testImagesContrast/roi.png', roi)
cv2.waitKey(0)