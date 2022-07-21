from data_augmentation_functions.salt_and_pepper import add_noise

import cv2
import matplotlib.pyplot as plt

import os
from PIL import Image, ImageEnhance
# assign directory and subdirectory for dataAgumentation
name = 'wire'
directory = f'D:/GitHub/Object Detection/trainingImages/{name}/'
 
# Switches for desired operations
rename = 0 #Rename all files in the directory
noise_filter = 0 #Add salt n pepper noise to existing images
brightness_filter = 1 #Alter brightness of existing images

i = 1
if rename:
    for filename in os.listdir(directory):
        """ 
            Renaming all files
        """
        f = os.path.join(directory, filename)
        img_type = f.rsplit('.', 1)[1]

        old_name = f.rsplit('/', 1)
        old_name = f.rsplit('.', 1)[0]
        new_name = f"{directory}{name}_0{i}.{img_type}"
        os.rename(f, new_name)
        i+=1

if noise_filter:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        # checking if it is a file
        img = cv2.imread(f)
        cv2.imwrite(f.rsplit('.', 1)[0] + str("_snp.") + f.rsplit('.', 1)[1], add_noise(img))


if brightness_filter:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        #read the image
        im = Image.open(f)

        #image brightness enhancer
        enhancer = ImageEnhance.Brightness(im)

        factor = 1 #gives original image

        factor = 0.5 #darkens the image
        im_output = enhancer.enhance(factor)
        im_output.save(f.rsplit('.', 1)[0] + str("_dark.") + f.rsplit('.', 1)[1])

        factor = 1.5 #brightens the image
        im_output = enhancer.enhance(factor)
        im_output.save(f.rsplit('.', 1)[0] + str("_bright.") + f.rsplit('.', 1)[1])