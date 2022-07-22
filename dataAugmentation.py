from data_augmentation_functions.salt_and_pepper import add_noise
from data_augmentation_functions.zoom import zoom
from data_augmentation_functions.rotate_image import rotate_img
from data_augmentation_functions.recolor import recolor
import cv2
import matplotlib.pyplot as plt

import os
from PIL import Image, ImageEnhance
# assign directory and subdirectory for dataAgumentation
name = 'lamp'
directory = f'D:/GitHub/Object Detection/trainingImages/{name}/'
 
# Switches for desired operations
rename = 0 #Rename all files in the directory
noise_filter = 0 #Add salt n pepper noise to existing images
brightness_filter = 0 #Alter brightness of existing images
augment_single_image = 0
zoom_images = 0
recolor_images = 0
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


if augment_single_image:
    f = r'd:\github\object detection\trainingimages\wire\wire_4_snp_snp.png'
    cv2.imwrite(f.rsplit('.', 1)[0] + str("_snp.") + f.rsplit('.', 1)[1], add_noise(cv2.imread(f)))
    im = Image.open(f)

    #image brightness enhancer
    enhancer = ImageEnhance.Brightness(im)

    factor = 0.5 #darkens the image
    im_output = enhancer.enhance(factor)
    im_output.save(f.rsplit('.', 1)[0] + str("_dark.") + f.rsplit('.', 1)[1])

    factor = 1.5 #brightens the image
    im_output = enhancer.enhance(factor)
    im_output.save(f.rsplit('.', 1)[0] + str("_bright.") + f.rsplit('.', 1)[1])


if zoom_images:
    files = []
    for filename in os.listdir(directory):
        files.append(filename)
        for FILE in files:
            f = os.path.join(directory, FILE)
        img = cv2.imread(f)
        zoomed = zoom(img, zoom_factor=2)
        cv2.imwrite(f.rsplit('.', 1)[0] + '_zoom.' + f.rsplit('.', 1)[1], zoomed)

if recolor_images:
    files = []
    for filename in os.listdir(directory):
        files.append(filename)

    for color in ['red', 'green', 'blue']:
        for FILE in files:
            f = os.path.join(directory, FILE)
            img = cv2.imread(f)
            recolored =  recolor(img, color=color)
            cv2.imwrite(f.rsplit('.', 1)[0] + '_' + color + '.' + f.rsplit('.', 1)[1], recolored)
