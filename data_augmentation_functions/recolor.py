# Recoloring image
import cv2
def recolor(img, color='red'):
    
    Conv_hsv_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    if color == 'red':
        new_mask = [0, 0, 255]
    elif color == 'blue':
        new_mask = [255, 0, 0]
    elif color == 'green':
        new_mask = [0, 255, 0]
    else:
        return print('Choose between colors "red", "green" and "blue"')
    img[mask == 255] = new_mask
    return img
