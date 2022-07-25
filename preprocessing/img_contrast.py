def increase_contrast(img, threshold):
    """ 
       Takes a grayscale img and a specificed threshold. 
       Every pixel-value larger than the threshold will be converted to white (255). 
       All beneath threshold will be set to black (0) 
    """
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            if img[i][j] > threshold:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img
