# Object Detection

An object detection algorithm for detection of elements in electrical circuits (See examples below)

## Training the neural network
The neural network uses data from the Image-Recognition project.
Model training is carried out via the CNN_TensorFlow.py file. NOTICE the model is saved at the end of training. Make sure to adjust the path to fit your own system.

After training and saving the trained model it can be tested one (1) An image, or (2) on a complete circuit.
Both test (1) and (2) load the previously saved model. Again, the path to the saved model must be adjusted to your case.

### (1) Test of image recognition
This is carried out from the "Testing model on one picture.py" file. Here, the path to the test image must be stated.

### (2) Test of Object Detection
This is carried out via the "detect_with_classifier.py" file.
The test is carried out with a combination of pyramid and sliding-window technique. The pyramid-part is creating copies of the original image at different scales. This can be adjusted from the initial parameteres.

The sliding window technique is simply creating sub-images, where a sliding window runs accros the original image and takes snippets. Afterwards, all snippets are run through the trained model to see if a known object is identified. Make sure to set a threshold for probability. This threshold is the required certainty the model has, that a particular "Region of Interest" / ROI is in fact a known object.

## Line detection
Using line_detector.py straight lines can be detected.
Line detection is carried out using Hough transform from opencv-library.

![alt text](https://github.com/lmschwenger/Object-Detection/blob/ec07775ef38e380a9cad02ff7593b7bcdf11d208/Plots/Line%20Detection.png?raw=true)

## Circle detection
Using circle_detector.py circle can be identified.
![alt text](https://github.com/lmschwenger/Object-Detection/blob/9b503d3d317d2a0fb71480e0ed34b9d3e3f165e8/Plots/Circle%20Detection.png?raw=true)
