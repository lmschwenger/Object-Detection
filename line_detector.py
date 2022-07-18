# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:02:35 2022

@author: lasse
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read in the image
image = cv2.imread(r'D:\GitHub\Object Detection\images\test_circuit2.png')
# Change color to RGB (from BGR)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Convert image to grayscale
#gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Define our parameters for Canny
low_threshold = 50
high_threshold = 55
edges = cv2.Canny(image, low_threshold, high_threshold)


# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 55
min_line_length = 5
max_line_gap = 20
line_image = np.copy(image) #creating an image copy to draw lines on
# Run Hough on the edge-detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
# Iterate over the output "lines" and draw lines on the image copy
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
from plots import plotLineDetector
if input("Show image processing steps? (y/n)") == "y":
    fig = plotLineDetector(image, edges, line_image)
    fig.show()
if input("Save figure? (y/n) ") == "y":
    fig = plotLineDetector(image, edges, line_image)
    fig.savefig('Plots/LineDetection.png', dpi=200, bbox_inches='tight')
