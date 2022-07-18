# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:31:36 2022

@author: lasse
"""

from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector
from collections import defaultdict
import matplotlib.pyplot as plt
# Load image:
input_image = Image.open('D:/GitHub/Image Recognition/Test Images/ammeter/ammeter_4.jpg')
input_image = Image.open('D:/GitHub/Object Detection/images/test_circuit2.png')
# Output image:
output_image = Image.new("RGB", input_image.size)
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)

# Find circles
rmin = 18
rmax = 45
steps = 100
threshold = 0.5

points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))
print(1)

acc = defaultdict(int)
for x, y in canny_edge_detector(input_image):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1
print(2)
circles = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, x, y, r)
        circles.append((x, y, r))
print(3)
for x, y, r in circles:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))    

fig, ax = plt.subplots(1, dpi=200)
ax.imshow(output_image)
ax.axis('off')
if input("Show result? (y/n) ") == "y":
    plt.show()

if input("Save figure? (y/n) ") == "y":
    fig.savefig('Plots/Circle Detection.png', dpi=200, bbox_inches='tight')
# Save output image
#output_image.save("result.png")