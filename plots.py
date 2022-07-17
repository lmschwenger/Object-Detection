# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:34:43 2022

@author: lasse
"""
import matplotlib.pyplot as plt
def performance_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    fig, ax = plt.subplots(1, 2, dpi=200, figsize=(12,5))
    ax[0].plot(epochs, acc, 'bo', label='Training acc')
    ax[0].plot(epochs, val_acc, 'b', label='Validation acc')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    
    
    ax[1].plot(epochs, loss, 'bo', label='Training loss')
    ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()

    plt.show()

def plotLineDetector(image, edges, line_image):
    fig, ax = plt.subplots(1, 3, dpi = 200)

    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(edges, cmap='gray')
    ax[2].imshow(line_image)

    ax[0].set_title('Original image')
    ax[1].set_title('Edges detected')
    ax[2].set_title('Identified Lines')
    for axis in ax.reshape(-1):
        axis.axis('off')
    plt.imshow(line_image)
    plt.show()