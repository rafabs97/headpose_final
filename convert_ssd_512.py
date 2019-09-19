#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the code needed to convert MatConvNet SSD512 models into Keras SSD512 models using the implementation
of SSD512 by Pierluigi Ferrari (https://github.com/pierluigiferrari/ssd_keras).
"""

import numpy as np
import pandas as pd

import keras.backend as K

from loadmat_stackoverflow import loadmat
from keras_ssd512 import ssd_512

def expand_to_4d(array):
    '''
    Reshapes 2D input array into 4D array by adding 2 length-1 dimensions to its shape.

    Arguments:
        array: The 2D input array.
    Returns:
        The reshaped array.
    '''
    return np.reshape(array, [1, 1, array.shape[0], array.shape[1]])

# Paths.

model_dir = 'models/'
model_name = 'head-detector'

matconvnet_model_path = model_dir + model_name + '.mat'
keras_model_path = model_dir + model_name + '.h5'

layer_mapping_path = model_dir + 'layers.csv'

# Model parameters.

img_height = 512
img_width = 512

# Model instance.

model = ssd_512(image_size=(img_height, img_width, 3), n_classes=1, min_scale=0.1, max_scale=1, mode='inference')

# Load weights.

model_ori = loadmat(matconvnet_model_path)

# Load layer mapping between MatConvNet and Keras models.

layers = pd.read_csv(layer_mapping_path)

# For each entry in the mapping, set its weights and biases from the original model:

for index, layer in layers.iterrows():

    print("Assigning layer " + layer['name'] + "...")

    if layer['name'] in ["fc7", "conv6_1", "conv7_1", "conv8_1", "conv9_1", "conv10_1"]:
        K.set_value(model.get_layer(layer['name']).weights[0], expand_to_4d(model_ori['vars'][layer['weights'] - 1]))
    else:
        K.set_value(model.get_layer(layer['name']).weights[0], model_ori['vars'][layer['weights'] - 1])

    print("Weights...")

    if layer['biases'] != -1:
        print("Biases...")
        K.set_value(model.get_layer(layer['name']).weights[1], model_ori['vars'][layer['biases'] - 1])

print("Done.")

# Save converted model at destination path.

model.save_weights(keras_model_path)