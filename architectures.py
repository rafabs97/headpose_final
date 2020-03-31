# -*- coding: utf-8 -*-

"""
This file contains a parameterized implementation of a CNN compatible with the architectures used in the article by
Patacchiola et al. at the publication below:

[15]  M. Patacchiola y A. Cangelosi. Head pose estimation in the wild using
convolutional neural networks and adaptive gradient methods. Pattern Recognition, 71,
June 2017.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def mpatacchiola_generic(in_size, num_conv_blocks, num_filters_start, num_dense_layers, dense_layer_size, dropout_rate=0, batch_size=None):
    '''
    Builds a Keras model based on the architectures used in the publication of Patacchiola et al. (see reference at the beginning
    of the file).

    Arguments:
        in_size: Length of each side of the image used as input for the model.
        num_conv_blocks: Number of convolutional blocks (defined as a convolutional layer followed by a pooling layer).
        num_filters_start: Number of filters in the first convolutional layer.
        num_dense_layers: Number of hidden dense layers in the model.
        dense_layer_size: Number of neurons per hidden dense layer.
        dropout_rate: Probability of ignoring a certain neuron at the output of each hidden dense layer at each training
        step.
        batch_size: Allows to set a fixed batch size, if needed.
    Returns:
        model: The configured Keras model.
    '''

    # Define the empty model as a linear stack of layers.
    model = Sequential()

    # At least, the model must have 1 convolutional block.
    model.add(Conv2D(num_filters_start, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(in_size, in_size, 1), batch_size=batch_size))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add the remaining convolutional blocks.
    for i in range(num_conv_blocks - 1):

        # Uncomment in order to add a layer with the same number of filters in the previous layer.
        #model.add(Conv2D(num_filters_start, kernel_size=(3, 3), padding='same', activation='relu'))

        # Uncomment in order to add a layer with the number of filters in the previous layer + the number of filters in
        # the first conv. layer.
        #model.add(Conv2D(num_filters_start * (i + 2), kernel_size=(3, 3), padding='same', activation='relu'))

        # Uncomment in order to add a layer with twice the number of filters in the previous layer.
        model.add(Conv2D(num_filters_start * (2 ** (i + 1)), kernel_size=(3, 3), padding='same', activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten (convert to a vector of values) the output of the last convolutional block.
    model.add(Flatten())

    # Add hidden dense layers; there is no minimum number of hidden dense layers (you can connect the output of the last convolutional block
    # directly to the output layer).
    for i in range(num_dense_layers):
        model.add(Dense(dense_layer_size))
        # If there is dropout, it is applied after every hidden dense layer.
        if dropout_rate != 0:
            model.add(Dropout(dropout_rate))

    # Add output layer with 2 outputs, each one outputting the prediction of an angle (tilt and pan).
    model.add(Dense(2))

    # Return the configured model.
    return model
