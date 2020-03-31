#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be used to test the performance of a trained pose estimator model on a given dataset.
"""

import numpy as np
import time

from clean_utils import array_from_npy
from architectures import mpatacchiola_generic

# Paths.

dataset_dir = 'clean/aflw_pointing04/'
csv_file = 'test.csv'
npy_file = 'test_img.npy'

dataset_csv = dataset_dir + csv_file
dataset_npy = dataset_dir + npy_file

models_path = 'models/'

estimator_file = 'pose-estimator.h5'
estimator_path = models_path + estimator_file

# Detector parameters.

in_size_detector = 512
confidence_threshold = 0.65

# Estimator parameters.

in_size_estimator = 64
num_conv_blocks = 6
num_filters_start = 32
num_dense_layers = 1
dense_layer_size = 512

# Normalization parameters.

mean = 0.408808
std = 0.237583

t_mean = -0.041212
t_std = 0.323931

p_mean = -0.000276
p_std = 0.540958

# Load image, tilt and pan arrays for the dataset.

img, tilt, pan = array_from_npy(dataset_npy, dataset_csv)

# Add extra dimension if needed (channels).

if len(img.shape) == 3:
    img = np.expand_dims(img, -1)

# Estimator model.

pose_estimator = mpatacchiola_generic(in_size_estimator, num_conv_blocks, num_filters_start, num_dense_layers, dense_layer_size)
pose_estimator.load_weights(estimator_path)

# Get score for the dataset (tilt, pan and global error).

start_time = time.time()
pred = pose_estimator.predict((img / 255.0 - mean) / std)
end_time = time.time()

mean_time = (end_time - start_time) / len(img)

mean_tilt_error = np.mean(np.abs(tilt - ((pred[:, 0] * t_std + t_mean) * 90.0)))
mean_pan_error = np.mean(np.abs(pan - ((pred[:, 1] * p_std + p_mean) * 90.0)))

score = (mean_pan_error + mean_tilt_error) / 2

# Print score.

print("Tilt: %.2fº Pan: %.2fº Global: %.2fº Mean time: %fs" % (mean_tilt_error, mean_pan_error, score, mean_time))