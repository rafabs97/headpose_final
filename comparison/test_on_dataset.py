#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be used to test the performance of a trained pose estimator model on a given dataset.
"""

import numpy as np
import time

import tensorflow as tf

from clean_utils import array_from_npy

from head_pose_estimation import CnnHeadPoseEstimator

# Paths.

dataset_dir = 'clean/aflw_haar_area_color_c/'
csv_file = 'labels.csv'
npy_file = 'img.npy'

dataset_csv = dataset_dir + csv_file
dataset_npy = dataset_dir + npy_file

models_path = 'models/'

tilt_estimator_file = 'pitch/cnn_cccdd_30k.tf'
pan_estimator_file = 'yaw/cnn_cccdd_30k.tf'

tilt_estimator_path = models_path + tilt_estimator_file
pan_estimator_path = models_path + pan_estimator_file

# Load image, tilt and pan arrays for the dataset.

img, tilt, pan = array_from_npy(dataset_npy, dataset_csv)
print(img.shape)

# Estimator model.

sess = tf.Session()
pose_estimator = CnnHeadPoseEstimator(sess)
pose_estimator.load_pitch_variables(tilt_estimator_path)
pose_estimator.load_yaw_variables(pan_estimator_path)

# Get score for the dataset (tilt, pan and global error).

pred = []

start_time = time.time()

for i in img:
    pred.append([pose_estimator.return_pitch(i)[0, 0, 0], pose_estimator.return_yaw(i)[0, 0, 0]])

end_time = time.time()

mean_time = (end_time - start_time) / len(img)

pred = np.asarray(pred)

mean_tilt_error = np.mean(np.abs(tilt - pred[:, 0]))
mean_pan_error = np.mean(np.abs(pan - pred[:, 1]))

score = (mean_pan_error + mean_tilt_error) / 2

# Print score.

print("Tilt: %.2fº Pan: %.2fº Global: %.2fº Mean time: %fs" % (mean_tilt_error, mean_pan_error, score, mean_time))