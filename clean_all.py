#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the code used to process the datasets AFLW and Pointing'04 using functions from files
clean_aflw.py and clean_pointing04.py, and to store them in a way useful for training head pose detection models.
Moreover, the code allows logging, for a certain confidence threshold, the number of cropped pictures obtained, P ratio
and F ratio for each dataset in order to select the best confidence threshold (refer to the Technical Manual,
section 8.3, in order to know the meaning of these values).
"""

import os
import shutil
import cv2

from keras_ssd512 import ssd_512

from clean_aflw import clean_aflw
from clean_pointing04 import clean_pointing04
from dataset_utils import class_assign, split_dataset, find_norm_parameters, store_dataset_arrays

# Source paths.

aflw_dir = 'original/aflw/'
aflw_mat = 'original/aflw/dataset_landmarks_and_pose_withfaceids.mat'
pointing04_dir = 'original/HeadPoseImageDatabase/'

# Destination paths.

destination_dir = 'clean/aflw_pointing04/'
detector_log_path = 'models/detector_log_corrected.csv'

# Model paths.

head_detector_path = 'models/head-detector.h5'

# Detection parameters.

confidence_threshold = 0.65
in_size = 512
out_size = 64

# Output parameters.

grayscale_output = True
downscaling_interpolation = cv2.INTER_LINEAR

# Number of splits for class assignation.

num_splits_tilt = 8
num_splits_pan = 8

# Ratios for train/test and train/validation split.

test_ratio = 0.2
validation_ratio = 0.2

# Detector model.

detector = ssd_512(image_size=(in_size, in_size, 3), n_classes=1, min_scale=0.1, max_scale=1, mode='inference')
detector.load_weights(head_detector_path)

# Check if output directory exists.

try:
    os.mkdir(destination_dir)
    print("Directory", destination_dir, "created.")

except FileExistsError:
    print("Directory", destination_dir, "already exists.")
    shutil.rmtree(destination_dir)
    os.mkdir(destination_dir)

# Actual cleaning.

count_aflw, t_ratio_aflw, f_ratio_aflw = clean_aflw(aflw_dir, aflw_mat, destination_dir, detector, confidence_threshold,
                                                    out_size, grayscale_output, downscaling_interpolation)
count_p04, t_ratio_p04, f_ratio_p04 = clean_pointing04(pointing04_dir, destination_dir, detector, confidence_threshold,
                                                       out_size, grayscale_output, downscaling_interpolation, count_aflw,
                                                       duplicate_until=-1)

"""
# Log found ratios.

if os.path.isfile(detector_log_path):
    file = open(detector_log_path, 'a')
    file.write("%.2f,%d,%f,%f,%d,%f,%f\n" % (confidence_threshold, count_aflw, t_ratio_aflw, f_ratio_aflw, count_p04 - count_aflw, t_ratio_p04, f_ratio_p04))
else:
    file = open(detector_log_path, 'w')
    file.write('threshold,count_aflw,t_ratio_aflw,f_ratio_aflw,count_p04,t_ratio_p04,f_ratio_p04\n')
    file.write("%.2f,%d,%f,%f,%d,%f,%f\n" % (confidence_threshold, count_aflw, t_ratio_aflw, f_ratio_aflw, count_p04 - count_aflw, t_ratio_p04, f_ratio_p04))

file.close()
"""

# Assign classes.

class_assign(destination_dir, num_splits_tilt, num_splits_pan)

# Split dataset.

split_dataset(destination_dir, test_ratio, validation_ratio)

# Get normalization parameters.

find_norm_parameters(destination_dir)

# OPTIONAL: Save dataset as numpy arrays (for uploading to Google Colab).

store_dataset_arrays(destination_dir)