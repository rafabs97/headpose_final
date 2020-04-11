#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use this script to check if everything was installed properly.
"""

import cv2
from math import sin, radians

from keras_ssd512 import ssd_512

from architectures import mpatacchiola_generic
from head_detector_utils import get_head_bboxes, get_cropped_pics
from pose_estimator_utils import get_pose

# Paths

demo_img = 'data/people_drinking.jpg'
detector_file = 'head-detector.h5'
estimator_file = 'pose-estimator.h5'

models_path = 'models/'

detector_path = models_path + detector_file
estimator_path = models_path + estimator_file

# Detector parameters.

in_size_detector = 512
confidence_threshold = 0.2

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

# Models.

head_detector = ssd_512(image_size=(in_size_detector, in_size_detector, 3), n_classes=1, min_scale=0.1, max_scale=1, mode='inference')
head_detector.load_weights(detector_path)

pose_estimator = mpatacchiola_generic(in_size_estimator, num_conv_blocks, num_filters_start, num_dense_layers, dense_layer_size)
pose_estimator.load_weights(estimator_path)

# Read image.

img = cv2.imread(demo_img)

# Get bounding boxes for every detected head in the picture.

bboxes = get_head_bboxes(img, head_detector, confidence_threshold)

# Get cropped pics for every valid bounding box.

gray_pic = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
heads = get_cropped_pics(gray_pic, bboxes, in_size_estimator, 0, cropping='small')

# For each cropped picture:
for i in range(len(heads)):

    # If it is a valid picture:
    if heads[i].shape == (in_size_estimator, in_size_estimator):

        # Get pose values.
        tilt, pan = get_pose(heads[i], pose_estimator, img_norm = [mean, std], tilt_norm = [t_mean, t_std],
                             pan_norm = [p_mean, p_std], rescale=90.0)

        # Get minimum and maximum values for both axes of the bounding box.
        xmin, ymin, xmax, ymax = bboxes[i]

        # Draw detection in the original picture..

        rect = cv2.rectangle(img, (xmax, ymin), (xmin, ymax), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(rect, 'TILT: ' + str(round(tilt, 2)) + ' PAN: ' + str(round(pan, 2)), (xmin, ymin - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

        # Draw arrow from the center of the picture in the direction of the pose in the original picture.

        centerx = int((xmin + xmax) / 2)
        centery = int((ymin + ymax) / 2)
        center = (centerx, centery)

        max_arrow_len = (xmax - xmin + 1) / 2

        offset_x = -1 * int(sin(radians(pan)) * max_arrow_len)
        offset_y = -1 * int(sin(radians(tilt)) * max_arrow_len)

        end = (centerx + offset_x, centery + offset_y)
        cv2.arrowedLine(img, center, end, (0, 0, 255), 2, line_type=cv2.LINE_AA)

# Show image with detections.

cv2.imshow('Detections', img)

# Exit

print('Press any key to exit...')
cv2.waitKey(0)
