#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use this script to check if everything was installed properly.
"""

import numpy as np
import cv2
from math import sin, radians

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from keras_ssd512 import ssd_512
from head_detector_utils import get_head_bboxes, get_cropped_pics

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import hopenet

# Paths

demo_img = '../data/people_drinking.jpg'
detector_file = 'head-detector.h5'
estimator_file = 'hopenet_robust_alpha1.pkl'

models_path = '../models/'

detector_path = models_path + detector_file
estimator_path = models_path + estimator_file

# Detector parameters.

in_size_detector = 512
in_size_estimator = 224
confidence_threshold = 0.2

# Transformations.

transformations = transforms.Compose([transforms.Scale(224),
transforms.CenterCrop(224), transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Other parameters.

cudnn.enabled = True
batch_size = 1
gpu = 0

# Models.

head_detector = ssd_512(image_size=(in_size_detector, in_size_detector, 3), n_classes=1, min_scale=0.1, max_scale=1, mode='inference')
head_detector.load_weights(detector_path)

pose_estimator = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
saved_state_dict = torch.load(estimator_path)
pose_estimator.load_state_dict(saved_state_dict)
pose_estimator.cuda(gpu)
pose_estimator.eval()

print(sum(p.numel() for p in pose_estimator.parameters()))

# Read image.

img = cv2.imread(demo_img)

# Get bounding boxes for every detected head in the picture.

bboxes = get_head_bboxes(img, head_detector, confidence_threshold)

# Get cropped pics for every valid bounding box.

heads = get_cropped_pics(img, bboxes, in_size_estimator, 0, cropping='small')

# Actual test.

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

# For each cropped picture:
for i in range(len(heads)):

    # If it is a valid picture:
    if heads[i].shape == (in_size_estimator, in_size_estimator, 3):

        # Transform crop to PIL image.
        crop = Image.fromarray(heads[i])

        # Get minimum and maximum values for both axes of the bounding box.
        xmin, ymin, xmax, ymax = bboxes[i]

        # Transform

        crop = transformations(crop)
        crop_shape = crop.size()
        crop = crop.view(1, crop_shape[0], crop_shape[1], crop_shape[2])
        crop = Variable(crop).cuda(gpu)

        # Get pose values.

        pan, tilt, _ = pose_estimator(crop)

        pan_predicted = F.softmax(pan)
        tilt_predicted = F.softmax(tilt)

        # Get continuous predictions in degrees.
        pan_predicted = torch.sum(pan_predicted.data[0] * idx_tensor) * 3 - 99
        tilt_predicted = torch.sum(tilt_predicted.data[0] * idx_tensor) * 3 - 99

        # Draw detection in the original picture.

        rect = cv2.rectangle(img, (xmax, ymin), (xmin, ymax), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(rect, 'TILT: ' + str(round(tilt_predicted.item(), 2)) + ' PAN: ' + str(round(pan_predicted.item(), 2)), (xmin, ymin - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

        # Draw arrow from the center of the picture in the direction of the pose in the original picture.

        centerx = int((xmin + xmax) / 2)
        centery = int((ymin + ymax) / 2)
        center = (centerx, centery)

        max_arrow_len = (xmax - xmin + 1) / 2

        offset_x = -1 * int(sin(radians(pan_predicted.item())) * max_arrow_len)
        offset_y = -1 * int(sin(radians(tilt_predicted.item())) * max_arrow_len)

        end = (centerx + offset_x, centery + offset_y)
        cv2.arrowedLine(img, center, end, (0, 0, 255), 2, line_type=cv2.LINE_AA)

# Show image with detections.

cv2.imshow('Detections', img)
cv2.imwrite('output.jpg', img)

# Exit

print('Press any key to exit...')
cv2.waitKey(0)
