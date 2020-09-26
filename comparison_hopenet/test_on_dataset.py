#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be used to test the performance of a trained pose estimator model on a given dataset.
"""

import numpy as np
import time

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from clean_utils import array_from_npy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import hopenet

# Paths.

dataset_dir = '../clean/aflw_dockerface_c/'
csv_file = 'labels.csv'
npy_file = 'img.npy'

dataset_csv = dataset_dir + csv_file
dataset_npy = dataset_dir + npy_file

models_path = '../models/'

estimator_file = 'hopenet_robust_alpha1.pkl'

estimator_path = models_path + estimator_file

# Transformations.

transformations = transforms.Compose([transforms.Scale(224),
transforms.CenterCrop(224), transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Other parameters.

cudnn.enabled = True
batch_size = 1
gpu = 0

# Load image, tilt and pan arrays for the dataset.

img, tilt, pan = array_from_npy(dataset_npy, dataset_csv)
print(img.shape)

# Estimator model.

pose_estimator = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
saved_state_dict = torch.load(estimator_path)
pose_estimator.load_state_dict(saved_state_dict)
pose_estimator.cuda(gpu)
pose_estimator.eval()

# Get score for the dataset (tilt, pan and global error).

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

pred = []

start_time = time.time()

for i in img:

    # Transform crop to PIL image.
    crop = Image.fromarray(i)

    # Transform
    crop = transformations(crop)
    crop_shape = crop.size()
    crop = crop.view(1, crop_shape[0], crop_shape[1], crop_shape[2])
    crop = Variable(crop).cuda(gpu)

    # Get pose values.
    pan_predicted, tilt_predicted, _ = pose_estimator(crop)

    pan_predicted = F.softmax(pan_predicted)
    tilt_predicted = F.softmax(tilt_predicted)

    # Get continuous predictions in degrees.
    pan_predicted = torch.sum(pan_predicted.data[0] * idx_tensor) * 3 - 99
    tilt_predicted = torch.sum(tilt_predicted.data[0] * idx_tensor) * 3 - 99

    pred.append([tilt_predicted.item(), pan_predicted.item()])

end_time = time.time()

mean_time = (end_time - start_time) / len(img)

pred = np.asarray(pred)

mean_tilt_error = np.mean(np.abs(tilt - pred[:, 0]))
mean_pan_error = np.mean(np.abs(pan - pred[:, 1]))

score = (mean_pan_error + mean_tilt_error) / 2

# Print score.

print("Tilt: %.2fº Pan: %.2fº Global: %.2fº Mean time: %fs" % (mean_tilt_error, mean_pan_error, score, mean_time))