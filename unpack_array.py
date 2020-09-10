#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run this utility to unpack the .npy file from a previously obtained clean dataset.
"""

import pandas as pd
import numpy as np
import cv2 

dataset_dir = 'clean/aflw_pointing04/'

source_csv_file = 'test.csv'
source_npy_file = 'test_img.npy'

source_csv = dataset_dir + source_csv_file
source_npy = dataset_dir + source_npy_file

df = pd.read_csv(source_csv)
img = np.load(source_npy)

for index, row in df.iterrows():
    cv2.imwrite(dataset_dir + row['file'], img[index])