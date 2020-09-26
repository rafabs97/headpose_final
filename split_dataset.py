#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run this utility to extract pictures from AFLW dataset from a previously obtained clean dataset, using the picture number as reference.
"""

import pandas as pd
import numpy as np
from clean_utils import array_from_csv

import os

dataset_dir = 'clean/aflw_pointing04/'

source_csv_file = 'test.csv'
destination_csv_file = 'test_aflw.csv'

destination_npy_file = 'test_aflw_img.npy'

source_csv = dataset_dir + source_csv_file

destination_csv = dataset_dir + destination_csv_file
destination_npy = dataset_dir + destination_npy_file

df = pd.read_csv(source_csv)

file = open(destination_csv, 'w')
file.write('file,tilt,pan\n')

for index, row in df.iterrows():
    if int(row['file'][4:-4]) < 44782:
        file.write(row['file'] + "," + str(row['tilt']) + "," + str(row['pan']) + "\n")

file.close()

img_array, _, _ = array_from_csv(destination_csv, dataset_dir)

np.save(destination_npy, img_array)