# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
This script can be used to find the crops from a dataset matching crops from a reference dataset using their pose values
as criterion, and create a new dataset from them.
"""

import pandas as pd
import numpy as np
import shutil
import os

from clean_utils import array_from_csv

# Directory of the dataset used as reference.

source_dir_1 = '../headpose_final/clean/aflw_pointing04/'

# Directory of the second dataset.

source_dir_2 = 'clean/aflw_haar_area_color/'

# Directory in which the new dataset will be stored.

destination_dir = 'clean/aflw_haar_area_color_c/'

# .csv file containing the labels of the dataset used as reference.

csv_path_1 = source_dir_1 + 'test_aflw.csv'

# .csv file containing the labels of the second dataset.

csv_path_2 = source_dir_2 + 'labels.csv'

# .csv file that will contain the labels for the new dataset.

destination_csv = destination_dir + 'labels.csv'

# .npy file that will contain the pictures of the new dataset.

destination_npy = destination_dir + 'img.npy'

# Check if output directory exists.

try:
    os.mkdir(destination_dir)
    print("Directory", destination_dir, "created.")

except FileExistsError:
    print("Directory", destination_dir, "already exists.")
    shutil.rmtree(destination_dir)
    os.mkdir(destination_dir)

# Load .csv files containing the labels from the reference dataset.

ref_df = pd.read_csv(csv_path_1)

# Create the pandas dataframe that will contain the labels of the pictures from the second dataset with a
# correspondence in the reference dataset.

destination_df = pd.DataFrame(columns=ref_df.columns.values)

# Load pictures and pose values from the second dataset.

img, tilt, pan = array_from_csv(csv_path_2, source_dir_2)

# List that will contain the pictures from the second dataset with a correspondence in the reference dataset.

destination_list = []

# Mask to avoid deletion.

mask = [True for i in range(len(img))]

# Naively search matching pictures between the two datasets using pose values as criterion.

for index, row in ref_df.iterrows():
    for i in range(len(mask)):
        if (mask[i] == True) and (row['tilt'] == tilt[i]) and (row['pan'] == pan[i]):

            print('Found match for %s%s...' % (source_dir_1, row['file']))

            destination_df = destination_df.append(row, ignore_index=True)
            destination_list.append(img[i])

            mask[i] = False

            break

# Write .csv file containing labels for the new dataset.

destination_df.to_csv(destination_csv, index=False)

# Save matching pictures as a .npy file.

np.save(destination_npy, np.asarray(destination_list))
