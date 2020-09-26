#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be used to find the crops from a dataset obtained using the Dockerface detector matching crops from a 
reference dataset using their pose values as criterion, and create a new dataset from them.
"""

from scipy.io import loadmat
import glob
import pandas as pd
import collections, itertools
from math import degrees
import shutil
import os
import numpy as np
import cv2

from clean_utils import bbox_match, array_from_csv

# Directory of the dataset used as reference.

source_dir = '../clean/aflw_pointing04/'

# Original dataset directory.

aflw_dir = '../original/aflw/'

# .mat file containing ground-truth annotations for the AFLW dataset.

mat_file = '../original/aflw/dataset_landmarks_and_pose_withfaceids.mat'

# Directory containing Dockerface detections.

dockerface_output_dir = '../clean/aflw_dockerface_output/'

# Directory in which the new dataset will be stored (detections from the reference dataset also found by Dockerface).

destination_dir = '../clean/aflw_dockerface_c/'

# .csv file containing the labels of the dataset used as reference.

csv_path = source_dir + 'test_aflw.csv'

# File containing every detection obtained by Dockerface over the AFLW picture folder.

dockerface_summary_file = dockerface_output_dir + 'dockerface_summary.csv'

# File containing for each valid Dockerface detection its pose value, alongside its source picture and coordinates for cropping.

dockerface_poses_file = dockerface_output_dir + 'dockerface_summary_poses.csv'

# .csv file that will contain the labels for the new dataset.

destination_csv = destination_dir + 'labels.csv'

# .npy files that will contain the pictures of the new dataset.

destination_npy = destination_dir + 'img.npy'
destination_npy_grayscale = destination_dir + 'img_grayscale.npy'

# Create Dockerface summary file.

summary = open(dockerface_summary_file, 'w')
summary.write('fileid,xmin,ymin,xmax,ymax,confidence\n')

for file in glob.iglob(dockerface_output_dir + '*.txt'):
    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        summary.write(line.replace(' ', ','))
    f.close()

summary.close()

# Create Dockerface poses file.

pose_file = open(dockerface_poses_file, 'w')
pose_file.write('fileid,xmin,ymin,xmax,ymax,tilt,pan\n')

mat = loadmat(mat_file)
values = sorted(zip(mat['fileids'], mat['bboxes'], mat['pose']), key = lambda x: x[0])

summary = pd.read_csv(dockerface_summary_file).sort_values(by='fileid', ignore_index=True)

iterator = range(len(values)).__iter__()
start_csv = 0

for tuple_index in iterator:

    # Create arrays containing ground truth detections and poses, and initialize them with the values corresponding
    # to the current tuple.
    true_bboxes = [values[tuple_index][1]]
    poses = [values[tuple_index][2]]

    i = 1
    id = values[tuple_index][0]

    '''
    For each following picture with the same fileid, add ground truth detections and poses to the arrays created
    before:
    '''

    while (tuple_index + i) < len(mat['fileids']) and values[tuple_index + i][0] == id:
        true_bboxes.append(values[tuple_index + i][1])
        poses.append(values[tuple_index + i][2])
        i = i + 1

    collections.deque(itertools.islice(iterator, i - 1))

    '''
    For each picture in the AFLW dataset with the same fileid, store detected face bboxes in an empty array:
    '''

    detected_bboxes = []

    for i in range(start_csv, len(summary.index)):
        if summary['fileid'][i] == id.strip():
            detected_bboxes.append([summary['xmin'][i], summary['ymin'][i], summary['xmax'][i], summary['ymax'][i]])
            start_csv = start_csv + 1
        else:
            break
    
    # If there are detected bounding boxes:
    if detected_bboxes:

        # Match detected bounding boxes with ground truth bounding boxes.
        indexes = bbox_match(detected_bboxes, true_bboxes)

        # For each matching:
        for box_index in range(len(indexes)):
    
            # If there is a valid matching between a detected bounding box and a ground truth bounding box:
            if indexes[box_index] != -1:

                # Pose values for the detection are the values corresponding to its matching ground truth annotated head.
                tilt = degrees(poses[indexes[box_index]][1])
                pan = degrees(poses[indexes[box_index]][2])

                # Store pose information for the detected bounding box.
                pose_file.write(id.strip() + "," + str(detected_bboxes[box_index][0]) + "," + str(detected_bboxes[box_index][1]) + "," + str(detected_bboxes[box_index][2]) + "," + str(detected_bboxes[box_index][3]) + "," + str(tilt) + "," + str(pan) + "\n")

pose_file.close()

# Actual matching:

# Check if output directory exists.

try:
    os.mkdir(destination_dir)
    print("Directory", destination_dir, "created.")

except FileExistsError:
    print("Directory", destination_dir, "already exists.")
    shutil.rmtree(destination_dir)
    os.mkdir(destination_dir)

# Load .csv file containing the labels from the reference dataset.

ref_df = pd.read_csv(csv_path)

# Load .csv file containing the labels of the second dataset.

sec_df = pd.read_csv(dockerface_poses_file)

# Sort both datasets.

ref_df.sort_values(by=['tilt', 'pan'], inplace=True)
sec_df.sort_values(by=['tilt', 'pan'], inplace=True)

# Create the pandas dataframe that will contain the labels of the pictures from the second dataset with a
# correspondence in the reference dataset.

destination_df = pd.DataFrame(columns=['file', 'tilt', 'pan'])

# Load pictures from the Dockerface detections.

img = array_from_csv(dockerface_poses_file, aflw_dir)[0]

# List that will contain the pictures from the Dockerface detections with a correspondence in the reference dataset.

destination_list = []

# Grayscale version (for testing with RealHePoNet).
destination_list_grayscale = []

# Naively search matching pictures between the two datasets using pose values as criterion.

start = 0

for row in sec_df.itertuples():

    found = False
    start_old = start

    for i in range(start, len(ref_df.index)):

        start = start + 1

        if (row[6] == ref_df['tilt'][ref_df.index[i]]) and (row[7] == ref_df['pan'][ref_df.index[i]]):

            print('Found match for %s%s...' % (source_dir, ref_df['file'][ref_df.index[i]]))

            destination_df = destination_df.append(ref_df.iloc[i], ignore_index=True)

            destination_list.append(img[row[0]])
            destination_list_grayscale.append(cv2.resize(cv2.cvtColor(img[row[0]], cv2.COLOR_BGR2GRAY), (64, 64)))

            found = True

            break

    if found == False:
        start = start_old

# Write .csv file containing labels for the new dataset.
destination_npy = destination_dir + 'img.npy'

destination_df.to_csv(destination_csv, index=False)

# Save matching pictures as .npy files.

np.save(destination_npy, np.asarray(destination_list))
np.save(destination_npy_grayscale, np.asarray(destination_list_grayscale))