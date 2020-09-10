# -*- coding: utf-8 -*-

"""
This file contains a set of functions used by other scripts in order to perform some operations related to the
preprocessing of the datasets used in this project.
"""

import numpy as np
import cv2
import pandas as pd

def overlapped_area(bbox_1, bbox_2):
    '''
    Returns overlapped area between two rectangles given their minimum and maximum row and column values.

    Arguments:
        bbox_1: First rectangle.
        bbox_2: Second rectangle.
    Returns:
        area: Value of the overlapped area between the two rectangles.
    '''

    # Find overlapped rectangle coordinates.
    x_min = max(bbox_1[0], bbox_2[0])
    y_min = max(bbox_1[1], bbox_2[1])
    x_max = min(bbox_1[2], bbox_2[2])
    y_max = min(bbox_1[3], bbox_2[3])

    # Calculate area of the overlapped rectangle.
    if (x_max - x_min) < 0 or (y_max - y_min) < 0:
        area = 0
    else:
        area = (x_max - x_min + 1) * (y_max - y_min + 1)

    # Return area value.
    return area

def union_area(bbox_1, bbox_2):
    '''
    Returns the area of the union of two rectangles given their minimum and maximum row and column values.

    Arguments:
        bbox_1: First rectangle.
        bbox_2: Second rectangle.
    Returns:
        area: Value of the overlapped area between the two rectangles.
    '''

    # Calculate the area of each rectangle.
    area_1 = (bbox_1[2] - bbox_1[0] + 1) * (bbox_1[3] - bbox_1[1] + 1)
    area_2 = (bbox_1[2] - bbox_1[0] + 1) * (bbox_1[3] - bbox_1[1] + 1)

    # Calculate union area.
    area = area_1 + area_2 - overlapped_area(bbox_1, bbox_2)

    # Return union area value.
    return area

def jaccard_index(real_bboxes, detected_bboxes):
    '''
    Returns the value of the Jaccard index between two lists of ground truth bounding boxes and detected bounding boxes.

    Arguments:
        real_bboxes: Ground truth bounding boxes.
        detected_bboxes: Detected bounding boxes.
    Returns:
        matrix: Matrix containing the values of the Jaccard index for every pair of rectangles; every row represents a
        ground truth bounding box; every column represents a detected bounding box.
    '''

    # Create empty matrix.
    matrix = np.zeros(shape=(len(real_bboxes), len(detected_bboxes)))

    # For every pair of rectangles calculate Jaccard index.
    for i in range(len(real_bboxes)):
        for j in range(len(detected_bboxes)):

            # Get overlapped area.
            overlap = overlapped_area(real_bboxes[i], detected_bboxes[j])

            # Get union area.
            union = union_area(real_bboxes[i], detected_bboxes[j])

            '''
            Jaccard index equals to the relation between overlapped area and union area for two rectangles, with a
            value ranging between 0 (minimum similarity) and 1 (maximum similarity).
            '''
            simil = overlap / union

            # Assign value to the corresponding position in the matrix.
            matrix[i, j] = simil

    # Return matrix containing Jaccard index values.

    return matrix

def bbox_match(detected_bboxes, real_bboxes):
    '''
    Use Jaccard index in order to find a match between a detected bounding box and a real bounding box, given two lists
    containing them.

    Arguments:
        detected_bboxes: List containing bounding boxes for every detection.
        real_bboxes: List containing ground truth bounding boxes.
    Returns:
        indexes: List containing, for the detected bounding box corresponding to a certain position (in the same order
        as in the list of detected bounding boxes), the index of its matching ground truth bounding box (in the list of
        ground truth bounding boxes).
    '''

    # Get Jaccard index for every pair of rectangles.
    matrix = jaccard_index(real_bboxes, detected_bboxes)

    # Create empty index array and fill it with a default value (-1).
    indexes = np.empty(shape=len(detected_bboxes), dtype=int)
    indexes.fill(-1)

    # For every real bounding box i:
    for i in range(len(real_bboxes)):

        max_simil = 0
        detected_index = 0

        # For every detected bounding box j:
        for j in range(len(detected_bboxes)):

            '''
            If the similarity between i and j is greater than the maximum recorded similarity between i and any 
            detected bounding box, the matching index for i in the detected bounding box list becomes j:
            '''
            if matrix[i, j] > max_simil:
                max_simil = matrix[i, j]
                detected_index = j

        '''
        After iterating over all possible matches for i, it becomes the matching index in the ground truth bounding
        box list for the detected bounding box with an index equal to the maximum similarity index recorded before.
        '''
        if (max_simil > 0.25 and indexes[detected_index] == -1) or max_simil > matrix[indexes[detected_index], detected_index]:
            indexes[detected_index] = i

    # Return list of ground truth indexes for every detection.
    return indexes

def array_from_csv(source_csv, img_dir):
    '''
    Loads every picture from a dataset (previously processed and listed in a .csv file), as well as their labels (tilt
    and pan values) into 3 Numpy arrays.

    Arguments:
        source_csv: .csv file containing the list of pictures from the dataset which we want to load.
        img_dir: Directory containing the pictures appearing in the .csv file.
    Returns:
        img_array: Numpy array containing the pictures from the dataset (in order of appearance in the .csv file).
        tilt: Numpy array containing tilt values for every picture in the dataset (in order of appearance in the .csv
        file).
        pan: Numpy array containing pan values for every picture in the dataset (in order of appearance in the .csv
        file).
    '''

    # Create empty image list.
    img_array = []

    # Load .csv file as a Pandas dataframe.
    df = pd.read_csv(source_csv)

    # For every annotation in the .csv:
    for index, row in df.iterrows():

        # Get picture path.
        img = row['file']
        path = img_dir + img

        # Load image and append it to the list.
        pic = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_array.append(pic)

    # Convert list of images into an array suitable for its use in model training.
    img_array = np.array(img_array)

    # Tilt and pan arrays can be directly obtained from the dataframe.
    tilt = np.array(df['tilt'])
    pan = np.array(df['pan'])

    # Return image, tilt and pan arrays.
    return img_array, tilt, pan

def array_from_npy(img_npy, source_csv):
    '''
    Loads every picture from a dataset (previously processed and listed in a .csv file), as well as their labels (tilt
    and pan values) into 3 Numpy arrays. Pictures must be stored  as a .npy file.

    Arguments:
        img_npy: File containing pictures to be loaded.
        source_csv: .csv file containing the list of pictures from the dataset which we want to load.
    Returns:
        img_array: Numpy array containing the pictures from the dataset (in order of appearance in the .csv file).
        tilt: Numpy array containing tilt values for every picture in the dataset (in order of appearance in the .csv
        file).
        pan: Numpy array containing pan values for every picture in the dataset (in order of appearance in the .csv
        file).
    '''

    # Load .csv file as a Pandas dataframe.
    df = pd.read_csv(source_csv)

    # Load images.
    img_array = np.load(img_npy)

    # Tilt and pan arrays can be directly obtained from the dataframe.
    tilt = np.array(df['tilt'])
    pan = np.array(df['pan'])

    # Return image, tilt and pan arrays.
    return img_array, tilt, pan