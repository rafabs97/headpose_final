#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a function used to preprocess the AFLW dataset (clean_aflw) as well as a main function allowing the
basic stand-alone processing of the dataset.
"""

from scipy.io import loadmat
import os
import cv2
import collections, itertools
from math import degrees
import shutil

from face_detection import HaarFaceDetector

from head_detector_utils import get_head_bboxes, get_cropped_pics
from clean_utils import bbox_match
from ../dataset_utils import class_assign, split_dataset, find_norm_parameters, store_dataset_arrays

def clean_aflw(aflw_dir, aflw_mat, destination_dir, detector, out_size, grayscale = False, interpolation = cv2.INTER_LINEAR, start_count = 0):
    '''
    Performs the basic processing of the AFLW dataset, obtaining cropped pictures for each head detection in the
    original images, and getting the ground truth pose values for each detection from the .mat file.

    Arguments:
        aflw_dir: Directory containing AFLW dataset pictures in its root.
        aflw_mat: Path to the .mat file containing pose values for each detection.
        destination_dir: Directory where cropped pictures and .csv containing pose values for each crop will be stored.
        detector: Keras model used for detecting human heads in pictures.
        out_size: Length of each side of final cropped pictures.
        start_count: Initial value of the picture count; used to assign filenames to cropped pictures, when AFLW dataset
        is not processed first.
    Returns:
        count: Total number of cropped pictures obtained (starting from start_count + 1).
        t_ratio: Ratio between the number of true detections and the number of annotated heads in the dataset.
        f_ratio: Ratio between the number of false detections and the total number of detections.
    '''

    # Loading .mat file.
    mat = loadmat(aflw_mat)
    values = sorted(zip(mat['fileids'], mat['bboxes'], mat['pose']), key = lambda x: x[0])

    # Initialize count.
    count = start_count

    # If labels.csv exists in destination dir, append pose values; if it doesn't exists then create it.
    if os.path.isfile(destination_dir + 'labels.csv'):
        file = open(destination_dir + 'labels.csv', 'a')
    else:
        file = open(destination_dir + 'labels.csv', 'w')
        file.write('file,tilt,pan\n')

    # Iterator for every annotated head in the dataset (tuples in the .mat file).
    iterator = range(len(values)).__iter__()

    # Initialize count of true detections, processed annotated heads, false detections and total detections.
    t_count = 0
    p_count = 0
    f_count = 0
    d_count = 0

    # Set number of channels for cropped pictures.
    if grayscale == True:
        out_shape = (out_size, out_size)
    else:
        out_shape = (out_size, out_size, 3)

    # For every detection in the dataset:
    for tuple_index in iterator:

        # Increase processed annotated heads count.
        p_count = p_count + 1

        # Load the picture containing the annotated head.
        pic = cv2.imread(aflw_dir + values[tuple_index][0].strip())

        # Get detections for the loaded picture.
        detected_bboxes = get_head_bboxes(pic, detector)

        # Increase detections count.
        d_count = d_count + len(detected_bboxes)

        # Get cropped pictures from the loaded picture.

        if grayscale == True:
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

        c_pics = get_cropped_pics(pic, detected_bboxes, out_size, 0, cropping='small', interpolation=interpolation)

        # Create arrays containing ground truth detections and poses, and initialize them with the values corresponding
        # to the current tuple.
        true_bboxes = [values[tuple_index][1]]
        poses = [values[tuple_index][2]]

        i = 1

        '''
        For each following picture with the same fileid, add ground truth detections and poses to the arrays created
        before:
        '''

        while (tuple_index + i) < len(mat['fileids']) and values[tuple_index + i][0] == id:
            p_count = p_count + 1
            true_bboxes.append(values[tuple_index + i][1])
            poses.append(values[tuple_index + i][2])
            i = i + 1

        collections.deque(itertools.islice(iterator, i - 1))

        # Update detection counters.
        if len(detected_bboxes) > len(true_bboxes):
            f_count = f_count + len(detected_bboxes) - len(true_bboxes)
            t_count = t_count + len(true_bboxes)
        else:
            t_count = t_count + len(detected_bboxes)

        # If there are cropped pictures:
        if c_pics:

            # Match detected bounding boxes with ground truth bounding boxes.
            indexes = bbox_match(detected_bboxes, true_bboxes)

            # For each matching:
            for box_index in range(len(indexes)):

                '''
                If there is a valid matching between a detected bounding box and a ground truth bounding box and the
                size of the cropped pic corresponding to the detected bounding box is the expected:
                '''
                if indexes[box_index] != -1 and c_pics[box_index].shape == out_shape:

                    # Pose values for the detection are the values corresponding to its matching ground truth annotated head.
                    tilt = degrees(poses[indexes[box_index]][1])
                    pan = degrees(poses[indexes[box_index]][2])

                    # Get cropped pic corresponding to the detection.
                    c_pic = c_pics[box_index]

                    # Store picture.
                    cv2.imwrite(destination_dir + "pic_" + str(count) + ".jpg", c_pic)
                    file.write("pic_" + str(count) + ".jpg," + str(tilt) + "," + str(pan) + "\n")

                    # Mirror picture.
                    pan = -1 * pan
                    c_pic = cv2.flip(c_pic, 1)

                    # Store mirrored picture.
                    cv2.imwrite(destination_dir + "pic_" + str(count + 1) + ".jpg", c_pic)
                    file.write("pic_" + str(count + 1) + ".jpg," + str(tilt) + "," + str(pan) + "\n")

                    # Increase cropped picture count.
                    count = count + 2
                    print("Count:", count)

    # Calculate t_ratio and f_ratio.
    t_ratio = t_count / p_count
    f_ratio = f_count / d_count

    # Return number of cropped pictures obtained (starting from start_count + 1), t_ratio and f_ratio.
    return count, t_ratio, f_ratio

def main():
    '''
    This function acts as a testbench for the function clean_aflw, using it to perform the basic processing of the AFLW
    dataset from a set of default values defined below.
    '''

    # Source paths.

    aflw_dir = '../original/aflw/'
    aflw_mat = '../original/aflw/dataset_landmarks_and_pose_withfaceids.mat'

    # Destination path.

    destination_dir = 'clean/aflw_haar_area/'

    # Detector model path.

    frontal_detector_path = 'models/haarcascade_frontalface_alt.xml'
    profile_detector_path = 'models/haarcascade_profileface.xml'

    # Detection parameters.

    out_size = 64

    # Output paramenters

    grayscale_output = True
    downscaling_interpolation = cv2.INTER_AREA

    # Number of splits for class assignation.

    num_splits_tilt = 8
    num_splits_pan = 8

    # Ratios for train/test and train/validation split.

    test_ratio = 0.2
    validation_ratio = 0.2

    # Detector model.

    detector = HaarFaceDetector(frontal_detector_path, profile_detector_path)

    # Check if output directory exists.

    try:
        os.mkdir(destination_dir)
        print("Directory", destination_dir, "created.")
    except FileExistsError:
        print("Directory", destination_dir, "already exists.")
        shutil.rmtree(destination_dir)
        os.mkdir(destination_dir)

    # Actual cleaning.

    clean_aflw(aflw_dir, aflw_mat, destination_dir, detector, out_size, grayscale_output, downscaling_interpolation)

    # Assign classes.

    class_assign(destination_dir, num_splits_tilt, num_splits_pan)

    # Split dataset.

    split_dataset(destination_dir, test_ratio, validation_ratio)

    # Get normalization parameters.

    find_norm_parameters(destination_dir)

    # OPTIONAL: Save dataset as numpy arrays (for uploading to Google Colab).

    store_dataset_arrays(destination_dir)

if __name__ == "__main__":
    main()