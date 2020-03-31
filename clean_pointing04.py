#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a function used to preprocess the Pointing'04 dataset (clean_pointing04) and the function to get the
ground truth pose from the filename of each image, as well as a main function allowing the basic stand-alone processing
of the dataset.
"""

import os
import re
import glob
import cv2
import shutil

from keras_ssd512 import ssd_512

from head_detector_utils import get_head_bboxes, get_cropped_pics
from dataset_utils import class_assign, split_dataset, find_norm_parameters, store_dataset_arrays

def pose_from_filename(img):
    '''
    Gets ground truth pose values for a picture in Pointing'04 dataset from its filename.

    Arguments:
        img: Path of the picture for which we want to get the ground truth pose values (if it has the format used
        for filenames in Pointing'04 dataset).
    Returns:
        tilt: Ground truth value for the tilt angle.
        pan: Ground truth value for the pan angle.
    '''

    # Get filename from path.
    raw = os.path.basename(img)[11:]
    raw = os.path.splitext(raw)[0]

    # Get absolute tilt and pan values from filename.
    tilt, pan = re.split('[-+]', raw[1:])

    # Get actual numerical values from the strings obtained before, plus the sign corresponding to each angle.
    tilt = int(raw[0] + tilt)
    pan = int(raw[len(raw) - len(pan) - 1] + pan)

    # Return tilt and pan values.
    return tilt, pan

def clean_pointing04(pointing04_dir, destination_dir, detector, confidence_threshold, out_size, grayscale = False, interpolation = cv2.INTER_LINEAR, start_count = 0, duplicate_until = 0):
    '''
    Performs the basic processing of the Pointing'04 dataset, obtaining cropped pictures for each head detection in the
    original images, and getting the ground truth pose values for each detection from the .mat file.

    Arguments:
        pointing04_dir: Directory containing Pointing'04 dataset pictures in its root.
        destination_dir: Directory where cropped pictures and .csv containing pose values for each crop will be stored.
        detector: Keras model used for detecting human heads in pictures.
        confidence_threshold: Value used to filter detections (detections must have a confidence value higher than this
        value in order to be considered valid).
        out_size: Length of each side of final cropped pictures.
        start_count: Initial value of the picture count; used to assign filenames to cropped pictures, when AFLW dataset
        is not processed first.
        duplicate_until: Target number of pictures per class; used in order to augment the size of the dataset by duplicating
        pictures on each class. Each pose in the dataset corresponds to a different class.
    Returns:
        count: Total number of cropped pictures obtained (starting from start_count + 1).
        t_ratio: Ratio between the number of true detections and the number of annotated heads in the dataset.
        f_ratio: Ratio between the number of false detections and the total number of detections.
    '''

    # Initialize count.
    count = start_count

    # If labels.csv exists in destination dir, append pose values; if it doesn't exists then create it.
    if os.path.isfile(destination_dir + 'labels.csv'):
        file = open(destination_dir + 'labels.csv', 'a')
    else:
        file = open(destination_dir + 'labels.csv', 'w')
        file.write('file,tilt,pan\n')

    # Initialize count of true detections, processed annotated heads, false detections and total detections.
    t_count = 0
    p_count = 0
    f_count = 0
    d_count = 0

    # Create empty arrays for storing pictures in each class.
    pics_by_class = [[] for i in range(169)]

    # Set number of channels for cropped pictures.
    if grayscale == True:
        out_shape = (out_size, out_size)
    else:
        out_shape = (out_size, out_size, 3)

    # For each person in the dataset:
    for i in range(1, 16):

        # Get the path for every picture of that person.
        num = '{0:02}'.format(i)
        path = pointing04_dir + "Person" + num
        images = glob.glob(path + "/*.jpg")

        # For each picture:
        for img in images:

            # Increase processed annotated heads count.
            p_count = p_count + 1

            # Get ground truth pose values from filename.
            tilt, pan = pose_from_filename(img)

            # Load the picture.
            pic = cv2.imread(img)

            # Get detections for the loaded picture.
            bboxes = get_head_bboxes(pic, detector, confidence_threshold)

            # Increase detections count.
            d_count = d_count + len(bboxes)

            # Update detection counters.
            if len(bboxes) > 1:
                f_count = f_count + len(bboxes) - 1
                t_count = t_count + 1
            else:
                t_count = t_count + len(bboxes)

            # Get cropped pictures from the loaded picture.

            if grayscale == True:
                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

            c_pics = get_cropped_pics(pic, bboxes, out_size, 0, cropping='small', interpolation=interpolation)

            # For each cropped picture:
            for c_pic in c_pics:

                # If the size of the cropped pic is the expected:
                if c_pic.shape == out_shape:

                    # Store picture.
                    cv2.imwrite(destination_dir + "pic_" + str(count) + ".jpg", c_pic)
                    file.write("pic_" + str(count) + ".jpg," + str(tilt) + "," + str(pan) + "\n")

                    # Assign class to picture from its pose values.
                    p_class = int(13 * ((tilt + 90) / 15) + ((pan + 90) / 15))
                    pics_by_class[p_class].append(count)

                    # Mirror picture
                    pan = -1 * pan
                    c_pic = cv2.flip(c_pic, 1)

                    # Store mirrored picture.
                    cv2.imwrite(destination_dir + "pic_" + str(count + 1) + ".jpg", c_pic)
                    file.write("pic_" + str(count + 1) + ".jpg," + str(tilt) + "," + str(pan) + "\n")

                    # Calculate class for mirrored picture.
                    p_class = int(13 * ((tilt + 90) / 15) + ((pan + 90) / 15))
                    pics_by_class[p_class].append(count + 1)

                    # Increase cropped picture count.
                    count = count + 2
                    print("Count:", count)

    '''
    If duplicate_until has a value of -1, the pictures in the dataset are duplicated until the number of cropped 
    pictures match the number of pictures indicated by start_count; else, the target number of pictures per class
    will match that value.
    '''
    if duplicate_until == -1:
        increase = start_count / (count - start_count)
    else:
        target_len = duplicate_until

    # If there are cropped pictures for the Pointing'04 dataset:
    if (count - start_count) > 0:

        # For each class:
        for i in range(len(pics_by_class)):

            # If there are pictures in the current class:
            if pics_by_class[i]:

                '''
                If duplicate_until has a value of -1, the target number of pictures for the current class is equal to the
                number of pictures already in that class times the proportion between the value of start_count and the number
                of cropped pictures obtained until this point.
                '''
                if duplicate_until == -1:
                    target_len = int(len(pics_by_class[i]) * increase)

                # If the number of pics in the current class is below the target number of pictures:
                if len(pics_by_class[i]) < target_len:

                    # Get pose values from class label.
                    tilt = int(i / 13) * 15 - 90
                    pan = int(i % 13) * 15 - 90

                    # Duplicate pictures in class until the target number of pictures is reached.
                    for j in range(target_len - len(pics_by_class[i])):
                        shutil.copyfile(destination_dir + "pic_" + str(pics_by_class[i][int(j % len(pics_by_class[i]))]) + ".jpg", destination_dir + "pic_" + str(count) + ".jpg")
                        file.write("pic_" + str(count) + ".jpg," + str(tilt) + "," + str(pan) + "\n")

                        # Increase cropped picture count.
                        count = count + 1
                        print("Count:", count)

    # Calculate t_ratio and f_ratio.
    t_ratio = t_count / p_count
    f_ratio = f_count / d_count

    # Return number of cropped pictures obtained (starting from start_count + 1), t_ratio and f_ratio.
    return count, t_ratio, f_ratio

def main():
    '''
    This function acts as a testbench for the function clean_pointing04, using it to perform the basic processing of the
    Pointing'04 dataset from a set of default values defined below.
    '''

    # Source paths.

    pointing04_dir = 'original/HeadPoseImageDatabase/'

    # Destination path.

    destination_dir = 'clean/pointing04/'

    # Detector model path.

    head_detector_path = 'models/head-detector.h5'

    # Detection parameters.

    in_size = 512
    out_size = 64
    confidence_threshold = 0.75

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

    clean_pointing04(pointing04_dir, destination_dir, detector, confidence_threshold, out_size, grayscale_output, downscaling_interpolation)

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