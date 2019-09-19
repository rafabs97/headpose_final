# -*- coding: utf-8 -*-

"""
This file contains functions that implement additional operations to be applied on previously preprocessed datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from clean_utils import array_from_csv

def class_assign(database_dir, num_splits_tilt, num_splits_pan):
    '''
    Divides the dataset in classes containing an equal number of pictures by splitting the dataset first by pan values
    and then by tilt values.

    Arguments:
        database_dir: Directory containing the preprocessed dataset.
        num_splits_tilt: Number of divisions to be made when splitting by tilt value.
        num_splits_pan: Number of divisions to be made when splitting by pan value.
    '''

    # Read .csv file containing labels for each picture in the dataset as a Pandas dataframe, and sort it by pan value.
    df = pd.read_csv(database_dir + 'labels.csv').sort_values('pan')

    '''
    Create a new .csv file which will contain the data from the original labels file plus the assigned class for 
    each picture.
    '''
    file = open(database_dir + 'labels_class.csv', 'w')

    # Number of elements per split in the first division (splitting by pan angle value).
    pan_split_size = int(df.shape[0] / num_splits_pan)

    # For each division by pan angle:
    for i in range(num_splits_pan):

        # Calculate the first element of the dataframe to be considered part of this pan split.
        lower_limit_1 = i * pan_split_size

        # Calculate the element after the last element of the dataframe to be considered part of this pan split.
        if i == (num_splits_pan - 1):
            upper_limit_1 = df.shape[0]
        else:
            upper_limit_1 = (i + 1) * pan_split_size

        # Get an split from the original dataset using the limits calculated before, and sort it by tilt value.
        data_slice_pan = df.iloc[lower_limit_1:upper_limit_1, :].sort_values('tilt')

        # Number of elements per split in the second division (splitting by tilt angle value).
        tilt_split_size = int(data_slice_pan.shape[0] / num_splits_tilt)

        # For each division by tilt angle:
        for j in range(num_splits_tilt):

            # Calculate class for the elements contained within the same tilt and pan splits.
            p_class = i * num_splits_tilt + j

            # Calculate the first element of the pan split to be considered part of this tilt split.
            lower_limit_2 = j * tilt_split_size

            # Calculate the element after the last element of the pan split to be considered part of this tilt split.
            if j == (num_splits_tilt - 1):
                upper_limit_2 = data_slice_pan.shape[0]
            else:
                upper_limit_2 = (j + 1) * tilt_split_size

            # Get an split from the pan split using the limits calculated before.
            data_slice_tilt = data_slice_pan.iloc[lower_limit_2:upper_limit_2, :].copy()

            # Add class value to every tuple in the split.
            data_slice_tilt['class'] = str(p_class)

            # Record data to the new .csv file.
            if p_class == 0:
                data_slice_tilt.to_csv(file, index=False, line_terminator='\n')
            else:
                data_slice_tilt.to_csv(file, index=False, header=False, line_terminator='\n')

    # Close the new .csv file.
    file.close()

def split_dataset(database_dir, test_ratio, validation_ratio):
    '''
    Divide dataset in train, test and validation subsets in a stratified way based on the class of each picture in the
    dataset, and record the pictures for each subset in a different .csv in the dataset directory.

    Arguments:
        database_dir: Directory containing the preprocessed dataset.
        test_ratio: Portion of the original dataset to be used as a test partition.
        validation_ratio: Portion of the remainder of the first split to be used as validation partition; train dataset
        will be the remainder of this second split.
    '''

    # Read .csv file containing labels and class for each picture in the dataset as a Pandas dataframe.
    df = pd.read_csv(database_dir + 'labels_class.csv')

    # Split dataset.
    train, test = train_test_split(df, test_size=test_ratio, stratify=df['class'])
    train, validation = train_test_split(train, test_size=validation_ratio, stratify=train['class'])

    # Record each subset in a different .csv file.
    train.to_csv(database_dir + 'train.csv', index=False)
    validation.to_csv(database_dir + 'validation.csv', index=False)
    test.to_csv(database_dir + 'test.csv', index=False)

def find_norm_parameters(database_dir):
    '''
    Find normalization parameters for images, tilt values and pan values from training subset and store them in a .csv
    file in the dataset directory.

    Arguments:
        database_dir: Directory containing the preprocessed dataset.
    '''

    # Get image, tilt and pan arrays from the preprocessed dataset.
    img, tilt, pan = array_from_csv(database_dir + 'train.csv', database_dir)

    # Calculate normalization parameters for the images.
    i_mean = np.mean(img / 255.0)
    i_std = np.std(img / 255.0)

    # Calculate normalization parameters for the tilt values.
    t_mean = np.mean(tilt / 90.0)
    t_std = np.std(tilt / 90.0)

    # Calculate normalization parameters for the pan values.
    p_mean = np.mean(pan / 90.0)
    p_std = np.std(pan / 90.0)

    # Store normalization values in a .csv file.
    file = open(database_dir + 'norm.csv', 'w')
    file.write('img_mean,img_std,t_mean,t_std,p_mean,p_std\n')
    file.write('%f,%f,%f,%f,%f,%f\n' % (i_mean, i_std, t_mean, t_std, p_mean, p_std))

    # Close .csv file.
    file.close()

def store_dataset_arrays(database_dir):
    '''
    Store preprocessed dataset images as a collection of .npy files.

    Arguments:
        database_dir: Directory containing the preprocessed dataset.
    '''

    # Load dataset as Numpy arrays.
    tr_img, tr_tilt, tr_pan = array_from_csv(database_dir + 'train.csv', database_dir)
    v_img, v_tilt, v_pan = array_from_csv(database_dir + 'validation.csv', database_dir)
    t_img, t_tilt, t_pan = array_from_csv(database_dir + 'test.csv', database_dir)

    # Store images as .npy arrays.
    np.save(database_dir + 'train_img.npy', tr_img)
    np.save(database_dir + 'validation_img.npy', v_img)
    np.save(database_dir + 'test_img.npy', t_img)