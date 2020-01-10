# -*- coding: utf-8 -*-

"""
This file implements the data generator used during the training of the head pose estimator models in this project.
"""

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from math import ceil
import numpy as np

class HeadPoseDataGenerator(Sequence):
    '''
    This class implements a basic Keras data generator overriding methods from its parent class Sequence. The purpose
    of this data generator is to deliver in each batch a set of images from the subset used to initalize this generator
    containing an equal number of members per class.
    '''

    def __init__(self, pose_dataframe, img_array, batch_size,
                 normalize=False, input_norm=None, tilt_norm=None, pan_norm=None,
                 augment=False, shift_range=None, zoom_range=None, brightness_range=None,
                 img_rescale=1, out_rescale=1):
        '''
        Initializes the data generator with the data from a given subset from the original dataset, and the values to
        use when doing data augmentation.

        Arguments:
            pose_dataframe: Dataframe containing a list of each picture in the given subset and its pose values.
            img_array: Numpy array containing the images from the given subset.
            batch_size: Number of pictures per batch.
            normalize: If the data shall be normalized or not.
            input_norm: Tuple containing mean and std values for normalizing pictures in the dataset.
            tilt_norm: Tuple containing mean and std values for normalizing tilt values in the dataset.
            pan_norm: Tuple containing mean and std values for normalizing pan values in the dataset.
            augment: If data augmentation shall be applied or not.
            shift_range: Value (between 0 and 1) indicating the portion of the length of the side of each picture that
            can be used to shift the picture (in both axes).
            zoom_range: Tuple containing the minimum and maximum values used to apply zoom to each picture.
            brightness_range: Tuple containing the minimum and maximum values used to apply a brightness transformation
            to each picture.
            img_rescale: Each pixel from every picture in the subset will be multiplied by this value.
            out_rescale: Tilt and pan values for every picture in the subset will be multiplied by this value.
        '''

        # Create empty arrays for pictures and labels from the subset.
        self.pics = []
        self.labels = []

        # Initialize batch size.
        self.batch_size = batch_size

        # Initialize normalization parameters.
        self.normalize = normalize
        self.input_norm = input_norm

        '''
        Initialize the parameter controlling if data augmentation shall be applied or not, and data augmentation 
        parameters.
        '''
        self.augment = augment

        if self.augment == True:
            self.generator = ImageDataGenerator(width_shift_range=shift_range, height_shift_range=shift_range,
                                                brightness_range=brightness_range, zoom_range=zoom_range)

        # Initialize scaling parameters.
        self.img_rescale = img_rescale
        self.out_rescale = out_rescale

        '''
        Initialize the iterator used to control the position of the next picture from every class that will be included
        in a batch.
        '''
        self.common_iterator = 0

        # Sort dataframe by class.
        df = pose_dataframe.sort_values('class')

        # Initialize the number of pictures in the dataset.
        self.total_size = len(df.index)

        # Load images and pose values into the previously created arrays.
        prev_class = -1
        class_index = -1

        # For each image in the (ordered) dataset:
        for index, row in df.iterrows():

            '''
            If the class for the current picture is different from the last class recorded, append an empty list for the
            new class.
            '''
            if row['class'] != prev_class:
                prev_class = row['class']
                self.pics.append([])
                self.labels.append([])
                class_index = class_index + 1

            # Append picture to corresponding class array.
            self.pics[class_index].append(np.squeeze(img_array[index]))

            # Append labels to corresponding class array (normalized and rescaled).
            self.labels[class_index].append([((row['tilt'] * out_rescale) - tilt_norm[0]) / tilt_norm[1] , ((row['pan'] * out_rescale) - pan_norm[0]) / pan_norm[1]])

        # Assert batch size is a multiple of the number of classes.
        assert(batch_size % len(self.pics) == 0)

    def __data_generation(self):
        '''
        Outputs a batch of pictures.

        Returns:
            X: Pictures in the batch.
            y: Labels for each picture in the batch.
        '''

        # Create empty lists for pictures and labels.
        X = []
        y = []

        # For each picture-per-class:
        for i in range(int(self.batch_size / len(self.pics))):

            # For each class:
            for j in range(len(self.pics)):

                # Select the next picture in the class list (start from beginning after the last picture).
                pic = self.pics[j][int(self.common_iterator % len(self.pics[j]))]
                pic = np.expand_dims(pic, axis=2)

                # Apply data augmentation.
                if self.augment == True:
                    transformation = self.generator.get_random_transform(pic.shape)
                    transformation['zx'] = transformation['zy']
                    pic = self.generator.apply_transform(pic, transformation)

                # Rescale each pixel value in image.
                pic = pic * self.img_rescale

                # Normalize image.
                if self.normalize == True:
                    pic = (pic - self.input_norm[0]) / self.input_norm[1]

                # Add image and labels to the batch.
                X.append(pic)
                y.append(self.labels[j][int(self.common_iterator % len(self.labels[j]))])

            # Update iterator.
            self.common_iterator = self.common_iterator + 1

        # Transform lists into Numpy arrays.
        X = np.array(X)
        y = np.array(y)

        # Return images and labels.
        return X, y

    def __len__(self):
        '''
        Outputs the length (number of batches) that the data generator can provide.

        Returns:
            l: The length of the data generator.
        '''

        '''
        Calculate the length of the data generator as the relation between the total number of images and the size of
        each batch; in order to function properly with uneven class lengths the number is rounded to the smaller integer 
        bigger or equal to the obtained result.
        '''
        l = ceil(self.total_size / self.batch_size)

        return l

    def __getitem__(self, index):
        '''
        Outputs a new batch given the batch index.

        Returns:
            X: Pictures in the batch.
            y: Labels for each picture in the batch.
        '''

        # Set the class iterator in the correct position for obtaining the requested batch.
        self.common_iterator = index * int(self.batch_size / len(self.pics))

        # Generate the batch.
        X, y = self.__data_generation()

        # Return images and labels for the requested batch.
        return X, y

    def reset(self):
        '''
        Resets the class iterator to its initial position.
        '''

        self.common_iterator = 0
