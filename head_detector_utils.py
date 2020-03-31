# -*- coding: utf-8 -*-

"""
This file contains the functions that allow to exploit the basic functionality of the head detector model.
"""

import cv2
import numpy as np
from keras.preprocessing import image

def get_head_bboxes(img_ori, model, confidence_threshold):
    '''
    Detects heads from an imput image and filters them by confidence value.

    Arguments:
        img_ori: Original input image from which we want to get detections.
        model: Head detector model.
        confidence_threshold: Value of confidence from which a detection will be valid.
    Returns:
        bboxes: List of rectangles enclosing heads in the original image, defined by their minimum and maximum row
        and column values.
    '''

    # Create an empty list of images which will be used as the input for the detector model.
    input_image = []

    # Input size for the detector.
    img_height = 512
    img_width = 512

    # Resize a copy of the original picture to the input size of the model.
    img_res = cv2.resize(img_ori, (img_width, img_height))

    # Append resized image to the list previously created, and convert it to a Numpy array.
    img = image.img_to_array(img_res)
    input_image.append(img)
    input_image = np.array(input_image)

    # Get predictions for the resized input image.
    y_pred = model.predict(input_image)

    # Filter predictions by confidence threshold.
    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    # Create empty bounding box list.
    bboxes = []

    # Append bounding boxes to the list previously created, transformed to the original image dimensions.
    for box in y_pred_thresh[0]:
        xmin = int(box[-4] * img_ori.shape[1] / img_width)
        ymin = int(box[-3] * img_ori.shape[0] / img_height)
        xmax = int(box[-2] * img_ori.shape[1] / img_width)
        ymax = int(box[-1] * img_ori.shape[0] / img_height)

        bboxes.append([xmin, ymin, xmax, ymax])

    # Return valid bounding boxes.
    return bboxes

def get_cropped_pics(img_ori, bboxes, crop_size, offset_perc, cropping = '', interpolation = cv2.INTER_LINEAR):
    '''

    Arguments:
        img_ori: Original input image from which we want to get cropped pics.
        bboxes: Bounding boxes previously obtained from the original input image.
        crop_size: Length of the side of the desired output images. When using cropping = 'small' or cropping = 'large',
        it will be the length of the side of the square; else it will be the length of the minor side.
        offset_perc: Percentage (between 0 and 1) over the original cropped picture length to be also included in the
        final output picture, around the original cropped picture borders; the final length of the side of the output
        pictures will be equal to crop_size * (1 + 2 * offset_perc).
        cropping: Cropping type. By default crop the original detection; other possible values are 'small' (crop using
        a box with a side length equal to the minimum length of the original detection) and 'large' (crop using
        a box with a side length equal to the maximum length of the original detection).
    Returns:
        pics: List containing output pictures.
    '''

    # Record original dimensions for the input picture.
    ori_height = img_ori.shape[0]
    ori_width = img_ori.shape[1]

    # Create empty output picture list.
    pics = []

    # For each bounding box:
    for box in bboxes:

        # Get maximum and minimum value for both axes.
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]

        # Crop picture using square box.
        if cropping == 'large':

            # Large box
            if (xmax - xmin) > (ymax - ymin):
                ymin = int((box[3] + box[1]) / 2 - (box[2] - box[0] + 1) / 2)
                ymax = int((box[3] + box[1]) / 2 + (box[2] - box[0] + 1) / 2) - 1

            elif (ymax - ymin) > (xmax - xmin):
                xmin = int((box[2] + box[0]) / 2 - (box[3] - box[1] + 1) / 2)
                xmax = int((box[2] + box[0]) / 2 + (box[3] - box[1] + 1) / 2) - 1

        elif cropping == 'small':

            # Small box
            if (xmax - xmin) > (ymax - ymin):
                xmin = int((box[2] + box[0]) / 2 - (box[3] - box[1] + 1) / 2)
                xmax = int((box[2] + box[0]) / 2 + (box[3] - box[1] + 1) / 2) - 1

            elif (ymax - ymin) > (xmax - xmin):
                ymin = int((box[3] + box[1]) / 2 - (box[2] - box[0] + 1) / 2)
                ymax = int((box[3] + box[1]) / 2 + (box[2] - box[0] + 1) / 2) - 1

        new_size = xmax - xmin

        # Increase box size

        xmin = xmin - int(new_size * offset_perc)
        ymin = ymin - int(new_size * offset_perc)

        xmax = xmax + int(new_size * offset_perc)
        ymax = ymax + int(new_size * offset_perc)

        # If box outside limits, try to fit inside picture dimensions.

        if xmin < 0:
            xmax = xmax - xmin
            xmin = 0

        if xmax >= (ori_width - 1):
            xmin = (ori_width - 1) - (xmax - xmin)
            xmax = ori_width - 1

        if ymin < 0:
            ymax = ymax - ymin
            ymin = 0

        if ymax >= (ori_height - 1):
            ymin = (ori_height - 1) - (ymax - ymin)
            ymax = ori_height - 1

        # Check if new box is valid: if it is valid, append to the output image list; if not, append a default value.
        if xmin >= 0 and ymin >= 0 and xmax < ori_width and ymax < ori_height:

            # Crop picture using the final boundaries.
            c_pic = img_ori[ymin:ymax, xmin:xmax]

            # If the box was to be reshaped, resize it.
            if cropping == 'small' or cropping == 'large':
                c_pic = cv2.resize(c_pic, (int(crop_size * (1 + 2 * offset_perc)), int(crop_size * (1 + 2 * offset_perc))), interpolation=interpolation)
            else:
                if c_pic.shape[0] > c_pic.shape[1]:
                    c_pic = cv2.resize(c_pic, (int((crop_size * c_pic.shape[0] / c_pic.shape[1]) * (1 + 2 * offset_perc)),
                                               int(crop_size * (1 + 2 * offset_perc))), interpolation=interpolation)
                else:
                    c_pic = cv2.resize(c_pic, (int(crop_size * (1 + 2 * offset_perc)),
                                               int((crop_size * c_pic.shape[1] / c_pic.shape[0]) * (1 + 2 * offset_perc))), interpolation=interpolation)

            # Append output picture.
            pics.append(c_pic)
        else:
            # Append default value.
            pics.append(np.empty(0))

    # Return output picture list.
    return pics