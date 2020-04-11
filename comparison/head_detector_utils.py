# -*- coding: utf-8 -*-

"""
This file contains the functions that allow to exploit the basic functionality of the head detector model.
"""

import cv2
import numpy as np

def get_head_bboxes(img_ori, model):

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)

    # Detect faces
    faces = model.returnMultipleFacesPosition(img_gray, runFrontal=True, runFrontalRotated=True,
                                                      runLeft=True, runRight=True,
                                                      frontalScaleFactor=1.2, rotatedFrontalScaleFactor=1.2,
                                                      leftScaleFactor=1.15, rightScaleFactor=1.15,
                                                      minSizeX=64, minSizeY=64,
                                                      rotationAngleCCW=30, rotationAngleCW=-30)

    bboxes = []

    for face in faces:
        xmin = int(face[0])
        ymin = int(face[1])
        xmax = int(xmin + face[2])
        ymax = int(ymin + face[3])

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