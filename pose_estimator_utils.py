# -*- coding: utf-8 -*-

"""
This file contains the function that allow to exploit the basic functionality of the pose estimator model.
"""

import numpy as np

def get_pose(cropped_img, model, img_norm = [0, 1], tilt_norm = [0, 1], pan_norm = [0, 1], rescale = 1):
    '''
    Estimates pose values (tilt and pan) from an input cropped picture.

    Arguments:
        cropped_img: Image containing the head for which we want to estimate pose.
        model: Estimator model used to find pose.
        img_norm: Tuple containing normalization values to be applied to picture before estimation (mean and std).
        tilt_norm: Tuple containing normalization values to be applied to tilt output (mean and std).
        pan_norm: Tuple containing normalization values to be applied to pan output (mean and std).
        rescale: Value used to rescale output if needed.
    Returns:
        tilt: Tilt value.
        pan: Pan value.
    '''

    # Normalize input picture.
    norm = ((cropped_img / 255.0) - img_norm[0]) / img_norm[1]
    norm = np.reshape(norm, (1, norm.shape[0], norm.shape[1], 1))

    # Get pose values.
    pred = model.predict(norm)

    # Revert normalization at pose values and rescale.
    tilt = (pred[0, 0] * tilt_norm[1] + tilt_norm[0]) * rescale
    pan = (pred[0, 1] * pan_norm[1] + pan_norm[0]) * rescale

    # Return tilt and pan values.
    return tilt, pan