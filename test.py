#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be used to test the basic functionality developed in this project, using a video stream as input and
getting estimated pose for every detected head in it. The usage of this script is described in the User's Manual.
"""

import cv2
from math import sin, radians
from datetime import datetime

from keras_ssd512 import ssd_512

from architectures import mpatacchiola_generic
from head_detector_utils import get_head_bboxes, get_cropped_pics
from pose_estimator_utils import get_pose

# Paths

detector_file = 'head-detector.h5'
estimator_file = 'pose-estimator.h5'

models_path = 'models/'

detector_path = models_path + detector_file
estimator_path = models_path + estimator_file

# Detector parameters.

in_size_detector = 512
confidence_threshold = 0.65

# Estimator parameters.

in_size_estimator = 64
num_conv_blocks = 6
num_filters_start = 64
num_dense_layers = 1
dense_layer_size = 512

# Normalization parameters.

mean = 0.407335
std = 0.236271

t_mean = -0.022308
t_std = 0.324841

p_mean = 0.000171
p_std = 0.518044

# Models.

head_detector = ssd_512(image_size=(in_size_detector, in_size_detector, 3), n_classes=1, min_scale=0.1, max_scale=1, mode='inference')
head_detector.load_weights(detector_path)

pose_estimator = mpatacchiola_generic(in_size_estimator, num_conv_blocks, num_filters_start, num_dense_layers, dense_layer_size)
pose_estimator.load_weights(estimator_path)

# Get video source.

video_source = input("Input stream: ")

# If video source is a device convert the string identifying it to integer.

try:
    video_source = int(video_source)
except:
    pass

# Initialize cam.

cam = cv2.VideoCapture(video_source)
ori_width = int(cam.get(3))
ori_height = int(cam.get(4))

cam.set(cv2.CAP_PROP_SETTINGS, 1)

# Set output file name, if it is to be recorded, or -1 if the output is to be discarded.

output_path = input("Output file name: ")

if output_path != "-1":
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), cam.get(cv2.CAP_PROP_FPS), (ori_width, ori_height))

# Set the value that controls if the picture should be flipped horizontally.

flip = None

while flip != 'Y' and flip != 'N':
    flip = input("Flip? (Y/N): ")

# Initialize output values and counters before processing the video stream.

out = True

frame_count = 0
fps_mean = 0

# While there is a frame from the video stream:
while out == True:

    # Get processing start time for the current frame.
    frame_start = datetime.now()

    # Try to get a frame from the camera.
    out, img = cam.read()

    # If there is no frame, exit.
    if out == False:
        break

    # Flip picture if needed.
    if flip == 'Y':
        img = cv2.flip(img, 1)

    # Get bounding boxes for every detected head in the picture.
    bboxes = get_head_bboxes(img, head_detector, confidence_threshold)

    # Get cropped pics for every valid bounding box.
    gray_pic = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heads = get_cropped_pics(gray_pic, bboxes, in_size_estimator, 0, cropping='small')

    # Initialize head counter.
    head_count = 0

    # For each cropped picture:
    for i in range(len(heads)):

        # If it is a valid picture:
        if heads[i].shape == (in_size_estimator, in_size_estimator):

            # Increase head counter.
            head_count = head_count + 1

            # Get pose values.
            tilt, pan = get_pose(heads[i], pose_estimator, img_norm = [mean, std], tilt_norm = [t_mean, t_std],
                                 pan_norm = [p_mean, p_std], rescale=90.0)

            # Get minimum and maximum values for both axes of the bounding box.
            xmin, ymin, xmax, ymax = bboxes[i]

            # Draw detection in the original picture..

            rect = cv2.rectangle(img, (xmax, ymin), (xmin, ymax), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(rect, 'Tilt: ' + str(round(tilt, 2)) + ' Pan: ' + str(round(pan, 2)), (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

            # Draw arrow from the center of the picture in the direction of the pose in the original picture.

            centerx = int((xmin + xmax) / 2)
            centery = int((ymin + ymax) / 2)
            center = (centerx, centery)

            max_arrow_len = (xmax - xmin + 1) / 2

            offset_x = -1 * int(sin(radians(pan)) * max_arrow_len)
            offset_y = -1 * int(sin(radians(tilt)) * max_arrow_len)

            end = (centerx + offset_x, centery + offset_y)
            cv2.arrowedLine(img, center, end, (0, 0, 255), 2, line_type=cv2.LINE_AA)

    # Show image with detections.

    cv2.imshow('Detections', img)

    if output_path != "-1":
        writer.write(img)

    # Get processing end time for the current frame.
    frame_end = datetime.now()

    # Calculate frametime.
    total_time = frame_end - frame_start

    # Calculate FPS at the current moment as the inverse of the frametime.
    fps = 1 / total_time.total_seconds()

    # Print for this frame the number of heads processed and the amount of FPS.
    print("Heads: %d, FPS: %.2f" % (head_count, fps))

    # Update FPS mean.
    fps_mean = fps_mean * (frame_count / (frame_count + 1)) + fps / (frame_count + 1)

    # Update processed frame counter.
    frame_count = frame_count + 1

    # If 'Esc' key is pressed, exit.
    if cv2.waitKey(1) == 27:
        break

# Before ending execution, show mean FPS value.
print("FPS (avg): %.2f" % fps_mean)

# Destroy CV2 windows.
cv2.destroyAllWindows()
