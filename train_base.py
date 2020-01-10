#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is used to train new pose estimator models on batch, setting parameters from another script.

THIS SCRIPT IS INTENDED TO BE CALLED FROM 'train_architectures.py'. FOR MANUAL PARAMETER SELECTION USE 'train.py' INSTEAD.
"""

# Try to set seeds for everything

import numpy as np
import tensorflow as tf
import random as rn
import os
import sys

seed = 0

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(seed)
rn.seed(seed)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True

from tensorflow.keras import backend as K

tf.set_random_seed(seed)

sess = tf.Session(graph = tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Other imports.

import time
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau

from architectures import mpatacchiola_generic
from data_generator_array import HeadPoseDataGenerator

# Control parameters.

batch_size = 128
epochs = 500
verbose = True
patience = 10

# Datagen parameters.

mean = 0.408808
std = 0.237583

t_mean = -0.041212
t_std = 0.323931

p_mean = -0.000276
p_std = 0.540958

# Paths.

clean_dir = '/gdrive/My Drive/headpose_final/clean/'
db_name = 'aflw_pointing04'

model_dir = '/gdrive/My Drive/headpose_final/models/'
model_csv = model_dir + 'models.csv'

# Callbacks.

stop = EarlyStopping(monitor='val_mean_absolute_error', patience=patience, verbose=verbose, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau()

# Dataset paths.

img_dir = clean_dir + db_name + '/'

train_csv = img_dir + 'train.csv'
validation_csv = img_dir + 'validation.csv'
test_csv = img_dir + 'test.csv'

# Load dataframes.

train_df = pd.read_csv(train_csv)
validation_df = pd.read_csv(validation_csv)
test_df = pd.read_csv(test_csv)

# Load image arrays.

train_array = np.load(img_dir + 'train_img.npy')
validation_array = np.load(img_dir + 'validation_img.npy')
test_array = np.load(img_dir + 'test_img.npy')

# Set input size for the estimator model.

in_size = 64

# Get the rest of parameters (architecture and data augmentation parameters) from command line.

if len(sys.argv) < 11:
    print("Usage: " + sys.argv[0] + " <num_conv_blocks> <num_filters_first_layer> <num_dense_layers> <dense_layer_size> <dropout_rate> <shift_range> <brightness_min> <brightness_max> <zoom_min> <zoom_max>")
    exit()

else:
    num_conv_blocks = int(sys.argv[1])
    num_filters_start = int(sys.argv[2])
    num_dense_layers = int(sys.argv[3])
    dense_layer_size = int(sys.argv[4])

    if float(sys.argv[5]) != 0:
        dropout_rate = float(sys.argv[5])
    else:
        dropout_rate = 0

    shift_range = float(sys.argv[6])
    brightness_range = [float(sys.argv[7]), float(sys.argv[8])]
    zoom_range = [float(sys.argv[9]), float(sys.argv[10])]

# Configure data generators.

train_generator = HeadPoseDataGenerator(train_df, train_array, batch_size, normalize=True, input_norm=[mean, std],
                                        tilt_norm=[t_mean, t_std], pan_norm=[p_mean, p_std], augment=True,
                                        shift_range=shift_range, zoom_range=zoom_range,
                                        brightness_range=brightness_range, img_rescale=1./255, out_rescale=1./90)

validation_generator = HeadPoseDataGenerator(validation_df, validation_array, batch_size, normalize=True, input_norm=[mean, std],
                                             tilt_norm=[t_mean, t_std], pan_norm=[p_mean, p_std], img_rescale=1./255,
                                             out_rescale=1./90)

STEP_SIZE_TRAIN = train_generator.__len__()
STEP_SIZE_VALID = validation_generator.__len__()

# Set new model name.

model_name = 'headpose' + str(int(time.time()))
model_path = model_dir + model_name + '.h5'
loss_csv = model_dir + model_name + '_loss.csv'

# Configure a callback for logging train progress in a .csv file.

csv_logger = CSVLogger(loss_csv)

# Get number of FLOPs.

run_meta = tf.RunMetadata()

with tf.Session(graph=tf.Graph()) as sess_2:
    K.set_session(sess_2)

    model = mpatacchiola_generic(in_size, num_conv_blocks, num_filters_start, num_dense_layers, dense_layer_size, dropout_rate, batch_size=1)

    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess_2.graph, run_meta=run_meta, cmd='op', options=opts).total_float_ops

# Restore session

K.set_session(sess)

# Configure estimator model from architecture parameters set before.

model = mpatacchiola_generic(in_size, num_conv_blocks, num_filters_start, num_dense_layers, dense_layer_size, dropout_rate)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Train the configured model on the train generator.

history = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=validation_generator,
                              validation_steps=STEP_SIZE_VALID, epochs=epochs, callbacks=[reduce_lr, stop, csv_logger], verbose=verbose)

# Get score for the dataset (tilt, pan and global error).

pred = model.predict((test_array / 255.0 - mean) / std)

mean_tilt_error = np.mean(np.abs(test_df['tilt'] - ((pred[:,0] * t_std + t_mean) * 90.0)))
mean_pan_error = np.mean(np.abs(test_df['pan'] - ((pred[:,1] * p_std + p_mean) * 90.0)))

score = (mean_pan_error + mean_tilt_error) / 2

# Save trained model.

model.save(model_path)

# Record configured architecture and data augmentation parameters, with the obtained score for that configuration.

t_epochs = len(history.history['loss'])

if os.path.exists(model_csv):
    with open(model_csv, "a") as file:
        file.write(model_name + '.h5,%d,%d,%d,%d,%d,%.2f,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d\n' %
                   (in_size, num_conv_blocks, num_filters_start, num_dense_layers, dense_layer_size, dropout_rate,
                    shift_range, brightness_range[0], brightness_range[1], zoom_range[0], zoom_range[1],
                    mean_tilt_error, mean_pan_error, score, t_epochs, model.count_params(), flops))
else:
    with open(model_csv, "w") as file:
        file.write('model,in_size,num_conv_blocks,num_filters_start,num_dense_layers,dense_layer_size,dropout_rate,'
                   'shift_range,brightness_min,brightness_max,zoom_min,zoom_max,tilt_error,pan_error,score,stop_epochs,num_weights,flops\n')

        file.write(model_name + '.h5,%d,%d,%d,%d,%d,%.2f,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d\n' %
                   (in_size, num_conv_blocks, num_filters_start, num_dense_layers, dense_layer_size, dropout_rate,
                    shift_range, brightness_range[0], brightness_range[1], zoom_range[0], zoom_range[1],
                    mean_tilt_error, mean_pan_error, score, t_epochs, model.count_params(), flops))