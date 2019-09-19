#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is a testbench for the script 'train_base.py', using it in order to test multiple architecture configurations
combined with a fixed set of data augmentation parameters.
"""

# Path to the 'train_base.py' script.

script = '/gdrive/My\ Drive/headpose_final/train_base.py'

# Id. of the machine in which we are running the code.

machine_id = 0

# Total number of machines.

num_machines = 1

# Dropout rate is set here to 0 initially; if overfitting changing this parameter may help reducing it.

dropout_rate = 0.0

# Data augmentation parameters.

shift_range = 0.0
brightness_min = 1.0
brightness_max = 1.0
zoom_min = 1.0
zoom_max = 1.0

# Initialize iteration counter.

iter = 0

# Set starting iteration.

start_from = 0

# Iterate over multiple values for each parameter used to configure the pose estimator model architecture.
for num_conv_blocks in range(1, 7):
    for num_filters_start in 32, 64, 128, 256:
        for num_dense_layers in range(1, 4):
            for dense_layer_size in 64, 128, 256, 512:

                #Execute only iterations after starting iteration, and only if they are assigned to this machine.
                if iter >= start_from and iter % num_machines == machine_id:

                    # Print iteration number.
                    print("Iteration:", iter)

                    # Configure the command to run from the parameters set before.
                    command = script + " " + str(num_conv_blocks) + " " + \
                    str(num_filters_start) + " " + str(num_dense_layers) + " " + str(dense_layer_size) + " " + \
                    str(dropout_rate) + " " + str(shift_range) + " " + \
                    str(brightness_min) + " " + str(brightness_max) + " " + \
                    str(zoom_min) + " " + str(zoom_max)

                    '''
                    Try executing the command configured before; if there is an error, print the configuration that 
                    may have caused it, and exit.
                    '''
                    try:
                        !python3 $command
                    except:
                        print("Error: %d %d %d %d %.2f %.1f %.2f %.2f %.2f %.2f\n" %
                              (num_conv_blocks, num_filters_start,
                               num_dense_layers, dense_layer_size, dropout_rate,
                               shift_range, brightness_min, brightness_max, zoom_min,
                               zoom_max))
                        exit()

                # Increase iteration counter.
                iter = iter + 1