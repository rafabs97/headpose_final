#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is a testbench for the script 'train_base.py', using it in order to test multiple combinations of data
augmentation parameters combined with a fixed architecture.
"""

# Path to the 'train_base.py' script.

script = '/gdrive/My\ Drive/headpose_final/train_base.py'

# Id. of the machine in which we are running the code.

machine_id = 0

# Total number of machines.

num_machines = 1

# Architecture parameters.

num_conv_blocks = 6
num_filters_start = 32
num_dense_layers = 1
dense_layer_size = 512
dropout_rate = 0

# Initialize iteration counter.

iter = 0

# Set starting iteration.

start_from = 0

# Iterate over multiple values for each parameter used to configure data augmentation.
for shift_range in 0.1, 0.2, 0.3:
    for brightness_range in 0, 0.25, 0.5:
        for zoom_range in 0, 0.25, 0.5:

            # Execute only iterations after starting iteration, and only if they are assigned to this machine.
            if iter >= start_from and iter % num_machines == machine_id:

                # Print iteration number.
                print("Iteration:", iter)

                # Configure the command to run from the parameters set before.
                command = script + " " + str(num_conv_blocks) + " " + \
                str(num_filters_start) + " " + str(num_dense_layers) + " " + str(dense_layer_size) + " " + \
                str(dropout_rate) + " " + str(shift_range) + " " + \
                str(1 - brightness_range) + " " + str(1 + brightness_range) + " " + \
                str(1 - zoom_range) + " " + str(1 + zoom_range)

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
                           shift_range, 1 - brightness_range, 1 + brightness_range, 1 - zoom_range,
                           1 + zoom_range))
                    exit()

            # Increase iteration counter.
            iter = iter + 1