import os
import shutil
import random
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence

###################
## ROBOTICS & RL ##
###################

def find_limit(simulation, axis, direction, step_size=0.1, max_steps=1000, silence=False):
    """
    Move the pipette along a specified axis in a given direction until it can no longer move.

    Args:
    - axis: 0 for x, 1 for y, 2 for z
    - direction: 1 for positive, -1 for negative
    - step_size: Incremental velocity to apply in the given direction
    - max_steps: Maximum number of steps to attempt

    Returns:
    - The coordinate value where movement stops
    """
    velocity = [0, 0, 0]
    previous_position = [0, 0, 0]
    boundary_position = None

    simulation.set_start_position(0, 0, 0.2)
    
    for _ in range(max_steps):
        velocity[axis] = step_size * direction
        actions = [[velocity[0], velocity[1], velocity[2], 0]]
        state = simulation.run(actions)
        
        # Extract current position from the state
        first_robot_key = list(state.keys())[0]
        current_position = state[first_robot_key]['pipette_position']
        
        # Check if the pipette is still moving along the axis
        if current_position[axis] == previous_position[axis]:
            boundary_position = current_position[axis]
            simulation.reset(num_agents=1)
            simulation.set_start_position(0, 0, 0.2)
            if not silence:
                print(boundary_position)
            return boundary_position
        
        # Update the previous position
        previous_position = current_position

def calculate_working_envelope(sim):
    """
    Given a simulation object, this function finds the limits along each axis,
    and calculates the coordinates of the working envelope's 8 corners.
    
    Args:
    - sim (Simulation): The simulation instance to be used for limit calculations.
    
    Returns:
    - list: A list of the coordinates of the 8 corners of the working envelope.
    """
    # Find the limits
    x_min = find_limit(sim, axis=0, direction=-1)
    x_max = find_limit(sim, axis=0, direction=1)
    y_min = find_limit(sim, axis=1, direction=-1)
    y_max = find_limit(sim, axis=1, direction=1)
    z_min = find_limit(sim, axis=2, direction=-1)
    z_max = find_limit(sim, axis=2, direction=1)
    
    # Define the 8 corners of the cube
    corners = [
        # Bottom square
        [x_min, y_min, z_min], # Bottom-left-back
        [x_min, y_max, z_min], # Bottom-right-back
        [x_max, y_min, z_min], # Bottom-left-front
        [x_max, y_max, z_min], # Bottom-right-front
        # Top square
        [x_min, y_min, z_max], # Top-left-back
        [x_min, y_max, z_max], # Top-right-back
        [x_max, y_min, z_max], # Top-left-front
        [x_max, y_max, z_max]  # Top-right-front
    ]
    return corners