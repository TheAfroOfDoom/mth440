# Jordan Williams
# MTH 440
# Training a network to differentiate between images of sine and gaussian waves
# Python 3.6

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import keras
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

# Helper libraries
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm

DEBUG = 0

np.random.seed()

# Tensorflow GPU stuff
config = tf.config.experimental
physical_devices = config.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config.set_memory_growth(physical_devices[0], True)

# NOTE(jordan): Change values here
train_n = 60000
test_n = 6000
total_n = train_n + test_n
# Percent sine
# Image resolution
resolution = [128, 128]

# Domain: [-1, 1], Range: [-1, 1]

# Define wave functions
# https://www.desmos.com/calculator/4jqgbsov2b

# Default
# range: [-1, 1]
# period: 2*pi
def sine_wave(x, SINE_AMPLITUDE, SINE_PERIOD, SINE_X_OFFSET, SINE_Y_OFFSET):
    return((SINE_AMPLITUDE * np.sin(SINE_PERIOD * (x - SINE_X_OFFSET))) + SINE_Y_OFFSET)

# Default:
# range: [0, 1]
def gaussian_wave(x, GAUSSIAN_MEAN, GAUSSIAN_STD_DEVIATION, GAUSSIAN_Y_OFFSET):
    return((1/(GAUSSIAN_STD_DEVIATION * np.sqrt(2 * math.pi))) * np.exp(-0.5 * np.power((x - GAUSSIAN_MEAN) / GAUSSIAN_STD_DEVIATION, 2)))

# Quadratic
def quadratic(x, QUADRATIC_AMPLITUDE, QUADRATIC_X_OFFSET, QUADRATIC_Y_OFFSET):
    return((QUADRATIC_AMPLITUDE * np.power(x - QUADRATIC_X_OFFSET, 2) + QUADRATIC_Y_OFFSET))

unique_functions = 3
p = [1 / unique_functions for i in range(unique_functions)] # Uniform distribution
#p = [0.5, 0.25, 0.25]  # Custom distribution

# Generate training/testing labels randomly according to the proportion
train_labels = np.random.choice(unique_functions, size = train_n)
test_labels = np.random.choice(unique_functions, size = test_n)

# Set bounds for wave parameters
SINE_AMPLITUDE = [0.1, 1.0]             # range[-0.1, 0.1] to range[-1, 1]
SINE_PERIOD = [math.pi, 2 * math.pi]    # 1 cycle to 2 cycles
SINE_X_OFFSET = [-1.0, 1.0]             # shifted 1 unit to the left or right
SINE_Y_OFFSET = [-0.5, 0.25]             # shifted 1 unit down or up

GAUSSIAN_MEAN = [-1.0, 1.0]             # same as SINE_X_OFFSET
GAUSSIAN_STD_DEVIATION = [0.25, 1]
GAUSSIAN_Y_OFFSET = [-1.5, 0.0]

QUADRATIC_AMPLITUDE = [-0.5, 0.5]
QUADRATIC_X_OFFSET = [-1, 1]
QUADRATIC_Y_OFFSET = [-0.5, 0.5]

# pixel x,y
image_pixels = np.zeros(resolution)  # image[y][x] = c

# # Randomize parameters
# Sine
samp = np.random.uniform(SINE_AMPLITUDE[0], SINE_AMPLITUDE[1], size = total_n)
sper = np.random.uniform(SINE_PERIOD[0], SINE_PERIOD[1], size = total_n)
sxoff = np.random.uniform(SINE_X_OFFSET[0], SINE_X_OFFSET[1], size = total_n)
syoff = np.random.uniform(SINE_Y_OFFSET[0], SINE_Y_OFFSET[1], size = total_n)

# Gaussian
gme = np.random.uniform(GAUSSIAN_MEAN[0], GAUSSIAN_MEAN[1], size = total_n)
gstd = np.random.uniform(GAUSSIAN_STD_DEVIATION[0], GAUSSIAN_STD_DEVIATION[1], size = total_n)
gyoff = np.random.uniform(GAUSSIAN_Y_OFFSET[0], GAUSSIAN_Y_OFFSET[1], size = total_n)

# Quadratic
qamp = np.random.uniform(QUADRATIC_AMPLITUDE[0], QUADRATIC_AMPLITUDE[1], size = total_n)
qxoff = np.random.uniform(QUADRATIC_X_OFFSET[0], QUADRATIC_X_OFFSET[1], size = total_n)
qyoff = np.random.uniform(QUADRATIC_Y_OFFSET[0], QUADRATIC_Y_OFFSET[1], size = total_n)

# Generate training data
print('Generating training data... (%dit)' % len(train_labels))
train_images = np.empty((len(train_labels), image_pixels.shape[0], image_pixels.shape[1]))
for index, label in tqdm.tqdm(enumerate(train_labels)):
    # Initialize image
    image = copy.deepcopy(image_pixels)

    # Sine
    if(label == 0):
        
        # Evaluate y = sin(x)
        for xi in range(resolution[0]):     # [0, 1, ... 128)
            x = -1 + (xi / resolution[0])   # [-1, -0.9921875, ... 1)
            y = sine_wave(x, samp[index], sper[index], sxoff[index], syoff[index])

            # If function is in range of [-1, 1)
            if(y >= -1 and y <= 1):

                # Calculate closest pixel to y value
                yi = int((resolution[1] / 2) * (y + 1))

                # Color in pixel
                image[yi][xi] = 1
    # Gaussian
    elif(label == 1):
        # Evaluate y = normal(x)
        for xi in range(resolution[0]):     # [0, 1, ... 128)
            x = -1 + (xi / resolution[0])   # [-1, -0.9921875, ... 1)
            y = gaussian_wave(x, gme[index], gstd[index], gyoff[index])

            # If function is in range of [-1, 1)
            if(y >= -1 and y <= 1):

                # Calculate closest pixel to y value
                yi = int((resolution[1] / 2) * (y + 1))

                # Color in pixel
                image[yi][xi] = 1

    # Quadratic
    else:
        # Evaluate y = x^2
        for xi in range(resolution[0]):     # [0, 1, ... 128)
            x = -1 + (xi / resolution[0])   # [-1, -0.9921875, ... 1)
            y = quadratic(x, qamp[index], qxoff[index], qyoff[index])

            # If function is in range of [-1, 1)
            if(y >= -1 and y <= 1):

                # Calculate closest pixel to y value
                yi = int((resolution[1] / 2) * (y + 1))

                # Color in pixel
                image[yi][xi] = 1

    # Append image to data
    train_images[index] = copy.deepcopy(image)

# Generate test data
test_images = np.empty((len(test_labels), image_pixels.shape[0], image_pixels.shape[1]))
print('Generating test data... (%d)' % len(test_labels))
for index, label in tqdm.tqdm(iterable = enumerate(test_labels)):

    # Initialize image
    image = copy.deepcopy(image_pixels)

    # Sine
    if(label == 0):

        # Evaluate y = sin(x)
        for xi in range(resolution[0]):     # [0, 1, ... 128)
            x = -1 + (xi / resolution[0])   # [-1, -0.9921875, ... 1)
            y = sine_wave(x, samp[index + train_n], sper[index + train_n],
                            sxoff[index + train_n], syoff[index + train_n])

            # If function is in range of [-1, 1)
            if(y >= -1 and y <= 1):

                # Calculate closest pixel to y value
                yi = int((resolution[1] / 2) * (y + 1))

                # Color in pixel
                image[yi][xi] = 1
    # Gaussian
    elif(label == 1):
        # Evaluate y = normal(x)
        for xi in range(resolution[0]):     # [0, 1, ... 128)
            x = -1 + (xi / resolution[0])   # [-1, -0.9921875, ... 1)
            y = gaussian_wave(x, gme[index + train_n], gstd[index + train_n],
                                gyoff[index + train_n])

            # If function is in range of [-1, 1)
            if(y >= -1 and y <= 1):

                # Calculate closest pixel to y value
                yi = int((resolution[1] / 2) * (y + 1))

                # Color in pixel
                image[yi][xi] = 1

    # Quadratic
    else:
        # Evaluate y = x^2
        for xi in range(resolution[0]):     # [0, 1, ... 128)
            x = -1 + (xi / resolution[0])   # [-1, -0.9921875, ... 1)
            y = quadratic(x, qamp[index + train_n], qxoff[index + train_n],
                            qyoff[index + train_n])

            # If function is in range of [-1, 1)
            if(y >= -1 and y <= 1):

                # Calculate closest pixel to y value
                yi = int((resolution[1] / 2) * (y + 1))

                # Color in pixel
                image[yi][xi] = 1

    # Append (image, label) to data
    test_images[index] = copy.deepcopy(image)

#'''
# Show images
if(not DEBUG):
    print("Displaying first 4 training images:")
    plt.figure(figsize=(7, 7))
    for i in range(4):
        plt.subplot(2,2,i+1)
        if(train_labels[i] == 0):
            plt.title("Sine")
        elif(train_labels[i] == 1):
            plt.title("Gaussian")
        else:
            plt.title("Quadratic")
        plt.grid(False)
        plt.imshow(train_images[i])
    plt.show()
#'''

# Check bad images for their parameters
if(DEBUG):
    i = 0
    j = 0
    while(True):
        print("Displaying next 4 training images:")
        plt.figure(figsize=(7, 7))
        for i in range(4):
            plt.subplot(2,2,i+1)
            if(train_labels[j] == 0):
                plt.title("Sine")
            elif(train_labels[j] == 1):
                plt.title("Gaussian")
            else:
                plt.title("Quadratic")
            plt.grid(False)
            plt.imshow(train_images[j])
            j += 1
        plt.show()

        in_ = input("Check parameters for: ")

        # Bad input = leave (not an int, or out of array range)
        if(not(type(in_) is int) or
            in_ >= train_labels or in_ < 0):
            break

        # Sine
        if(train_labels[in_] == 0):
            print("SINE\nsamp = %.2f\nsper = %.2f\nsxoff = %.2f\nsyoff = %.2f" %
                   (samp[in_], sper[in_], sxoff[in_], syoff[in_]))
        # Gaussian
        if(train_labels[in_] == 1):
            print("GAUSSIAN\ngme = %.2f\ngstd = %.2f\ngyoff = %.2f" %
                    (gme[in_], gstd[in_], gyoff[in_]))
        # Quadratic
        else:
            print("QUADRATIC\nqamp = %.2f\nqxoff = %.2f\nqyoff = %.2f" %
                    (qamp[in_], qxoff[in_], qyoff[in_]))

# Create the structure of the model
# Layers
model = keras.Sequential([
    # Turn image from 2D array into 1D row
    keras.layers.Flatten(input_shape = resolution),
    # 128-node dense layer - what does this mean???
    keras.layers.Dense(256, activation = tf.nn.relu),
    # Output layer: probability/confidence array for each class
    keras.layers.Dense(unique_functions, activation = tf.nn.softmax)
])

# Compilation parameters
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Fit model to the data
model.fit(train_images, train_labels, epochs = 5)

# Evaluate our model's accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)
