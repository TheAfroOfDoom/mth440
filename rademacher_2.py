# Jordan Williams
# MTH 440
# Calculating the Rademacher complexity for Sine and Gaussian Waves
# and relating them to the training time of a neural network to detect them.

# https://www.youtube.com/watch?v=gR9Q8pS03ZE

from __future__ import absolute_import, division, print_function, unicode_literals

# Helper libraries
import copy
import numpy as np
import math
import tqdm

train_n = 60000
total_n = train_n #+ test_n
# Image resolution
resolution = [128, 128]

DEBUG = 0

def constant(x):
    return(1)
f_const = np.vectorize(constant)

def linear(x):
    return(x)
f_lin = np.vectorize(linear)

#def sine_wave(x, SINE_AMPLITUDE, SINE_PERIOD, SINE_X_OFFSET, SINE_Y_OFFSET):
#    return((SINE_AMPLITUDE * np.sin(SINE_PERIOD * (x - SINE_X_OFFSET))) + SINE_Y_OFFSET)
def sine_wave(x):
    return(1 * np.exp(np.sin(2 * math.pi * x)))
f_sine = np.vectorize(sine_wave)

def gaussian_wave(x, GAUSSIAN_MEAN, GAUSSIAN_STD_DEVIATION, GAUSSIAN_Y_OFFSET):
    return((1/(GAUSSIAN_STD_DEVIATION * np.sqrt(2 * math.pi))) * np.exp(-0.5 * np.power((x - GAUSSIAN_MEAN) / GAUSSIAN_STD_DEVIATION, 2)))
f_gaus = np.vectorize(gaussian_wave)

def quadratic(x, QUADRATIC_AMPLITUDE, QUADRATIC_X_OFFSET, QUADRATIC_Y_OFFSET):
    return((QUADRATIC_AMPLITUDE * np.power(x - QUADRATIC_X_OFFSET, 2) + QUADRATIC_Y_OFFSET))
f_quad = np.vectorize(quadratic)

fx_s = ["f_const", "f_sine"]
fxs = [f_const, f_sine, f_lin]

# Calculate average of sig*y
def sigy(y, n_samples):
    # Randomize Rademacher variables
    sigma = np.random.choice([-1, 1], size = n_samples)
    if(DEBUG):
        print("\t\tsigma: ", sigma[0:5])
    # Return max of the average value for sigma * f(x)
    #return(np.amax(np.average(sigma * y)))
    y *= sigma

# Calculate Experimental Rademacher Complexity, respect to sigma distributions
def exp_rademacher(n_sigs, n_samples, base):
    # Randomize "n_samples" samples from [0, train_n)
    #x = np.random.choice(a = range(train_n), size = n_samples, replace = False) # Z -> data
    # [-1, 1)\
    x = (2 * np.random.random(size = n_samples)) - 1    # (2 * [0, 1]) - 1

    y = np.tile(x, (len(fxs), 1))                       # Generate an array of shape <len(fxs) x n_samples>,
                                                        # where each row corresponds to each function's outputs
                                                        # in the list of fxs

    # Apply each function to its corresponding row from fxs
    for i, function in enumerate(fxs):
        y[i] = function(y[i])

    # Calculate experimental rademacher complexities by
    # averaging many different Supremum variations
    Y = np.zeros(len(fxs))  # Y is a list of the average y*sigs for each function
    for i in range(len(Y)):
        if(DEBUG):
            print("\ty: ", y[i][0:5])
        #ysigs = y[i] # Take the average of n_sigs different sigma variations for y * sigma
        #kwargs = {'y': y[i], 'n_samples': n_samples}
        #np.apply_along_axis(sigy, 0, ysigs, **kwargs)
        sigy(y, n_samples)
        if(DEBUG):
            #print("\tysigs: ", ysigs[0:5], "\n\t= = =")
            print("\tysigs: ", y[i][0:5], "\n\t= = =")
        Y[i] = np.average(y[i])
    return(np.amax(Y))      # Return supremum of Y
Vexp_rademacher = np.vectorize(exp_rademacher)

# Calculate Rademacher Complexity, respect to samples
def rademacher(n_exprad, n_sigs, n_samples):
    # array of experimental Rademachers, n_exprad long
    ex_rads = np.zeros(n_exprad)
    ex_rads = Vexp_rademacher(n_sigs, n_samples, base = ex_rads)
    if(DEBUG):
        print("erads: ", ex_rads[0:5])
    return(np.average(ex_rads))

# Loops
n_exprad = 100000
n_sigs = 1000
n_samples = 1000

unique_functions = 3
# Generate training/testing labels randomly according to the proportion
train_labels = np.random.choice(unique_functions, size = train_n)

''''''
# Set bounds for wave parameters
SINE_AMPLITUDE = [0.1, 1.0]             # range[-0.1, 0.1] to range[-1, 1]
SINE_PERIOD = [math.pi, 2 * math.pi]    # 1 cycle to 2 cycles
SINE_X_OFFSET = [-1.0, 1.0]             # shifted 1 unit to the left or right
SINE_Y_OFFSET = [-0.5, 0.25]            # shifted 1 unit down or up

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
'''
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
            if(y > -1 and y < 1):

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
            if(y > -1 and y < 1):

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
            if(y > -1 and y < 1):

                # Calculate closest pixel to y value
                yi = int((resolution[1] / 2) * (y + 1))

                # Color in pixel
                image[yi][xi] = 1

    # Append image to data
    train_images[index] = copy.deepcopy(image)
'''
print("fxs: ", fx_s)

# Calculate Rademacher complexity by averaging a lot of
# experimental Rademacher complexities, which are based on our samples
r = rademacher(n_exprad, n_sigs, n_samples)
print(r)
