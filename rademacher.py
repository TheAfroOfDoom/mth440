# Jordan Williams
# MTH 440
# Calculating the Rademacher complexity for Sine and Gaussian Waves
# and relating them to the training time of a neural network to detect them.

from __future__ import absolute_import, division, print_function, unicode_literals

# Helper libraries
import numpy as np
import math

# Define wave functions

'''
# sine parameters
SINE_AMPLITUDE = 1
SINE_PERIOD = 1
SINE_Y_OFFSET = 0
SINE_X_OFFSET = 0
'''

# Default
# range: [-1, 1]
# period: 2*pi
def sine_wave(x, SINE_AMPLITUDE, SINE_PERIOD, SINE_X_OFFSET, SINE_Y_OFFSET):
    return((SINE_AMPLITUDE * np.sin(SINE_PERIOD * (x - SINE_X_OFFSET))) + SINE_Y_OFFSET)

vsin = np.vectorize(sine_wave)

# Loops
n0 = 100
n1 = 1000
n2 = 1000
samples = 1000

# Layer 0
# Calculate Rademacher complexity by averaging a lot of
# experimental Rademacher complexities, which are based on our samples
r = 0
for i in range(n0):

    print("%5.2f%%" % (100 * i / n0))
    # Layer 1
    # Randomize samples from [0, 1)
    x_samples = np.random.random_sample(samples)
    y_samples = (vsin(x_samples, 1, 2 * math.pi, 0, 0))
    #y_samples = [10 for k in range(samples)]

    #print("x_samples: ", x_samples[0:10])
    #print("y_samples: ", y_samples[0:10])

    # Calculate experimental rademacher complexities by
    # averaging many different sigma variations
    re = 0
    for j in range(n1):

        # Layer 2
        # Randomize Rademacher variables
        sigma = np.random.choice([-1, 1], size = samples)

        #print("Rademacher Variables: ", sigma[0:10])

        # Find the supremum of the expected (sigma * g(Z))
        supremum = -(2 ** 31) - 1
        for m in range(n2):

            # Layer 3
            # Find the expected value for sigma * samples
            samples_expected = 0
            for l in range(samples):
                samples_expected += sigma[l] * y_samples[l]
            samples_expected /= samples

            if(samples_expected > supremum):
                supremum = samples_expected

        re += supremum
        #print("Supremum: ", supremum)

    re /= n1
    r += re
    print("Empirical Rademacher Complexity: ", re)

r /= n0

# Print Rademacher Complexity for sin(2pi*x), 0 <= x < 1
print("Rademacher Complexity: ", r)