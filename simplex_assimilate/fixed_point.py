# check the arguments for the assimilation
#

import numpy as np

SIG_BITS = 31
ONE = np.uint32(1<<SIG_BITS)
DELTA = np.uint32(1)

def check_samples(samples):
    assert samples.ndim == 2, "Samples must be a 2D array"
    assert samples.dtype == np.uint32, "Samples must be provided as uint32. (1<<31) represents 1.0"
    assert np.all(samples.sum(axis=1) == ONE), "Samples must sum to 1"
