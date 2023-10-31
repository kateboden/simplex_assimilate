import warnings
from numpy.typing import NDArray
import numpy as np

from simplex_assimilate.fixed_point import ONE, DELTA


def quantize(float_samples: NDArray[np.float32]) -> NDArray[np.uint32]:
    # check inputs lie on the simplex to within tolerance DELTA
    assert float_samples.ndim == 2, "Samples must be a 2D array"
    assert np.all(float_samples >= 0), "Samples must be non-negative"
    assert np.all(1 - 1e-6 < float_samples.sum(axis=1)) and np.all(float_samples.sum(axis=1) < 1 + 1e-6), f"Samples must sum to 1 to within 1e-6"
    float_samples = float_samples.astype(np.float64)
    float_samples /= float_samples.sum(axis=1, keepdims=True)  # normalize
    # take cumulative sum, round to nearest quantized value, and take differences
    cumsum = np.cumsum(float_samples, axis=1)
    cumsum = np.insert(cumsum, 0, 0, axis=1)
    cumsum = np.round(cumsum * ONE).astype(np.uint32)  # multiply by ONE to convert from float to uint32
    samples = np.diff(cumsum, axis=1)
    if not np.all((samples > 0) == (float_samples > 0)):
        warnings.warn(f"Truncation performed in quantization. Inputs should be thresholded before quantization."
                      f"Recommended threshold is at least Δ={DELTA / ONE}.")
    assert (samples.sum(axis=1) == ONE).all(), "Samples do not sum to 1 after quantization"
    return samples

def dequantize(samples: NDArray[np.uint32]) -> NDArray[np.float32]:
    return samples.astype(np.float32) / ONE