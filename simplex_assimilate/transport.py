import numpy as np
from numpy.typing import NDArray

from simplex_assimilate.dirichlet import MixedDirichlet
from simplex_assimilate.cdf import uniformize, deuniformize

from simplex_assimilate.fixed_point import check_samples

def transport_pipeline(X: NDArray[np.uint32], x_0: NDArray[np.uint32]) -> NDArray[np.uint32]:
    assert x0.dtype == np.uint32, "x_0 must be provided as uint32. (1<<31) represents 1.0"
    check_samples(X)

    """
    Transport X- to X+ based on the observation x_0
    - Set X- to zero where it is less than threshold
    - Estimate the prior using X-
    - quantize X- and x_0 to fixed point representations
    - compute the cdf of X- under the prior to get U
    - compute the inverse cdf of U under the posterior to get X+
    - dequantize X+ to floating point representation
    """
    # threshold
    # TODO: thresholding and quantizing should be done by the user.
    # The sample even should be quantized by the user
    # X = np.where(X < threshold, 0, X)
    # X /= X.sum(axis=1, keepdims=True)
    # estimate prior
    prior = MixedDirichlet.est_from_samples(X)
    # convert to fixed point
    # X = quantize(X)
    # x_0 = (x_0 * ONE).astype(np.uint32)  #
    # map to uniforms and back
    U = uniformize(X, prior)
    X = deuniformize(U, prior, x_0)
    # convert back to floating point
    # X = dequantize(X)
    return X

