import numpy as np
from numpy.typing import NDArray

from simplex_assimilate.dirichlet import MixedDirichlet
from simplex_assimilate.cdf import uniformize, deuniformize
from simplex_assimilate.quantize import quantize, dequantize, ONE


def transport_pipeline(X: NDArray[np.float64], x_0: NDArray[np.float64], threshold = 1e-6) -> NDArray[np.float64]:
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
    X = np.where(X < threshold, 0, X)
    X /= X.sum(axis=1, keepdims=True)
    # estimate prior
    prior = MixedDirichlet.est_from_samples(X)
    # convert to fixed point
    X = quantize(X)
    x_0 = (x_0 * ONE).astype(np.uint32)
    # map to uniforms and back
    U = uniformize(X, prior)
    X = deuniformize(U, prior, x_0)
    # convert back to floating point
    X = dequantize(X)
    return X

