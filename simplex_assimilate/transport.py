import numpy as np
from numpy.typing import NDArray

from simplex_assimilate._dirichlet import MixedDirichlet
from simplex_assimilate.cdf import uniformize, deuniformize
from simplex_assimilate.fixed_point import check_samples

def transport(X: NDArray[np.uint32], x_0: NDArray[np.uint32]) -> NDArray[np.uint32]:
    assert x_0.dtype == np.uint32, "x_0 must be provided as uint32. (1<<31) represents 1.0"
    check_samples(X)
    """
    Transport X- to X+ based on the observation x_0
    - Estimate the prior using X-
    - compute the cdf of X- under the prior to get U
    - compute the inverse cdf of U under the posterior to get X+
    """
    prior = MixedDirichlet.est_from_samples(X)
    U = uniformize(X, prior)
    #  TODO: add support for beta likelihood as well as point likelihood
    X = deuniformize(U, prior, x_0)
    return X

