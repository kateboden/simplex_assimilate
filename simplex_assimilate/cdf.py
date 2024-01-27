import logging
import warnings
import numpy as np
from scipy import stats
from numpy.typing import NDArray

from simplex_assimilate import _dirichlet
from simplex_assimilate.fixed_point import ONE, DELTA, SIG_BITS

def log_likelihood(alpha: NDArray[np.float64], pre_x: NDArray[np.uint32]) -> np.float64:
    # check inputs
    assert pre_x.dtype == np.uint32, 'pre_x must use uint32 representation of components'
    assert pre_x.ndim == 1, 'pre_x must be a 1D array'
    assert alpha.ndim == 1, 'alpha must be a 1D array'
    assert len(pre_x) <= len(alpha), 'pre_x must have fewer components than alpha'
    j = len(pre_x)
    agrees = np.all((alpha[:j] > 0) == (pre_x > 0)) and (alpha[j:].sum() > 0) == (
                pre_x.sum() < ONE)  # zeros and non-zeros agree
    alpha = np.append(alpha[:j], alpha[j:].sum())  # collapse x_(>=j) into a single component
    out = stats.dirichlet.logpdf(pre_x[pre_x > 0] / ONE, alpha[alpha > 0]) if agrees else -np.inf  # return the pdf if pre_x fits alpha, else return 0
    # check outputs
    return np.float64(out)


def vectorized_log_likelihood(alpha: NDArray[np.float64], pre_samples: NDArray[np.uint32]) -> NDArray[np.float64]:
    # check inputs
    N, j = pre_samples.shape
    K, J = alpha.shape
    assert j < J, 'pre_samples must have fewer components than alpha'
    # calculate likelihood for each sample against each class
    out = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            out[i, k] = log_likelihood(alpha[k], pre_samples[i])
    # check outputs
    # assert out.any(axis=1).all(), 'every sample must have at least one compatible class'
    # assert out.shape == (N, K)
    return out


def cdf(x_j, prior: _dirichlet.MixedDirichlet, pre_samples: NDArray[np.uint32]) -> NDArray[np.float64]:
    #   There are three kinds of classes:
    #     lower_classes: x_j = 0    (delta distribution)
    #     middle_classes: 0 < x_j < (1 - Σx_(<j))   (beta distribution)
    #     upper_classes: x_j = (1 - Σx_(<j))   (delta distribution)

    # check inputs
    assert pre_samples.dtype == np.uint32, 'pre_samples and x_j must use uint32 representation of components'
    assert pre_samples.ndim == 2, 'pre_samples must be a 2D array'
    N, j = pre_samples.shape
    assert x_j.shape == (N,)
    # CALCULATE THE POSTERIOR MIXTURE WEIGHTS
    prior_pi = prior.mixture_weights
    log_likelihood = vectorized_log_likelihood(prior.full_alpha, pre_samples)
    log_posterior_pi = np.log(prior_pi) + log_likelihood  # the log posterior mixture weights
    bad_samples = (log_posterior_pi == -np.inf).all(axis=1)  # samples with no compatible classes
    if bad_samples.any():
        warnings.warn(f'every sample should have at least one compatible class, but samples {np.where(bad_samples)} do not. pre_Samples: {pre_samples[bad_samples]}')
        # assert (ONE == pre_samples[bad_samples].sum(axis=1)).all(), 'samples with no compatible classes should have full mass'
    # get the posterior weights using numpys log softmax
    log_posterior_pi[bad_samples] = 0  # uniform probability of each class if each class had log_prob -inf
    log_posterior_pi -= log_posterior_pi.max(axis=1, keepdims=True)  # subtract the max to avoid numerical issues
    posterior_pi = np.exp(log_posterior_pi)  # convert back to linear space
    # TODO: if the sample is bad (i.e. in a novel class) and there is mass remaining, we shouldn't
    # allow it to be considered part of a class which doesn't have mass remaining
    finished_classes = prior.full_alpha[:, j:].sum(axis=1) == 0  # classes with no mass remaining
    # posterior_pi[bad_samples][:, finished_classes] = 0  # samples with no compatible classes should have zero probability of being in a finished class
    unfinished_bad_samples = bad_samples & (pre_samples.sum(axis=1) < ONE)
    posterior_pi[unfinished_bad_samples[:,None] & finished_classes] = 0  # samples with no compatible classes should have zero probability of being in a finished class if they have mass remaining

    posterior_pi /= posterior_pi.sum(axis=1, keepdims=True)  # normalize
    assert not np.isnan(posterior_pi).any(), 'posterior_pi should not contain NaNs'
    # CREATE MASKS FOR LOWER, MIDDLE, AND UPPER
    # classes with no mass in component j
    lower_classes = (prior.full_alpha[:, j] == 0)
    # classes with all the remaining mass in component j
    upper_classes = (prior.full_alpha[:, j] > 0) & (~ prior.full_alpha[:, j + 1:].any(axis=1))
    middle_classes = ~ (lower_classes | upper_classes)
    # every compatible class must belong to exactly one of the three categories
    assert np.all(lower_classes + upper_classes + middle_classes == 1)
    # CALCULATE THE CDFs of the MIXTURE MIDDLE CLASSES
    # x_j/(1-Σx_(<j)) | x_(<j) ~ Beta(alpha_j, sum(alpha_(j+1:)))
    upper = ONE - pre_samples.sum(axis=1)
    mixture_cdfs = np.zeros((N, prior.K), dtype=np.float64)
    mixture_cdfs[:, lower_classes] = 1.
    mixture_cdfs[:, upper_classes] = (x_j >= upper)[:, None]
    if middle_classes.any():
        middle_alphas = prior.full_alpha[middle_classes]
        betas = np.column_stack((middle_alphas[:, j], middle_alphas[:, j + 1:].sum(axis=1)))
        # BUILD THE CDF from the LOWER, MIDDLE, and UPPER CLASSES
        # assert not posterior_pi[upper==0][:, middle_classes].any(), 'When sample\'s upper==0, middle_classes should be impossible'
        frac = np.ones_like(x_j, dtype=np.float64)  # fraction of the remaining area covered by x_j
        frac[upper > 0] = x_j[upper > 0] / upper[upper > 0]  # we need to be safe in case upper=0
        mixture_cdfs[:, middle_classes] = np.column_stack(tuple(stats.beta(*beta).cdf(frac) for beta in betas))
    out = (posterior_pi * mixture_cdfs).sum(axis=1)
    out[upper==0] = 1.  # if upper=0, the cdf must be one.
    # this special case occurs when we are pushed to a novel class and have used up the mass.
    # check output
    assert (out < 1 + 1e-10).all(), 'cdf must be less than 1'
    out = np.minimum(out,
                     1)  # clip to 1 (numerical error in posterior_weights can cause it to be slightly greater than 1
    assert (out <= 1).all(), 'cdf must be less than 1'
    assert (0 <= out).all(), 'cdf must be greater than 0'
    assert out.shape == x_j.shape
    assert out.dtype == np.float64
    return out


def inv_cdf(uniforms: NDArray[np.float64], prior: _dirichlet.MixedDirichlet,
            pre_samples: NDArray[np.uint32]) -> NDArray[np.uint32]:
    # check inputs
    assert uniforms.ndim == 1, 'uniforms must be a 1D array'
    assert len(uniforms) == len(pre_samples), 'uniforms and pre_samples must have the same length'
    X = np.zeros_like(uniforms, dtype=np.uint32)
    # the uniform can map back to a delta on either end of the interval or to a value in the middle
    # we need to check all three cases
    lower, upper = np.zeros_like(uniforms).astype(np.uint32), ONE - pre_samples.sum(axis=1)  # bounds
    lower_mask = uniforms <= cdf(lower, prior, pre_samples)  # the mask of samples that map to the lower bound
    upper_mask = uniforms >= cdf(upper - DELTA, prior, pre_samples)  # the samples that map to the upper bound
    # middle_mask = np.logical_not(np.logical_or(lower_mask, upper_mask))  # the samples that map to the middle
    middle_mask = ~ (lower_mask | upper_mask)
    X[lower_mask] = lower[lower_mask]  # map the samples to the lower bound
    X[upper_mask] = upper[upper_mask]  # map the samples to the upper bound
    # map the samples to the middle. Invert the cdf by binary search
    X[middle_mask] = vector_binary_search(lambda x: cdf(x, prior, pre_samples[middle_mask]), uniforms[middle_mask])
    # check output
    assert (X >= 0).all(), 'samples must be non-negative'
    assert (X <= ONE).all(), 'samples must be less than 1'
    assert X.dtype == np.uint32
    assert X.shape == uniforms.shape
    return X


def uniformize(samples: NDArray[np.uint32], prior: _dirichlet.MixedDirichlet) -> NDArray[np.float64]:
    # check inputs
    assert samples.dtype == np.uint32, 'samples must use uint32 representation of components'
    assert samples.ndim == 2, 'samples must be a 2D array'
    assert samples.shape[1] == prior.full_alpha.shape[1], 'samples must have the same number of components as prior'
    assert (samples.sum(axis=1) == ONE).all(), 'samples must sum to 1'
    # take the conditional cdf of the samples one component at a time
    U = np.zeros_like(samples, dtype=np.float64)
    I, J = samples.shape
    for j in range(J):
        pre_samples = samples[:, :j]
        x_j = samples[:, j]
        lower, upper = 0, ONE - pre_samples.sum(axis=1)  # legal bounds for x_j
        low_samples = x_j == 0
        upper_samples = (x_j == upper) & (~ low_samples)
        middle_samples = ~ (low_samples | upper_samples)
        assert (low_samples + upper_samples + middle_samples == 1).all(), 'samples must be in exactly one category'
        # samples on the boundary come from a delta distribution so the cdf is
        # discontinuous and we need to sample from a uniform distribution
        U[low_samples, j] = stats.uniform(
            0,
            cdf(np.zeros_like(x_j[low_samples]), prior, pre_samples[low_samples])
        ).rvs()
        cdf_before_upper_delta = cdf(upper[upper_samples] - DELTA, prior, pre_samples[upper_samples])
        U[upper_samples, j] = stats.uniform(
            cdf_before_upper_delta,
            1 - cdf_before_upper_delta
        ).rvs()
        U[middle_samples, j] = cdf(x_j[middle_samples], prior, pre_samples[middle_samples])
        # round to the interval [0, 1]
        u_j = U[:, j]
        assert (0 - 1e-6 < u_j).all() and (u_j < 1 + 1e-6).all(), 'cdf must be in the interval [0, 1]'
        U[:, j] = np.maximum(np.minimum(U[:, j], 1), 0)

    # check output
    assert np.all((0 <= U) & (U <= 1))
    return U


def deuniformize(U: NDArray[np.float64], prior: _dirichlet.MixedDirichlet, x_0 = None) -> NDArray[np.uint32]:
    # check inputs
    assert U.shape[1] == prior.full_alpha.shape[1], 'uniform samples must have the same number of components as prior'
    assert (0 <= U).all() and (U <= 1).all(), 'uniform samples must be in the interval (0, 1)'
    # build the samples
    X = np.ones_like(U, dtype=np.uint32)
    if x_0 is not None:
        X[:, 0] = x_0
    _, J = U.shape
    j_start = 1 if x_0 is not None else 0
    for j in range(j_start, J):
        pre_samples = X[:, :j]
        u_j = U[:, j]
        X[:, j] = inv_cdf(u_j, prior, pre_samples)
    # check output
    assert (X >= 0).all() and (X <= ONE).all(), 'samples must be in the interval [0, 1]'
    assert X.dtype == np.uint32, 'samples must use uint32 representation of components'
    assert X.shape == U.shape, 'samples must have the same shape as U'
    return X


def vector_binary_search(f, Y):
    ''' Given a scalar function f and a vector Y, return the vector X such that f(X) = Y. Where
    0 <= X <= 1 and X is represented with a 32-bit unsigned integer. '''
    X = np.zeros_like(Y, dtype=np.uint32)
    assert np.all(f(X) <= Y), "f(0) must be less than or equal to Y"
    for sig_bit in 2 ** np.arange(SIG_BITS - 1, -1, -1):
        X = np.where(f(X) <= Y, X + sig_bit, X - sig_bit)
    assert np.all(0 < X) and np.all(X < ONE)
    # check the values on either side of X
    three = np.column_stack((X - DELTA, X, X + DELTA))  # X and the two values on either side
    X = three[
        np.arange(len(X)), np.argmin(np.abs(np.column_stack(tuple(f(COL) for COL in three.T)) - Y[:, None]), axis=1)]
    if np.any(X == 0):
        warnings.warn("Binary search selected a value on the boundary 0. "
                      "This could cause a sample to shift to an incompatible class.")
    if np.any(X == ONE):
        warnings.warn("Binary search selected a value on the boundary ONE."
                      "This could cause a sample to shift to an incompatible class.")
    return X
