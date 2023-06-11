import logging
import warnings
import numpy as np
from scipy import stats
from numpy.typing import NDArray
from simplex_assimilate import dirichlet

SIG_BITS = 31
ONE = np.uint32(2 ** SIG_BITS)
DELTA = np.uint32(1)


def quantize(float_samples: NDArray[np.float64]) -> NDArray[np.uint32]:
    # check inputs lie on the simplex to within tolerance DELTA
    assert float_samples.ndim == 2, "Samples must be a 2D array"
    assert np.all(float_samples >= 0), "Samples must be non-negative"
    assert np.all(1 - DELTA / ONE < float_samples.sum(axis=1)) and np.all(
        float_samples.sum(axis=1) < 1 + DELTA / ONE), f"Samples must sum to 1, to tolerance {DELTA / ONE}"
    # take cumulative sum, round to nearest quantized value, and take differences
    cumsum = np.cumsum(float_samples, axis=1)
    cumsum = np.insert(cumsum, 0, 0, axis=1)
    cumsum = np.round(cumsum * ONE).astype(np.uint32)  # multiply by ONE to convert from float to uint32
    samples = np.diff(cumsum, axis=1)
    if not np.all((samples > 0) == (float_samples > 0)):
        warnings.warn(f"Truncation performed in quantization. Inputs should be thresholded before quantization."
                      f"Recommended threshold is at least 10*Δ={DELTA / ONE * 10}, preferably greater.")
    if not np.all(samples.sum(axis=1) == ONE):
        raise ValueError("Samples do not sum to 1 after quantization")
    return samples


def cdf(x_j, prior: dirichlet.MixedDirichlet,
        pre_class: NDArray[bool], pre_samples: NDArray[np.uint32]) -> NDArray[np.float64]:
    assert pre_samples.dtype == np.uint32, 'pre_samples and x_j must use uint32 representation of components'
    assert check_pre_samples(pre_samples, pre_class), 'pre_samples do not belong to pre_class'
    j = len(pre_class) - 1  # we want the cdf of the component with index j
    pre_class_matrix = np.column_stack((prior.class_matrix[:, :j], prior.class_matrix[:, j:].any(axis=1)))
    compat_classes = np.all(pre_class_matrix == pre_class, axis=1)  # find the classes which agree with pre_class
    assert np.any(compat_classes), 'At least one mixture class must be consistent with pre_class'
    if not pre_class[-1]:  # no mass remaining in this or any future components so the pdf is a delta at 0
        return np.ones_like(x_j)
        # trying to evaluate the dirichlet likelihood would throw an error
    # the cdf is the sum of the cdf of the mixture components
    # calculate the posterior mixture weights
    prior_pi = prior.mixture_weights
    # if we treat x_(>=j) as a single component, then the marginal pdf of x_(<j) is a dirichlet
    N, K = len(x_j), sum(compat_classes)  # K is the number of compatible classes
    if j > 0:
        # compute the likelihood of each (compatible) class given the presamples
        alphas = prior.full_alpha[compat_classes, :j][:, pre_class[:-1]]  # the alpha of the x_(<j) components
        alphas = np.column_stack((alphas, prior.full_alpha[compat_classes, j:].sum(axis=1)))  # the alpha of the x_(>=j) combined component
        # each column stats.dirichlet(alpha).pdf(pre_samples/ONE) is the likelihood of the pre_samples belonging a given class
        likelihood = np.column_stack(tuple(stats.dirichlet(alpha).pdf(pre_samples[:, pre_class[:-1]]/ONE) for alpha in alphas))
        assert likelihood.shape == (N, K), 'Likelihood has wrong shape'
    else:
        likelihood = np.ones((N, K))
    # Bayes rule to compute the posterior mixture weights
    posterior_pi = prior_pi[compat_classes] * likelihood  # the posterior mixture weights
    posterior_pi /= posterior_pi.sum(axis=1)  # normalize the posterior mixture weights for each sample
    # create boolean three boolean masks to indicate if the class puts x_j in the 0-delta, the 1-delta, or in the interval.
    compat_lower_classes = ~ prior.class_matrix[compat_classes, j]  # compatible classes with no mass in component j
    # compatible classes with all the remaining mass in component j
    compat_upper_classes = prior.class_matrix[compat_classes, j] & ~ prior.class_matrix[compat_classes, j+1:].any(axis=1)
    compat_middle_classes = ~ (compat_lower_classes | compat_upper_classes)
    # every compatible class must belong to exactly one of the three categories
    assert np.all(compat_lower_classes + compat_upper_classes + compat_middle_classes == 1), 'Every class must me lower, middle, or upper'
    # calculate the cdf of the mixture components. Marginally these are beta
    # x_j | x_(<j) ~ beta(alpha_j, sum(alpha_(j+1:)))
    middle_alphas = prior.full_alpha[compat_classes][compat_middle_classes]
    betas = stats.beta(middle_alphas[:, j], middle_alphas[:, j+1:].sum(axis=1))  # the beta distribution of the x_j component
    # the cdf is the sum of the lower delta, the middle dirichlet cdf, and the upper delta
    upper = ONE - pre_samples.sum(axis=1)
    lower_mass, middle_mass, upper_mass = posterior_pi[:,compat_lower_classes].sum(axis=1), posterior_pi[:,compat_middle_classes].sum(axis=1), posterior_pi[:,compat_upper_classes].sum(axis=1)
    return lower_mass + \
           np.where(x_j <= upper, posterior_pi[:,compat_middle_classes].dot(betas.cdf(x_j/upper).T), middle_mass) + \
           np.where(x_j >= upper, upper_mass, 0)

def check_pre_samples(pre_samples: NDArray[np.uint32], pre_class: NDArray[bool]) -> bool:
    # check that the pre_samples really belong to the pre_class
    assert pre_samples.dtype == np.uint32, 'pre_samples must use uint32 representation of components'
    assert pre_samples.shape[1] + 1 == len(pre_class), 'pre_samples must have one fewer component than pre_class'
    return np.all(np.column_stack((pre_samples > 0, pre_samples.sum(axis=1) < ONE)) == pre_class)

def inv_cdf(uniforms: NDArray[np.float64], prior: dirichlet.MixedDirichlet,
            pre_class: NDArray[bool], pre_samples: NDArray[np.uint32]) -> NDArray[np.uint32]:
    assert uniforms.ndim == 1, 'uniforms must be a 1D array'
    assert check_pre_samples(pre_samples, pre_class), 'pre_samples do not belong to pre_class'
    assert len(uniforms) == len(pre_samples), 'uniforms and pre_samples must have the same length'
    X = np.zeros_like(uniforms, dtype=np.uint32)
    # the uniform can map back to a delta on either end of the interval or to a value in the middle
    # we need to check all three cases
    lower, upper = np.zeros_like(uniforms), ONE - pre_samples.sum(axis=1)  # the lower and upper bounds of the interval
    lower_mask = uniforms < cdf(lower, prior, pre_class, pre_samples)  # the mask of samples that map to the lower bound
    upper_mask = uniforms > cdf(upper - DELTA, prior, pre_class, pre_samples)  # the samples that map to the upper bound
    # middle_mask = np.logical_not(np.logical_or(lower_mask, upper_mask))  # the samples that map to the middle
    middle_mask = ~ (lower_mask | upper_mask)
    X[lower_mask] = lower  # map the samples to the lower bound
    X[upper_mask] = upper  # map the samples to the upper bound
    # map the samples to the middle. Invert the cdf by binary search
    X[middle_mask] = vector_binary_search(lambda x: cdf(x, prior, pre_class, pre_samples), uniforms[middle_mask])
    return X


def uniformize(samples: NDArray[np.uint32], prior: dirichlet.MixedDirichlet) -> NDArray[np.float64]:
    # take the cdf of each sample
    U = np.zeros_like(samples, dtype=np.float64)
    # i is the sample index, j is the component index, k is the class index
    I, J = samples.shape
    for j in range(J):
        pre_samples = samples[:, :j]
        x_j = samples[:, j]
        sample_pre_classes = np.column_stack((pre_samples > 0, pre_samples.sum(axis=1) < ONE))
        for pc in np.unique(sample_pre_classes, axis=0):
            pc_idx = np.all(sample_pre_classes == pc, axis=1)  # index of samples belong to this pre_class (pc)
            # we have limited ourselves to a certain "pre_class", but x_j itself can be low, middle, or upper
            lower, upper = 0, ONE - pre_samples[pc_idx].sum(axis=1)  # legal bounds for x_j
            low_samples = x_j[pc_idx] == 0
            upper_samples = x_j[pc_idx] == upper
            middle_samples = ~ (low_samples | upper_samples)
            u = np.zeros_like(x_j[pc_idx], dtype=np.float64)
            u[low_samples] = stats.uniform(0, cdf(0, prior, pc, pre_samples[pc_idx][low_samples])).rvs()
            u[upper_samples] = stats.uniform(
                cdf(upper[upper_samples] - DELTA, prior, pc, pre_samples[pc_idx][upper_samples]),
                1 - cdf(upper[upper_samples] - DELTA, prior, pc, pre_samples[pc_idx][upper_samples])).rvs()
            u[middle_samples] = cdf(x_j[pc_idx][middle_samples], prior, pc, pre_samples[pc_idx][middle_samples])
            U[pc_idx, j] = u

    assert np.all((0 < U) & (U < 1))
    return U


def deuniformize(U: NDArray[np.float64], prior: dirichlet.MixedDirichlet) -> NDArray[np.uint32]:
    X = np.zeros_like(U, dtype=np.uint32)
    # i is the sample index, j is the component index, k is the class index
    I, J = U.shape
    for j in range(J):
        pre_samples = X[:, :j]
        x_j = X[:, j]
        sample_pre_classes = np.column_stack((pre_samples > 0, pre_samples.sum(axis=1) < ONE))
        for pc in np.unique(sample_pre_classes, axis=0):
            pc_idx = np.all(sample_pre_classes == pc, axis=1)  # index of samples belong to this pre_class (pc)
            X[pc_idx, j] = inv_cdf(U[pc_idx, j], prior, pc, pre_samples[pc_idx])
    return X


def vector_binary_search(f, Y):
    ''' Given a scalar function f and a vector Y, return the vector X such that f(X) = Y. Where
    0 <= X <= 1 and is represented with a 32-bit unsigned integer. '''
    X = np.zeros_like(Y, dtype=np.uint32)
    assert np.all(f(X) <= Y), "f(0) must be less than or equal to Y"
    for sig_bit in 2 ** np.arange(SIG_BITS - 1, -1, -1):
        X = np.where(f(X) <= Y, X + sig_bit, X - sig_bit)
    assert np.all(0 < X) and np.all(X < ONE)
    # check the values on either side of X
    three = np.column_stack((X - DELTA, X, X + DELTA))  # X and the two values on either side
    X = three[np.arange(len(X)), np.argmin(np.abs(f(three) - Y[:, None]), axis=1)]
    if np.any(X == 0) or np.any(X == ONE):
        warnings.warn("Binary search selected a value on the bounds of [0,1].")
    return X