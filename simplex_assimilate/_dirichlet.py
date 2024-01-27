import numpy as np
from numpy.typing import NDArray
import warnings

from simplex_assimilate.fixed_point import check_samples, ONE

MAX_ALPHA = 1e4
DEFAULT_ALPHA = 1e1

class MixedDirichlet:

    def __init__(self, full_alpha: NDArray[np.float64], mixture_weights: NDArray[np.float64]):
        full_alpha, mixture_weights = np.array(full_alpha), np.array(mixture_weights)  # cast to np.ndarray
        assert len(full_alpha) == len(mixture_weights), 'full_alpha and mixing_weights must have the same length.'
        assert (mixture_weights > 0).all(), 'Mixing weights must be greater than zero.'
        assert np.isclose(mixture_weights.sum(), 1.0, atol=1e-32), 'Mixing weights must sum to one.'
        self.full_alpha = full_alpha
        self.mixture_weights = mixture_weights
        self.class_matrix = self.full_alpha > 0
        self.K, self.J = self.full_alpha.shape
        # check that every row in class_matrix is unique
        if not len(np.unique(self.class_matrix, axis=0)) == len(self.class_matrix):
            warnings.warn('Class_matrix has duplicate rows. Modelling a class with multiple Dirichlets is not supported.')
        return

    def __repr__(self):
        return f'MixedDirichlet(full_alpha={self.full_alpha}, mixture_weights={self.mixture_weights})'

    @classmethod
    def est_from_samples(cls, samples: NDArray[np.uint32]):
        # convert to floating point
        check_samples(samples)
        #
        classes = np.unique(samples > 0, axis=0)
        alphas = []
        s_determined = []
        mixture_weights = []
        for c in classes:
            class_rows = ((samples > 0) == c).all(axis=1)  # rows of samples which agree with class c
            class_samples = samples[class_rows][:, c]  # nonzero components
            s, a = cls.fit_class_dirichlet(class_samples)
            alphas.append(a)
            s_determined.append(s)
            mixture_weights.append(len(class_samples)/len(samples))
        if not any(s_determined):
            warnings.warn('No two samples are in the same class. Cannot estimate magnitude of alpha. Using default alpha.')
            alphas = [DEFAULT_ALPHA*alpha for alpha in alphas]
        elif not all(s_determined):
            max_observed_alpha = max([sum(a) for a, s in zip(alphas, s_determined) if s])
            alphas = [max_observed_alpha*alpha if (not s) else alpha for alpha, s in zip(alphas, s_determined)]
        K, J = len(alphas), len(samples[0])
        full_alpha = np.zeros((K, J))
        for i, alpha in enumerate(alphas):
            full_alpha[i, classes[i]] = alpha
        # print('estimation of alpha: ')
        # print(full_alpha, mixture_weights)
        return cls(full_alpha=full_alpha, mixture_weights=np.array(mixture_weights))

    @staticmethod
    def fit_class_dirichlet(samples: NDArray[np.uint32]):
        check_samples(samples)
        samples = samples.astype(np.float64) / ONE  # convert to floating point
        assert np.all(samples > 0), 'All components of all samples must be positive.'
        log_avg = np.log(samples).mean(axis=0)
        geo_mean = np.exp(log_avg)
        mean = geo_mean / geo_mean.sum()  # normalized geometric mean
        if len(samples) == 1:
            return False, mean  # cannot determine s, but we can still provide the normalized mean
        # if all the samples are in a tight envelope, use the max_alpha instead
        if np.allclose(samples.min(axis=0), samples.max(axis=0)):
            warnings.warn('Multiple samples are in a tight envelope. Using max_alpha instead of MLE. '
                          f'class samples = {samples}')
            return True, MAX_ALPHA * mean
        # use the alpha of the closed form MLE using Stirling's approximation
        N, J = samples.shape
        s = ((J-1)/2) / (np.inner(mean, np.log(mean) - log_avg))
        s = min(s, MAX_ALPHA)
        alpha = s * mean
        return True, alpha


'''
else:
    gammaln, digamma, polygamma = scipy.special.gammaln, scipy.special.digamma, scipy.special.polygamma
    f = lambda alpha: gammaln(alpha.sum()) - gammaln(alpha).sum() + (log_avg * (alpha - 1)).sum()  # likelihood
    alphas = [np.ones_like(log_avg)]  # initialize alpha_0
    for _ in range(5):
        alpha = alphas[-1]
        grad = digamma(alpha.sum()) - digamma(alpha) + log_avg
        hessian = - np.eye(len(alpha)) * (polygamma(1, alpha))
        hessian += polygamma(1, alpha.sum())
        invH = np.linalg.inv(hessian)
        da = - np.dot(invH, grad)
        alphas.append(alpha + da)
'''
