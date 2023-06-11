import numpy as np
from numpy.typing import NDArray
import warnings

class MixedDirichlet:

    def __init__(self, full_alpha: NDArray[np.float64], mixture_weights: NDArray[np.float64]):
        full_alpha, mixture_weights = np.array(full_alpha), np.array(mixture_weights)  # cast to np.ndarray
        assert len(full_alpha) == len(mixture_weights), 'full_alpha and mixing_weights must have the same length.'
        assert (mixture_weights > 0).all(), 'Mixing weights must be greater than zero.'
        assert np.isclose(mixture_weights.sum(), 1.0, atol=1e-32), 'Mixing weights must sum to one.'
        self.full_alpha = full_alpha
        self.mixture_weights = mixture_weights
        self.class_matrix = self.full_alpha > 0
        # check that every row in class_matrix is unique
        if not len(np.unique(self.class_matrix)) == len(self.class_matrix):
            warnings.warn('Class_matrix has duplicate rows. Modelling a class with multiple Dirichlets is not supported.')
        return

    @classmethod
    def est_from_samples(cls, samples: NDArray[np.float64]):
        pass