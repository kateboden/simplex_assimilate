import numpy as np

import pytest

from simplex_assimilate.dirichlet import MixedDirichlet


def test_mixed_dirichlet_instantiation_warns_when_mixture_component_classes_are_not_unique():
    with pytest.warns(UserWarning):
        alphas = np.array([[1, 3, 5],
                            [1, 6, 10]])
        pi = [0.3, 0.7]
        MixedDirichlet(alphas, pi)

def test_mixed_dirichlet_mixing_weights_must_be_positive_and_sum_to_one():
    # wrong number of mixing weights
    alphas = np.array([[1, 3, 5],
                       [1, 6, 10]])
    with pytest.raises(AssertionError):
        pi = [0.3, 0.2, 0.5]
        MixedDirichlet(alphas, pi)
    # negative mixing weight
    with pytest.raises(AssertionError):
        pi = [0.3, -0.3]
        MixedDirichlet(alphas, pi)
    # mixing weights don't sum to one
    with pytest.raises(AssertionError):
        pi = [0.3, 0.7, 0.1]
        MixedDirichlet(alphas, pi)
