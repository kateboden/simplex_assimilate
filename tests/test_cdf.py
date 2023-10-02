import numpy as np

import pytest

from simplex_assimilate.utils.quantize import quantize
from simplex_assimilate.fixed_point import ONE
from simplex_assimilate.cdf import vector_binary_search, likelihood, vectorized_likelihood, \
    cdf, inv_cdf, uniformize, deuniformize
from simplex_assimilate.dirichlet import MixedDirichlet

class TestCDF:
    def test_vector_binary_search_finds_inverse_function_and_can_find_boundary(self):
        def f(x):
            x = x / ONE
            return 2 * x ** 2
        Y = np.array([0, 0.5, 1, 1.5, 2])
        with pytest.warns(UserWarning):
            X = vector_binary_search(f, Y)
            assert np.allclose(f(X), Y)
            assert np.allclose((X / ONE), np.array([0, 0.5, 0.707, 0.866, 1.]), atol=1e-3)
            assert X[0] == 1 and X[-1] == ONE
            assert X.dtype == np.uint32


    def test_likelihood(self):
        alpha = np.array([1.0, 2.0, 7.0])
        # incompatible class -> 0 likelihood
        pre_x = quantize(np.array([[0.0, 0.3, 0.7], ]))[0]
        assert likelihood(alpha, pre_x) == 0
        # sample is close to alpha_mean -> high likelihood
        pre_x = quantize(np.array([[0.1, 0.2, 0.7], ]))[0]
        assert likelihood(alpha, pre_x) == 11.85901920236669
        # including the last component is optional
        pre_x = quantize(np.array([[0.1, 0.2, 0.7], ]))[0][:-1]
        assert likelihood(alpha, pre_x) == 11.85901920236669
        # sample is far from alpha_mean -> low likelihood
        pre_x = quantize(np.array([[0.7, 0.2, 0.1], ]))[0]
        assert likelihood(alpha, pre_x) == 0.00010080000042244779


    def test_vectorized_likelihood(self):
        alphas = np.array([[1.0, 2.0, 7.0],
                           [4.0, 3.0, 3.0],
                           [0.0, 10.0, 0.0]])
        pre_samples = quantize(np.array([[0.1, 0.2, 0.7],
                                         [0.7, 0.2, 0.1],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 1.0, 0.0]]))
        pre_samples = pre_samples[:, :-1]  # only give the first two components
        output = np.array([[1.18590192e+01, 2.96352000e-01, 0.00000000e+00],
                           [1.00800000e-04, 2.07446400e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        assert np.allclose(vectorized_likelihood(alphas, pre_samples), output)
        pre_samples = pre_samples[:, :1]  # only give the first component
        output = np.array([[3.87420489e+00, 2.97606961e-01, 0.00000000e+00],
                           [5.90489997e-04, 4.20078959e-01, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        assert np.allclose(vectorized_likelihood(alphas, pre_samples), output)


    def test_cdf_1(self):
        alphas = np.array([[1.0, 2.0, 7.0]])
        pi = np.array([1.0, ])
        prior = MixedDirichlet(alphas, pi)
        samples = quantize(np.array([[0.1, 0.2, 0.7]]))
        pre_samples = samples[:, :0]
        # x_0 ~ Beta(1, 9)
        x_j = samples[:, 0]
        output = np.array([0.61257951])
        assert np.allclose(cdf(x_j, prior, pre_samples), output)
        # x_2 is a delta
        x_j = samples[:, 2]
        pre_samples = samples[:, :2]
        assert np.allclose(cdf(x_j - 1, prior, pre_samples), [0.])  # no chance <x_j
        assert np.allclose(cdf(x_j, prior, pre_samples), [1.])  # must be =x_j


    def test_cdf_2(self):
        alphas = np.array([[1., 1., 0.],
                           [0., 1., 1.]])
        pi = np.array([0.5, 0.5])
        prior = MixedDirichlet(alphas, pi)
        samples = quantize(np.array([[0.5, 0.5, 0.0],
                                     [0.0, 0.5, 0.5],
                                     [0.0, 0.5, 0.5]]))
        # x_0 ~ Beta(1, 1) with 50% chance, otherwise 0
        pre_samples = samples[:, :0]
        x_j = samples[:, 0]
        output = np.array([0.75, 0.5, 0.5])
        assert np.allclose(cdf(x_j, prior, pre_samples), output)
        # if we condition on x_0=0, then x_1 ~ Beta(1, 1)
        # if we condition on x_0=0.5 then x_1 is a delta at 0.5
        pre_samples = samples[:, :1]
        x_j = samples[:, 1]
        output = np.array([1., 0.5, 0.5])
        assert np.allclose(cdf(x_j, prior, pre_samples), output)


    def test_inv_cdf_1(self):
        alphas = np.array([[1.0, 2.0, 7.0]])
        pi = np.array([1.0, ])
        prior = MixedDirichlet(alphas, pi)
        samples = quantize(np.array([[0.1, 0.2, 0.7]]))
        pre_samples = samples[:, :0]
        u_j = np.array([0.5])
        output = np.array([0.07412528758868575])  # the median is less than the mean
        assert np.allclose(inv_cdf(u_j, prior, pre_samples) / ONE, output)
        # any uniform will map to the delta
        pre_samples = samples[:, :2]  # only give the first two components
        u_j = np.array([0.5])
        output = samples[:, 2]
        assert np.allclose(inv_cdf(u_j, prior, pre_samples), output)


    def test_uniformize_1(self):
        alphas = np.array([[1.0, 2.0, 7.0],
                           [4.0, 3.0, 3.0],
                           [0.0, 10.0, 0.0]])
        pi = np.array([0.5, 0.25, 0.25])
        prior = MixedDirichlet(alphas, pi)
        samples = quantize(np.array([[0.1, 0.2, 0.7],
                                     [0.7, 0.2, 0.1],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 1.0, 0.0]]))
        np.random.seed(0)
        out = np.array([[0.55837253, 0.54209212, 0.43758721],
                        [0.99366645, 0.79070457, 0.891773],
                        [0.13720338, 0.60276338, 0.4236548],
                        [0.17879734, 0.54488318, 0.64589411]])
        assert np.allclose(uniformize(samples, prior), out)


    def test_deuniformize_1(self):
        # check that we invert the uniformization of test_uniformize_1
        # the inversion is EXACT
        U = np.array([[0.55837253, 0.54209212, 0.43758721],
                      [0.99366645, 0.79070457, 0.891773],
                      [0.13720338, 0.60276338, 0.4236548],
                      [0.17879734, 0.54488318, 0.64589411]])
        alphas = np.array([[1.0, 2.0, 7.0],
                           [4.0, 3.0, 3.0],
                           [0.0, 10.0, 0.0]])
        pi = np.array([0.5, 0.25, 0.25])
        samples = quantize(np.array([[0.1, 0.2, 0.7],
                                     [0.7, 0.2, 0.1],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 1.0, 0.0]]))
        prior = MixedDirichlet(alphas, pi)
        assert np.allclose(deuniformize(U, prior), samples)

