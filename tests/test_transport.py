import pytest
import numpy as np

from simplex_assimilate.utils.quantize import quantize, dequantize
from simplex_assimilate.fixed_point import ONE

from simplex_assimilate.transport import transport

class TestTransport:

    def test_two_components(self):
        X = quantize(np.array([[0.1, 0.9],
                               [0.9, 0.1]]))
        x_0 = (ONE * np.array([[0.1, 0.9]])).astype(np.uint32)
        expected_out = X
        out = transport(X, x_0)
        assert np.allclose(expected_out, out)

    def test_no_change(self):
        X = quantize(np.array([[0.2, 0.3, 0.5],
                               [0.3, 0.2, 0.5]]))
        x_0 = (ONE * np.array([[0.2, 0.3]])).astype(np.uint32)
        expected_out = X
        out = transport(X, x_0)
        assert np.allclose(expected_out, out, atol=1)  # we do not move by more than DELTA=1 in any component

    def test_x0_increases(self):
        X = quantize(np.array([[0.2, 0.3, 0.5],
                               [0.3, 0.2, 0.5]]))
        x_0 = (ONE * np.array([[0.25, 0.25]])).astype(np.uint32)
        expected_out = [[0.25, 0.2813, 0.4688],
                        [0.25, 0.2143, 0.5357]]
        out = dequantize(transport(X, x_0))
        assert np.allclose(expected_out, out, atol=1e-4)  # we do not move by more than DELTA=1 in any component

    def test_multiple_classes_no_change(self):
        X = quantize(np.array([[0.2, 0.3, 0.5],
                               [0.3, 0.2, 0.5],
                               [0.0, 0.0, 1.0]]))
        x_0 = (ONE * np.array([[0.2, 0.3, 0.0]])).astype(np.uint32)
        expected_out = X
        out = transport(X, x_0)
        assert np.allclose(expected_out, out, atol=1)  # we do not move by more than DELTA=1 in any component

    def test_new_water_jumps_classes(self):
        X = quantize(np.array([[0.2, 0.3, 0.5],
                               [0.3, 0.2, 0.5],
                               [0.0, 0.0, 1.0]]))
        x_0 = (ONE * np.array([[0.2, 0.3, 0.3]])).astype(np.uint32)
        expected_out = X
        expected_out = [[0.2, 0.3, 0.5],
                        [0.3, 0.2, 0.5],
                        [0.3, 0.2321, 0.4679]]
        out = dequantize(transport(X, x_0))
        assert np.allclose(expected_out, out, atol=1)  # we do not move by more than DELTA=1 in any component

    def test_extreme_value_jumps_classes(self):
        X = [[0.2, 0.7, 0.1, 0.0],
             [0.2, 0.1, 0.7, 0.0],
             [0.7, 0.0, 0.1, 0.2],
             [0.7, 0.0, 0.2, 0.1]]
        X = quantize(np.array(X))
        x_0 = [0.8, 0.8, 0.1, 0.1]
        x_0 = (ONE * np.array([x_0])).astype(np.uint32)
        expected_out = X
        expected_out = np.array([[0.80000001, 0.        , 0.10647821, 0.09352179],
                                 [0.80000001, 0.        , 0.10280609, 0.0971939],
                                 [0.1       , 0.46332979, 0.43667021, 0.],
                                 [0.1       , 0.59351826, 0.30648175, 0.]])
        np.random.seed(0)
        out = dequantize(transport(X, x_0))
        assert np.allclose(expected_out, out, atol=1e-4)

    def test_tight_envelope(self):
        X = [[0.1, 0.3, 0.6],
             [0.1, 0.3, 0.6]]
        expected_out = X
        # Make sure that identical samples don't cause a problem
        X = quantize(np.array(X))
        x_0 = [0.1, 0.1]
        x_0 = (ONE * np.array([x_0])).astype(np.uint32)
        with pytest.warns(UserWarning):  # tight envelope
            out = dequantize(transport(X, x_0))
        assert np.allclose(expected_out, out, atol=1e-4)

    def test_observation_incompatible_with_existing_classes(self):
        """ We observe a nonzero first component, but the existing classes have zero first component."""
        X = [[0.0, 0.5, 0.5],
             [0.0, 0.4, 0.6]]
        X = quantize(np.array(X))
        x_0 = [0.1, 0.1]
        x_0 = (ONE * np.array([x_0])).astype(np.uint32)
        expected_out = [[0.1, 0.45, 0.45],
                        [0.1, 0.36, 0.54]]
        with pytest.warns(UserWarning):  # bad samples
            out = dequantize(transport(X, x_0))
        assert np.allclose(expected_out, out, atol=1e-4)




