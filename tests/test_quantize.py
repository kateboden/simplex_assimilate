import numpy as np

import pytest

# from simplex_assimilate.quantize import quantize, dequantize, ONE
from simplex_assimilate.utils.quantize import quantize
from simplex_assimilate.fixed_point import ONE

class TestQuantize:
    def test_quantize_raises_error_when_inputs_dont_sum_to_one(self):
        with pytest.raises(AssertionError):
            quantize(np.array([[0.0, 0.3, 0.6]]))
    def test_quantize_raises_error_on_negative_inputs(self):
        with pytest.raises(AssertionError):
            quantize(np.array([[-1, 1, 0]]))

    def test_quantize_output_equals_ONE(self):
        output = quantize(np.array([[0.0, 0.3, 0.7],
                                    [0.2, 0.3, 0.5]]))
        assert (output.sum(axis=1) == ONE).all()

    def test_quantize_warns_when_nonzero_is_truncated_to_zero(self):
        with pytest.warns(UserWarning):
            quantize(np.array([[1e-32, 1-1e-32, 0]]))

    def test_dequantize_inverts_quantize(self):
        input = np.array([[0.0, 0.3, 0.7],
                          [0.2, 0.3, 0.5]])
        quantized = quantize(input)
        dequantized = quantized / ONE
        assert np.allclose(input, dequantized)
