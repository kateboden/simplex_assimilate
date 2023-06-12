import simplex_assimilate as sa
import numpy as np
import pytest

@pytest.fixture
def mixed_dirichlet():
    return sa.dirichlet.MixedDirichlet(np.array([[1, 1, 1], [1, 0, 1]]), np.array([0.5, 0.5]))


def test_vector_binary_search_correctly_finds_inverse_function_and_can_find_boundary():
    def f(x):
        x = x / sa.ONE
        return 2 * x ** 2
    Y = np.array([0, 0.5, 1, 1.5, 2])
    with pytest.warns(UserWarning):
        X = sa.vector_binary_search(f, Y)
        assert np.allclose(f(X), Y)
        assert np.allclose((X/sa.ONE), np.array([0, 0.5, 0.707, 0.866, 1.]), atol=1e-3)
        assert X[0] == 0 and X[-1] == sa.ONE
        assert X.dtype == np.uint32


def test_quantize_raises_error_when_inputs_dont_sum_to_one():
    with pytest.raises(AssertionError):
        sa.quantize(np.array([[0.0, 0.3, 0.6]]))
def test_quantize_raises_error_on_negative_inputs():
    with pytest.raises(AssertionError):
        sa.quantize(np.array([[-1, 1, 0]]))

def test_quantize_output_equals_ONE():
    output = sa.quantize(np.array([[0.0, 0.3, 0.7],
                                   [0.2, 0.3, 0.5]]))
    assert (output.sum(axis=1) == sa.ONE).all()

def test_quantize_warns_when_nonzero_is_truncated_to_zero():
    with pytest.warns(UserWarning):
        sa.quantize(np.array([[1e-32, 1-1e-32, 0]]))

def test_mixed_dirichlet_instantiation_warns_when_mixture_component_classes_are_not_unique():
    with pytest.warns(UserWarning):
        sa.dirichlet.MixedDirichlet(np.array([[1, 3, 5],
                                    [1, 6, 10]]), [0.3, 0.7])

def test_mixed_dirichlet_mixing_weights_must_be_positive_and_sum_to_one():
    # wrong number of mixing weights
    with pytest.raises(AssertionError):
        sa.dirichlet.MixedDirichlet(np.array([[1, 3, 5],
                                    [1, 6, 10]]), [0.3, 0.2, 0.5])
    # negative mixing weight
    with pytest.raises(AssertionError):
        sa.dirichlet.MixedDirichlet(np.array([[1, 3, 5],
                                    [1, 6, 10]]), [0.3, -0.3])
    # mixing weights don't sum to one
    with pytest.raises(AssertionError):
        sa.dirichlet.MixedDirichlet(np.array([[1, 3, 5],
                                    [1, 6, 10]]), [0.3, 0.7, 0.1])

def test_cdf_asserts_at_least_one_mixture_class_consistent_w_pre_class(mixed_dirichlet):
    with pytest.raises(AssertionError):
        sa.cdf(x_j=np.array([[0.5]]),
               prior=mixed_dirichlet,
               pre_samples=np.array([[0]]))

def test_cdf_asserts_pre_samples_and_x_j_have_dtype_uint32(mixed_dirichlet):
    with pytest.raises(AssertionError):
        sa.cdf(x_j=np.array([[0.5]]),  # wrong dtype
               prior=mixed_dirichlet,
               pre_samples=np.array([[np.uint32(1)]]))
    with pytest.raises(AssertionError):
        sa.cdf(x_j=np.array([[np.uint32(1)]]),
               prior=mixed_dirichlet,
               pre_samples=np.array([[0.5]])) # wrong dtype

def test_cdf_correct_1(mixed_dirichlet):
    # if we observe small x_0 this favors the first mixture component
    # which means the probability that x_1 is zero is now less than half
    assert np.allclose(sa.cdf(x_j=(np.array([0])*sa.ONE).astype(np.uint32),
                              prior=mixed_dirichlet,
                              pre_samples=(np.array([[0.1]])*sa.ONE).astype(np.uint32),
                            ),
                       np.array([[0.357]]), atol=1e-3)

def test_cdf_correct_2(mixed_dirichlet):
    # show that if x_1 is zero then the distribution of x_2
    # must be a delta function at 1-x_0
    open_water = round(0.4*sa.ONE)
    upper = sa.ONE - open_water
    assert np.allclose(sa.cdf(x_j=(np.array([upper - sa.DELTA])).astype(np.uint32),
                                prior=mixed_dirichlet,
                                pre_samples=np.array([[open_water, 0]]).astype(np.uint32),
                            ),
                          np.array([[0.]]), atol=1e-30)
    assert np.allclose(sa.cdf(x_j=(np.array([upper])).astype(np.uint32),
                                prior=mixed_dirichlet,
                                pre_samples=np.array([[open_water, 0]]).astype(np.uint32),
                            ),
                            np.array([[1.]]), atol=1e-30)

def test_cdf_correct_3():
    # demonstrate a component compelled to be zero
    assert np.allclose(sa.cdf(x_j=(np.array([0])).astype(np.uint32),
                                prior=sa.dirichlet.MixedDirichlet(np.array([[4, 0, 5],]), mixture_weights=[1.]),
                                pre_samples=(np.array([[0.1]])*sa.ONE).astype(np.uint32),
                            ),
                            np.array([[1.]]), atol=1e-30)


def test_invert_cdf_checks_uniforms_len_equals_pre_samples_len():
    with pytest.raises(AssertionError):
        sa.inv_cdf(uniforms=np.array([[0.5],
                                      [0.5]]),
                      prior=sa.dirichlet.MixedDirichlet(np.array([[4, 0, 5],]), mixture_weights=[1.]),
                      pre_samples=(np.array([[0.1]])*sa.ONE).astype(np.uint32),
        )

def test_inv_cdf_correct_1(mixed_dirichlet):
    # the inverse of u_0=0.5 should be somewhere between the means of the two mixture components
    # in the first components which are 1/3 and 1/2
    assert np.allclose((sa.inv_cdf(uniforms=np.array([0.5, ]),
                                    prior=mixed_dirichlet,
                                    pre_samples=(np.array([[0.5],])*sa.ONE).astype(np.uint32),
                                )/sa.ONE).astype(np.float64),
                            np.array([[0.200]]), atol=1e-3)