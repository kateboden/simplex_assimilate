# Simplex Assimilate: Data assimilation on the simplex

![Coverage](tests/coverage.svg)

## Installation 
```bash
pip install simplex-assimilate
```

## File Structure
 - `simplex_assimilate/` source code
 - `simplex_assimilate/cdf.py` Conditional distribution functions for the mixed dirichlet prior.
 - `simplex_assimilate/quantize.py` Convert floating point samples to fixed point representation.
 - `simplex_assimilate/dirichlet.py` Model samples on the simplex with a mixture of Dirichlet distributions. Fit parameters by maximum likelihood.
 - `simplex_assimilate/transport.py` Optimal transport of samples on the simplex.
 - `tests/` unit tests
 - `coverage_report/` coverage reports

## See Also
  - [Ice Simplex Assimilate Repository](https://github.com/oscarlaird/ice_simplex_assimilate)
    - Represent an ice thickness distribution as a point on the simplex and perform transport.