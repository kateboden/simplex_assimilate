import scipy
from dataclasses import dataclass

# UPDATE ON OBSERVATION
def update_x0(uniform, md: MixedDirichlet, observation: Observation):
    # N.B. Dirichlet is conjugate prior to multinomial observation, but not to binomial observation
    # so the observation does not give us a mixed dirichlet for the posterior
    # But the distribution for x0 is beta and is conjugate to the observation
    # We transform x0 and then transform the other components.
    betas = [scipy.stats.beta(  cd.alpha[0]+observation.r, sum(cd.alpha[1:])+(observation.n-observation.r)  ) for cd in md.dirichlets]
    def cdf(x0):
        return sum([beta.cdf(x0)*pi for beta,pi in zip(betas, md.mixing_rates)])
    delta = 1e-10
    return scipy.optimize.root_scalar(lambda x0: cdf(x0)-uniform, bracket=[delta, 1-delta ]).root
