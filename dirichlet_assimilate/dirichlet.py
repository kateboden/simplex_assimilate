import numpy as np
import scipy
from .shared_classes import Sample, SampleClass, Ensemble, ClassEnsemble, UniformSample, UniformEnsemble, ClassDirichlet, MixedDirichlet

# CONVERT TO UNIFORM
def class_likelihood(x_i, cd: ClassDirichlet) -> float:
    x_i = np.array(x_i)
    if x_i.size==0:
        return 1
    # check that x_i is zero where the class is zero and nonzero where the class is nonzero
    if not matches_class(x_i, cd.sample_class):
        # print(cd.sample_class)
        return 0
    pos_x_i = x_i[x_i>0]
    alpha_i = cd.alpha[:len(pos_x_i)]
    alpha_k = cd.alpha[len(pos_x_i):]
    # if we have all the components we can simply evaluate the pdf
    if len(pos_x_i)==len(cd.alpha):
        ret = scipy.stats.dirichlet(alpha=cd.alpha).pdf(pos_x_i[:-1])
        # print(ret, cd.alpha, pos_x_i)
        return ret
    else:
        x = np.append(pos_x_i, 1-sum(x_i))
        a = np.append(alpha_i, sum(alpha_k))
        return scipy.stats.dirichlet(a).pdf(x)

def matches_class(x_i, sample_class: SampleClass) -> bool:
    # we have
    zeroes_match = np.array_equal( x_i>0, sample_class[:len(x_i)])
    x_all_mass = np.isclose(1., x_i.sum(), atol=1e-15, rtol=1e-15)  # have we used up all the probability mass yet (Σx_i=1)
    class_all_mass = np.all(sample_class[len(x_i):]==0)  # does the class have any remaining nonzero categories?
    # if sample_class[-2]:
        # print(x_all_mass, len(x_i), x_i.sum(), zeroes_match and x_all_mass==class_all_mass)
    return zeroes_match and x_all_mass==class_all_mass

def find_post_class_probs(x_i, md: MixedDirichlet):
    # print('x_i: ', x_i, x_i.sum())
    prior_class_probs = md.mixing_rates
    class_likelihoods = np.array( [class_likelihood(x_i, cd) for cd in md.dirichlets] )
    # print(class_likelihoods)
    post_class_probs = prior_class_probs * class_likelihoods
    return post_class_probs / sum(post_class_probs)

def get_conditional_cdf(x_i, md: MixedDirichlet):
    # the pdf is a mixture of betas with a deltas on either end
    # return the probability mass of 0 and 1 as well as a function to evaluate the cdf
    x_i = np.array(x_i)
    pos_x_i = x_i[x_i>0]
    post_class_probs = find_post_class_probs(x_i, md)
    j = len(x_i)
    p0 = post_class_probs[[cd.sample_class[j]==False for cd in md.dirichlets]].sum()
    p1 = post_class_probs[[cd.sample_class[j]==True and not np.any(cd.sample_class[j+1:]) for cd in md.dirichlets]].sum()

    def cdf(x_j):
        cdf_total = p0

        if sum(x_i)==1:
            return 1
        if np.isclose(0.,         x_j):
            return p0
        if np.isclose(1-sum(x_i), x_j):
            return 1-p1


        scaled_x_j = x_j/(1-sum(x_i))
        assert 0<scaled_x_j<1, scaled_x_j

        for cd, pcp in zip(md.dirichlets, post_class_probs):
            # the ClassDirichlet matches our components
            if not matches_class(np.append(x_i, x_j), cd.sample_class):
                continue
            # x_j is distributed beta in the remaining space
            alpha_j = cd.alpha[len(pos_x_i)]
            alpha_k = cd.alpha[len(pos_x_i)+1:]
            cdf_total += scipy.stats.beta(alpha_j, sum(alpha_k)).cdf(scaled_x_j) * pcp
        return cdf_total
    return p0, p1, cdf

def uniformize_component(x_i, x_j, md: MixedDirichlet) -> float:
    # apply the cdf to transform x_j to a uniform random variable (needed for optimal transport)
    # the pdf is a mixture of scaled betas possibly with deltas at either end
    x_i = np.array(x_i)
    p0, p1, cdf = get_conditional_cdf(x_i, md)
    if   np.isclose(x_j, 0.):
        return scipy.stats.uniform(0, p0).rvs()
    elif np.isclose(x_j, 1-sum(x_i)):
        delta = 1e-10 # todo
        return scipy.stats.uniform(1-p1-delta, p1+delta).rvs()
    else:
        return cdf(x_j)

def uniformize_sample(sample: Sample, md: MixedDirichlet) -> UniformSample:
    u = []
    for j, x_j in enumerate(sample):
        u.append(uniformize_component(x_i=sample[:j], x_j=x_j, md=md))
    return UniformSample(u)

def uniformize_ensemble(ensemble: Ensemble, md: MixedDirichlet) -> UniformEnsemble:
    uniform_samples = []
    for s in ensemble.samples:
        uniform_samples.append(uniformize_sample(s, md))
    return UniformEnsemble(samples=uniform_samples)

def update_xj(x_i, uniform, md):
    p0, p1, cdf = get_conditional_cdf(x_i, md)
    if uniform < p0:
        return 0.
    elif uniform < 1-p1:
        delta = 1e-10
        return scipy.optimize.root_scalar(lambda x_j: cdf(x_j)-uniform, bracket=[delta, 1-sum(x_i)-delta]).root
    else:
        return 1-sum(x_i)

def update_sample(uniform_sample: UniformSample, md: MixedDirichlet, observation: float) -> Sample:
    x = []
    x0 = update_x0(uniform_sample[0], md, observation)
    x.append(x0)
    for j in range(1, len(uniform_sample)):
        x.append(update_xj(x, uniform_sample[j], md))
    return Sample(x)

def update_ensemble(uniform_ensemble: UniformEnsemble, md: MixedDirichlet, observation: float) -> Ensemble:
    samples = []
    for uniform_sample in uniform_ensemble.samples:
        samples.append(update_sample(uniform_sample, md, observation))
    return Ensemble(samples=samples)