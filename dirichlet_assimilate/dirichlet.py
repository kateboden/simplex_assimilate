import numpy as np
import scipy
from dataclasses import dataclass
from typing import List

class Sample(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)
    @property
    def sample_class(self):
        return SampleClass(self > 0)


class UniformSample(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

class SampleClass(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array, dtype=bool).view(cls)

@dataclass
class Ensemble:
    samples: List[Sample]
    def __post_init__(self):
        self.sample_classes = [SampleClass(c) for c in {tuple(s.sample_class) for s in self.samples}]
        self.sample_classes = sorted(self.sample_classes, key = lambda x: list(x))
        self.class_ensembles = [ClassEnsemble(samples=[s for s in self.samples if np.all(s.sample_class==c)]) for c in self.sample_classes]

class ClassEnsemble(Ensemble):
    def __post_init__(self):
        self.sample_class = self.samples[0].sample_class

@dataclass
class ClassDirichlet:
    alpha: np.ndarray
    sample_class: SampleClass

    @property
    def full_alpha(self):
        a = np.zeros_like(self.sample_class, dtype=float)
        a[self.sample_class] = self.alpha
        return a

    @property
    def mean_sample(self):
        return Sample(self.full_alpha / self.full_alpha.sum())

    def errorbars(self, a=0.1):
        dists = scipy.stats.beta(self.alpha, self.alpha.sum()-self.alpha)
        lower_error = dists.isf(1 - a)
        upper_error = dists.isf(a)
        y_error = np.array([lower_error, upper_error])
        return y_error

@dataclass
class MixedDirichlet:
    mixing_rates: np.ndarray
    dirichlets: List[ClassDirichlet]

@dataclass
class UniformEnsemble:
    samples: List[UniformSample]


# DIRICHLET PARAMETER ESTIMATION
def fit_dirichlet(class_ensemble: ClassEnsemble) -> ClassDirichlet:
    samples = class_ensemble.samples
    assert len(samples)>1

    log_sum = sum( [np.log(s[class_ensemble.sample_class]) for s in samples] )
    log_avg = log_sum / len(samples)

    gammaln, digamma, polygamma = scipy.special.gammaln, scipy.special.digamma, scipy.special.polygamma
    f = lambda alpha: gammaln(alpha.sum()) - gammaln(alpha).sum() + (log_avg * (alpha - 1)).sum()  # likelihood
    alphas = [np.ones_like(log_avg)]  # initialize alpha_0
    for _ in range(15):
        alpha = alphas[-1]
        grad = digamma(alpha.sum()) - digamma(alpha) + log_avg
        hessian = - np.eye(len(alpha)) * (polygamma(1, alpha))
        # print(alpha)
        hessian += polygamma(1, alpha.sum())
        invH = np.linalg.inv(hessian)
        da = - np.dot(invH, grad)
        alphas.append(alpha + da)
    # print(np.exp(alpha))
    alpha = np.exp(alpha)  # TODO: why?
    return ClassDirichlet(alpha=alpha, sample_class=class_ensemble.sample_class)


def fit_mixed_dirichlet(ensemble: Ensemble) -> MixedDirichlet:
    dirichlets = [None for _ in ensemble.class_ensembles]
    for i, ce in enumerate(ensemble.class_ensembles):
        if len(ce.samples)>1:
            dirichlets[i] = fit_dirichlet(ce)
    max_alpha = max( [sum(d.alpha) for d in dirichlets if d is not None] )
    for i, ce in enumerate(ensemble.class_ensembles):
        if len(ce.samples)==1:
            alpha = ce.samples[0][ce.sample_class]*max_alpha
            dirichlets[i] = ClassDirichlet(alpha=alpha, sample_class=ce.sample_class)
    mixing_rates = np.array( [len(ce.samples)/len(ensemble.samples) for ce in ensemble.class_ensembles] )
    return MixedDirichlet(mixing_rates=mixing_rates, dirichlets=dirichlets)

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

def update_sample(uniform_sample: UniformSample, md: MixedDirichlet, observation: Observation) -> Sample:
    x = []
    x0 = update_x0(uniform_sample[0], md, observation)
    x.append(x0)
    for j in range(1, len(uniform_sample)):
        x.append(update_xj(x, uniform_sample[j], md))
    return Sample(x)

def update_ensemble(uniform_ensemble: UniformEnsemble, md: MixedDirichlet, observation: Observation) -> Ensemble:
    samples = []
    for uniform_sample in uniform_ensemble.samples:
        samples.append(update_sample(uniform_sample, md, observation))
    return Ensemble(samples=samples)