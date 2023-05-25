import numpy as np
import scipy
from dataclasses import dataclass
from typing import List
from matplotlib import pyplot as plt

## classes
class Component(float):
    pass

class Category(int):
    pass

@dataclass
class RawSample:
    area: np.ndarray
    volume: np.ndarray
    snow: np.ndarray
    def __post_init__(self):
        if not len(self.area)==len(self.volume)==len(self.snow):
            raise ValueError('Area, Volume, and Snow vectors must have the same length.')

class HeightBounds(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)
    @property
    def intervals(self):
        return zip(self[:-1], self[1:])

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
class Observation:
    n: int
    r: int


@dataclass
class RawEnsemble:
    samples: List[RawSample]

@dataclass
class Ensemble:
    samples: List[Sample]
    def __post_init__(self):
        self.sample_classes = [SampleClass(c) for c in {tuple(s.sample_class) for s in self.samples}]
        self.sample_classes = sorted(self.sample_classes, key = lambda x: list(x))

        # all_classes = [s.sample_class for s in self.samples]
        # self.sample_classes = []
        # for c in all_classes:
            # if not any(np.array_equal(c, existing_c) for existing_c in self.sample_classes):
                # self.sample_classes.append(c)
        # create the class ensembles and estimate their dirichlet parameters
        self.class_ensembles = [ClassEnsemble(samples=[s for s in self.samples if np.all(s.sample_class==c)]) for c in self.sample_classes]

    def visualize_classes(self, h_bnd):
        nrows = (len(self.class_ensembles)+1)//2
        fig = plt.figure(figsize=(40, 8*nrows))
        axes = fig.subplots(ncols=2, nrows=nrows, sharey=True)
        for ce, ax in zip(self.class_ensembles, axes.flatten()):
            num = len(ce.samples)
            title = f'Example from class {ce.sample_class}. Count = {num}'
            ax = ce.samples[0].visualize(h_bnd, ax=ax)
            ax.title.set_text(title)
        fig.tight_layout()
        return fig

    def visualize(self, h_bnd):
        color = 'b'
        ax = self.samples[0].visualize(h_bnd, style='scatter', show_raw=False, show_labels=False, color=color)
        for s in self.samples[1:]:
            ax = s.visualize(h_bnd, ax=ax, style='scatter', show_raw=False, show_labels=False, show_bounds=False, color=color)
        return ax

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


class Visualisation:

    def __init__(self, h_bnd, ax=None):
        self.h_bnd = h_bnd
        if not ax:
            fig, ax = plt.subplots(figsize=(20,8))
        self.width = 0.25
        self.bottom=-15
        ax.set_xlabel('Height')
        ax.set_ylabel('Log mass')
        self.ax = ax
        self.ax.set_ylim(self.bottom, 0)
        self.ax.set_xlim(-1, h_bnd[-1]+1)
        self.show_bounds()

    def show_bounds(self):
        for bound in self.h_bnd:
            self.ax.axvline(x=bound, color='r')
        self.ax.set_xticks(self.h_bnd)

    @staticmethod
    def random_color():
        return tuple(scipy.stats.uniform(0,1).rvs(3))

    def show_sample(self, sample: Sample):
        self.add_sample(sample, style='bar', show_labels=True, show_raw=True)

    def show_class_dirichlet(self, cd: ClassDirichlet, color=None):
        if not color:
            color = self.random_color()
        self.add_sample(cd.mean_sample, style='bar', show_labels=True, show_raw=True, color=color)
        self.ax.set_title(f'$\\alpha = {cd.alpha.sum():.2f}$')

    def show_dirichlet_bars(self, cd: ClassDirichlet, color=None, legend=None):
        if not color:
            color = self.random_color()
        x_r = self.h_bnd[cd.full_alpha[::2] > 0]  - self.width/2
        x_l = self.h_bnd[:-1][cd.full_alpha[1::2]> 0] + self.width/2
        x = np.sort(np.concatenate((x_r, x_l)))
        y_lims = np.maximum(np.log(cd.errorbars()), self.bottom)
        u = cd.alpha / cd.alpha.sum()
        y_error = np.maximum( (y_lims-np.log(u))*np.array([[-1.],[1.]]), 0)
        # print(y_lims, np.log(u), y_error)
        self.ax.errorbar(x=x, y=np.log(u), yerr=y_error, fmt='x', capsize=8, zorder=1, color=color, label=legend)
        self.ax.legend()
        self.ax.set_ylim(self.bottom, 0)

    def show_mixed_dirichlet(self, md: MixedDirichlet):
        for cd in md.dirichlets:
            self.show_dirichlet_bars(cd)


    def show_dirichlet_plus_samples(self, ce: ClassEnsemble, cd: ClassDirichlet, legend=None):
        color = self.random_color()
        self.show_class_ensemble(ce, color)
        self.show_dirichlet_bars(cd, color, legend=legend)

    def show_mixed_dirichlet_plus_samples(self, en: Ensemble, md: MixedDirichlet):
        for ce, cd, mr in zip(en.class_ensembles, md.dirichlets, md.mixing_rates):
            legend = f'$n = {len(ce.samples)}, \\pi = {mr:.2f}$'
            self.show_dirichlet_plus_samples(ce, cd, legend=legend)

    def show_class_ensemble(self, ce: ClassEnsemble, color=None):
        if not color:
            color = self.random_color()
        for sample in ce.samples:
            self.add_sample(sample, style='scatter', color=color, dot_size=50)

    def show_ensemble(self, ensemble: Ensemble):
        for ce in ensemble.class_ensembles:
            self.show_class_ensemble(ce)


    def add_sample(self, sample, style='bar', color='b', show_labels=False, show_raw=False, dot_size=10):
        assert len(sample) == 2*len(self.h_bnd) - 1

        x_r = self.h_bnd[sample[::2] > 0]  - self.width/2
        x_l = self.h_bnd[:-1][sample[1::2]> 0] + self.width/2
        top_r = np.log(sample[::2][sample[::2] > 0])
        top_l = np.log(sample[1::2][sample[1::2] > 0])
        if style=='bar':
            bars_r = self.ax.bar(x=x_r, height=top_r-self.bottom, width=self.width, color=color, bottom=self.bottom)
            bars_l = self.ax.bar(x=x_l, height=top_l-self.bottom, width=self.width, color=color, bottom=self.bottom)
            bars = [b for pair in zip(bars_r[:-1], bars_l) for b in pair] + [bars_r[-1],]
        elif style=='scatter':
            self.ax.scatter(x=x_r, y=top_r, color=color, s=dot_size)
            self.ax.scatter(x=x_l, y=top_l, color=color, s=dot_size )
        else:
            raise ValueError(f'Invalid style={style}, must be bar or scatter')

        pretty_float = lambda f: f'{f:.1e}' if 0<f<0.001 else f'{f:.3f}'

        if show_labels:
            assert style=='bar', 'labels can only be shown on bar charts. Try show_labels=False or style="bar"'
            labels = np.vectorize(pretty_float)(sample[sample>0])
            for bar, label in zip(bars, labels):
                height = bar.get_height()
                self.ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, self.bottom + height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
                )

        if show_raw:
            for interval, l, r, in zip(self.h_bnd.intervals, sample[1::2], sample[2::2]):
                x_pos = sum(interval) / 2
                y_pos = self.bottom + 3
                a, v = l+r, interval[0]*l + interval[1]*r
                self.ax.annotate(
                    f'a={pretty_float(a)}\nv={pretty_float(v)}',
                    xy=(x_pos, y_pos),
                    ha='center',
                    va='bottom',
                    textcoords='offset points',
                    xytext=(0,5),
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
                )

        self.ax.set_ylim(self.bottom, 0)

## FUNCTIONS
# PROCESS TO DELTIZED FORM
def process_sample(raw_sample: RawSample, h_bnd: HeightBounds, threshold=1e-7) -> Sample:
    x = []
    for i, interval in enumerate(h_bnd.intervals):
        M = np.array([[1., 1.,],
                      interval  ])
        x += list(np.linalg.inv(M) @ np.array([raw_sample.area[i], raw_sample.volume[i]]))
    x.insert(0, 1-sum(x))  # first component is open water. How much isn't covered in ice?
    x = np.array(x)
    x[x < threshold] = 0.  # set small measurements to exactly zero
    x /= x.sum()  # renormalize to one
    assert np.isclose(x.sum(), 1., atol=1e-10)
    return Sample(x)

def process_ensemble(raw_ensemble: RawEnsemble, h_bnd: HeightBounds, threshold=1e-7) -> Ensemble:
    return Ensemble(samples=[process_sample(raw_sample, h_bnd, threshold) for raw_sample in raw_ensemble.samples])

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

# CONVERT BACK TO RAW FORM
def post_process_sample(sample: Sample, h_bnd: HeightBounds) -> RawSample:
    l, r = sample[1::2], sample[2::2]  # delta size on left and right of each interval
    a = l + r
    v = h_bnd[:-1]*l + h_bnd[1:]*r
    return RawSample(area=np.array(a), volume=np.array(v), snow=np.zeros_like(a))

def post_process_ensemble(ensemble: Ensemble, h_bnd: HeightBounds) -> RawEnsemble:
    raw_samples = []
    for sample in ensemble.samples:
        raw_samples.append(post_process_sample(sample, h_bnd=h_bnd))
    return RawEnsemble(samples=raw_samples)

# PUT IT ALL TOGETHER
def transport_ensemble(ensemble: Ensemble, observation: Observation):
    mixed_dirichlet = fit_mixed_dirichlet(ensemble)
    uniform_ensemble = uniformize_ensemble(ensemble, mixed_dirichlet)
    post_ensemble = update_ensemble(uniform_ensemble, mixed_dirichlet, observation=observation)
    return post_ensemble

def transport_raw_ensemble(raw_ensemble: RawEnsemble, h_bnd: HeightBounds, observation: Observation):
    ensemble = process_ensemble(raw_ensemble, h_bnd)
    post_ensemble = transport_ensemble(ensemble, observation=observation)
    post_raw_ensemble = post_process_ensemble(post_ensemble, h_bnd)
    return post_raw_ensemble




#%%

#%%
