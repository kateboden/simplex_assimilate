from .shared_classes import ClassEnsemble, Ensemble, ClassDirichlet, MixedDirichlet
import numpy as np
import scipy

# DIRICHLET PARAMETER ESTIMATION
def fit_dirichlet(class_ensemble: ClassEnsemble) -> ClassDirichlet:
    samples = class_ensemble.samples
    assert len(samples)>1

    samples = np.array(class_ensemble.samples)[:, class_ensemble.sample_class]
    log_avg = np.log(samples).mean(axis=0)

    gammaln, digamma, polygamma = scipy.special.gammaln, scipy.special.digamma, scipy.special.polygamma
    f = lambda alpha: gammaln(alpha.sum()) - gammaln(alpha).sum() + (log_avg * (alpha - 1)).sum()  # likelihood
    alphas = [np.ones_like(log_avg)]  # initialize alpha_0
    for _ in range(15):
        alpha = alphas[-1]
        grad = digamma(alpha.sum()) - digamma(alpha) + log_avg
        hessian = - np.eye(len(alpha)) * (polygamma(1, alpha))
        hessian += polygamma(1, alpha.sum())
        invH = np.linalg.inv(hessian)
        da = - np.dot(invH, grad)
        alphas.append(alpha + da)
    return ClassDirichlet(alpha=alpha, sample_class=class_ensemble.sample_class)


def fit_mixed_dirichlet(ensemble: Ensemble) -> MixedDirichlet:
    dirichlets = [None for _ in ensemble.class_ensembles]
    for i, ce in enumerate(ensemble.class_ensembles):
        if len(ce.samples)>1:
            dirichlets[i] = fit_dirichlet(ce)
    max_s = max( [sum(d.alpha) for d in dirichlets if d is not None] )
    for i, ce in enumerate(ensemble.class_ensembles):
        if len(ce.samples)==1:
            alpha = ce.samples[0][ce.sample_class]*max_s
            dirichlets[i] = ClassDirichlet(alpha=alpha, sample_class=ce.sample_class)

    mixing_rates = np.array( [len(ce.samples)/len(ensemble.samples) for ce in ensemble.class_ensembles] )

    return MixedDirichlet(mixing_rates=mixing_rates, dirichlets=dirichlets)

