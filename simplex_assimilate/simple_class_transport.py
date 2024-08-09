import numpy as np
import scipy
from simplex_assimilate import dirichlet
from scipy.optimize import linprog
# CHANGES import from _dirichlet
from simplex_assimilate._dirichlet import MixedDirichlet
from simplex_assimilate.utils.quantize import quantize


DEFAULT_ALPHA = 1e11

def diff(class_samples):
    n = np.size(class_samples, 0)     # determine number of ensemble members in the class
    diff = 0                          # initialize difference to be 0
    i = 1                             # initialize looping index
    if n == 1:
        return 0
    else:
        while diff == 0 and i < n:              # If the difference is zero keep checking, if not return the difference- stop when you have checked all rows
            diff = class_samples[0]-class_samples[i]
            diff = np.linalg.norm(diff)
            i +=1
        return diff
            


def est_mixed_dirichlet(samples):
    """
    given (N, D) samples which may contain zeros
    estimate the parameters of a mixed dirichlet distribution (K, D)
    where K is the number of dirichlet components
    and each component may contain zeros
     
    >>> np.random.seed(1)
    >>> samples_1 = scipy.stats.dirichlet.rvs([1, 2, 3], size=5)
    >>> samples_1 = np.hstack([samples_1, np.zeros((5, 1))])
    >>> samples_2 = scipy.stats.dirichlet.rvs([3, 2, 1], size=10)
    >>> samples_2 = np.hstack([np.zeros((10, 1)), samples_2])
    >>> samples = np.vstack([samples_1, samples_2])
    >>> classes, class_idxs, alphas, pi = est_mixed_dirichlet(samples)
    >>> classes
    array([[False,  True,  True,  True],
           [ True,  True,  True, False]])
    >>> class_idxs
    array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> alphas
    array([[0.        , 3.92525376, 1.96737369, 1.5965454 ],
           [0.58799722, 1.91248973, 1.89843362, 0.        ]])
    >>> pi
    array([0.66666667, 0.33333333])
    """

    # first determine the classes
    N, D = samples.shape
    classes, class_idxs, class_counts = np.unique(samples > 0, axis=0, return_inverse=True, return_counts=True)
    pi = class_counts / N
    K = len(classes)
    print('There are', classes.shape[0] ,'classes')
    
    # then estimate the parameters for each class
    alphas = np.zeros_like(classes, dtype=np.float64)
    for i, c in enumerate(classes):
        class_samples = samples[class_idxs == i][:, c]
        Diff = diff(class_samples)
        # New code- logic to check that the there is separation between class samples (also that there is more than one class sample)
        if Diff == 0:
            print('Class ' + str(i) + ' has ' + str(class_counts[i]) + ' ensemble member(s) that are the same, cannot estimate magnitude of alpha. Using default alpha')
            alpha = np.tile(DEFAULT_ALPHA, np.size(class_samples,1))*class_samples.mean(axis = 0)
            alphas[i][c] = alpha  
        else: 
            alpha = dirichlet.mle(class_samples)
            alphas[i][c] = alpha
    return classes, class_idxs, alphas, pi


def unif_dirichlet_samples(samples, alpha):
    """
    given (N, E) samples from a dirichlet distribution with parameter alpha
    sequentially evaluate the inverse cdf of the marginal dirichlet distribution (beta)
    to map the samples to uniform rvs

    >>> np.random.seed(1)
    >>> samples = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    >>> alpha = np.array([1, 2, 3])
    >>> unif_dirichlet_samples(samples, alpha)
    array([[0.40951   , 0.21582076, 0.417022  ],
           [0.83193   , 0.57351104, 0.72032449]])
    """
    a = alpha # alpha weight per entry
    b = alpha.sum() - alpha.cumsum()  # remaining alpha weight
    betas = scipy.stats.beta(a[:-1], b[:-1])
    remaining_mass = 1 - samples.cumsum(axis=1) # remaining mass per entry
    rel_samples = (samples) / (samples + remaining_mass) # how much of the remaining mass is assigned to each entry
    u = betas.cdf(rel_samples[:, :-1])
    # stack a col of unifs at the end
    u = np.hstack([u, np.random.uniform(size=(samples.shape[0], 1))])
    return u

def unif_dirichlet_mixed_samples(samples, classes, class_idxs, alphas):
    """
    Given (N, D) samples from a mixed dirichlet distribution and their classes
    map the samples to a uniform distribution using the inverse cdf of the marginal dirichlet distribution (beta)

    >>> np.random.seed(10)
    >>> samples = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.1, 0.9, 0.0], [0.2, 0.8, 0.0]])
    >>> classes = np.array([[True, True, True], [True, True, False]])
    >>> class_idxs = np.array([0, 0, 1, 1])
    >>> alphas = np.array([[1, 2, 3], [3, 3, 0]])
    >>> unif_dirichlet_mixed_samples(samples, classes, class_idxs, alphas)
    array([[0.40951   , 0.21582076, 0.00394827],
           [0.83193   , 0.57351104, 0.51219226],
           [0.00856   , 0.81262096, 0.16911084],
           [0.05792   , 0.61252607, 0.95339335]])
    """
    U = np.random.uniform(size=samples.shape)
    for i, c in enumerate(classes):
        row_idxs = np.where(class_idxs == i)[0]
        col_idxs = np.where(c)[0]
        entry_idxs = np.ix_(row_idxs, col_idxs)
        class_samples = samples[entry_idxs]
        alpha = alphas[i, c]
        U[entry_idxs] = unif_dirichlet_samples(class_samples, alpha)
    return U


def get_class_log_likelihood(x0, classes, alphas):
    """
    likelihood of each class given the x0
    
    >>> np.random.seed(1)
    >>> classes = np.array([[False, True, True], [True, True, False], [True, True, True]])
    >>> alphas = np.array([[0, 1, 2], [3, 4, 0], [1, 2, 3]])
    >>> get_class_log_likelihood(0.5, classes, alphas)
    array([       -inf,  0.62860866, -1.16315081])
    """
    no_water_classes = classes[:, 0] == 0
    all_water_classes = classes[:, 1:].sum(axis=1) == 0
    if x0 == 0.0:
        return no_water_classes.astype(np.float64)
    if x0 == 1.0:
        return all_water_classes.astype(np.float64)
    compat_classes = np.logical_not(np.logical_or(no_water_classes, all_water_classes))
    a = alphas[compat_classes, 0]
    b = alphas[compat_classes, 1:].sum(axis=1)
    betas = scipy.stats.beta(a, b)                                             # K different beta distributions, one for each class
    compat_likelihoods = betas.logpdf(x0)
    likelihoods = np.zeros(len(classes), dtype=np.float64)
    likelihoods[:] = -np.inf
    likelihoods[compat_classes] = compat_likelihoods
    return likelihoods


def get_class_log_posterior(pi, log_likelihoods):
    return np.log(pi) + log_likelihoods

 
def get_class_transition_matrix(pi, posterior, costs=None):
    """
    Given the prior and posterior distributions, and a transition cost matrix,
    find the optimal transition matrix A that minimizes the cost of transitioning from the prior to the posterior distribution.

    >>> pri, post = np.array([0., 0.3, 0.3, 0.4]), np.array([0.1, 0., 0.6, 0.3])
    >>> A_solution = get_class_transition_matrix(pri, post)
    >>> A_solution
    array([[ 0.25,  0.25,  0.25,  0.25],
           [-0.  ,  0.  ,  1.  ,  0.  ],
           [ 0.  ,  0.  ,  1.  ,  0.  ],
           [ 0.25,  0.  ,  0.  ,  0.75]])
    """
    n = len(pi)
    if not costs:
        costs = np.ones(n * n)
        costs[::n+1] = 0
    # Equality constraints
    # A @ x = b
    # For A1 = p- and 1^T A = p+, we reshape A as a vector and construct the matrix
    A_eq = np.zeros((2 * n, n * n))
    for i in range(n):
        A_eq[i, i*n:(i+1)*n] = 1   # Constraints for A*1 = 1
        A_eq[n+i, i::n] = pi       # Constraints for pi^T*A = posterior
    b_eq = np.concatenate([np.ones(n), posterior])
    # Bounds for each variable in A to be between 0 and 1
    bounds = [(0, 1) for _ in range(n * n)]
    # Solve the linear programming problem
    result = linprog(costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds', options = {'presolve': True, 'autoscale': True})
    # Check if the optimization was successful
    if result.success:
        # Reshape the result back into a matrix form
        A_solution = abs(np.reshape(result.x, (n, n)))
        # normalize
        A_solution[A_solution.sum(axis=1) == 0] = 1 / n
        A_solution /= A_solution.sum(axis=1, keepdims=True)
        return A_solution
    else:
        print("Optimization failed:", result.message)
        print("norm(pi-posterior): ", np.linalg.norm(pi-posterior))
        return np.eye(n)
        
        
def get_post_class_idxs(class_idxs, transition_matrix):
    """
    Apply a transition matrix to the class indices to get the posterior class indices
    >>> np.random.seed(1)
    >>> class_idxs = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    >>> transition_matrix = np.array([[0.25,0.25,0.25,0.25],[-0.,0.,1.,0.],[0.,0.,1.,0.],[0.25,0.,0.,0.75]])
    >>> get_post_class_idxs(class_idxs, transition_matrix)
    array([2, 2, 2, 2, 2, 2, 0, 3, 3])
    """
    post_class_idxs = np.random.choice(len(transition_matrix), p=transition_matrix[class_idxs])
    return post_class_idxs
    
    
def invert_unifs(alpha, uniforms, obs=None):
    """
    map uniform rvs to dirichlet rvs using the inverse cdf of the marginal dirichlet distribution (beta)
    
    >>> alpha = np.array([1, 2, 3])
    >>> uniforms = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
    >>> invert_unifs(alpha, uniforms)
    array([[0.02085164, 0.20788997, 0.77125839],
           [0.36904266, 0.36750336, 0.26345398]])
    """
    X = np.zeros_like(uniforms, dtype=np.float64)
    if obs:
        X[0] = obs
    for i in range(1 if obs else 0, len(alpha) - 1):  # cutting the ribbon 
        a = alpha[i]
        b = alpha[i:].sum() - alpha[i]
        beta = scipy.stats.beta(a, b)
        remaining_mass = 1 - X[:i].sum()
        X[i] = beta.ppf(uniforms[i]) * remaining_mass
    X[-1] = 1 - X.sum()
    return X


def invert_mixed_unifs(post_class_idxs, classes, alphas, uniforms, obs=None):
    """
    Invert the uniform rvs for each class to get the posterior dirichlet rvs
    by applying the inverse cdf of the marginal dirichlet distribution (beta)

    >>> np.random.seed(1)
    >>> post_class_idxs = np.array([0, 0, 1])
    >>> classes = np.array([[True, True, True], [True, True, False]])
    >>> alphas = np.array([[1, 2, 3], [3, 3, 0]])
    >>> uniforms = np.array([[0.1, 0.1, 0.1], [0.9, 0.5, 0.5], [0.1, 0.1, 0.1]])
    >>> invert_mixed_unifs(post_class_idxs, classes, alphas, uniforms)
    array([[0.02085164, 0.13958672, 0.83956164],
           [0.36904266, 0.24337764, 0.3875797 ],
           [0.24663645, 0.75336355, 0.        ]])
    """
    X = np.zeros_like(uniforms, dtype=np.float64)
    cols = np.where(classes[post_class_idxs,:])[0]
    alpha = alphas[post_class_idxs,cols]
    X[cols] = invert_unifs(alpha, uniforms[cols], obs)
    return X


def get_post_class_idxs_pipeline(x0, classes, class_idxs, alphas, pi):
    """
    Given the prior and the x0, get the posterior class indices

    >>> np.random.seed(1)
    >>> x0 = 0.5
    >>> classes = np.array([[False, True, True], [True, True, False], [True, True, True]])
    >>> class_idxs = np.array([0, 0, 0, 1, 1, 1])
    >>> alphas = np.array([[0, 1, 2], [3, 4, 0], [1, 2, 3]])
    >>> pi = np.array([0.3, 0.3, 0.4])
    >>> get_post_class_idxs_pipeline(x0, classes, class_idxs, alphas, pi)
    array([1, 1, 1, 1, 1, 1])
    """
    # Conditioning the class weights on the prior x_0 (open water fraction)
    ll = get_class_log_likelihood(x0[0], classes, alphas)
    lp = get_class_log_posterior(pi, ll)
    lp -= lp.max()
    pi_x0 = np.exp(lp) / np.exp(lp).sum()
    
    # Conditioning the class weights on the posterior x_0 (open water fraction)
    ll = get_class_log_likelihood(x0[1], classes, alphas)
    lp = get_class_log_posterior(pi, ll)
    lp -= lp.max()
    post = np.exp(lp) / np.exp(lp).sum()
    
    A = get_class_transition_matrix(pi_x0, post)
    post_class_idxs = get_post_class_idxs(class_idxs, A)
    return post_class_idxs


def transport_pipeline(samples, x0):
    """
    Empirical Bayes transport pipeline
    
    >>> np.random.seed(1)
    >>> samples = np.array([[0.4, 0.3, 0.3], [0.3, 0.3, 0.4], [0.1, 0.9, 0.0], [0.2, 0.8, 0.0]])
    >>> x0 = 0.25
    >>> transport_pipeline(samples, x0)
    array([[0.25      , 0.75      , 0.        ],
           [0.25      , 0.32142857, 0.42857143],
           [0.25      , 0.75      , 0.        ],
           [0.25      , 0.75      , 0.        ]])
    """
    classes, class_idxs, alphas, pi = est_mixed_dirichlet(samples)
    U = unif_dirichlet_mixed_samples(samples, classes, class_idxs, alphas)
    X = np.zeros_like(samples)
    post_class_idxs = np.zeros_like(class_idxs)
    for n in range(samples.shape[0]):  # each ensemble member one at a time
        post_class_idxs[n] = get_post_class_idxs_pipeline(x0[n], classes, class_idxs[n], alphas, pi)
        X[n,:] = invert_mixed_unifs(post_class_idxs[n], classes, alphas, U[n,:], x0[n,1])
    return X
