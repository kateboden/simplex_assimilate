import numpy as np
import scipy
from simplex_assimilate import dirichlet
from scipy.optimize import linprog
# CHANGES import from _dirichlet
from simplex_assimilate._dirichlet import MixedDirichlet
from simplex_assimilate.utils.quantize import quantize


DEFAULT_ALPHA = 1e11

# This function is used to see if class members are different, if there is only one class member then diff = 0
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


def get_scale_factor(a0_prior, a0_posterior):
    try:
        return (1-a0_posterior)/(1-a0_prior)
    except ZeroDivisionError:
        return 1
    
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


# Kate addition, one ensemble member at a time
# Be careful to only use values that are non_zero
def dir_to_unif(sample, class_idx, alphas):
    """
    Given a single ensemble member, its class, and the alphas
    map the sample to a uniform distribution using the cdf of the marginal dirichlet distribution (beta)

    """
    # Initialize random uniform sample
    U = np.random.uniform(size=sample.shape)
    
    # Find location where the original sample is nonzero
    entry_idxs = np.where(sample)
    sample = sample[entry_idxs]
    
    # Build the beta pdfs
    if np.ndim(alphas) == 1:
        a = alphas[entry_idxs]
    else:
        a = alphas[class_idx][entry_idxs]
    b = a.sum() - a.cumsum()                       
    betas = scipy.stats.beta(a[:-1], b[:-1])
    remaining_mass = 1 - sample.cumsum()           
    rel_sample = (sample) / (sample + remaining_mass) 
    # Apply the beta cdf
    u = betas.cdf(rel_sample[:-1])
    # add a uniform at the end
    u = np.append(u, np.random.uniform(size=1))
    # Update the uniform sample
    U[entry_idxs] = u
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

 
def get_class_transition_matrix(pi, posterior, costs):
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
    
    # First check if the weights are the same- if so just make A the identity, no need for optimization
    if np.linalg.norm(pi-posterior) < 1e-12:
        return np.eye(n)

    A_eq = np.zeros((2 * n, n * n))
    for i in range(n):
        A_eq[i, i*n:(i+1)*n] = 1   # Constraints for A*1 = 1
        A_eq[n+i, i::n] = pi       # Constraints for pi^T*A = posterior
    b_eq = np.concatenate([np.ones(n), posterior])
    # Bounds for each variable in A to be between 0 and 1
    bounds = [(0, 1) for _ in range(n * n)]
    # Solve the linear programming problem
    result = linprog(np.ndarray.flatten(costs), A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds', options = {'presolve': True})
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
        return np.tile(posterior, (n,1))
        
# No longer using     
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


def get_post_class_idxs_pipeline(x0, classes, class_idx, alphas, pi):
    """
    Given the prior and the x0, get the posterior class indices

    >>> np.random.seed(1)
    >>> x0 = np.array([0.5, 0.45])
    >>> classes = np.array([[False, True, True], [True, True, False], [True, True, True]])
    >>> class_idx = 1
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
    
    # Implement the hamming distance given we start in class = class_idx
    n = classes.shape[0]
    costs = np.zeros((n,n))
    i = class_idx
    for j in range(n):
        costs[i,j] = scipy.spatial.distance.hamming(classes[i,:],classes[j,:])    

    A = get_class_transition_matrix(pi_x0, post, costs)
    post_class_idx = np.random.choice(len(A), p=A[class_idx])
    return post_class_idx


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
    X = np.zeros_like(samples)
    post_class_idxs = np.zeros_like(class_idxs)
    counts = 0                                                  # count the number of ens mems that change class
    scale_factor = np.full(samples.shape[0], np.nan)
    for n in range(samples.shape[0]):  # each ensemble member one at a time
        post_class_idxs[n] = get_post_class_idxs_pipeline(x0[n], classes, class_idxs[n], alphas, pi)
        # Check if the ensemble member changes class
        if post_class_idxs[n] == class_idxs[n]:                         # Scale
            X[n, 0] = x0[n,1]
            scale_factor[n] = get_scale_factor(x0[n,0], x0[n,1])
            X[n,1:] = scale_factor[n]*samples[n,1:]
        else:
            counts = counts + 1
            U_new = dir_to_unif(samples[n], class_idxs[n], alphas)                                                           # Class transport
            X[n,:] = invert_mixed_unifs(post_class_idxs[n], classes, alphas, U_new, x0[n,1])
    print(str(counts) + " ensemble members changed class")
    return X




