import numpy as np
import scipy
import dirichlet
from scipy.optimize import linprog


def est_mixed_dirichlet(samples):
    # given (N, D) samples which may contain zeros
    # estimate the parameters of a mixed dirichlet distribution (K, D)
    # where K is the number of dirichlet components
    # and D is the dimensionality of the dirichlet components

    # first determine the classes
    print(samples)
    N, D = samples.shape
    classes, class_idxs, class_counts = np.unique(samples > 0, axis=0, return_inverse=True, return_counts=True)
    pi = class_counts / N
    K = len(classes)
    # then estimate the parameters for each class
    alphas = np.zeros_like(classes, dtype=np.float64)
    for i, c in enumerate(classes):
        class_samples = samples[class_idxs == i][:, c]
        alpha = dirichlet.mle(class_samples)
        alphas[i][c] = alpha
    return classes, class_idxs, alphas, pi
def unif_dirichlet_samples(samples, alpha):
    # given (N, E) samples from a dirichlet distribution with parameter alpha
    # sequentially evaluate the inverse cdf of the dirichlet distribution
    # to map the samples to uniform rvs
    # we want to use beta dists
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
    U = np.zeros_like(samples, dtype=np.float64)
    for i, c in enumerate(classes):
        row_idxs = np.where(class_idxs == i)[0]
        col_idxs = np.where(c)[0]
        entry_idxs = np.ix_(row_idxs, col_idxs)
        class_samples = samples[entry_idxs]
        alpha = alphas[i, c]
        U[entry_idxs] = unif_dirichlet_samples(class_samples, alpha)
    return U
def get_class_log_likelihood(observation, classes, alphas):
    no_water_classes = classes[:, 0] == 0
    all_water_classes = classes[:, 1:].sum(axis=1) == 0
    if observation == 0.0:
        return no_water_classes.astype(np.float64)
    if observation == 1.0:
        return all_water_classes.astype(np.float64)
    compat_classes = np.logical_not(np.logical_or(no_water_classes, all_water_classes))
    a = alphas[compat_classes, 0]
    b = alphas[compat_classes, 1:].sum(axis=1)
    betas = scipy.stats.beta(a, b)
    compat_likelihoods = betas.logpdf(observation)
    likelihoods = np.zeros(len(classes), dtype=np.float64)
    likelihoods[:] = -np.inf
    likelihoods[compat_classes] = compat_likelihoods
    return likelihoods
def get_class_log_posterior(pi, log_likelihoods):
    return np.log(pi) + log_likelihoods
def get_class_transition_matrix(pi, posterior, costs=None):
    n = len(pi)
    if not costs:
        costs = np.ones(n * n)
        costs[::n+1] = 0
    # Equality constraints
    # A @ x = b
    # For A1 = p- and 1^T A = p+, we reshape A as a vector and construct the matrix
    A_eq = np.zeros((2 * n, n * n))
    for i in range(n):
        A_eq[i, i*n:(i+1)*n] = 1  # Constraints for A1 = p-
        A_eq[n+i, i::n] = 1  # Constraints for 1^T A = p+
    b_eq = np.concatenate([pi, posterior])
    # Bounds for each variable in A to be greater than 0
    bounds = [(0, 1) for _ in range(n * n)]
    # Solve the linear programming problem
    result = linprog(costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    # Check if the optimization was successful
    if result.success:
        # Reshape the result back into a matrix form
        A_solution = np.reshape(result.x, (n, n))
        print("Optimized Matrix A:\n", A_solution)
        # normalize
        A_solution = A_solution / A_solution.sum(axis=1, keepdims=True)
        return A_solution
    else:
        print("Optimization failed:", result.message)
def get_post_class_idxs(class_idxs, transition_matrix):
    post_class_idxs = [np.random.choice(len(transition_matrix), p=transition_matrix[i]) for i in class_idxs]
    return post_class_idxs
def invert_unifs(alpha, uniforms):
    X = np.zeros_like(uniforms, dtype=np.float64)
    for i in range(len(alpha) - 1):
        a = alpha[i]
        b = alpha[i:].sum() - alpha[i]
        beta = scipy.stats.beta(a, b)
        remaining_mass = 1 - X[:, :i].sum(axis=1)
        X[:, i] = beta.ppf(uniforms[:, i]) * remaining_mass
    X[:, -1] = 1 - X.sum(axis=1)
    return X
def invert_mixed_unifs(post_class_idxs, classes, alphas, uniforms):
    X = np.zeros_like(uniforms, dtype=np.float64)
    for i, c in enumerate(classes):
        rows = np.where(post_class_idxs == i)[0]
        cols = np.where(c)[0]
        alpha = alphas[i, c]
        entry_idxs = np.ix_(rows, cols)
        print(rows, cols, entry_idxs, alpha)
        X[entry_idxs] = invert_unifs(alpha, uniforms[entry_idxs])
    return X

def get_post_class_idxs_pipeline(observation, classes, class_idxs, alphas, pi):
    ll = get_class_log_likelihood(observation, classes, alphas)
    lp = get_class_log_posterior(pi, ll)
    post = np.exp(lp) / np.exp(lp).sum()
    A = get_class_transition_matrix(pi, post)
    post_class_idxs = get_post_class_idxs(class_idxs, A)
    return post_class_idxs
def transport_pipeline(samples, observation):
    classes, class_idxs, alphas, pi = est_mixed_dirichlet(samples)
    print(f"classes: {classes}\nclass_idxs: {class_idxs}\nalphas: {alphas}\npi: {pi}")
    U = unif_dirichlet_mixed_samples(samples, classes, class_idxs, alphas)
    print(f"U: {U}")
    post_class_idxs = get_post_class_idxs_pipeline(observation, classes, class_idxs, alphas, pi)
    print(f"post_class_idxs: {post_class_idxs}")
    X = invert_mixed_unifs(post_class_idxs, classes, alphas, U)
    print(f"X: {X}")
    return X




